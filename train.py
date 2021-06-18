import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from dataset import CPCDataset_sameSeq as CPCDataset
from scheduler import WarmupScheduler
from model_encoder import Encoder, CPCLoss_sameSeq, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk
from mi_estimators import CLUBSample_group, CLUBSample_reshape

import apex.amp as amp
import os
import time

torch.manual_seed(137)
np.random.seed(137)

def save_checkpoint(encoder, encoder_lf0, cpc, encoder_spk, \
                    cs_mi_net, ps_mi_net, cp_mi_net, decoder, \
                    optimizer, optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net, scheduler, amp, epoch, checkpoint_dir, cfg):
    if cfg.use_amp:
        amp_state_dict = amp.state_dict()
    else:
        amp_state_dict = None
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "encoder_lf0": encoder_lf0.state_dict(),
        "cpc": cpc.state_dict(),
        "encoder_spk": encoder_spk.state_dict(),
        "ps_mi_net": ps_mi_net.state_dict(),
        "cp_mi_net": cp_mi_net.state_dict(),
        "cs_mi_net": cs_mi_net.state_dict(), 
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_cs_mi_net": optimizer_cs_mi_net.state_dict(),
        "optimizer_ps_mi_net": optimizer_ps_mi_net.state_dict(),
        "optimizer_cp_mi_net": optimizer_cp_mi_net.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp": amp_state_dict,
        "epoch": epoch
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))



def mi_first_forward(mels, lf0, encoder, encoder_lf0, encoder_spk, cs_mi_net, optimizer_cs_mi_net,
                     ps_mi_net, optimizer_ps_mi_net, cp_mi_net, optimizer_cp_mi_net, cfg):
    optimizer_cs_mi_net.zero_grad()
    optimizer_ps_mi_net.zero_grad()
    optimizer_cp_mi_net.zero_grad()
    z, _, _, _, _ = encoder(mels)
    z = z.detach()
    lf0_embs = encoder_lf0(lf0).detach()
    spk_embs = encoder_spk(mels).detach()
    if cfg.use_CSMI:
        lld_cs_loss = -cs_mi_net.loglikeli(spk_embs, z)
        if cfg.use_amp:
            with amp.scale_loss(lld_cs_loss, optimizer_cs_mi_net) as sl:
                sl.backward()
        else:
            lld_cs_loss.backward()
        optimizer_cs_mi_net.step()
    else:
        lld_cs_loss = torch.tensor(0.)
    
    if cfg.use_CPMI:
        lld_cp_loss = -cp_mi_net.loglikeli(lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0],-1,2,lf0_embs.shape[-1]).mean(2), z)
        if cfg.use_amp:
            with amp.scale_loss(lld_cp_loss, optimizer_cp_mi_net) as slll:
                slll.backward()
        else:
            lld_cp_loss.backward()
        torch.nn.utils.clip_grad_norm_(cp_mi_net.parameters(), 1)
        optimizer_cp_mi_net.step()
    else:
        lld_cp_loss = torch.tensor(0.)
        
    if cfg.use_PSMI:
        lld_ps_loss = -ps_mi_net.loglikeli(spk_embs, lf0_embs)
        if cfg.use_amp:
            with amp.scale_loss(lld_ps_loss, optimizer_ps_mi_net) as sll:
                sll.backward()
        else:
            lld_ps_loss.backward()
        optimizer_ps_mi_net.step()
    else:
        lld_ps_loss = torch.tensor(0.)
            
    return optimizer_cs_mi_net, lld_cs_loss, optimizer_ps_mi_net, lld_ps_loss, optimizer_cp_mi_net, lld_cp_loss


def mi_second_forward(mels, lf0, encoder, encoder_lf0, cpc, encoder_spk, cs_mi_net, ps_mi_net, cp_mi_net, decoder, cfg, optimizer, scheduler):
    optimizer.zero_grad()
    z, c, _, vq_loss, perplexity = encoder(mels)
    cpc_loss, accuracy = cpc(z, c)
    spk_embs = encoder_spk(mels)
    lf0_embs = encoder_lf0(lf0)
    recon_loss, pred_mels = decoder(z, lf0_embs, spk_embs, mels.transpose(1,2))
    
    loss = recon_loss + cpc_loss + vq_loss
    
    if cfg.use_CSMI:
        mi_cs_loss = cfg.mi_weight*cs_mi_net.mi_est(spk_embs, z)
    else:
        mi_cs_loss = torch.tensor(0.).to(loss.device)
    
    if cfg.use_CPMI:
        mi_cp_loss = cfg.mi_weight*cp_mi_net.mi_est(lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0],-1,2,lf0_embs.shape[-1]).mean(2), z)
    else:
        mi_cp_loss = torch.tensor(0.).to(loss.device)
        
    if cfg.use_PSMI:
        mi_ps_loss = cfg.mi_weight*ps_mi_net.mi_est(spk_embs, lf0_embs)
    else:
        mi_ps_loss = torch.tensor(0.).to(loss.device)
    
    loss = loss + mi_cs_loss + mi_ps_loss + mi_cp_loss
    
    if cfg.use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    
    optimizer.step()
    return optimizer, recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, mi_ps_loss, mi_cp_loss


def calculate_eval_loss(mels, lf0, \
                        encoder, encoder_lf0, cpc, \
                        encoder_spk, cs_mi_net, ps_mi_net, \
                        cp_mi_net, decoder, cfg):
    with torch.no_grad():
        z, c, z_beforeVQ, vq_loss, perplexity = encoder(mels)
        c = c
        lf0_embs = encoder_lf0(lf0)
        spk_embs = encoder_spk(mels)
        
        if cfg.use_CSMI:
            lld_cs_loss = -cs_mi_net.loglikeli(spk_embs, z)
            mi_cs_loss = cfg.mi_weight*cs_mi_net.mi_est(spk_embs, z)
        else:
            lld_cs_loss = torch.tensor(0.)
            mi_cs_loss = torch.tensor(0.)
        
        # z, c, z_beforeVQ, vq_loss, perplexity = encoder(mels)
        cpc_loss, accuracy = cpc(z, c)
        recon_loss, pred_mels = decoder(z, lf0_embs, spk_embs, mels.transpose(1,2))
        
        if cfg.use_CPMI:
            mi_cp_loss = cfg.mi_weight*cp_mi_net.mi_est(lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0],-1,2,lf0_embs.shape[-1]).mean(2), z)
            lld_cp_loss = -cp_mi_net.loglikeli(lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0],-1,2,lf0_embs.shape[-1]).mean(2), z)
        else:
            mi_cp_loss = torch.tensor(0.)
            lld_cp_loss = torch.tensor(0.)
            
        if cfg.use_PSMI:
            mi_ps_loss = cfg.mi_weight*ps_mi_net.mi_est(spk_embs, lf0_embs)
            lld_ps_loss = -ps_mi_net.loglikeli(spk_embs, lf0_embs)
        else:
            mi_ps_loss = torch.tensor(0.)
            lld_ps_loss = torch.tensor(0.)
            
        return recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, lld_cs_loss, mi_ps_loss, lld_ps_loss, mi_cp_loss, lld_cp_loss


def to_eval(all_models):
    for m in all_models:
        m.eval()
        
        
def to_train(all_models):
    for m in all_models:
        m.train()
        
        
def eval_model(epoch, checkpoint_dir, device, valid_dataloader, encoder, encoder_lf0, cpc, encoder_spk, cs_mi_net, ps_mi_net, cp_mi_net, decoder, cfg):
    stime = time.time()
    average_cpc_loss = average_vq_loss = average_perplexity = average_recon_loss = 0
    average_accuracies = np.zeros(cfg.training.n_prediction_steps)
    average_lld_cs_loss = average_mi_cs_loss = average_lld_ps_loss = average_mi_ps_loss = average_lld_cp_loss = average_mi_cp_loss = 0
    all_models = [encoder, encoder_lf0, cpc, encoder_spk, cs_mi_net, ps_mi_net, cp_mi_net, decoder]
    to_eval(all_models)
    for i, (mels, lf0, speakers) in enumerate(valid_dataloader, 1):
        lf0 = lf0.to(device)
        mels = mels.to(device) # (bs, 80, 128)
        recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, lld_cs_loss, mi_ps_loss, lld_ps_loss, mi_cp_loss, lld_cp_loss = \
            calculate_eval_loss(mels, lf0, \
                        encoder, encoder_lf0, cpc, \
                        encoder_spk, cs_mi_net, ps_mi_net, \
                        cp_mi_net, decoder, cfg)
       
        average_recon_loss += (recon_loss.item() - average_recon_loss) / i
        average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
        average_vq_loss += (vq_loss.item() - average_vq_loss) / i
        average_perplexity += (perplexity.item() - average_perplexity) / i
        average_accuracies += (np.array(accuracy) - average_accuracies) / i
        average_lld_cs_loss += (lld_cs_loss.item() - average_lld_cs_loss) / i
        average_mi_cs_loss += (mi_cs_loss.item() - average_mi_cs_loss) / i
        average_lld_ps_loss += (lld_ps_loss.item() - average_lld_ps_loss) / i
        average_mi_ps_loss += (mi_ps_loss.item() - average_mi_ps_loss) / i
        average_lld_cp_loss += (lld_cp_loss.item() - average_lld_cp_loss) / i
        average_mi_cp_loss += (mi_cp_loss.item() - average_mi_cp_loss) / i
        
    
    ctime = time.time()
    print("Eval | epoch:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}, used time:{:.3f}s"
          .format(epoch, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss, average_lld_cp_loss, average_mi_cp_loss, ctime-stime))
    print(100 * average_accuracies)
    results_txt = open(f'{str(checkpoint_dir)}/results.txt', 'a')
    results_txt.write("Eval | epoch:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}"
          .format(epoch, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss, average_lld_cp_loss, average_mi_cp_loss)+'\n')
    results_txt.write(' '.join([str(cpc_acc) for cpc_acc in average_accuracies])+'\n')
    results_txt.close()
    
    to_train(all_models)
    
    
@hydra.main(config_path="config/train.yaml")
def train_model(cfg):
    cfg.checkpoint_dir = f'{cfg.checkpoint_dir}/useCSMI{cfg.use_CSMI}_useCPMI{cfg.use_CPMI}_usePSMI{cfg.use_PSMI}_useAmp{cfg.use_amp}'
    if cfg.encoder_lf0_type == 'no_emb': # default
        dim_lf0 = 1
    else:
        dim_lf0 = 64
    
    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # define model
    encoder = Encoder(**cfg.model.encoder)
    encoder_lf0 = Encoder_lf0(cfg.encoder_lf0_type)
    cpc = CPCLoss_sameSeq(**cfg.model.cpc)
    encoder_spk = Encoder_spk()
    cs_mi_net = CLUBSample_group(256, cfg.model.encoder.z_dim, 512)
    ps_mi_net = CLUBSample_group(256, dim_lf0, 512)
    cp_mi_net = CLUBSample_reshape(dim_lf0, cfg.model.encoder.z_dim, 512)
    decoder = Decoder_ac(dim_neck=cfg.model.encoder.z_dim, dim_lf0=dim_lf0, use_l1_loss=True)
    
    encoder.to(device)
    cpc.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    cs_mi_net.to(device)
    ps_mi_net.to(device)
    cp_mi_net.to(device)
    decoder.to(device)

    optimizer = optim.Adam(
        chain(encoder.parameters(), encoder_lf0.parameters(), cpc.parameters(), encoder_spk.parameters(), decoder.parameters()),
        lr=cfg.training.scheduler.initial_lr)
    optimizer_cs_mi_net = optim.Adam(cs_mi_net.parameters(), lr=cfg.mi_lr)
    optimizer_ps_mi_net = optim.Adam(ps_mi_net.parameters(), lr=cfg.mi_lr)
    optimizer_cp_mi_net = optim.Adam(cp_mi_net.parameters(), lr=cfg.mi_lr)
    # TODO: use_amp is set default to True to speed up training; no-amp -> more stable training? => need to be verified
    if cfg.use_amp: 
        [encoder, encoder_lf0, cpc, encoder_spk, decoder], optimizer = amp.initialize([encoder, encoder_lf0, cpc, encoder_spk, decoder], optimizer, opt_level='O1')
        [cs_mi_net], optimizer_cs_mi_net = amp.initialize([cs_mi_net], optimizer_cs_mi_net, opt_level='O1')
        [ps_mi_net], optimizer_ps_mi_net = amp.initialize([ps_mi_net], optimizer_ps_mi_net, opt_level='O1')
        [cp_mi_net], optimizer_cp_mi_net = amp.initialize([cp_mi_net], optimizer_cp_mi_net, opt_level='O1')
    
    root_path = Path(utils.to_absolute_path("data"))
    dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames, # 128
        mode='train')
    valid_dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames, # 128
        mode='valid')
    
    warmup_epochs = 2000 // (len(dataset)//cfg.training.batch_size)
    print('warmup_epochs:', warmup_epochs)
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        initial_lr=cfg.training.scheduler.initial_lr,
        max_lr=cfg.training.scheduler.max_lr,
        milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)
    
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size, # 256
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=False)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.training.batch_size, # 256
        shuffle=False,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=False)
    
    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
        cpc.load_state_dict(checkpoint["cpc"])
        encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        cs_mi_net.load_state_dict(checkpoint["cs_mi_net"])
        ps_mi_net.load_state_dict(checkpoint["ps_mi_net"])
        if cfg.use_CPMI:
            cp_mi_net.load_state_dict(checkpoint["cp_mi_net"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_cs_mi_net.load_state_dict(checkpoint["optimizer_cs_mi_net"])
        optimizer_ps_mi_net.load_state_dict(checkpoint["optimizer_ps_mi_net"])
        optimizer_cp_mi_net.load_state_dict(checkpoint["optimizer_cp_mi_net"])
        if cfg.use_amp:
            amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 1
    
    if os.path.exists(f'{str(checkpoint_dir)}/results.txt'):
        wmode = 'a'
    else:
        wmode = 'w'
    results_txt = open(f'{str(checkpoint_dir)}/results.txt', wmode)
    results_txt.write('save training info...\n')
    results_txt.close()
    
    global_step = 0
    stime = time.time()
    for epoch in range(start_epoch, cfg.training.n_epochs + 1):
        average_cpc_loss = average_vq_loss = average_perplexity = average_recon_loss = 0
        average_accuracies = np.zeros(cfg.training.n_prediction_steps)
        average_lld_cs_loss = average_mi_cs_loss = average_lld_ps_loss = average_mi_ps_loss = average_lld_cp_loss = average_mi_cp_loss = 0

        for i, (mels, lf0, speakers) in enumerate(dataloader, 1):
            lf0 = lf0.to(device)
            mels = mels.to(device) # (bs, 80, 128)
            if cfg.use_CSMI or cfg.use_CPMI or cfg.use_PSMI:
                for j in range(cfg.mi_iters):
                    optimizer_cs_mi_net, lld_cs_loss, optimizer_ps_mi_net, lld_ps_loss, optimizer_cp_mi_net, lld_cp_loss = mi_first_forward(mels, lf0, encoder, encoder_lf0, encoder_spk, cs_mi_net, optimizer_cs_mi_net, \
                                                               ps_mi_net, optimizer_ps_mi_net, cp_mi_net, optimizer_cp_mi_net, cfg)
            else:
                lld_cs_loss = torch.tensor(0.)
                lld_ps_loss = torch.tensor(0.)
                lld_cp_loss = torch.tensor(0.)
                
            optimizer, recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, mi_ps_loss, mi_cp_loss = mi_second_forward(mels, lf0, \
                                                                                                                                encoder, encoder_lf0, cpc, \
                                                                                                                                encoder_spk, cs_mi_net, ps_mi_net, \
                                                                                                                                cp_mi_net, decoder, cfg, \
                                                                                                                                optimizer, scheduler)
           
            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i
            average_accuracies += (np.array(accuracy) - average_accuracies) / i
            average_lld_cs_loss += (lld_cs_loss.item() - average_lld_cs_loss) / i
            average_mi_cs_loss += (mi_cs_loss.item() - average_mi_cs_loss) / i
            average_lld_ps_loss += (lld_ps_loss.item() - average_lld_ps_loss) / i
            average_mi_ps_loss += (mi_ps_loss.item() - average_mi_ps_loss) / i
            average_lld_cp_loss += (lld_cp_loss.item() - average_lld_cp_loss) / i
            average_mi_cp_loss += (mi_cp_loss.item() - average_mi_cp_loss) / i
            
            
            ctime = time.time()
            print("epoch:{}, global step:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}, used time:{:.3f}s"
                  .format(epoch, global_step, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss, average_lld_cp_loss, average_mi_cp_loss, ctime-stime))
            print(100 * average_accuracies)
            stime = time.time()
            global_step += 1
            # scheduler.step()
            
        results_txt = open(f'{str(checkpoint_dir)}/results.txt', 'a')
        results_txt.write("epoch:{}, global step:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}"
              .format(epoch, global_step, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss, average_lld_cp_loss, average_mi_cp_loss)+'\n')
        results_txt.write(' '.join([str(cpc_acc) for cpc_acc in average_accuracies])+'\n')
        results_txt.close()
        scheduler.step()
        
        
        if epoch % cfg.training.log_interval == 0 and epoch != start_epoch:
            eval_model(epoch, checkpoint_dir, device, valid_dataloader, encoder, encoder_lf0, cpc, encoder_spk, cs_mi_net, ps_mi_net, cp_mi_net, decoder, cfg)

            ctime = time.time()
            print("epoch:{}, global step:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}, used time:{:.3f}s"
                  .format(epoch, global_step, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss, average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss, average_lld_cp_loss, average_mi_cp_loss, ctime-stime))
            print(100 * average_accuracies)
            stime = time.time()
            
        if epoch % cfg.training.checkpoint_interval == 0 and epoch != start_epoch:
            save_checkpoint(encoder, encoder_lf0, cpc, encoder_spk, \
                            cs_mi_net, ps_mi_net, cp_mi_net, decoder, \
                            optimizer, optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net, scheduler, amp, epoch, checkpoint_dir, cfg)


if __name__ == "__main__":
    train_model()
