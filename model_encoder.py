import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder_lf0(nn.Module):
    def __init__(self, typ='no_emb'):
        super(Encoder_lf0, self).__init__()
        self.type = typ
        if typ != 'no_emb':
            convolutions = []
            for i in range(3):
                conv_layer = nn.Sequential(
                    ConvNorm(1 if i==0 else 256, 256,
                             kernel_size=5, stride=2 if i==2 else 1,
                             padding=2,
                             dilation=1, w_init_gain='relu'),
                    nn.GroupNorm(256//16, 256),
                    nn.ReLU())
                convolutions.append(conv_layer)
            self.convolutions = nn.ModuleList(convolutions)
            self.lstm = nn.LSTM(256, 32, 1, batch_first=True, bidirectional=True)

    def forward(self, lf0):
        if self.type != 'no_emb':
            if len(lf0.shape) == 2:
                lf0 = lf0.unsqueeze(1) # bz x 1 x 128
            for conv in self.convolutions:
                lf0 = conv(lf0) # bz x 256 x 128
            lf0 = lf0.transpose(1,2) # bz x 64 x 256
            self.lstm.flatten_parameters()
            lf0, _ = self.lstm(lf0) # bz x 64 x 64
        else:
            if len(lf0.shape) == 2:
                lf0 = lf0.unsqueeze(-1) # bz x 128 x 1 # no downsampling
        return lf0



def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()


class SpeakerEncoder(nn.Module):
    '''
    reference from speaker-encoder of AdaIN-VC: https://github.com/jjery2243542/adaptive_voice_conversion/blob/master/model.py
    '''
    def __init__(self, c_in=80, c_h=128, c_out=256, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=128, 
            n_conv_blocks=6, n_dense_blocks=6, 
            subsample=[1, 2, 1, 2, 1, 2], act='relu', dropout_rate=0):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp):
        out = inp
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        # dense layers
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)
        # avg pooling
        out = self.pooling_layer(out).squeeze(2)
        # dense blocks
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out



class Encoder(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''
    def __init__(self, in_channels, channels, n_embeddings, z_dim, c_dim):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels, channels, 4, 2, 1, bias=False)
        self.encoder = nn.Sequential(
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, channels, bias=False),
            nn.LayerNorm(channels),
            nn.ReLU(True),
            nn.Linear(channels, z_dim),
        )
        self.codebook = VQEmbeddingEMA(n_embeddings, z_dim)
        self.rnn = nn.LSTM(z_dim, c_dim, batch_first=True)

    def encode(self, mel):
        z = self.conv(mel)
        z_beforeVQ = self.encoder(z.transpose(1, 2))
        z, r, indices = self.codebook.encode(z_beforeVQ)
        c, _ = self.rnn(z)
        return z, c, z_beforeVQ, indices

    def forward(self, mels):
        z = self.conv(mels.float()) # (bz, 80, 128) -> (bz, 512, 128/2)
        z_beforeVQ = self.encoder(z.transpose(1, 2)) # (bz, 512, 128/2) -> (bz, 128/2, 512) -> (bz, 128/2, 64)
        z, r, loss, perplexity = self.codebook(z_beforeVQ) # z: (bz, 128/2, 64)
        c, _ = self.rnn(z) # (64, 140/2, 64) -> (64, 140/2, 256)
        return z, c, z_beforeVQ, loss, perplexity
    


class VQEmbeddingEMA(nn.Module):
    '''
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / 512
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding) # only change during forward
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        residual = x - quantized
        return quantized, residual, indices.view(x.size(0), x.size(1))

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0) # calculate the distance between each ele in embedding and x

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training: # EMA based codebook learning
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss
        
        residual = x - quantized
        
        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, residual, loss, perplexity


class CPCLoss(nn.Module):
    '''
    CPC-loss calculation: negative samples are drawn within-speaker
    reference from: https://github.com/bshall/VectorQuantizedCPC/blob/master/model.py
    '''
    def __init__(self, n_speakers_per_batch, n_utterances_per_speaker, n_prediction_steps, n_negatives, z_dim, c_dim):
        super(CPCLoss, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps // 2
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c): # z:(64, 70, 64), c:(64, 70, 256)
        length = z.size(1) - self.n_prediction_steps # 64

        z = z.reshape(
            self.n_speakers_per_batch,
            self.n_utterances_per_speaker,
            -1,
            self.z_dim
        ) # (64, 70, 64) -> (8, 8, 70, 64)
        c = c[:, :-self.n_prediction_steps, :] # (64, 64, 256)

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, :, k:length + k, :] # (8, 8, 64, 64), positive samples

            Wc = self.predictors[k-1](c) # (64, 64, 256) -> (64, 64, 64)
            Wc = Wc.view(
                self.n_speakers_per_batch,
                self.n_utterances_per_speaker,
                -1,
                self.z_dim
            ) # (64, 64, 64) -> (8, 8, 64, 64)

            batch_index = torch.randint(
                0, self.n_utterances_per_speaker,
                size=(
                    self.n_utterances_per_speaker,
                    self.n_negatives
                ),
                device=z.device
            )
            batch_index = batch_index.view(
                1, self.n_utterances_per_speaker, self.n_negatives, 1
            ) # (1, 8, 17, 1)

            # seq_index: (8, 8, 17, 64)
            seq_index = torch.randint(
                1, length,
                size=(
                    self.n_speakers_per_batch,
                    self.n_utterances_per_speaker,
                    self.n_negatives,
                    length
                ),
                device=z.device
            ) 
            seq_index += torch.arange(length, device=z.device) #(1)
            seq_index = torch.remainder(seq_index, length) #(2) (1)+(2) ensures that the current positive frame will not be selected as negative sample...
            
            speaker_index = torch.arange(self.n_speakers_per_batch, device=z.device) # within-speaker sampling
            speaker_index = speaker_index.view(-1, 1, 1, 1)
            
            # z_negatives: (8,8,17,64,64); z_negatives[0,0,:,0,:] is (17, 64) that is negative samples for first frame of first utterance of first speaker...
            z_negatives = z_shift[speaker_index, batch_index, seq_index, :] # speaker_index has the original order (within-speaker sampling)
                                                                            # batch_index is randomly sampled from 0~7, each point has 17 negative samples
                                                                            # seq_index is randomly sampled from 0~115
                                                                        # so for each positive frame with time-id as t, the negative samples will be selected from 
                                                                        # another or the current utterance and the seq-index (frame-index) will not conclude t  

            zs = torch.cat((z_shift.unsqueeze(2), z_negatives), dim=2) # (8, 8, 1+17, 64, 64)

            f = torch.sum(zs * Wc.unsqueeze(2) / math.sqrt(self.z_dim), dim=-1) # (8, 8, 1+17, 64), vector product in fact...
            f = f.view(
                self.n_speakers_per_batch * self.n_utterances_per_speaker,
                self.n_negatives + 1,
                -1
            ) # (64, 1+17, 64)

            labels = torch.zeros(
                self.n_speakers_per_batch * self.n_utterances_per_speaker, length,
                dtype=torch.long, device=z.device
            ) # (64, 64)

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels # (64, 116)
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies


class CPCLoss_sameSeq(nn.Module):
    '''
    CPC-loss calculation: negative samples are drawn within-sequence/utterance
    '''
    def __init__(self, n_speakers_per_batch, n_utterances_per_speaker, n_prediction_steps, n_negatives, z_dim, c_dim):
        super(CPCLoss_sameSeq, self).__init__()
        self.n_speakers_per_batch = n_speakers_per_batch
        self.n_utterances_per_speaker = n_utterances_per_speaker
        self.n_prediction_steps = n_prediction_steps 
        self.n_negatives = n_negatives
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.predictors = nn.ModuleList([
            nn.Linear(c_dim, z_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, z, c): # z:(256, 64, 64), c:(256, 64, 256)
        length = z.size(1) - self.n_prediction_steps # 64-6=58, length is the total time-steps of each utterance used for calculated cpc loss
        n_speakers_per_batch = z.shape[0] # each utterance is treated as a speaker
        c = c[:, :-self.n_prediction_steps, :] # (256, 58, 256)

        losses, accuracies = list(), list()
        for k in range(1, self.n_prediction_steps+1):
            z_shift = z[:, k:length + k, :] # (256, 58, 64), positive samples

            Wc = self.predictors[k-1](c) # (256, 58, 256) -> (256, 58, 64)

            # seq_index: (256, 10, 58)
            seq_index = torch.randint(
                1, length,
                size=(
                    n_speakers_per_batch,
                    self.n_negatives,
                    length
                ),
                device=z.device
            ) 
            seq_index += torch.arange(length, device=z.device) #(1)
            seq_index = torch.remainder(seq_index, length) #(2) (1)+(2) ensures that the current positive frame will not be selected as negative sample...
            
            speaker_index = torch.arange(n_speakers_per_batch, device=z.device) # within-utterance sampling
            speaker_index = speaker_index.view(-1, 1, 1)
            
            
            z_negatives = z_shift[speaker_index, seq_index, :] # (256,10,58,64), z_negatives[i,:,j,:] is the negative samples set for ith utterance and jth time-step

            zs = torch.cat((z_shift.unsqueeze(1), z_negatives), dim=1) # (256,11,58,64) 

            f = torch.sum(zs * Wc.unsqueeze(1) / math.sqrt(self.z_dim), dim=-1) # (256,11,58), vector product in fact...
            
            labels = torch.zeros(
                n_speakers_per_batch, length,
                dtype=torch.long, device=z.device
            ) 

            loss = F.cross_entropy(f, labels)

            accuracy = f.argmax(dim=1) == labels # (256, 58)
            accuracy = torch.mean(accuracy.float())

            losses.append(loss)
            accuracies.append(accuracy.item())

        loss = torch.stack(losses).mean()
        return loss, accuracies
    


