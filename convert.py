import hydra
import hydra.utils as utils

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

import soundfile as sf

from model_encoder import Encoder, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk
import os
import random

from glob import glob
import subprocess
from spectrogram import logmelspectrogram
import kaldiio

import resampy
import pyworld as pw

def select_wavs(paths, min_dur=2, max_dur=8):
    pp = []
    for p in paths:
        x, fs = sf.read(p)
        if len(x)/fs>=min_dur and len(x)/fs<=8:
            pp.append(p)
    return pp


def extract_logmel(wav_path, mean, std, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    #wav, _ = librosa.effects.trim(wav, top_db=15)
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=80,
                n_fft=400,
                n_shift=160,
                win_length=400,
                window='hann',
                fmin=80,
                fmax=7600,
            )
    mel = (mel - mean) / (std + 1e-8)
    tlen = mel.shape[0]
    frame_period = 160/fs*1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return mel, lf0

@hydra.main(config_path="config/convert.yaml")
def convert(cfg):
    src_wav_paths = glob('/Dataset/VCTK-Corpus/wav48_silence_trimmed/p225/*mic1.flac') # modified to absolute wavs path, can select any unseen speakers
    src_wav_paths = select_wavs(src_wav_paths)
    
    tar1_wav_paths = glob('/Dataset/VCTK-Corpus/wav48_silence_trimmed/p231/*mic1.flac') # can select any unseen speakers
    tar2_wav_paths = glob('/Dataset/VCTK-Corpus/wav48_silence_trimmed/p243/*mic1.flac') # can select any unseen speakers
    # tar1_wav_paths = select_wavs(tar1_wav_paths)
    # tar2_wav_paths = select_wavs(tar2_wav_paths)
    tar1_wav_paths = [sorted(tar1_wav_paths)[0]]
    tar2_wav_paths = [sorted(tar2_wav_paths)[0]]
    
    print('len(src):', len(src_wav_paths), 'len(tar1):', len(tar1_wav_paths), 'len(tar2):', len(tar2_wav_paths))

    tmp = cfg.checkpoint.split('/')
    steps = tmp[-1].split('-')[-1].split('.')[0]
    out_dir = f'test/{tmp[-3]}-{tmp[-2]}-{steps}'
    out_dir = Path(utils.to_absolute_path(out_dir))
    out_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    encoder_lf0 = Encoder_lf0()
    encoder_spk = Encoder_spk()
    decoder = Decoder_ac(dim_neck=64)
    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    decoder.to(device)

    print("Load checkpoint from: {}:".format(cfg.checkpoint))
    checkpoint_path = utils.to_absolute_path(cfg.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    encoder_spk.eval()
    decoder.eval()
    
    mel_stats = np.load('./data/mel_stats.npy')
    mean = mel_stats[0]
    std = mel_stats[1]
    feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir)+'/feats.1'))
    for i, src_wav_path in tqdm(enumerate(src_wav_paths, 1)):
        if i>10:
            break
        mel, lf0 = extract_logmel(src_wav_path, mean, std)
        if i % 2 == 1:
            ref_wav_path = random.choice(tar2_wav_paths)
            tar = 'tarMale_'
        else:
            ref_wav_path = random.choice(tar1_wav_paths)
            tar = 'tarFemale_'
        ref_mel, _ = extract_logmel(ref_wav_path, mean, std)
        
        mel = torch.FloatTensor(mel.T).unsqueeze(0).to(device)
        lf0 = torch.FloatTensor(lf0).unsqueeze(0).to(device)
        ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
        
        out_filename = os.path.basename(src_wav_path).split('.')[0] 
        with torch.no_grad():
            z, _, _, _ = encoder.encode(mel)
            lf0_embs = encoder_lf0(lf0)
            spk_embs = encoder_spk(ref_mel)
            output = decoder(z, lf0_embs, spk_embs)
            
            logmel = output.squeeze(0).cpu().numpy()
            feat_writer[out_filename] = logmel
            feat_writer[out_filename+'_src'] = mel.squeeze(0).cpu().numpy().T
            feat_writer[out_filename+'_ref'] = ref_mel.squeeze(0).cpu().numpy().T
            
        subprocess.call(['cp', src_wav_path, out_dir])
    
    feat_writer.close()
    print('synthesize waveform...')
    cmd = ['parallel-wavegan-decode', '--checkpoint', \
           '/vocoder/checkpoint-3000000steps.pkl', \
           '--feats-scp', f'{str(out_dir)}/feats.1.scp', '--outdir', str(out_dir)]
    subprocess.call(cmd)

if __name__ == "__main__":
    convert()
