# -*- coding: utf-8 -*-

from spectrogram import logmelspectrogram
import numpy as np
from joblib import Parallel, delayed
import librosa
import soundfile as sf
import os
from glob import glob
from tqdm import tqdm
import random
import json
import resampy
import pyworld as pw

def extract_logmel(wav_path, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    wav, _ = librosa.effects.trim(wav, top_db=60)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
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
    
    tlen = mel.shape[0]
    frame_period = 160/fs*1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    
    wav_name = os.path.basename(wav_path).split('.')[0]
    # print(wav_name, mel.shape, duration)
    return wav_name, mel, lf0, mel.shape[0]


def normalize_logmel(wav_name, mel, mean, std):
    mel = (mel - mean) / (std + 1e-8)
    return wav_name, mel


def save_one_file(save_path, arr):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)


def save_logmel(save_root, wav_name, melinfo, mode):
    mel, lf0, mel_len = melinfo
    spk = wav_name.split('_')[0]
    mel_save_path = f'{save_root}/{mode}/mels/{spk}/{wav_name}.npy'
    lf0_save_path = f'{save_root}/{mode}/lf0/{spk}/{wav_name}.npy'
    save_one_file(mel_save_path, mel)
    save_one_file(lf0_save_path, lf0)
    return mel_len, mel_save_path, lf0_save_path

# def get_wavs_names(spks, data_root)
data_root = '/Dataset/VCTK-Corpus/wav48_silence_trimmed'
save_root = 'data'
os.makedirs(save_root, exist_ok=True)

spk_info_txt = '/Dataset/VCTK-Corpus/speaker-info.txt'
f = open(spk_info_txt, 'r')
gen2spk = {}
all_spks = []
for i, line in enumerate(f):
    if i == 0:
        continue
    else:
        tmp = line.split()
        # print(tmp)
        spk = tmp[0]
        all_spks.append(spk)
        gen = tmp[2]
        if gen not in gen2spk:
            gen2spk[gen] = [spk]
        else:
            gen2spk[gen].append(spk)

random.shuffle(all_spks)
train_spks = all_spks[:-20]
test_spks = all_spks[-20:]

train_wavs_names = []
valid_wavs_names = []
test_wavs_names = []
    
print('all_spks:', all_spks)
for spk in train_spks:
    spk_wavs = glob(f'{data_root}/{spk}/*mic1.flac')
    print('len(spk_wavs):', len(spk_wavs))
    spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
    valid_names = random.sample(spk_wavs_names, int(len(spk_wavs_names)*0.1))
    train_names = [n for n in spk_wavs_names if n not in valid_names]
    train_wavs_names += train_names
    valid_wavs_names += valid_names
    
for spk in test_spks:
    spk_wavs = glob(f'{data_root}/{spk}/*mic1.flac')
    print('len(spk_wavs):', len(spk_wavs))
    spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
    test_wavs_names += spk_wavs_names
    
print(len(train_wavs_names))
print(len(valid_wavs_names))
print(len(test_wavs_names))
# extract log-mel
print('extract log-mel...')
all_wavs = glob(f'{data_root}/*/*mic1.flac')
results = Parallel(n_jobs=-1)(delayed(extract_logmel)(wav_path) for wav_path in tqdm(all_wavs))
wn2mel = {}
for r in results:
    wav_name, mel, lf0, mel_len = r
    # print(wav_name, mel.shape, duration)
    wn2mel[wav_name] = [mel, lf0, mel_len]

# normalize log-mel
print('normalize log-mel...')
mels = []
spk2lf0 = {}
for wav_name in train_wavs_names:
    mel, _, _ = wn2mel[wav_name]
    mels.append(mel)

mels = np.concatenate(mels, 0)
mean = np.mean(mels, 0)
std = np.std(mels, 0)
mel_stats = np.concatenate([mean.reshape(1,-1), std.reshape(1,-1)], 0)
np.save(f'{save_root}/mel_stats.npy', mel_stats)

results = Parallel(n_jobs=-1)(delayed(normalize_logmel)(wav_name, wn2mel[wav_name][0], mean, std) for wav_name in tqdm(wn2mel.keys()))
wn2mel_new = {}
for r in results:
    wav_name, mel = r
    lf0 = wn2mel[wav_name][1]
    mel_len = wn2mel[wav_name][2]
    wn2mel_new[wav_name] = [mel, lf0, mel_len]

# save log-mel
print('save log-mel...')
train_results = Parallel(n_jobs=-1)(delayed(save_logmel)(save_root, wav_name, wn2mel_new[wav_name], 'train') for wav_name in tqdm(train_wavs_names))
valid_results = Parallel(n_jobs=-1)(delayed(save_logmel)(save_root, wav_name, wn2mel_new[wav_name], 'valid') for wav_name in tqdm(valid_wavs_names))
test_results = Parallel(n_jobs=-1)(delayed(save_logmel)(save_root, wav_name, wn2mel_new[wav_name], 'test') for wav_name in tqdm(test_wavs_names))

def save_json(save_root, results, mode):
    fp = open(f'{save_root}/{mode}.json', 'w')
    json.dump(results, fp, indent=4)
    fp.close()
    
save_json(save_root, train_results, 'train')
save_json(save_root, valid_results, 'valid')
save_json(save_root, test_results, 'test')


    


