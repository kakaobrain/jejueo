# -*- coding: utf-8 -*-

from __future__ import print_function, division

import numpy as np
import librosa
import os, copy
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from scipy import signal

from hyperparams import Hyperparams as hp
import tensorflow as tf

def get_spectrograms(fpath):
    '''Parse the wave file in `fpath` and
    Returns normalized melspectrogram and linear spectrogram.

    Args:
      fpath: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) and dtype of float32.
      mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=hp.sr)

    # # Trimming
    # y, _ = librosa.effects.trim(y, top_db=40)

    # Preemphasis
    y = np.append(y[0], y[1:] - hp.preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=hp.n_fft,
                          hop_length=hp.hop_length,
                          win_length=hp.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(hp.sr, hp.n_fft, hp.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)
    mag = np.clip((mag - hp.ref_db + hp.max_db) / hp.max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram

    Args:
      mag: A numpy array of (T, 1+n_fft//2)

    Returns:
      wav: A 1-D numpy array.
    '''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * hp.max_db) - hp.max_db + hp.ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**hp.power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -hp.preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav, top_db=40)
    # wav = trim(wav)

    return wav.astype(np.float32)

def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(hp.n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y

def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def plot_alignment(alignment, gs, dir=hp.logdir):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir): os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(alignment)

    fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/alignment_{}.png'.format(dir, gs), format='png')

def guided_attention(n, t, g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    gt = np.ones((hp.max_N, hp.max_T), np.float32)
    for n_pos in range(n):
        for t_pos in range(t):
            gt[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(t) - n_pos / float(n)) ** 2 / (2 * g * g))

    return gt

def learning_rate_decay(init_lr, global_step, warmup_steps = 4000.0):
    '''Noam scheme from tensor2tensor'''
    step = tf.to_float(global_step + 1)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)

def load_spectrograms(fpath):
    '''Read the wave file in `fpath`
    and extracts spectrograms'''

    fname = os.path.basename(fpath)
    mel, mag = get_spectrograms(fpath)
    t = mel.shape[0]

    # Marginal padding for reduction shape sync.
    num_paddings = hp.r - (t % hp.r) if t % hp.r != 0 else 0
    mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
    mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

    # Reduction
    mel = mel[::hp.r, :]
    t = mel.shape[0]

    return fname, mel, mag, t

#This is adapted by
# https://github.com/keithito/tacotron/blob/master/util/audio.py#L55-62
def trim(wav, top_db=40, min_silence_sec=0.8):
    frame_length = int(hp.sr * min_silence_sec)
    hop_length = int(frame_length / 4)
    endpoint = librosa.effects.split(wav, frame_length=frame_length,
                               hop_length=hop_length,
                               top_db=top_db)[0, 1]
    return wav[:endpoint]

def load_j2hcj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)
    Returns:
      A dictionary that converts jamo into Hangul Compatibility Jamo(0x03130 - 0x0318F) Character
    '''
    j   = 'ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒᅌᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲ' \
      'ᅳᅴᅵᆨᆩᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂᆞ'
    hcj = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅇㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠ' \
          'ㅡㅢㅣㄱㄲㄴㄵㄶㄷㄹㄺㄻㄼㄾㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎㆍ'
    assert len(j) == len(hcj)
    j2hcj = {j_: hcj_ for j_, hcj_ in zip(j, hcj)}
    return j2hcj

def load_j2sj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)
    Returns:
      A dictionary that decomposes double consonants into two single consonants.
    '''
    j = 'ᄁᄄᄈᄊᄍᆩᆬᆭᆰᆱᆲᆴᆶᆹᆻ'
    sj = 'ᄀᄀ|ᄃᄃ|ᄇᄇ|ᄉᄉ|ᄌᄌ|ᆨᆨ|ᆫᆽ|ᆫᇂ|ᆯᆨ|ᆯᆷ|ᆯᆸ|ᆯᇀ|ᆯᇂ|ᆸᆺ|ᆺᆺ'
    assert len(j)==len(sj.split("|"))
    j2sj = {j_: sj_ for j_, sj_ in zip(j, sj.split("|"))}
    return j2sj

def load_j2shcj():
    '''
    Arg:
      jamo: A Hangul Jamo character(0x01100-0x011FF)
    Returns:
      A dictionary that converts jamo into Hangul Compatibility Jamo(0x03130 - 0x0318F) Character.
      Double consonants are further decomposed into single consonants.
    '''
    j   = 'ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑᄒᅌᅡᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲ' \
      'ᅳᅴᅵᆨᆩᆫᆬᆭᆮᆯᆰᆱᆲᆴᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂᆞ'
    shcj = 'ㄱ|ㄱㄱ|ㄴ|ㄷ|ㄷㄷ|ㄹ|ㅁ|ㅂ|ㅂㅂ|ㅅ|ㅅㅅ|ㅇ|ㅈ|ㅈㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ|ㅇ|ㅏ|ㅐ|ㅑ|ㅒ|ㅓ|ㅔ|ㅕ|ㅖ|ㅗ|ㅘ|ㅙ|ㅚ|ㅛ|ㅜ|ㅝ|ㅞ|ㅟ|ㅠ|' \
    'ㅡ|ㅢ|ㅣ|ㄱ|ㄱㄱ|ㄴ|ㄴㅈ|ㄴㅎ|ㄷ|ㄹ|ㄹㄱ|ㄹㅁ|ㄹㅂ|ㄹㅌ|ㄹㅎ|ㅁ|ㅂ|ㅂㅅ|ㅅ|ㅅㅅ|ㅇ|ㅈ|ㅊ|ㅋ|ㅌ|ㅍ|ㅎ|ㆍ'

    assert len(j)==len(shcj.split("|"))
    j2shcj = {j_: shcj_ for j_, shcj_ in zip(j, shcj.split("|"))}
    return j2shcj