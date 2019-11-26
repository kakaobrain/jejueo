#-*- coding: utf-8 -*-

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import os
from jamo import h2j
from itertools import chain

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    # load conversion dictionaries
    j2hcj, j2sj, j2shcj = load_j2hcj(), load_j2sj(), load_j2shcj()

    # Parse
    fpaths, text_lengths, texts = [], [], []
    transcript = os.path.join(hp.data, 'jss.v1.0.txt')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    if mode == "train":
        lines = lines[:-100]
    else:
        lines = lines[-100:]

    for line in lines:
        fname, text = line.strip().split("|")
        fpath = os.path.join(hp.data, fname)
        fpaths.append(fpath)

        text += "␃"  # ␃: EOS
        if hp.token_type == "char": # syllable
            text = list(text)
        else:
            text = [h2j(char) for char in text]
            text = chain.from_iterable(text)
            if hp.token_type == "j": # jamo
                text = [h2j(char) for char in text]
            elif hp.token_type == "sj":  # single jamo
                text = [j2sj.get(j, j) for j in text]
            elif hp.token_type == "hcj": # hangul compatibility jamo
                text = [j2hcj.get(j, j) for j in text]
            elif hp.token_type == "shcj": # single hangul compatibility jamo
                text = [j2shcj.get(j, j) for j in text]
        text = chain.from_iterable(text)

        text = [char2idx[char] for char in text if char in char2idx]
        text_lengths.append(len(text))
        if mode == "train":
            texts.append(np.array(text, np.int32).tostring())
        else:
            texts.append(text + [0]*(hp.max_N-len(text)))

    return fpaths, text_lengths, texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)
        print("maxlen=", maxlen, "minlen=", minlen)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        fname, mel, mag, t = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32, tf.int64])
        gt, = tf.py_func(guided_attention, [text_length, t], [tf.float32])

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft//2+1))
        gt.set_shape((hp.max_N, hp.max_T))

        # Batching
        _, (texts, mels, mags, gts, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, gt, fname],
                                            batch_size=hp.B,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 40)],
                                            num_threads=8,
                                            capacity=hp.B*10,
                                            dynamic_pad=True)

    return texts, mels, mags, gts, fnames, num_batch
