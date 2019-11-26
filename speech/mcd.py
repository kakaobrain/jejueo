#-*- coding: utf-8 -*-

from tqdm import tqdm
import soundfile as sf
import numpy as np
import pysptk
import pyworld
from nnmnkwii.preprocessing.alignment import DTWAligner
import nnmnkwii.metrics

aligner = DTWAligner()

def get_mc(wav):
    y, sr = sf.read(wav)
    y = y.astype(np.float64)
    f0, timeaxis = pyworld.dio(y, sr, frame_period=5)
    f0 = pyworld.stonemask(y, f0, timeaxis, sr)
    spectrogram = pyworld.cheaptrick(y, f0, timeaxis, sr)
    mc = pysptk.sp2mc(spectrogram, order=24, alpha=0.41)
    mc = mc.astype(np.float32)

    return mc


def get_mcd(inp, ref):
    # extract mc
    inp_mc = get_mc(inp)
    ref_mc = get_mc(ref)

    # alignment
    inp = np.expand_dims(inp_mc, 0) # rank=3
    ref = np.expand_dims(ref_mc, 0) # rank=3

    inp_aligned, ref_aligned = aligner.transform((inp, ref))

    inp_aligned = np.squeeze(inp_aligned)
    ref_aligned = np.squeeze(ref_aligned)

    # calc mcd
    mcd = nnmnkwii.metrics.melcd(inp_aligned, ref_aligned)

    return mcd


if __name__ == "__main__":
    def run(token_type):
        mcd_li = []
        for i in tqdm(range(1, 101)):
            inp = 'samples/{}/{}.wav'.format(token_type, i)
            ref = '/data/public/rw/jss/jss/{}.wav'.format(9900-1+i)
            mcd = get_mcd(inp, ref)
            mcd_li.append(mcd)
        mcd_li = np.array(mcd_li)
        print('{}'.format(token_type))
        print('mean =', mcd_li.mean())
        print('var =', mcd_li.var())

    # run("char")
    # run("j")
    # run("hcj")
    # run("shcj")
    run("sj")