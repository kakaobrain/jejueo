#-*- coding: utf-8 -*-
'''
Train bpe model and bpe-segment train/valid/test datasets.

e.g.,
python bpe_segment.py --jit /data/private/jejueo/jit/jit --vocab_size 8000
'''
import codecs
import os
import sentencepiece as spm

from jamo import h2j # syllable to jamo (letter)
import argparse

def train_bpe(fpath, vocab_size):
    dir = os.path.dirname(fpath)
    train = f'--input={fpath} \
              --normalization_rule_name=identity \
              --model_prefix={dir}/bpe \
              --character_coverage=0.995 \
              --vocab_size={vocab_size} \
              --model_type=bpe'
    spm.SentencePieceTrainer.Train(train)

    # modify Dictionary
    lines = [line.replace("\t", " ") for line in codecs.open(f'{dir}/bpe.vocab', 'r', 'utf8').read().splitlines()[3:]]
    with codecs.open(f'{dir}/bpe.dict', 'w', 'utf8') as fout:
        fout.write("\n".join(lines))
    os.system(f'rm {dir}/bpe.vocab')

def apply_bpe(sp, sents, out_file):
    with codecs.open(out_file, 'w', 'utf8') as fout:
        fout.write("\n".join(" ".join(sp.EncodeAsPieces(sent)) for sent in sents))


if __name__ == "__main__":
    # arguments setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit', type=str, required=True,
                        help="JIT directory path")
    parser.add_argument('--vocab_size', type=int, default=8000,
                        help='Total number of BPE tokens')
    hp = parser.parse_args()

    # train/valid/test
    train_je = codecs.open(f"{hp.jit}/je.train", 'r', 'utf8').read().splitlines()
    dev_je = codecs.open(f"{hp.jit}/je.dev", 'r', 'utf8').read().splitlines()
    test_je = codecs.open(f"{hp.jit}/je.test", 'r', 'utf8').read().splitlines()
    train_ko = codecs.open(f"{hp.jit}/ko.train", 'r', 'utf8').read().splitlines()
    dev_ko = codecs.open(f"{hp.jit}/ko.dev", 'r', 'utf8').read().splitlines()
    test_ko = codecs.open(f"{hp.jit}/ko.test", 'r', 'utf8').read().splitlines()

    # bpe train
    dir = 'data/{}k/bpe'.format(str(hp.vocab_size)[:-3])
    os.makedirs(dir, exist_ok=True)

    with codecs.open(f"{dir}/bpe.train", 'w', 'utf8') as fout:
        fout.write("\n".join(train_je + train_ko))
    train_bpe(f'{dir}/bpe.train', hp.vocab_size)

    # apply
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{dir}/bpe.model')
    apply_bpe(sp, train_je, f'{dir}/train.je')
    apply_bpe(sp, dev_je, f'{dir}/dev.je')
    apply_bpe(sp, test_je, f'{dir}/test.je')
    apply_bpe(sp, train_ko, f'{dir}/train.ko')
    apply_bpe(sp, dev_ko, f'{dir}/dev.ko')
    apply_bpe(sp, test_ko, f'{dir}/test.ko')

