# Jejueo-Korean Machine Translation with JIT dataset

## Requirements
  * python >= 3.6
  * NumPy >= 1.11.1
  * Fairseq
  * Sentencepiece
  * tqdm
  
## Training
* STEP 0. Download and extract the [JIT dataset](https://www.kaggle.com/bryanpark/jit-dataset)
* STEP 1. bpe segment for training
```
python bpe_segment.py --jit jit --vocab_size 4000
```
* STEP 2.fairseq-prepro
```
python prepro.py --src ko --tgt je --vocab_size 4000
python prepro.py --src je --tgt ko --vocab_size 4000
```
* STEP 3. train
```
export lang1="ko"
export lang2="je"
fairseq-train data/4k/${lang1}-${lang2}-bin \
    --arch transformer       \
    --optimizer adam \
    --lr 0.0005 \
    --label-smoothing 0.1 \
    --dropout 0.3       \
    --max-tokens 4000 \
    --min-lr '1e-09' \
    --lr-scheduler inverse_sqrt       \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy       \
    --max-epoch 100 \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07'    \
    --adam-betas '(0.9, 0.98)'       \
    --save-dir train/4k/${lang1}-${lang2}/ckpt  \
    --save-interval 10
```

```
export lang1="je"
export lang2="ko"
fairseq-train data/4k/${lang1}-${lang2}-bin \
    --arch transformer       \
    --optimizer adam \
    --lr 0.0005 \
    --label-smoothing 0.1 \
    --dropout 0.3       \
    --max-tokens 4000 \
    --min-lr '1e-09' \
    --lr-scheduler inverse_sqrt       \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy       \
    --max-epoch 100 \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07'    \
    --adam-betas '(0.9, 0.98)'       \
    --save-dir train/4k/${lang1}-${lang2}/ckpt  \
    --save-interval 10
```

* STEP 4. Evaluation
```
export lang1="ko"
export lang2="je"

fairseq-generate data/4k/${lang1}-${lang2}-bin \
  --ckpt CKPT \
  --subset {valid,test} \
  --beam-width 5
```


