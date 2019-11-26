# Speech Synthesis with JSS dataset

## Requirements
  * python >= 3.6
  * NumPy >= 1.11.1
  * TensorFlow==1.3
  * librosa
  * tqdm
  * matplotlib
  * scipy
  * pysptk
  * pyworld
  * nnmnkwii
  
## Training
  * STEP 0. Download and extract [JSS](https://www.kaggle.com/bryanpark/jejueo-single-speaker-speech-dataset).
  * STEP 1. Run `python train.py 1` for training Text2Mel.
  * STEP 2. Run `python train.py 2` for training SSRN.

You can do STEP 1 and 2 at the same time, if you have more than one gpu card.


## Sample Synthesis
  * Run `synthesize.py` and check the files in `samples`.

## MCD Evalution
  * Run `python mcd.py`.
```
