# Gender-classification-based-on-audio
This repository contains a machine learning pipeline for automatic gender classification based on short audio samples. The project uses a dataset of 24,000 utterances from male and female speakers (LibriSpeech) to combine signal processing techniques (like MFCC and fundamental frequency extraction) with classical ML models (Logistic Regression, SVM, MLP).

## Structure:
1) classifier.py – complete audio preprocessing pipeline:

- pre-emphasis filtering

- custom MFCC extraction using FFT and DCT

- fundamental frequency analysis

- training and evaluating models (SVM, Logistic Regression, MLP)

2) extract.py – utility to extract and resample audio files from the dataset based on gender metadata
3) ff.py - fundamental frequency extractor

## Conclusions
- Achieved 98% accuracy using SVM with RBF kernel

- Avoided deep learning to test the power of traditional ML

- Focus on interpretable audio features and clean preprocessing

## Requirements
- LibriSpeech audio samples
- Python 3.12
 
## Instruction:
First, to use the program, one must install the Python compiler. To do so, enter the site and follow the given steps: https://code.visualstudio.com/

Then, install the requirements from requirements.txt 
```
pip install -r requirements.txt
```
