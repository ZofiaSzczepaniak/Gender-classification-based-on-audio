import librosa
import numpy as np
from scipy.fftpack import dct
import soundfile as sf
import os
import pandas as pd
result = np.array([])
path = 'path/to/LibriSpeech/male_voices'
i = 1
for file in os.listdir(path):
    input_path = os.path.join(path, file)
    if file.lower().endswith(('.mp3', '.wav', '.flac', '.mov')):
        if i % 100 == 0:
            print(i)
        y, sr = librosa.load(input_path)

        f0 = librosa.yin(y,sr=sr,fmin=50,fmax=500)
        sd = np.std(f0)
        m = np.mean(f0)
        c = np.where(np.abs((f0 - m) / sd) < 1, f0, -1)
        cd = np.delete(c, np.where(c == -1))
        result = np.append(result,np.median(cd))
        i+=1
result2 = np.array([])
path = 'path/to/LibriSpeech/female_voices'
i = 1
for file in os.listdir(path):
    input_path = os.path.join(path, file)
    if file.lower().endswith(('.mp3', '.wav', '.flac', '.mov')):
        if i % 100 == 0:
            print(i)
        try:
            y, sr = librosa.load(input_path)
            f0 = librosa.yin(y, sr=sr, fmin=50, fmax=500)
            sd = np.std(f0)
            m = np.mean(f0)
            c = np.where(np.abs((f0 - m) / sd) < 1, f0, -1)
            cd = np.delete(c, np.where(c == -1))
            result2 = np.append(result2, np.median(cd))
        except:
            print(input_path)
        i+=1
labels = np.hstack((np.full(len(result),"M"),np.full(len(result2),"F")))
data = np.hstack((result,result2))
df = pd.DataFrame()
df['gender'] = labels
df['f0'] = data
df.to_csv("ff.csv",index=False)