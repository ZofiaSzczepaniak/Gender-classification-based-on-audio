import librosa
import soundfile as sf
import os
import pandas as pd
path = 'path/to/LibriSpeech'
path_out1 = 'path/to/LibriSpeech/male_voices/'
path_out2 = 'path/to/LibriSpeech/female_voices/'
df = pd.read_csv("male.csv",sep=';')
df2 = pd.read_csv("female.csv",sep=';')
df['totext'] = df['id'].astype(str)
df2['totext'] = df2['id'].astype(str)

for dirs in df2['totext']:
    for roots, dirs2, files in os.walk(path+dirs+'/'):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.mov')):
                try:
                    y, sr = librosa.load(roots+ "/" +file,sr = None)
                    sf.write(path_out2 + file, y, 16000,
                                 subtype="PCM_16")
                except:
                    #os.remove(roots+"/" + file)
                    print(f"error{file}")

