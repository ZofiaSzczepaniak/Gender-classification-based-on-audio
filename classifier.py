import librosa
import numpy as np
from scipy.fftpack import dct
import soundfile as sf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def preemphasis(file):
    y, sr = librosa.load(file)
    y_filt = librosa.effects.preemphasis(y)
    return y_filt, sr
def mfcc(y_filt,sr):
    L = len(y_filt)
    frame_len = int(sr * 0.03)
    frame_jump = int(
        sr * 0.02)
    num = int((L - frame_len) // (frame_jump)) + 1
    sig_len = num * frame_jump + frame_len
    signal = y_filt
    signal = np.concatenate((signal, np.zeros(
        sig_len - L)))
    frames = []
    for x in range(num):
        frames.append(signal[x * frame_jump:x * frame_jump + frame_len])
    frequencises = []
    frames *= np.hamming(frame_len)
    for frame in frames:
        frames_freq = np.fft.rfft(frame,
                                  n=frame_len)
        frequencises.append(frames_freq)
    freqq = []
    for freq in frequencises:
        freqq.append(abs(1 / len(frame) * freq * np.conj(freq)))
    return freqq
def mel_filter(freqq,sr):
    frame_len = int(sr * 0.03)
    Low_fr = 200
    High_fr = 8000
    n_mel = 30
    Hz_to_mel = lambda f: 1125 * np.log(1 + f / 700)
    mel_to_Hz = lambda m: 700 * (np.exp(m / 1125) - 1)
    mel_threshs = np.linspace(Hz_to_mel(Low_fr), Hz_to_mel(High_fr),
                              n_mel + 2)
    freq_threshs = mel_to_Hz(mel_threshs)
    bins = np.floor((frame_len + 1) * freq_threshs / sr)
    fbanks = np.zeros(331)

    for i in range(1, n_mel):
        lbound = int(bins[i - 1])
        mbound = int(bins[i])
        hbound = int(bins[i + 1])
        l = np.linspace(0, 0, lbound + 1)
        lm = np.linspace(0, 1, mbound - lbound + 1)
        mh = np.linspace(1, 0, hbound - mbound + 1)
        h = np.linspace(0, 0, len(freqq[0]) - hbound + 1)
        some = np.concatenate((l[:-1], lm[:-1], mh[:-1], h[:-1]), axis=None)
        fbanks = np.vstack((fbanks, some))
    fbanks = fbanks[1:n_mel]
    mels = np.zeros(n_mel - 1)
    for fr in freqq:
        mels = np.vstack((mels, np.array([np.dot(fr, filter) for filter in fbanks])))

    mels = 20 * np.log10(mels)
    return mels
def melcoeff(mels):

    num_ceps = 12
    mfcc = dct(mels, type=2, axis=1, norm='ortho')[:, :num_ceps]
    mfcc_avg = [np.mean(cc) for cc in mfcc.T[:, 1:]]
    return mfcc_avg

dict = {}
i = 0
path = 'path/to/LibriSpeech/male_voices'
for file in os.listdir(path):
    input_path = os.path.join(path, file)
    if file.lower().endswith(('.mp3', '.wav', '.flac', '.mov')):
        if i %100 ==0:
            print(i)

        start, sr=preemphasis(input_path)
        first_step=mfcc(start,sr)
        second_step =mel_filter(first_step,sr)
        third_step = melcoeff(second_step)
        dict[file] = third_step

        i += 1

df = pd.DataFrame.from_dict(dict, orient='index')
df.to_csv('path=/male_coeff.csv')

file_M = 'male_coeff.csv'
file_F = 'female_coeff.csv'

df_M = pd.read_csv(file_M,index_col=None)
df_F = pd.read_csv(file_F, index_col=None)

df_M['gender']='M'
df_F['gender']='F'
Y_M = df_M['gender']
Y_F = df_F['gender']
df_M.drop(['gender'], axis=1, inplace=True)
df_F.drop(['gender'], axis=1,inplace=True)
df_M.drop(['abc'], axis=1, inplace=True)
df_F.drop(['abc'], axis=1,inplace=True)

X_M = df_M.to_numpy()
X_F = df_F.to_numpy()
dff = pd.read_csv("ff.csv")
X = np.vstack((X_M, X_F))
df_n = pd.DataFrame(X)
df_n['gender'] = dff['gender']
df_n['f0'] = dff['f0']
df_n.dropna(axis=0,inplace=True)
Y=df_n['gender']
df_n.drop(['gender'], axis=1,inplace=True)
X = df_n.to_numpy()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(data_scaled, Y, test_size=0.3)
model = SVC(kernel='rbf')
model.fit(X_train, Y_train)
prediction = model.predict(X_test)
accuracy = accuracy_score(Y_test, prediction)
print(f'Accuaracy: {accuracy}')

