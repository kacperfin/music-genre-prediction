import librosa
import json
import numpy as np

from settings import NUM_OF_MFCC

SAMPLE_RATE = 22050

def save_features(file_path: str, json_path: str, n_mfcc: int=NUM_OF_MFCC, n_fft: int=2048, hop_length: int=512):
    
    # print('Single song feature extraction')
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # extract mfcc and tempo
    mfcc = librosa.feature.mfcc(y=signal,
                                sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length)
    mfcc = mfcc.T
    mfcc_mean = np.mean(mfcc, axis=0)
    tempo = librosa.feature.tempo(y=signal,
                                  sr=sr)

    print(f"{file_path.split('/')[-1]}")        

    # save data to a json file
    with open(json_path, "w") as fp:
        json.dump([np.hstack((mfcc_mean, tempo)).tolist()], fp, indent=4)