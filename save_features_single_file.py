import librosa
import json

SAMPLE_RATE = 22050

def save_mfcc(file_path: str, json_path: str, n_mfcc: int=13, n_fft: int=2048, hop_length: int=512):
    # print('Single song feature extraction')

    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # extract mfcc
    mfcc = librosa.feature.mfcc(y=signal,
                                sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length)
    mfcc = mfcc.T

    # check if there are enough mfcc vectors so that the data is coherent

    print(f"{file_path.split('/')[-1]}")        

    # save data to a json file
    with open(json_path, "w") as fp:
        json.dump([mfcc.tolist()], fp, indent=4)