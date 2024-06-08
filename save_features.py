import os
import librosa
import json
import numpy as np
from settings import *

DATASET_PATH = "songs"
SAMPLE_RATE = 22050
DURATION = 30 # of a single track, in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
JSON_PATH = "data/data.json"

# dictionary to store data
data = {
    "mfcc": [],
    "genres": [],
    "labels": []
}

def save_features(dataset_path: str, json_path: str, n_mfcc: int=NUM_OF_MFCC, n_fft: int=2048, hop_length: int=512, num_segments: int=1, song_limit_per_genre: int=None):
    num_samples_per_segment = SAMPLES_PER_TRACK / num_segments

    # loop through all the genres
    for i, (root, dirs, files) in enumerate(os.walk(dataset_path)):
        # get the genres' names
        if i == 0:
            data['genres'] = dirs

        # load audio files
        for j, file in enumerate(files):
            if j == song_limit_per_genre:
                break
            file_path = os.path.join(root, file)
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

            # extract mfcc from segments
            for segment in range(num_segments):
                start_sample_rate = num_samples_per_segment * segment
                finish_sample_rate = start_sample_rate + num_samples_per_segment

                start_sample_rate = int(start_sample_rate)
                finish_sample_rate = int(finish_sample_rate)

                # extract mfcc
                mfcc = librosa.feature.mfcc(y=signal[start_sample_rate:finish_sample_rate],
                                            sr=sr,
                                            n_mfcc=n_mfcc,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
                mfcc = mfcc.T
                mfcc_mean = np.mean(mfcc, axis=0)

                tempo = librosa.feature.tempo(y=signal[start_sample_rate:finish_sample_rate],
                                              sr=sr)

                data["mfcc"].append(np.hstack((mfcc_mean, tempo)).tolist())
                data["labels"].append(i-1)

            print(f"{file}")        

        # save data to a json file
        with open(json_path, "w") as fp:
            json.dump(data, fp, indent=4)

if __name__ == '__main__':
    os.system('clear')
    save_features(DATASET_PATH, JSON_PATH, num_segments=5)