import os
import joblib
from save_features_single_file import *
import json
import numpy as np

MODEL = 'model/model.joblib'
SCALER = 'model/scaler.joblib'
DATA = 'data/data.json'

def predict(song_path: str, song_features_path):
    # Get the genres' names
    with open(DATA, "r") as data:
        data = json.load(data)

    genres = data['genres']

    # Extract features from the songs
    save_mfcc(song_path, song_features_path)
    with open(song_features_path, "r") as fp:
        X = json.load(fp)

    X = np.array(X)
    X = np.mean(X, axis=1)
    scaler = joblib.load(SCALER)
    X = scaler.transform(X)

    model = joblib.load(MODEL)
    y_pred = model.predict(X)

    genre = y_pred[0]

    print(f"The genre is: {genres[genre]} ({genre})")

    return genres[genre]
