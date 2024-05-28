from os import listdir
from os.path import join
from extract_features import extract_features
import pandas as pd
import os

def get_features_dataframe(audio_dataset_path, limit=None):
    genre_list = [f for f in listdir(audio_dataset_path)]

    data = []

    for genre in genre_list:
        path = join(audio_dataset_path, genre)

        for index, file in enumerate(listdir(path)):
            if index == limit:
                break

            path_to_file = audio_dataset_path + genre + '/' + file
            features = extract_features(path_to_file)
            if features is not None:
                data.append([features, genre])
                os.system('clear')
                print(f"Hold on... {genre}/{index+1}")

    return pd.DataFrame(data, columns=['features', 'genre'])