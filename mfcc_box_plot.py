import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler

from settings import NUM_OF_MFCC

DATA_PATH = 'data/data.json'

def load_data(data_path: str):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    return data

def main():
    data = load_data(DATA_PATH)

    mfcc = np.array(data['mfcc'])
    scaler = StandardScaler()
    mfcc = scaler.fit_transform(mfcc)

    labels = np.array(data['labels'])
    genres = data['genres']

    df = pd.DataFrame(mfcc)
    df = df.iloc[:, :]
    columns_names = [f'mfcc_mean_{x+1}' for x in range(NUM_OF_MFCC)]
    columns_names.append('tempo')
    df.columns = columns_names
    df['genre'] = [genres[label] for label in labels]

    print(df)

    for index, genre in enumerate(genres):
        df_genre = df[df.genre == genre] 
        f, ax = plt.subplots(figsize=(11, 8))
        sns.boxplot(data=df_genre, color=f'C{index}')
        ax.set_title(f"Boxplot of mfccs for the genre: {genre}", fontsize=20)
        ax.set_ylabel('mean value of the mfcc after standardizing',fontsize=16)
        plt.show()

if __name__ == '__main__':
    os.system('clear')
    main()