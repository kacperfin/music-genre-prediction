import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = 'data/data.json'

def load_data(data_path: str):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    return data

def main():
    data = load_data(DATA_PATH)
    tempos = np.array(data['mfcc'])[:, -1]
    labels = np.array(data['labels'])
    genres = data['genres']

    genres = [genres[label] for label in labels]

    df = pd.DataFrame({'tempo': tempos, 'genre': genres})

    f, ax = plt.subplots(figsize=(11, 8))

    sns.boxplot(x='genre', y='tempo', data=df, palette='husl')

    ax.set_title('Tempo boxplot for genres', fontsize=18)
    ax.set_xlabel('genre', fontsize=16)
    ax.set_ylabel('tempo', fontsize=16)

    plt.show()

if __name__ == '__main__':
    os.system('clear')
    main()