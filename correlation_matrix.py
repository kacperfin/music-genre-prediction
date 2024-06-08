import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from settings import *


DATA_PATH = 'data/data.json'

def load_data(data_path: str):
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    data = data['mfcc']
    return data

def main():
    data = load_data(DATA_PATH)
    data = np.array(data)

    df = pd.DataFrame(data)

    columns_names = []
    for x in range(NUM_OF_MFCC):
        columns_names.append("mfcc_mean_"+str(x+1))
    columns_names.append('tempo')
    
    df.columns = columns_names

    correlation_matrix = df.corr()

    fig, ax = plt.subplots(figsize=(11, 8))

    cmap = sns.color_palette("vlag", as_cmap=True)

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                annot=False, fmt="0.2f", square=True)
    
    ax.set_title('Correlation map between features')

    plt.show()

if __name__ == '__main__':
    os.system('clear')
    main()