import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def show_features_correlation_matrix(data):
    matrix = pd.DataFrame(data=data.features.to_list())
    correlation_matrix = matrix.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    f, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(correlation_matrix, cmap='flare', ax=ax,
                vmin=0, vmax=1,
                mask=mask, square=True)
    plt.show()