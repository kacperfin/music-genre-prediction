from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from get_features_dataframe import get_features_dataframe
from show_features_correlation_matrix import show_features_correlation_matrix


# Extracting features of the songs

audio_dataset_path = 'songs/'  
data = get_features_dataframe(audio_dataset_path, limit=5)

if data.shape[0] == 0:
    print('There is no songs provided.')
    exit()

# Preparing the data for training

x = np.array(data.features.to_list())
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = np.array(data.genre.tolist())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)