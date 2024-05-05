import librosa
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from extract_features import extract_features

# Extracting features of the songs

audio_dataset_path = 'songs/'
metadata_path = 'metadata.csv'

metadata = pd.read_csv(metadata_path)

data = []

for index, row in metadata.iterrows():
    if pd.isna(row['file_name']):
        print(f'The file name is missing. Index: {index}')
    elif pd.isna(row['genre']):
        print(f'The genre is missing. Index: {index}')
    else:
        file_path = audio_dataset_path + row['file_name']
        genre = row['genre']
        features = extract_features(file_path)
        data.append([features, genre])

data = pd.DataFrame(data, columns=['features', 'genre'])

if data.shape[0] == 0:
    print('There is no songs provided.')
    exit()

# Preparing the data for training

x = np.array(data.features.tolist())
scaler = StandardScaler()
x = scaler.fit_transform(x)
y = np.array(data.genre.tolist())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(y_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
