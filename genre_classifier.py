import json
from matplotlib.pylab import standard_t
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os

JSON_PATH = "data.json"

def load_data(json_path):
    with open(json_path, "r") as fp :
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    genres = data["genres"]

    return inputs, targets, genres

def main():
    X, y, genres = load_data(JSON_PATH)
    X = np.mean(X, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred, target_names=genres))


if __name__ == "__main__":
    os.system('clear')
    main()
    