import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def main():
    pickle_file_path = "../Models/EncodeFile.pkl"
    # Load encoding file
    with open(pickle_file_path, 'rb') as f:
        x, y= pickle.load(f)

    # DataFrame
    df = pd.DataFrame(x, columns=np.arange(512))
    x = df.copy()
    x = x.astype('float64')

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Start training
    svm_model  = SVC(kernel='linear', probability=True)

    svm_model.fit(x_train, y_train)

    y_pred = svm_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy of the SVM model:", accuracy)

    pickle_file_path = "../Models/facemodel.pkl"

    with open(pickle_file_path, 'wb') as f:
        pickle.dump(svm_model, f)

    print(f'[INFO] Embeddings saved to {pickle_file_path}')

if __name__ == "__main__":
    main()