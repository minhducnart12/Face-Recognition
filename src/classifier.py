import argparse
import cv2
import glob
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", type=str, default='../Datasets/Pictures/Normalised',
                    help="path to Norm/dir")
    ap.add_argument("-o", "--save", type=str, default='../Models/model.h5',
                    help="path to save .h5 model, eg: dir/model.h5")
    ap.add_argument("-l", "--le", type=str, default='../Models/le.pickle',
                    help="path to label encoder")
    ap.add_argument("-b", "--batch_size", type=int, default=16,
                    help="batch Size for model training")
    ap.add_argument("--epochs", type=int, default=200,
                    help="Epochs for Model Training")
    
    return ap.parse_args(argv)

def main():
    parsed_args = parse_arguments(sys.argv[1:])

    import pickle

    pickle_file_path = "../Models/EncodeFile.pkl"

    # Load encoding file
    with open(pickle_file_path, 'rb') as f:
        x, y= pickle.load(f)

    print(y)
    # DataFrame
    df = pd.DataFrame(x, columns=np.arange(512))
    x = df.copy()
    x = x.astype('float64')


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Start training

    svm_model  = SVC(kernel='linear', probability=True)

    svm_model.fit(x_train, y_train)

    y_pred = svm_model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Độ chính xác của mô hình SVM:", accuracy)

    # save label encoder
    # f = open(parsed_args.le, "wb")
    # f.write(pickle.dumps(svm_model), le)
    # f.close()
    # print('[INFO] Successfully Saved models/le.pickle')

    pickle_file_path = "../Models/facemodel.pkl"

    with open(pickle_file_path, 'wb') as f:
        pickle.dump(svm_model, f)


if __name__ == "__main__":
    main()