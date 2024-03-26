import ArcFace
import argparse
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import tensorflow as tf

def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--dataset", type=str, default='../Datasets/Pictures/Normalised',
                    help="path to Norm/dir")
    return ap.parse_args(argv)

def main():
    # Load ArcFace Model
    parsed_args = parse_arguments(sys.argv[1:])
    path_to_dir = parsed_args.dataset
    model = ArcFace.loadModel()
    model.load_weights("../Models/arcface_weights.h5")
    print("ArcFace expects ", model.layers[0].input_shape[0][1:], " inputs")
    print("and it represents faces as ",
        model.layers[-1].output_shape[1:], " dimensional vectors")
    target_size = model.layers[0].input_shape[0][1:3]
    print('target_size: ', target_size)

    # Variable for store img Embedding
    x = []
    y = []

    names = os.listdir(path_to_dir)
    names = sorted(names)

    for name in names:
        img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
        img_list = sorted(img_list)

        for img_path in img_list:
            img = cv2.imread(img_path)
            img_resize = cv2.resize(img, target_size)

            img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_norm = img_pixels/255  # normalize input in [0, 1]
            img_embedding = model.predict(img_norm)[0]

            x.append(img_embedding)
            y.append(name)
            print(f'[INFO] Embedding {img_path}')
        print(f'[INFO] Completed {name} Part')
    print('[INFO] Image Data Embedding Completed...')

    pickle_file_path = "../Models/EncodeFile.pkl"

    with open(pickle_file_path, 'wb') as f:
        pickle.dump((x, y), f)

    print(f'[INFO] Embeddings saved to {pickle_file_path}')

if __name__ == "__main__":
    main()
