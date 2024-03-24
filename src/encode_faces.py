import ArcFace
import argparse
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, default='Datasets/Pictures/Normalised',
                help="path to Norm/dir")
# ap.add_argument("-o", "--save", type=str, default='/Models/model.h5',
#                 help="path to save .h5 model")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
# checkpoint_path = args['save']

# Load ArcFace Model
model = ArcFace.loadModel()
model.load_weights("Models/arcface_weights.h5")
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
class_number = len(names)

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
