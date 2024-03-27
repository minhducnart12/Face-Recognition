from keras.models import load_model
from mtcnn import MTCNN
from my_utils import alignment_procedure
import tensorflow as tf
import ArcFace
import cv2
import numpy as np
import pandas as pd
import argparse
import pickle

# Load SVM model
with open("../Models/facemodel.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Initialize MTCNN for face detection
mtcnn = MTCNN()

model = ArcFace.loadModel()
model.load_weights("../Models/arcface_weights.h5")
target_size = model.layers[0].input_shape[0][1:3]

def recognize_face(img):
    # Detect faces using MTCNN
    detections = mtcnn.detect_faces(img)

    for detect in detections:
        # Extract face bounding box
        bbox = detect['box']
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), \
                    int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])

        right_eye = detect['keypoints']['right_eye']
        left_eye = detect['keypoints']['left_eye']
        norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
        img_resize = cv2.resize(norm_img_roi, target_size)

        img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_norm = img_pixels/255  # normalize input in [0, 1]
        img_embedding = model.predict(img_norm)[0]
        img_embedding = img_embedding.reshape(1, -1)

        # Predict probabilities using SVM
        predicted_probabilities = svm_model.predict_proba(img_embedding)
        max_probability = np.max(predicted_probabilities)
        print(max_probability)
        predicted_label = svm_model.classes_[np.argmax(predicted_probabilities)]

        # Check if prediction exceeds threshold
        if max_probability > 0.85:
            # Draw bounding box and label on face
            cv2.rectangle(
                    img, (xmin, ymin), (xmax, ymax),
                    (0, 255, 0), 2
                    )
            
            cv2.putText(
                        img, predicted_label,
                        (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2
                    )
        else:
            cv2.rectangle(
                    img, (xmin, ymin), (xmax, ymax),
                    (0, 0, 255), 2
                    )
            
            cv2.putText(
                        img, "Unknow",
                        (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 0, 255), 2
                    )
    return img

# Open camera
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Recognize faces in frame
    frame_with_faces = recognize_face(frame)

    # Display the result
    cv2.imshow('Face Recognition', frame_with_faces)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()