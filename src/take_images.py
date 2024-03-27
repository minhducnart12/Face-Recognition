import os
import cv2
import sys
import argparse

def parse_arguments(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--source", type=str, required=True,
                    help="Video folder path or webcam-id")
    ap.add_argument("-n", "--name", type=str, required=True,
                    help="name of the person")
    ap.add_argument("-o", "--save", type=str, default='../Datasets/Pictures/Raw',
                    help="path to save dir")
    ap.add_argument("-c", "--conf", type=float, default=0.8,
                    help="min prediction conf (0<conf<1)")
    ap.add_argument("-x", "--number", type=int, default=100,
                    help="number of data wants to collect")
    
    return ap.parse_args(argv)

def main():
    
    # Load ArcFace Model
    parsed_args = parse_arguments(sys.argv[1:])
    source = parsed_args.source
    name_of_person = parsed_args.name
    path_to_save = parsed_args.save
    min_confidence = parsed_args.conf

    os.makedirs((os.path.join(path_to_save, name_of_person)), exist_ok=True)
    path_to_save = os.path.join(path_to_save, name_of_person)

    opencv_dnn_model = cv2.dnn.readNetFromCaffe(
        prototxt="../Models/deploy.prototxt",
        caffeModel="../Models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    )

    if source.isnumeric():
        source = int(source)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0
    img_name = 0
    while True:
        success, img = cap.read()
        if not success:
            print('[INFO] Cam NOT working!!')
            break

        # Caffe Model - Face Detection
        h, w, _ = img.shape
        preprocessed_image = cv2.dnn.blobFromImage(
            img, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 117.0, 123.0), swapRB=False, crop=False
        )
        opencv_dnn_model.setInput(preprocessed_image)
        results = opencv_dnn_model.forward() 

        face_count = 0
        for face in results[0][0]:

            face_confidence = face[2]
            if face_confidence > min_confidence:
                face_count += 1
                bbox = face[3:]
                img_without_rect = img.copy()
                x1 = int(bbox[0] * w)
                y1 = int(bbox[1] * h)
                x2 = int(bbox[2] * w)
                y2 = int(bbox[3] * h)
                cv2.rectangle(
                    img, pt1=(x1, y1), pt2=(x2, y2),
                    color=(0, 255, 0), thickness=w//200
                )
            
        # Save Image
        if  count % int(fps/5) == 0:
            if face_count == 1:
                img_name = len(os.listdir(path_to_save))
                cv2.imwrite(f'{path_to_save}/{img_name}.jpg',img_without_rect)
                print(f'[INFO] Successfully Saved {img_name}.jpg')
            else:
                print(f'[WARNING]More than one face appears')
        count += 1
        cv2.imshow('Webcam', img)
        cv2.waitKey(1)

        if img_name == parsed_args.number-1:
            print(f"[INFO] Collected {parsed_args.number} Images")
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
