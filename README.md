# FaceRecognition-MTCNN-ArcFace
FaceRecognition with MTCNN using ArcFace


### Clone this Repository
```
git clone https://github.com/minhducnart12/Face-Recognition.git
cd Face-Recognition
```

### Install dependency
```
pip3 install -r requirements.txt
```
```
cd src
### 1.Collect Data using Web-cam or RTSP

<details>
  <summary>Args</summary>
  
  `-i`, `--source`: RTSP link or webcam-id <br>
  `-n`, `--name`: name and ID of the person <br>
  `-o`, `--save`: path to save dir <br>
  `-c`, `--conf`: min prediction conf (0<conf<1) <br>
  `-x`, `--number`: number of data wants to collect

</details>

**Example:**
```
python take_images.py --source 0 --name MinhDuc_000001 --save ../Datasets/Pictures/Raw --conf 0.8 --number 100
```
:book: **Note:** <br>
Repeate this process for all people, that we need to detect on CCTV, Web-cam or in Video.<br>
In side save Dir, contain folder with name and ID of people. Inside that, it contain collected image data of respective people.<br>
**Structure of Save Dir:** <br>
```
├── data_dir
│   ├── name1_000001
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   ├── name2_000002
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
.   .
.   .
```

### 2.Normalize Collected Data
It will Normalize all data inside path to save Dir and save same as like Data Collected Dir

<details>
  <summary>Args</summary>
  
  `-i`, `--dataset`: path to dataset/dir <br>
  `-o`, `--save`: path to save dir

</details>

**Example:**
```
python normalize_images.py --dataset ../Datasets/Pictures/Raw
```
**Structure of Normalized Data Dir:** <br>
```
├── normalize
│   ├── name1_000001
│   │   ├── 1_norm.jpg
│   │   ├── 2_norm.jpg
│   │   ├── ...
│   ├── name2_000002
│   │   ├── 1_norm.jpg
│   │   ├── 2_norm.jpg
│   │   ├── ...
.   .
.   .
```

### 3.Encoding Face using Normalized Data

<details>
  <summary>Args</summary>
  
  `-i`, `--dataset`: path to Norm/dir <br>

</details>

**Example:**
```
python encode_faces.py --dataset ../Datasets/Pictures/Normalised
```

### 3.Train a Model Classifier using Encoding Face

**Example:**
```
python classifier.py
```

## Inference

<details>
  <summary>Args</summary>
  
  `-i`, `--source`: path to Video or webcam or image <br>

</details>

### On Image 
**Example:**
```
python inference.py -i 0
```
**To Exit Window - Press Q-Key**