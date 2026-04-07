# Facial Emotion Recognition using CNN

## 📌 Project Overview

This project detects human emotions from facial images using a Convolutional Neural Network (CNN). It supports both image prediction and real-time webcam detection.

## 🎯 Features

* Detects 7 emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
* Real-time webcam emotion detection
* Image-based prediction
* Built using TensorFlow, Keras, and OpenCV

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy

## 📊 Model Details

* CNN architecture with Conv2D, MaxPooling, Dense layers
* Trained on FER dataset
* Achieved ~93% training accuracy and ~59% validation accuracy

## ⚠️ Limitations

* Model shows slight overfitting
* Can be improved using data augmentation and regularization

## 🚀 Future Improvements

* Use Transfer Learning (ResNet, VGG16)
* Improve accuracy with data augmentation
* Deploy using Streamlit or Flask

## ▶️ How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Train model

```
python train.py
```

### Image prediction

```
python predict.py
```

### Webcam detection

```
python webcam.py
```

## 📷 Output

(Add screenshots here)

## 👨‍💻 Author

Rakesh V
