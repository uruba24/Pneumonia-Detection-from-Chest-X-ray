# Pneumonia Detection from Chest X-rays

## Overview
This project is a deep learning-based system that detects pneumonia from chest X-ray images using a Convolutional Neural Network (CNN). A user-friendly UI is provided for real-time predictions, allowing users to upload X-ray images and receive diagnostic results.

## Features
- **Deep Learning Model**: ResNet50-based CNN for binary classification (Normal/Pneumonia).
- **Data Augmentation**: Improved generalization with preprocessing techniques.
- **Grad-CAM Visualization**: Heatmap for explainability.
- **User Interface**: Built with Streamlit for ease of use.
- **Deployment**: Can be run locally or deployed on the cloud.

---

## Installation and Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow/Keras
- OpenCV, NumPy, Pandas
- Matplotlib, Seaborn (for visualization)
- Streamlit (for UI)

### Clone Repository
```bash
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage
### 1. Train the Model
To train the CNN model from scratch:
```bash
python train.py
```
The trained model will be saved as `pneumonia_model.h5`.

### 2. Run the UI
Launch the Streamlit interface:
```bash
streamlit run app.py
```
Upload a chest X-ray image, and the model will predict whether it indicates pneumonia or is normal.

### 3. Model Inference
To make a prediction using a saved model:
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('pneumonia_model.h5')

# Load and preprocess image
image = cv2.imread('path_to_xray.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (224, 224))
image = image / 255.0
image = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)
print("Pneumonia Detected" if prediction > 0.5 else "Normal")
```

---

## Model Performance
- **Training Accuracy**: 75%
- **Validation Accuracy**: 50%
- **Test Accuracy**: 80%
- **Grad-CAM visualization** highlights important regions in the X-ray for model predictions.

---

## Challenges and Future Improvements
### Challenges
- Data imbalance resolved using class weighting.
- Prevented overfitting with dropout and regularization.
- Reduced model size for efficient deployment.

### Future Enhancements
- Experiment with EfficientNet/MobileNet for better accuracy.
- Add multi-class classification (detect other lung diseases).
- Deploy the model on AWS/GCP for wider accessibility.

---

## Contributors
- **Urooba Aftab** – [GitHub](https://github.com/uruba24)
