# 🌿 AgriBot — Leaf Disease Detection System

An intelligent agricultural robot built with **Raspberry Pi** that detects and classifies plant leaf diseases in real-time using **SVM** and **CNN** machine learning models. Designed to assist farmers in early disease identification, reducing crop loss and minimizing pesticide usage.

---

## 📌 Problem Statement

Crop diseases account for significant agricultural losses worldwide. Manual inspection is time-consuming, error-prone, and requires expert knowledge. AgriBot automates this process by capturing leaf images and classifying diseases instantly using trained ML models.

## 🔧 How It Works

1. **Image Capture** — A camera module mounted on the Raspberry Pi captures leaf images in the field.
2. **Preprocessing** — Images are resized, normalized, and prepared for model inference.
3. **Disease Classification** — The system runs the image through a trained SVM or CNN model to identify the disease category.
4. **Sensor Data** — Environmental sensors (ultrasonic, soil moisture, etc.) provide additional agricultural data.
5. **Output** — The predicted disease class is displayed, enabling timely intervention.

## 📁 Project Structure

```
leafdiseaseproject/
│
├── leaf disease svm/           # SVM-based disease classification
│   ├── accuracycheck.py        # SVM model evaluation & metrics
│   ├── accuracycheck_cnn.py    # CNN accuracy comparison
│   ├── pred_ui.py              # Prediction UI for testing
│   ├── val.py                  # Validation scripts
│   ├── s1.py                   # SVM training pipeline
│   ├── m.py                    # Model utilities
│   ├── pi_cam.py               # Raspberry Pi camera capture
│   ├── hcsro4.py               # HC-SR04 ultrasonic sensor driver
│   └── model_comparison.md     # SVM vs CNN performance comparison
│
├── LeafCnnmodel/               # CNN-based disease classification
│   └── New Plant Diseases Dataset (Augmented)/
│       ├── train/              # Training images (38 disease classes)
│       └── valid/              # Validation images
│
├── mainsensorcode.py           # Main sensor integration code (Raspberry Pi)
└── README.md
```

## 🧠 Models Used

### SVM (Support Vector Machine)
- Trained on extracted features from leaf images
- Uses an SVM classifier with optimized hyperparameters
- Evaluation includes confusion matrix, per-class precision/recall/F1, and learning curves

### CNN (Convolutional Neural Network)
- Trained on the **New Plant Diseases Dataset (Augmented)** — 87,000+ images across 38 classes
- Covers diseases for crops including Tomato, Potato, Corn, Apple, Grape, and more
- Deep learning approach for higher accuracy on complex disease patterns

## 🌱 Supported Crop Diseases

The model can classify diseases across multiple crops, including:

- **Tomato** — Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Mosaic Virus, etc.
- **Potato** — Early Blight, Late Blight
- **Corn (Maize)** — Cercospora Leaf Spot, Common Rust, Northern Leaf Blight
- **Apple** — Apple Scab, Black Rot, Cedar Apple Rust
- **Grape** — Black Rot, Esca, Leaf Blight
- And many more (38 classes total including healthy leaves)

## 🔌 Hardware Components

- Raspberry Pi (main controller)
- Pi Camera Module (image capture)
- HC-SR04 Ultrasonic Sensor (obstacle detection / distance measurement)
- Soil Moisture Sensor
- Motor Driver + DC Motors (robot mobility)

## 🛠️ Tech Stack

- **Language:** Python
- **ML/DL:** scikit-learn (SVM), TensorFlow/Keras (CNN)
- **Image Processing:** OpenCV, PIL
- **Visualization:** Matplotlib, Seaborn
- **Hardware Interface:** RPi.GPIO
- **Dataset:** New Plant Diseases Dataset (Augmented) from Kaggle

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy opencv-python scikit-learn matplotlib seaborn pillow tensorflow
```

### Running the SVM Model
```bash
cd "leaf disease svm"
python s1.py              # Train the SVM model
python accuracycheck.py   # Evaluate model performance
python pred_ui.py         # Launch prediction UI
```

### Running on Raspberry Pi
```bash
python mainsensorcode.py  # Start sensor data collection + disease detection
```

## 📊 Model Evaluation

The project includes comprehensive evaluation tools:
- Confusion Matrix visualization
- Per-class Precision, Recall, and F1-Score
- Learning Curves (training vs validation accuracy)
- SVM vs CNN comparative analysis

## 👨‍💻 Author

Ece-c batch c5 students 2022-2026 batch

## 📄 License

This project is for academic and educational purposes.
