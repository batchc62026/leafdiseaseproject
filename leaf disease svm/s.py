import tkinter as tk
import tkinter.filedialog
import cv2
import numpy as np
import joblib
import os
import pickle

# Model file path
model_path = "finger_vein_svm_model.joblib"

# Define the image size
img_size = 64

# Function to load the model with error handling
def load_model():
    if not os.path.exists(model_path):
        print("❌ Error: Model file not found!")
        return None

    try:
        # Try loading with joblib
        clf = joblib.load(model_path)
        print("✅ Model loaded successfully with Joblib!")
        return clf
    except Exception as e:
        print("⚠️ Joblib failed. Trying Pickle...")
    
    try:
        # Try loading with pickle
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        print("✅ Model loaded successfully with Pickle!")
        return clf
    except Exception as e:
        print("❌ Error: Unable to load the model. Model might be corrupted or incompatible.")
        return None

# Load the SVM model
clf = load_model()
if clf is None:
    exit()

# Function to browse and predict label
def predict_label():
    file_path = tk.filedialog.askopenfilename()
    
    if not file_path:
        return  # If no file is selected, exit the function

    try:
        # Load and preprocess the image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))

        # Convert image to numpy array
        X = np.array([img])
        X = X.reshape(X.shape[0], -1)  # Flatten the image for SVM

        # Predict the label
        y_pred = clf.predict(X)
        disease_info = get_disease_info(str(y_pred[0]))

        # Display result
        label.config(text=disease_info, fg="blue")
    except Exception as e:
        label.config(text="Error in prediction!", fg="red")
        print(f"❌ Prediction Error: {e}")

# Function to map predictions to disease names and remedies
def get_disease_info(prediction):
    diseases = {
        "1": ("Bacterial Spot", "Spray copper-based fungicide."),
        "2": ("Early Blight", "Spray Bonide Liquid Copper Fungicide."),
        "3": ("Healthy", "No action needed."),
        "4": ("Late Blight", "Apply fungicides immediately."),
        "5": ("Leaf Mold", "Mix 1 tsp of baking soda with 1 quart of water and spray."),
    }
    disease_name, remedy = diseases.get(prediction, ("Unknown", "No remedy available."))
    return f"Disease: {disease_name}\nRemedy: {remedy}"

# Create GUI window
root = tk.Tk()
root.title("Leaf Disease Prediction")

# Create GUI elements
browse_button = tk.Button(root, text="Browse Image", command=predict_label)
browse_button.pack(pady=10)

label = tk.Label(root, text="Select an image to predict.", font=("Arial", 12))
label.pack(pady=10)

# Start the GUI loop
root.mainloop()
