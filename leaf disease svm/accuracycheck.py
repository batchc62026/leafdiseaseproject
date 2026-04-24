import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
MODEL_PATH = "finger_vein_svm_model.joblib"
IMG_SIZE = 64

# Mapping from label ID to Disease Name
DISEASE_MAP = {
    1: "Bacterial spot",
    2: "Early blight",
    3: "Healthy",
    4: "Late blight",
    5: "Leaf mold"
}

def load_svm_model():
    """Loads the pre-trained SVM model."""
    if not os.path.exists(MODEL_PATH):
        # Fallback if there is a space in the filename
        alt_path = "finger_vein_svm model.joblib"
        if os.path.exists(alt_path):
            return joblib.load(alt_path)
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return None
    return joblib.load(MODEL_PATH)

def preprocess_frame(frame):
    """Preprocess the camera frame to match the mode training input."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize to the same size used in training
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    # Flatten the image into a 1D array
    flattened = resized.reshape(1, -1)
    return flattened

def run_camera_test(clf):
    """Runs the camera, makes real-time predictions, and records user input for true classes."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open laptop camera.")
        return [], []

    print("\n================ LIVE CAMERA TEST ================")
    print("  Controls:")
    print("  - 's': Capture a sample, see prediction, input reality")
    print("  - 'q': Quit camera & display final metrics plot")
    print("==================================================\n")

    y_true = []
    y_pred = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break
            
        # Get raw prediction
        processed_img = preprocess_frame(frame)
        prediction = int(clf.predict(processed_img)[0])
        disease_name = DISEASE_MAP.get(prediction, f"Unknown Class {prediction}")

        # Display on frame
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (0, 0), (700, 80), (0, 0, 0), -1)
        
        cv2.putText(display_frame, f"Prediction: Class {prediction} - {disease_name}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to capture and test, 'q' to quit", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Leaf Disease - SVM Camera Test", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nEnding live session...")
            break
        elif key == ord('s'):
            print(f"\n--- Sample Captured ---")
            print(f"Model Predicted: Class {prediction} ({disease_name})")
            
            # The cv2 input stops the feed temporarily so we can ask the user in the terminal
            try:
                true_label_str = input("Enter actual plant class (1-5), or '0' to cancel capture: ")
                true_label = int(true_label_str.strip())
                if 1 <= true_label <= 5:
                    y_true.append(true_label)
                    y_pred.append(prediction)
                    print(f"✓ Recorded: True Class={true_label}, Predicted Class={prediction}")
                elif true_label == 0:
                    print("Capture cancelled.")
                else:
                    print("❌ Invalid class. Please enter a number from 1 to 5.")
            except ValueError:
                print("❌ Invalid input. Needs to be a number.")

    cap.release()
    cv2.destroyAllWindows()
    
    return y_true, y_pred

def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        print("\nNo samples were recorded during the camera session. Cannot calculate metrics.")
        return

    print("\n============== EVALUATION METRICS ==============")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"Total Samples Tested: {len(y_true)}")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("==============================================\n")

    # Draw Confusion Matrix
    classes = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Live Camera Test")
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Annotate numeric values inside the cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Class Label')
    plt.xlabel('Predicted Class Label')
    plt.tight_layout()
    plt.show()

    # Bar chart for Accuracy, Precision, Recall, and F1-score
    plt.figure(figsize=(8, 6))
    x_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    y_values = [acc, prec, rec, f1]
    
    # Create bar plot
    bars = plt.bar(x_labels, y_values, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    plt.ylim(0, 1.1)  # slightly above 1 to make room for text
    plt.title('Performance Metrics - Live Camera Test')
    plt.ylabel('Score')
    
    # Add numerical labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontweight='bold')
                 
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trained_model = load_svm_model()
    
    if trained_model:
        y_true, y_pred = run_camera_test(trained_model)
        calculate_metrics(y_true, y_pred)
    else:
        print("\nCould not find the model file.")
