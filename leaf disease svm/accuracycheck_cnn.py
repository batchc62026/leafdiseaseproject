import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
# Pointing to the LeafCnnmodel folder on the Desktop
MODEL_PATH = r"C:\Users\pgiri\Desktop\LeafCnnmodel\leaf_model.tflite"

# Mapping from label ID to Disease Name (based on test_model_camera.py)
CLASS_NAMES = {
    0: "Healthy",
    1: "Tomato_Early_Blight",
    2: "Tomato_Late_Blight",
    3: "Tomato_Bacterial_Spot",
    4: "Tomato_Leaf_Mold",
    5: "Potato_Early_Blight",
    6: "Potato_Late_Blight",
    7: "Pepper_Bacterial_Spot",
    8: "Tomato_Yellow_Leaf_Virus",
    9: "Tomato_Septoria_Leaf_Spot"
}

def load_tflite_model():
    """Loads the pre-trained TFLite model and returns the interpreter."""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found!")
        return None, None, None
        
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def preprocess_frame(frame):
    """Preprocess the camera frame to match the mode training input (160x160, normalized)."""
    img = cv2.resize(frame, (160, 160))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def run_camera_test(interpreter, input_details, output_details):
    """Runs the camera, makes real-time predictions with CNN, and records actual labels."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open laptop camera.")
        return [], []

    print("\n================ LIVE CAMERA TEST (CNN) ================")
    print("  Controls:")
    print("  - 's': Capture a sample, see prediction, input reality")
    print("  - 'q': Quit camera & display final metrics plot")
    print("========================================================\n")
    print("Class Mapping Reference:")
    for key, name in CLASS_NAMES.items():
        print(f"  {key}: {name}")
    print("========================================================\n")

    y_true = []
    y_pred = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break
            
        # Get raw prediction
        processed_img = preprocess_frame(frame)
        interpreter.set_tensor(input_details[0]['index'], processed_img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        prediction = int(np.argmax(output))
        confidence = np.max(output)
        disease_name = CLASS_NAMES.get(prediction, f"Unknown Class {prediction}")

        # Display on frame
        display_frame = frame.copy()
        cv2.rectangle(display_frame, (0, 0), (700, 80), (0, 0, 0), -1)
        
        cv2.putText(display_frame, f"Pred: {prediction} ({disease_name}) Conf: {confidence*100:.1f}%", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to capture and test, 'q' to quit", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Leaf Disease - CNN TFLite Camera Test", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nEnding live session...")
            break
        elif key == ord('s'):
            print(f"\n--- Sample Captured ---")
            print(f"Model Predicted: Class {prediction} ({disease_name})")
            
            try:
                true_label_str = input("Enter actual plant class (0-9), or '-1' to cancel capture: ")
                true_label = int(true_label_str.strip())
                if 0 <= true_label <= 9:
                    y_true.append(true_label)
                    y_pred.append(prediction)
                    print(f"✓ Recorded: True Class={true_label}, Predicted Class={prediction}")
                elif true_label == -1:
                    print("Capture cancelled.")
                else:
                    print("❌ Invalid class. Please enter a number from 0 to 9.")
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

    # Draw Confusion Matrix (0 through 9)
    classes = list(range(10)) 
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Live Camera Test (CNN)")
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
    plt.ylim(0, 1.1)
    plt.title('Performance Metrics - Live Camera Test (CNN)')
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
    interpreter, input_details, output_details = load_tflite_model()
    
    if interpreter:
        y_true, y_pred = run_camera_test(interpreter, input_details, output_details)
        calculate_metrics(y_true, y_pred)
    else:
        print("\nCould not load the model.")
