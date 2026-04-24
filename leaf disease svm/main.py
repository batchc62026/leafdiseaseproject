import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import joblib
# Set the path to your finger vein image dataset
#data_dir = "/home/pi/Desktop/leaf disease svm/Dataset"
data_dir = "Dataset"


# Define the image size
img_size = 64

# Load the images and labels into arrays
images = []
labels = []
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for img_file in os.listdir(label_dir):
        img_path = os.path.join(label_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        images.append(img)
        labels.append(int(label))

# Convert the images and labels to numpy arrays
X = np.array(images)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the training and testing data to be compatible with SVM
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Define the SVM classifier with a linear kernel
clf = svm.SVC(kernel='linear')

# Train the SVM classifier
clf.fit(X_train, y_train)

# Save the SVM model
joblib.dump(clf, "finger_vein_svm_model.joblib")

# Test the SVM classifier
y_pred = clf.predict(X_test)

# Calculate the confusion matrix, precision, recall, and F1-score
cm = confusion_matrix(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y), rotation=45)
plt.yticks(tick_marks, np.unique(y))
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plot the accuracy, precision, and recall
plt.figure(figsize=(8, 6))
x_labels = ['Accuracy', 'Precision', 'Recall']
y_values = [accuracy, precision, recall]
plt.bar(x_labels, y_values)
plt.ylim(0, 1)
plt.title('Accuracy, Precision, and Recall')
plt.show()

# Print the precision, recall, and F1-score
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1_score)
print("Accuracy: ", accuracy)


# Add this at the end of main.py for live camera testing

print("\n===== LIVE CAMERA TEST =====")

# Function to find available camera
def find_available_camera():
    """Try to find an available camera on the system"""
    for camera_index in range(5):  # Try indices 0-4
        print(f"Trying camera index {camera_index}...", end=" ")
        cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better compatibility
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Try to grab a frame
        ret, frame = cap.read()
        
        if ret and frame is not None:
            print(f"✓ Success!")
            return cap
        else:
            print("✗ Failed")
            cap.release()
    
    return None

# Initialize camera
cap = find_available_camera()

if cap is None:
    print("\n❌ ERROR: No camera found!")
    print("\nTroubleshooting steps:")
    print("  1. Check if camera is physically connected")
    print("  2. Try restarting the script")
    print("  3. Check Device Manager (search 'devmgmt.msc') for camera devices")
    print("  4. Disable any other applications using the camera")
    print("  5. Update camera drivers from manufacturer website")
else:
    test_count = 0
    correct_count = 0
    
    print("\n✓ Camera ready!")
    print("Instructions:")
    print("  - Press 's' to test a sample and enter the correct class (1-5)")
    print("  - Press 'q' to quit")
    print("\nStarting camera feed...\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture frame")
            break
        
        # Flip frame for mirror effect (optional)
        frame = cv2.flip(frame, 1)
        
        # Preprocess frame same way as training
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (img_size, img_size))
        X_test_cam = resized.reshape(1, -1)
        
        # Predict
        prediction = clf.predict(X_test_cam)[0]
        
        # Display prediction on frame
        cv2.putText(frame, f"Prediction: Class {prediction}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to test, 'q' to quit", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if test_count > 0:
            accuracy_pct = (correct_count / test_count) * 100
            cv2.putText(frame, f"Accuracy: {accuracy_pct:.1f}% ({correct_count}/{test_count})", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.imshow('Leaf Disease Detection - ML Model Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # Press 's' to test a sample
            try:
                true_label = int(input(f"\nModel predicted: Class {prediction}\nEnter actual class (1-5): "))
                if 1 <= true_label <= 5:
                    if true_label == prediction:
                        correct_count += 1
                    test_count += 1
                    accuracy_pct = (correct_count / test_count) * 100
                    print(f"✓ Recorded! Accuracy so far: {correct_count}/{test_count} = {accuracy_pct:.2f}%\n")
                else:
                    print("❌ Invalid class number. Please enter 1-5\n")
            except ValueError:
                print("❌ Invalid input. Please enter a number\n")

    cap.release()
    cv2.destroyAllWindows()

    print("\n===== RESULTS =====")
    if test_count > 0:
        accuracy_pct = (correct_count / test_count) * 100
        print(f"✓ Camera Test Accuracy: {accuracy_pct:.2f}%")
        print(f"  Correct: {correct_count}/{test_count} samples")
    else:
        print("✓ No samples tested")