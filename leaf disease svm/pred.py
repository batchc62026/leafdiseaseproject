import cv2
import numpy as np
import joblib

# Set the path to your finger vein image
img_path = "2.bmp"

# Define the image size
img_size = 64

# Load the image and resize it
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (img_size, img_size))

# Convert the image to a numpy array
X = np.array([img])

# Reshape the data to be compatible with SVM
X = X.reshape(X.shape[0], -1)

# Load the SVM model
clf = joblib.load("finger_vein_svm_model.joblib")

# Predict the label of the image
y_pred = clf.predict(X)

# Print the predicted label
print(y_pred[0])
