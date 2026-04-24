import cv2
import urllib.request
import numpy as np
import requests
import joblib
import time

def nothing(x):
    pass

# Set the path to the SVM model file
model_path = "finger_vein_svm_model.joblib"

# Define the image size
img_size = 64

# Load the SVM model
clf = joblib.load(model_path)



# Create a function to browse for an image file and predict its label
def predict_label():
    # Ask the user to select an image file

    img = cv2.imread(frame, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    
    # Convert the image to a numpy array
    X = np.array([img])
    
    # Reshape the data to be compatible with SVM
    X = X.reshape(X.shape[0], -1)
    
    # Predict the label of the image
    y_pred = clf.predict(X)
    
    print("Predicted label: " + str(y_pred[0]))

vs = cv2.VideoCapture(0)
time.sleep(2)
 
while True:
    (grabbed, frame) = vs.read()
    cv2.imshow('input',frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('test.jpg',frame)
        cv2.waitKey(1)
        img = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        
        # Convert the image to a numpy array
        X = np.array([img])
        
        # Reshape the data to be compatible with SVM
        X = X.reshape(X.shape[0], -1)
        
        # Predict the label of the image
        y_pred = clf.predict(X)
        
        print("Predicted label: " + str(y_pred[0]))
        if(str(y_pred[0])=='1'):
           print('abc')
        if(str(y_pred[0])=='2'):
           print('def')
        if(str(y_pred[0])=='3'):
           print('ee')
        if(str(y_pred[0])=='4'):
           print('ff')
        if(str(y_pred[0])=='5'):
           print('gg')
        
