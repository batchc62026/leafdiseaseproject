import cv2
import numpy as np
import tensorflow as tf

# ================= LOAD TFLITE MODEL =================
interpreter = tf.lite.Interpreter(model_path="leaf_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ================= CLASS NAMES =================
class_names = [
    "Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Bacterial_Spot",
    "Tomato_Leaf_Mold",
    "Potato_Early_Blight",
    "Potato_Late_Blight",
    "Pepper_Bacterial_Spot",
    "Tomato_Yellow_Leaf_Virus",
    "Tomato_Septoria_Leaf_Spot"
]

# ================= OPEN WEBCAM =================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("Camera Started. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for model
    img = cv2.resize(frame, (160, 160))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.argmax(output)
    confidence = np.max(output)

    disease_name = class_names[prediction]

    # Display result
    cv2.putText(frame,
                f"{disease_name} ({confidence*100:.1f}%)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.imshow("Leaf Disease Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()