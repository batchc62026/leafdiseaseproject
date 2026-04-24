import cv2
import numpy as np
import joblib
import time
import requests
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ================= TELEGRAM =================
BOT_TOKEN = "8235870280:AAFNYHc90pAjZiO_bDkwwGubBePS46Y_Qho"
CHAT_ID = "1762714591"

def send_telegram_message(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg},
            timeout=5
        )
    except:
        pass

# ================= GPIO =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motors
m1, m2, m3, m4 = 21, 20, 16, 12
GPIO.setup([m1, m2, m3, m4], GPIO.OUT)

# Ultrasonic
TRIG, ECHO = 14, 15
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# ================= MOTOR =================
def move_forward():
    GPIO.output(m1,1); GPIO.output(m2,0)
    GPIO.output(m3,1); GPIO.output(m4,0)

def stop_robot():
    GPIO.output(m1,0); GPIO.output(m2,0)
    GPIO.output(m3,0); GPIO.output(m4,0)

def turn_left():
    GPIO.output(m1,0); GPIO.output(m2,1)
    GPIO.output(m3,1); GPIO.output(m4,0)

# ================= ULTRASONIC =================
def get_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    start = time.time()
    while GPIO.input(ECHO) == 0:
        if time.time() - start > 0.05:
            return 100

    start = time.time()
    while GPIO.input(ECHO) == 1:
        if time.time() - start > 0.05:
            return 100

    duration = time.time() - start
    return round(duration * 17150, 2)

# ================= CAMERA =================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (320, 240)}  # REDUCED RESOLUTION
))
picam2.start()
time.sleep(2)

# ================= MODEL =================
clf = joblib.load("finger_vein_svm_model.joblib")
img_size = 64

# ================= TIMERS =================
last_disease_check = 0
last_sent_label = None

print("🚀 Optimized Robot Started")

# ================= MAIN LOOP =================
while True:

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # -------- OBSTACLE DETECTION --------
    dist = get_distance()
    print(f"Distance: {dist} cm")

    if dist < 20:
        stop_robot()
        turn_left()
        time.sleep(1.5)
    else:
        move_forward()

    current = time.time()

    # -------- DISEASE CHECK (EVERY 15 SEC) --------
    if current - last_disease_check > 15:

        print("Checking Leaf Disease...")

        stop_robot()
        time.sleep(2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (img_size, img_size))
        X = img.reshape(1, -1)

        label = str(clf.predict(X)[0])

        disease_map = {
            '1': "Bacterial Spot",
            '2': "Early Blight",
            '3': "Healthy",
            '4': "Late Blight",
            '5': "Leaf Mold"
        }

        disease = disease_map.get(label, "Unknown")

        print("Disease:", disease)

        if disease != "Healthy" and disease != last_sent_label:
            send_telegram_message(f"🚨 Disease Detected: {disease}")
            last_sent_label = disease

        move_forward()
        last_disease_check = current

    # -------- DISPLAY --------
    cv2.imshow("Robot Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# ================= CLEANUP =================
GPIO.cleanup()
picam2.stop()
cv2.destroyAllWindows()