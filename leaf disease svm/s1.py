import cv2
import numpy as np
import joblib
import time
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Motor Pins
m1, m2, m3, m4 = 21, 20, 16, 12

# Ultrasonic Pins
TRIG = 14
ECHO = 15

GPIO.setup([m1, m2, m3, m4], GPIO.OUT)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)

# ================= MOTOR FUNCTIONS =================
def move_forward():
    GPIO.output(m1,1)
    GPIO.output(m2,0)
    GPIO.output(m3,1)
    GPIO.output(m4,0)

def stop_robot():
    GPIO.output(m1,0)
    GPIO.output(m2,0)
    GPIO.output(m3,0)
    GPIO.output(m4,0)

def turn_left():
    GPIO.output(m1,0)
    GPIO.output(m2,1)
    GPIO.output(m3,1)
    GPIO.output(m4,0)

# ================= SAFE ULTRASONIC =================
def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.0002)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.05

    while GPIO.input(ECHO) == 0:
        if time.time() > timeout:
            return 100

    pulse_start = time.time()

    while GPIO.input(ECHO) == 1:
        if time.time() > timeout:
            return 100

    pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 2)

# ================= PICAMERA =================
print("Starting Camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()
time.sleep(2)

print("Robot Started")

# ================= MAIN LOOP =================
while True:

    frame = picam2.capture_array()

    # Fix blue color
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Measure distance
    distance = get_distance()

    # Properly print distance
    print(f"Obstacle Distance: {distance} cm")

    # Show distance on screen
    cv2.putText(frame, f"Distance: {distance} cm",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    # Obstacle logic
    if distance < 20:
        print("Obstacle detected! Turning LEFT for 5 seconds")
        stop_robot()
        turn_left()
        time.sleep(5)
        stop_robot()
        move_forward()
    else:
        move_forward()

    cv2.imshow("Robot Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# ================= CLEANUP =================
picam2.stop()
cv2.destroyAllWindows()
GPIO.cleanup()
