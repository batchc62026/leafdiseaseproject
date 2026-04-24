import RPi.GPIO as GPIO
import time
import urllib.request
import Adafruit_DHT
import spidev

# ================= GPIO SETUP =================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Buzzer
buz = 2

# Servo
servo_pin = 17

GPIO.setup(buz, GPIO.OUT)
GPIO.setup(servo_pin, GPIO.OUT)

# ================= SERVO SETUP =================
pwm = GPIO.PWM(servo_pin, 50)  # 50Hz
pwm.start(0)

def set_angle(angle):
    duty = 2 + (angle / 18)
    GPIO.output(servo_pin, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(servo_pin, False)
    pwm.ChangeDutyCycle(0)

# ================= MCP3208 SETUP =================
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_adc(channel):
    adc = spi.xfer2([6 | (channel >> 2), (channel & 3) << 6, 0])
    data = ((adc[1] & 15) << 8) + adc[2]
    return data

# ================= LOOP =================
last_servo_time = time.time()

while True:

    # -------- DHT SENSOR --------
    humidity, temperature = Adafruit_DHT.read_retry(Adafruit_DHT.DHT11, 4)

    if humidity is not None and temperature is not None:
        humidity = int(humidity)
        temperature = int(temperature)
    else:
        humidity = 0
        temperature = 0

    # -------- MCP3208 ANALOG SENSORS --------
    soil = read_adc(0)
    rain_analog = read_adc(1)
    ldr = read_adc(2)
    ph_raw = read_adc(3)

    # -------- pH CALCULATION --------
    voltage = (ph_raw * 3.3) / 4095
    ph_value = round(5.0 * voltage, 2)

    print("Temperature:", temperature)
    print("Humidity:", humidity)
    print("Soil Moisture:", soil)
    print("Rain Analog:", rain_analog)
    print("LDR:", ldr)
    print("pH Raw:", ph_raw)
    print("pH Value:", ph_value)
    print("--------------------------------")

    # -------- SERVO ROTATION EVERY 10 SECONDS --------
    current_time = time.time()
    if current_time - last_servo_time >= 10:
        print("Servo Rotating 0 → 90 → 0")
        set_angle(0)
        set_angle(90)
        time.sleep(2)
        set_angle(0)
        last_servo_time = current_time

    # -------- THINGSPEAK UPLOAD --------
    try:
        base_url = "thinkspeak url "
        api_key = "public api key needed"

        url = base_url + "?api_key=" + api_key
        url += "&field1=" + str(temperature)
        url += "&field2=" + str(humidity)
        url += "&field3=" + str(soil)
        url += "&field4=" + str(rain_analog)
        url += "&field5=" + str(ldr)
        url += "&field6=" + str(ph_value)

        response = urllib.request.urlopen(url)
        result = response.read().decode()

        if result == "0":
            print("ThingSpeak rejected data (Check timing/API key)")
        else:
            print("Uploaded Successfully, Entry ID:", result)

    except Exception as e:
        print("ThingSpeak Upload Failed")
        print("Error:", e)

    time.sleep(15)  # ThingSpeak free limit
