import tkinter as tk
import tkinter.filedialog
import cv2
import numpy as np
import joblib
import RPi.GPIO as  GPIO
import time
import telepot
import serial
import time
import string
import pynmea2
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
servo=2
GPIO.setup(servo, GPIO.OUT)
vs = cv2.VideoCapture(0)
time.sleep(2)
pwm = GPIO.PWM(servo, 100)
pwm.start(5)

def handle(msg):
  global telegramText
  global chat_id
  global receiveTelegramMessage
  
  chat_id = msg['chat']['id']
  telegramText = msg['text']
  
  print("Message received from " + str(chat_id))
  
  if telegramText == "/track":
    ik=0
    while(ik==0):
        port="/dev/ttyUSB0"
        ser=serial.Serial(port, baudrate=9600, timeout=0.5)
        dataout = pynmea2.NMEAStreamReader()
        newdata=str(ser.readline())
        newdata=newdata[2:-5]
        #print(str(newdata[0:6]))

        if newdata[0:6] == "$GPRMC":
                newmsg=pynmea2.parse(newdata)
                lat=newmsg.latitude
                lng=newmsg.longitude
                gps = "Latitude=" + str(lat) + "and Longitude=" + str(lng)
                print(gps)
                bot.sendMessage(chat_id, 'https://www.google.com/maps/search/?api=1&query='+str(lat) + "," + str(lng))
                ik=1
              
  


def capture():
    
    print("Sending photo to " + str(chat_id))
    bot.sendPhoto(chat_id, photo = open('image.jpg', 'rb'))


bot = telepot.Bot('6070988162:AAHHschIkOEjD3yQMmJwy6kiVRUgtmfl27M')
chat_id='6176665415'
bot.message_loop(handle)



#bot.sendMessage(chat_id, 'BOT STARTED')
time.sleep(2)


angle=10
duty = float(angle) / 10.0 + 2.5

pwm.ChangeDutyCycle(duty)

# Set the path to the SVM model file
model_path = "finger_vein_svm_model.joblib"

# Define the image size
img_size = 64

# Load the SVM model
clf = joblib.load(model_path)



LCD_RS = 19
LCD_E  = 13
LCD_D4 = 6
LCD_D5 = 5
LCD_D6 = 11
LCD_D7 = 9

buz=26
GPIO.setup(LCD_E, GPIO.OUT)  # E
GPIO.setup(LCD_RS, GPIO.OUT) # RS
GPIO.setup(LCD_D4, GPIO.OUT) # DB4
GPIO.setup(LCD_D5, GPIO.OUT) # DB5
GPIO.setup(LCD_D6, GPIO.OUT) # DB6
GPIO.setup(LCD_D7, GPIO.OUT) # DB7

GPIO.setup(buz, GPIO.OUT) # DB7
GPIO.output(buz,0)
# Define some device constants
LCD_WIDTH = 16    # Maximum characters per line
LCD_CHR = True
LCD_CMD = False

LCD_LINE_1 = 0x80 # LCD RAM address for the 1st line
LCD_LINE_2 = 0xC0 # LCD RAM address for the 2nd line

# Timing constants
E_PULSE = 0.0005
E_DELAY = 0.0005


def lcd_init():
  # Initialise display
  lcd_byte(0x33,LCD_CMD) # 110011 Initialise
  lcd_byte(0x32,LCD_CMD) # 110010 Initialise
  lcd_byte(0x06,LCD_CMD) # 000110 Cursor move direction
  lcd_byte(0x0C,LCD_CMD) # 001100 Display On,Cursor Off, Blink Off
  lcd_byte(0x28,LCD_CMD) # 101000 Data length, number of lines, font size
  lcd_byte(0x01,LCD_CMD) # 000001 Clear display
  time.sleep(E_DELAY)

def lcd_byte(bits, mode):
  # Send byte to data pins
  # bits = data
  # mode = True  for character
  #        False for command

  GPIO.output(LCD_RS, mode) # RS

  # High bits
  GPIO.output(LCD_D4, False)
  GPIO.output(LCD_D5, False)
  GPIO.output(LCD_D6, False)
  GPIO.output(LCD_D7, False)
  if bits&0x10==0x10:
    GPIO.output(LCD_D4, True)
  if bits&0x20==0x20:
    GPIO.output(LCD_D5, True)
  if bits&0x40==0x40:
    GPIO.output(LCD_D6, True)
  if bits&0x80==0x80:
    GPIO.output(LCD_D7, True)

  # Toggle 'Enable' pin
  lcd_toggle_enable()

  # Low bits
  GPIO.output(LCD_D4, False)
  GPIO.output(LCD_D5, False)
  GPIO.output(LCD_D6, False)
  GPIO.output(LCD_D7, False)
  if bits&0x01==0x01:
    GPIO.output(LCD_D4, True)
  if bits&0x02==0x02:
    GPIO.output(LCD_D5, True)
  if bits&0x04==0x04:
    GPIO.output(LCD_D6, True)
  if bits&0x08==0x08:
    GPIO.output(LCD_D7, True)

  # Toggle 'Enable' pin
  lcd_toggle_enable()

def lcd_toggle_enable():
  # Toggle enable
  time.sleep(E_DELAY)
  GPIO.output(LCD_E, True)
  time.sleep(E_PULSE)
  GPIO.output(LCD_E, False)
  time.sleep(E_DELAY)

def lcd_string(message,line):
  # Send string to display




  message = message.ljust(LCD_WIDTH," ")

  lcd_byte(line, LCD_CMD)

  for i in range(LCD_WIDTH):
    lcd_byte(ord(message[i]),LCD_CHR)



lcd_init()
lcd_byte(0x01,LCD_CMD)
lcd_string("   WELCOME",LCD_LINE_1)



# Create a function to browse for an image file and predict its label
def predict_label():
    # Ask the user to select an image file
    file_path = tk.filedialog.askopenfilename()
    
    # Load the image and resize it
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    
    # Convert the image to a numpy array
    X = np.array([img])
    
    # Reshape the data to be compatible with SVM
    X = X.reshape(X.shape[0], -1)
    
    # Predict the label of the image
    y_pred = clf.predict(X)
    
    # Print the predicted label
    if(str(y_pred[0])=="1"):
      print("Disease Name:Bacterial spot")
      print("Remidies:spray copper")
    
    if(str(y_pred[0])=="2"):
      print("Disease Name:early blight")
      print("Remidies:spray bonide liquid copper fungicide concentrate")

    if(str(y_pred[0])=="3"):
      print("Disease Name:healty")
      print("Remidies:.....")
    if(str(y_pred[0])=="4"):
      print("Disease Name:late blight")
      print("Remidies:spray fungicides")
    if(str(y_pred[0])=="5"):
      print("Disease Name:leaf mold")
      print("Remidies:mix 1 teaspoon of baking soda with 1 quert of water and spray.")



# Create a GUI window
root = tk.Tk()
root.title("Finger Vein Image Prediction")

# Create a button to browse for an image file
browse_button = tk.Button(root, text="Browse", command=predict_label)
browse_button.pack(pady=10)

# Create a label to display the predicted label
label = tk.Label(root, text="")
label.pack(pady=10)

# Start the GUI event loop
root.mainloop()
