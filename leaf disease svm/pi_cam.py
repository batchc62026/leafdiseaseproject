import cv2
import time

vs = cv2.VideoCapture(0)
time.sleep(2)
usr=input('Enter Name:')
cnt=0
while True:
        
        # read the next frame from the file

        (grabbed, frame) = vs.read()
        cv2.imshow('input',frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
                cnt=cnt+1
                cv2.imwrite('./data/'+str(usr)+"_"+ str(cnt)+'.jpg',frame)
                cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
