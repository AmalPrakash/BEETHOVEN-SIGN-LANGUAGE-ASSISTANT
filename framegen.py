import cv2
import numpy as np
import os

# object creation, background subtraction
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

video = cv2.VideoCapture(
    "C:\\Users\\LENOVO\\Documents\\GitHub\\OpenCV\\A.mp4")
try:
    if not os.path.exists('A'):
        os.makedirs('A')

except OSError:
    print('Error: Creating directory of dataset')

currentframe = 0

while(True):

    ret, frame = video.read()

    if ret:
        name = './A/bigframe' + str(currentframe) + '.jpg'
        print('Creating...' + name)
        halfsize = cv2.resize(frame, (300, 300))
        img = fgbg.apply(halfsize)
        #img = cv2.flip(halfsize,1)

        cv2.imwrite(name, img)
        currentframe += 1
    else:
        break
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
