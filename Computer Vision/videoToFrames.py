# -*- coding: utf-8 -*-
import cv2
import math

videoFile = "video1.mp4"
imagesFolder = "Frames"                 #address of the folder where the images.
capture = cv2.VideoCapture(videoFile)
frameRate = capture.get(5)              #frame rate
while(capture.isOpened()):
    frameId = capture.get(1)            #current frame number
    ret, frame = capture.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename = imagesFolder + "\\image_" +  str(int(frameId)) + ".jpg"
        cv2.imwrite(filename, frame)
capture.release()
print ("Done!")