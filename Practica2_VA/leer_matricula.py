import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
listFiles = os.listdir("testing_ocr")
face_cascade_matriculas = cv.CascadeClassifier('matriculas.xml')
for file in listFiles:
    img = cv.imread("testing_ocr/" + file, 0)
    imgRGB = cv.imread("testing_ocr/" + file)
    imgThreshold = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    im2, contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    facesMatricula = face_cascade_matriculas.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30),flags=cv.CASCADE_SCALE_IMAGE)
    listFacesMatricula = []
    for (x_, y_, w_, h_) in facesMatricula:
        x = x_
        y = y_
        w = w_
        h = h_
        listFacesMatricula.append((x, y, w, h))

    for face in listFacesMatricula:
        imgCropped = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
        imgCroppedThreshold = cv.adaptiveThreshold(imgCropped, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        im2, contours, hierarchy = cv.findContours(imgCroppedThreshold, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if ((h >= 25) & (h <= 30)) & ((w >= 15) & (w <= 25)):
                cv.rectangle(imgCropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv.imshow(file, imgCropped)
                # cv.waitKey()
                plt.imshow(imgCropped)
                plt.show()


