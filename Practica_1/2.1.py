import cv2 as cv
import numpy as np
imagenes = []
ruta = "training/frontal_"
formato = ".jpg"
for i in range(48):
    i = i + 1
    m = cv.imread(ruta + str(i) + formato, 0)
    imagenes.append(m)

orb = cv.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)
for imagen in imagenes:
    kp = orb.detect(imagen, None)
    kp, des = orb.compute(imagen, kp)
    s = cv.drawKeypoints(imagen, kp, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("keypoints", s)
    cv.waitKey()


