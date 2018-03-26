import cv2 as cv
import numpy as np
from Practica_1.detector_car.Detector import Detector

imagenes = []
rutaTraining = "../training/frontal_"
formato = ".jpg"
for i in range(59):
    i = i + 1
    m = cv.imread(rutaTraining + str(i) + formato, 0)
    m = cv.equalizeHist(m)
    imagenes.append(m)

    #Keypoint
    orb = cv.ORB_create(nfeatures=150, nlevels=4, scaleFactor=1.15)
    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                  multi_probe_level = 1) #2
    search_params = dict(checks=50) # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)

d = Detector(orb, flann, imagenes)
flann = d.training(imagenes)
for z in range(33):
    z = z + 1
    imgTest = cv.imread("../testing/test" + str(z) + formato, 0)
    imgTest = cv.equalizeHist(imgTest)
    pos_coche = d.test(imgTest=imgTest)
    print("El coche esta en la coordenada: ", (pos_coche[0] * 10, pos_coche[1] * 10))
    for x in range(imgTest.shape[0]):
        for y in range(imgTest.shape[1]):
            imgTest[pos_coche[1]*10][y] = 0
            imgTest[pos_coche[1]*10 - 10][y] = 0
            imgTest[x][pos_coche[0]*10] = 0
            imgTest[x][pos_coche[0]*10 - 10] = 0

    cv.imshow("Coche: " + str(z), imgTest)
    cv.waitKey()
