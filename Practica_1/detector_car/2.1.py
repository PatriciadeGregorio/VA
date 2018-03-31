import cv2 as cv
import numpy as np
import os, os.path
from detector_car.Detector import Detector


#Constantes necesarias para cargar las imagenes
imagenes = []
rutaTraining = "../training/"
formato = ".jpg"
rutaTesting = "../testing/"

imagenes_directorio_training = os.listdir(rutaTraining)
#Se cargan las imagenes de test
for name_img in imagenes_directorio_training:
    m = cv.imread(rutaTraining + name_img, 0)
    if m is not None:
        m = cv.equalizeHist(m)
        imagenes.append(m)

#Creacion de los objetos necesarios para la votacion de Hough
orb = cv.ORB_create(nfeatures=300, nlevels=5, scaleFactor=1.1)
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                  multi_probe_level = 1) #2
search_params = dict(checks=50) # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)


#Creamos un objeto de la clase Detector
d = Detector(orb, flann, imagenes)
#Entrenamos a nuestro sistema
flann = d.training(imagenes)

imagenes_directorio_testing = os.listdir(rutaTesting)

#Se procede al cargado de las imagenes de test
for name_img in imagenes_directorio_testing:
    imgTest = cv.imread(rutaTesting + name_img, 0)
    if imgTest is not None:
        imgTest = cv.equalizeHist(imgTest)
        pos_coche, matrizVotacion = d.test(imgTest=imgTest)
        #print("El coche esta en la coordenada: ", (pos_coche[0] * 10, pos_coche[1] * 10))
        for x in range(imgTest.shape[0]):
            for y in range(imgTest.shape[1]):
                imgTest[pos_coche[1]*10][y] = 0
                imgTest[pos_coche[1]*10 - 10][y] = 0
                imgTest[x][pos_coche[0]*10] = 0
                imgTest[x][pos_coche[0]*10 - 10] = 0
        matrizVotacion[pos_coche[1]][pos_coche[0]] = 0
        pos_coche = np.unravel_index(matrizVotacion.argmax(), matrizVotacion.shape)

        for x in range(imgTest.shape[0]):
            for y in range(imgTest.shape[1]):
                imgTest[pos_coche[1]*10][y] = 255
                imgTest[pos_coche[1]*10 - 10][y] = 255
                imgTest[x][pos_coche[0]*10] = 255
                imgTest[x][pos_coche[0]*10 - 10] = 255
        cv.imshow("Coche: " + str(name_img), imgTest)
        cv.waitKey()
        print ("El archivo: " + name_img + " no es una imagen")
