
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
class CarDetector:
    def detectar_coche(self, img):

        face_cascade_matriculas = cv.CascadeClassifier('coches.xml')
        img = cv.equalizeHist(img)

        facesMatricula = face_cascade_matriculas.detectMultiScale(img, scaleFactor=1.1, minNeighbors=6)
        listFacesMatricula = []
        for (x_, y_, w_, h_) in facesMatricula:
            x = x_
            y = y_
            w = w_
            h = h_
            listFacesMatricula.append((x, y, w, h))
        lista_coches = []
        for face in listFacesMatricula:
            lista_coches.append(img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]])
        return lista_coches