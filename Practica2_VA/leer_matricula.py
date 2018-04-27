import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
h_min = 160;
h_max = 230;
w_min = 24;
w_max = 115;
w_img = 1200;
h_img = 400;
white_color = (255, 255, 255)
black_color = (0, 0, 0)
kernel = (9, 9)
white_value = 255
num_colors = 8
scale_factor = 1.07
neighbors = 6
#Matriculas: 1867CGS, 0195CDT, 9597CNW, 2387DHJ, 1512DLC, 1851DGZ, 3069CDF, 2976DBT

def clean_img(img, threshold_value):
    img = cv.resize(img, None, fx=w_img/img.shape[1], fy=h_img/img.shape[0], interpolation=cv.INTER_LINEAR)
    cv.imshow("Reescalada", img)
    img = cv.equalizeHist(img)
    cv.imshow("Ecualizada", img)
    img = cv.cvtColor(reduce_colors(cv.cvtColor(img, cv.COLOR_GRAY2BGR), num_colors), cv.COLOR_BGR2GRAY)
    cv.imshow("8 colores", img)
    img = cv.GaussianBlur(img, kernel, 0)
    ret3, img = cv.threshold(img, threshold_value, white_value, cv.THRESH_BINARY)
    cv.imshow("Umbralizada", img)
    img = cv.bitwise_not(img)
    cv.imshow("Negativa final", img)
    cv.waitKey()
    return img
def reduce_colors(img, n):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret,label,center=cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def detectar_num_matriculas():
    listFiles = os.listdir("testing_ocr")
    face_cascade_matriculas = cv.CascadeClassifier('matriculas.xml')

    for file in listFiles:
        file = "frontal_1.jpg"
        img = cv.imread("testing_ocr/" + file, 0)
        img = cv.equalizeHist(img)
        plt.imshow(img, cmap='gray')
        plt.show()
        imgRGB = cv.imread("testing_ocr/" + file)
        facesMatricula = face_cascade_matriculas.detectMultiScale(img, scaleFactor=scale_factor, minNeighbors=neighbors)
        listFacesMatricula = []
        for (x_, y_, w_, h_) in facesMatricula:
            x = x_
            y = y_
            w = w_
            h = h_
            listFacesMatricula.append((x, y, w, h))
        for face in listFacesMatricula:
            imgCropped = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            imgCleaned = clean_img(imgCropped, 127)
            contours = cv.findContours(imgCleaned, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[1]
            listaDigitos = []
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                if ((h >= h_min) & (h <= h_max)) & ((w >= w_min) & (w <= w_max)):
                    digito = imgCleaned[y:y + h, x:x + w]
                    cv.rectangle(imgCleaned, (x, y), (x + w, y + h), white_color, 2)
                    listaDigitos.append(digito)
                # else:
                #     digito = imgCleaned[y:y + h, x:x + w]
                #     cv.rectangle(imgCleaned, (x, y), (x + w, y + h), black_color, 2)
                #     listaDigitos.append(digito)

            num_digitos = len(listaDigitos)
            while (num_digitos <= 5):
                imgCleaned = clean_img(imgCropped, 90)
                contours = cv.findContours(imgCleaned, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[1]
                listaDigitos = []
                for cnt in contours:
                    x, y, w, h = cv.boundingRect(cnt)
                    if ((h >= h_min) & (h <= h_max)) & ((w >= w_min) & (w <= w_max)):
                        digito = imgCleaned[y:y + h, x:x + w]
                        cv.rectangle(imgCleaned, (x, y), (x + w, y + h), white_color, 2)
                        listaDigitos.append(digito)
                    # else:
                    #     digito = imgCleaned[y:y + h, x:x + w]
                    #     cv.rectangle(imgCleaned, (x, y), (x + w, y + h), black_color, 2)
                    #     listaDigitos.append(digito)
                num_digitos = len(listaDigitos)
                if (num_digitos <= 4):
                    threshold_value = threshold_value - 10


                    #cv.rectangle(imgCleaned, (x, y), (x + w, y + h), white_color, 2)

            plt.imshow(imgCleaned, cmap='gray')
            plt.show()
            i = 0
            for d in listaDigitos:
                i = i + 1
                plt.imshow(d, cmap='gray')
                plt.show()
            listaDigitos = []

detectar_num_matriculas();