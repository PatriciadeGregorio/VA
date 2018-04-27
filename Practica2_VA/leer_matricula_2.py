import cv2 as cv
import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

h_min = 160
h_max = 230
w_min = 24
w_max = 115
w_img = 1200
h_img = 400
white_color = (255, 255, 255)
black_color = (0, 0, 0)
kernel = (9, 9)
white_value = 255
num_colors = 8
scale_factor = 1.07
neighbors = 6

def clean_img(img, threshold_value):
    img = cv.resize(img, None, fx=1200.0/img.shape[1], fy=400.0/img.shape[0], interpolation=cv.INTER_LINEAR)
    img = cv.equalizeHist(img)
    img = cv.cvtColor(reduce_colors(cv.cvtColor(img, cv.COLOR_GRAY2BGR), 8), cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (9, 9), 0)
    ret3, img = cv.threshold(img, threshold_value, 255, cv.THRESH_BINARY)
    img = cv.bitwise_not(img)
    return img
def clean_digito(img):
    img = cv.bilateralFilter(img, 1, 750, 7, borderType=cv.BORDER_CONSTANT)
    ret3, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return img

def reduce_colors(img, n):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def get_class(lista_digitos_resize):
    alphabet = []
    digitos_comprimidos = []
    letters_rep= []
    i = 0
    for k in range(10):
        for j in range (250):
            alphabet.append(str(k))

    letters = [chr(ord('A') + i) for i in range(25, -1, -1)]
    letters.insert(5, 'ESP')

    j = 0
    for j in range(250):
        letters_rep = letters_rep + letters
    letters_rep.sort()

    for digito in lista_digitos_resize:
        converted = digito.flatten()
        digitos_comprimidos.append(converted)

    return (digitos_comprimidos, alphabet+letters_rep)

def imprimir_fichero(prediccion):
    f = open("kk.txt", "a")
    for p in prediccion:
        f.write(p + "\n")
    f.close()

def detectar_num_matriculas():
    listFiles = os.listdir("testing_ocr")
    face_cascade_matriculas = cv.CascadeClassifier('matriculas.xml')
    listaDigitos = []
    resultado = []
    i = 0
    img = cv.imread("testing_ocr/" + 'frontal_12.jpg', 0)
    img = cv.equalizeHist(img)

    facesMatricula = face_cascade_matriculas.detectMultiScale(img, scaleFactor=1.07, minNeighbors=6)
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
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if ((h >= 160) & (h <= 230)) & ((w >= 70) & (w <= 115)):
                digito = imgCleaned[y:y + h, x:x + w]
                digito = tratamiento_digito_1(digito)

                digito = digito.flatten()
                listaDigitos.append((digito, x))
            else:
                digito = imgCleaned[y:y + h, x:x + w]

        #PARA RECONOCER MAS CASOS
        num_digitos = len(listaDigitos)
        threshold_value = 90
        while (num_digitos <= 5):
            imgCleaned = clean_img(imgCropped, threshold_value)
            contours = cv.findContours(imgCleaned, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[1]
            listaDigitos = []
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                if ((h >= h_min) & (h <= h_max)) & ((w >= w_min) & (w <= w_max)):
                    digito = imgCleaned[y:y + h, x:x + w]

                    plt.imshow(digito, cmap='gray')
                    plt.show()
                    digito = tratamiento_digito_1(digito)
                    plt.imshow(digito, cmap='gray')
                    plt.show()
                    digito = digito.flatten()
                    listaDigitos.append((digito, x))
            num_digitos = len(listaDigitos)
            if (num_digitos <= 4):
                threshold_value = threshold_value - 10

    sorted_by_second = sorted(listaDigitos, key=lambda listaDigitos: listaDigitos[1])
    listaDigitos = []
    for t in sorted_by_second:
        resultado.append(t[0])
    return resultado


def tratamiento_digito(imgThreshold, y, h, x, w):
    digito = imgThreshold[y:y + h, x:x + w]
    digito = 255 - digito
    height = digito.shape[0]
    resize = cv.resize(digito, (5, 10), interpolation=cv.INTER_LINEAR)
    rango = 10 - resize.shape[1]
    #resize = np.pad(resize, [(0, 0), (rango, 0)], mode='constant', constant_values=255)
    return resize

def tratamiento_digito_1(digito):
    height = digito.shape[0]
    digito = 255 - digito
    #v = round((digito.shape[1] * 10) / height)
    resize = cv.resize(digito, (5, 10), interpolation=cv.INTER_LINEAR)
    rango = 10 - resize.shape[1]
    #resize = np.pad(resize, [(0, 0), (rango, 0)], mode='constant', constant_values=255)
    return resize

#Redimensionar las imagenes de training_ ocr
def reconocimiento_digitos():
    i = 0
    z = 0
    training_ocr = os.listdir("training_ocr")
    training_sorted = sorted(training_ocr)
    lista_digitos= []

    for image_training in training_sorted:
        img = cv.imread("training_ocr/" + image_training, 0)
        imgThreshold = clean_digito(img)
        im2, contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        valid = 0
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if ((cv.contourArea(cnt)>=4) & ((w/h) <1) & (not valid)):
                digito_tratado = tratamiento_digito(imgThreshold, y, h, x, w)
                lista_digitos.append(digito_tratado)
                i = i + 1
                if (valid):
                    print('bounding box repetidos', image_training)
                valid = 1

        if (not valid):
            z = z + 1
            print(z)
            print('NOT VALID', image_training)

    digitos_comprimidos = get_class(lista_digitos)
    clf = LinearDiscriminantAnalysis()
    clf.fit(digitos_comprimidos[0], digitos_comprimidos[1])

    resultado = detectar_num_matriculas()

    prediction = clf.predict(resultado)
    imprimir_fichero(prediction)

reconocimiento_digitos()
