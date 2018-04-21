import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
def clean_img(img):

    img = cv.resize(img, None, fx=1200.0/img.shape[1], fy=400.0/img.shape[0], interpolation=cv.INTER_CUBIC)
    cv.imshow("Reescalada", img)
    img = cv.equalizeHist(img)
    cv.imshow("Ecualizada", img)
    # #imgCropped = reduce_colors(imgCropped, 8)
    img = cv.cvtColor(reduce_colors(cv.cvtColor(img, cv.COLOR_GRAY2BGR), 8), cv.COLOR_BGR2GRAY)
    # cv.imshow("A 8 colores", img)
    img = cv.GaussianBlur(img, (5, 5), 0)
    ret3, img = cv.threshold(img, 87, 255, cv.THRESH_BINARY)
    # imgCroppedThreshold = cv.adaptiveThreshold(imgCropped, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imshow("Umbralizada", img)    # cv.imshow("Umbralizada", img)
    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # img = cv.erode(img, kernel, iterations = 1)
    # cv.imshow("Erosionada", img)
    # cv.waitKey()

    img = cv.bitwise_not(img)
    cv.imshow("Erosionada", img)
    cv.waitKey()
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

listFiles = os.listdir("testing_full_system")
face_cascade_matriculas = cv.CascadeClassifier('matriculas.xml')
listaDigitos = []
for file in listFiles:
    img = cv.imread("testing_full_system/" + file, 0)
    img = cv.equalizeHist(img)
    plt.imshow(img, cmap='gray')
    plt.show()
    imgRGB = cv.imread("testing_full_system/" + file)
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
        imgCleaned = clean_img(imgCropped)
        contours = cv.findContours(imgCleaned, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[1]
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            # print("Bounding box", x, y, w, h)
            if ((h >= 75) & (h <= 150)) & ((w >= 25) & (w <= 125)):
                digito = imgCleaned[y:y + h, x:x + w]
                # cv.rectangle(imgCroppedThreshold, (x, y), (x + w, y + h), (0, 255, 0), 1)
                pixelesTotales = digito.shape[0] * digito.shape[1]
                pixelesNegros = round(cv.countNonZero(digito) / pixelesTotales * 100)
                pixelesBlancos = round((pixelesTotales - cv.countNonZero(digito)) / pixelesTotales * 100)

                if (pixelesNegros <= 20) | (pixelesNegros >= 80):
                    print("Pixeles blancos: ", pixelesBlancos, " Pixeles negros: ", pixelesNegros)

                else:
                    #print("RECONOCIDA: Pixeles blancos: ", pixelesBlancos, " Pixeles negros: ", pixelesNegros)
                    listaDigitos.append(digito)
                    cv.rectangle(imgCleaned, (x, y), (x + w, y + h), (255, 255, 255), 2)

        #print("-----------------")
        plt.imshow(imgCleaned, cmap='gray')
        plt.show()
        i = 0
        for d in listaDigitos:
            i = i + 1
            #print(cv.countNonZero(d))
            plt.imshow(d, cmap='gray')
            plt.show()
        listaDigitos = []