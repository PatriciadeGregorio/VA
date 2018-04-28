import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
class LicensePlateCleaner:
    __h_min = 160
    __h_max = 230
    __w_min = 24
    __w_max = 115
    __w_img = 1200
    __h_img = 400
    __white_color = (255, 255, 255)
    __black_color = (0, 0, 0)
    __kernel = (9, 9)
    __white_value = 255
    __num_colors = 8
    __scale_factor = 1.07
    __neighbors = 6
    def __clean_img(self, img, threshold_value):
        img = cv.resize(img, None, fx=self.__w_img / img.shape[1], fy=self.__h_img / img.shape[0], interpolation=cv.INTER_LINEAR)
        img = cv.equalizeHist(img)
        img = cv.cvtColor(self.__reduce_colors(cv.cvtColor(img, cv.COLOR_GRAY2BGR), 8), cv.COLOR_BGR2GRAY)
        img = cv.GaussianBlur(img, self.__kernel, 0)
        ret3, img = cv.threshold(img, threshold_value, self.__white_value, cv.THRESH_BINARY)
        img = cv.bitwise_not(img)
        # cv.imshow("Imagen final", img)
        # cv.waitKey()
        return img

    def __reduce_colors(self, img, n):
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = n
        ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        return res2

    def __tratamiento_digito_1(self, digito):
        digito = 255 - digito
        resize = cv.resize(digito, (5, 10), interpolation=cv.INTER_LINEAR)
        return resize

    def obtener_num_matriculas(self, coche):
        face_cascade_matriculas = cv.CascadeClassifier('matriculas.xml')
        resultado = []
        i = 0
        listaDigitos = []
        coche = cv.equalizeHist(coche)

        facesMatricula = face_cascade_matriculas.detectMultiScale(coche, scaleFactor=1.07, minNeighbors=6)
        listFacesMatricula = []
        for (x_, y_, w_, h_) in facesMatricula:
            x = x_
            y = y_
            w = w_
            h = h_
            listFacesMatricula.append((x, y, w, h))
        centros_largos = []
        for face in listFacesMatricula:
            imgCropped = coche[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            imgCleaned = self.__clean_img(imgCropped, 127)

            contours = cv.findContours(imgCleaned, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[1]
            centros_largos.append((face[0]/2, face[1]/2, w))
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                if ((h >= self.__h_min) & (h <= self.__h_max)) & ((w >= self.__w_min) & (w <= self.__w_max)):
                    digito = imgCleaned[y:y + h, x:x + w]
                    digito = self.__tratamiento_digito_1(digito)

                    digito = digito.flatten()
                    listaDigitos.append((digito, x))
            num_digitos = len(listaDigitos)
            threshold_value = 90
            num_intentos = 0
            t = 1
            while (num_digitos <= 5) & (num_intentos < 150):
                if (threshold_value > 255):
                    break
                imgCleaned = self.__clean_img(imgCropped, threshold_value)
                contours = cv.findContours(imgCleaned, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)[1]
                listaDigitos = []
                for cnt in contours:
                    x, y, w, h = cv.boundingRect(cnt)
                    if ((h >= self.__h_min) & (h <= self.__h_max)) & ((w >= self.__w_min) & (w <= self.__w_max)):
                        digito = imgCleaned[y:y + h, x:x + w]
                        digito = self.__tratamiento_digito_1(digito)
                        # plt.imshow(digito)
                        # plt.show()
                        digito = digito.flatten()
                        listaDigitos.append((digito, x))
                num_digitos = len(listaDigitos)
                if (num_digitos <= 5):
                    if (threshold_value > 0) & (t):
                        threshold_value = threshold_value - 10
                    else:
                        t = 0
                        if (num_intentos <= 4):
                            threshold_value = threshold_value + 30
                        else:
                            threshold_value = threshold_value + 15
                num_intentos = num_intentos + 1
            sorted_by_second = sorted(listaDigitos, key=lambda listaDigitos: listaDigitos[1])
            if (len(sorted_by_second) > 0):
                resultado = tuple([list(tup) for tup in zip(*sorted_by_second)])[0]
                #resultado = sorted_by_second
            listaDigitos = []
        return resultado, centros_largos
