import cv2 as cv
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LicensePlateNumbersDetector:

    __predictor = LinearDiscriminantAnalysis()
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

    def train(self, lista_digitos_entrenamiento):
        lista_digitos = []
        for image_training in lista_digitos_entrenamiento:
            imgThreshold = self.__clean_digito(image_training)
            im2, contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            valid = 0
            for cnt in contours:
                x, y, w, h = cv.boundingRect(cnt)
                if ((cv.contourArea(cnt) >= 4) & ((w / h) < 1) & (not valid)):
                    digito_tratado = self.__tratamiento_digito(imgThreshold, y, h, x, w)
                    lista_digitos.append(digito_tratado)
                    valid = 1

        digitos_comprimidos = self.__get_class(lista_digitos)
        self.__predictor = LinearDiscriminantAnalysis()
        self.__predictor.fit(digitos_comprimidos[0], digitos_comprimidos[1])

    def predict_numbers(self, number):
        prediction = self.__predictor.predict(number)
        return prediction

    def __get_class(self, lista_digitos_resize):
        alphabet = []
        digitos_comprimidos = []
        letters_rep = []
        i = 0
        for k in range(10):
            for j in range(250):
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

        return (digitos_comprimidos, alphabet + letters_rep)
    def __tratamiento_digito(self, imgThreshold, y, h, x, w):
        digito = imgThreshold[y:y + h, x:x + w]
        digito = 255 - digito
        resize = cv.resize(digito, (5, 10), interpolation=cv.INTER_LINEAR)
        return resize

    def __clean_digito(self, img):
        img = cv.bilateralFilter(img, 1, 750, 7, borderType=cv.BORDER_CONSTANT)
        ret3, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        return img