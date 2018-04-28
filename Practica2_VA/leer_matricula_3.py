import cv2 as cv
import os
from LicensePlateCleaner import LicensePlateCleaner
from LicensePlateNumbersDetector import LicensePlateNumbersDetector
from CarDetector import CarDetector




def imprimir_testing_ocr():
    lpc = LicensePlateCleaner()
    cd = CarDetector()
    lpnd = LicensePlateNumbersDetector()
    training_numbers = os.listdir("training_ocr")

    lista_digitos_entrenamiento = []
    file_opened = open("testing_ocr.txt", "a")
    for t_n in training_numbers:
        n = cv.imread("training_ocr/" + t_n, 0)
        lista_digitos_entrenamiento.append(n)
    lpnd.train(lista_digitos_entrenamiento)

    # Para las imagenes en testing_ocr
    imagenes = os.listdir("testing_ocr")
    print(imagenes)
    for i in imagenes:
        info_escritura = []
        img = cv.imread("testing_ocr/" + i, 0)
        #lista_coches = cd.detectar_coche(img)
        texto_matricula = ""
        resultado, centros_largos = lpc.obtener_num_matriculas(coche=img)

        numbers = []
        for r in resultado:
            numbers.append(r)
        for n in numbers:
            predict = lpnd.predict_numbers([n])
            texto_matricula = texto_matricula + predict[0]
        if (len(centros_largos) > 0):
            info_escritura.append((i, centros_largos[0][0], centros_largos[0][1], texto_matricula, centros_largos[0][2]))
        for i in info_escritura:
            file_opened.write("<" + str(i[0]) + ">" + "<" + str(i[1]) + ">" + "<" + str(i[2]) + ">" + "<" + str(i[3]) + ">" + "<" + str(i[4]) + ">" + "\n")
            file_opened.flush()
    file_opened.close()
#Para las imagenes en testing_full_system
def imprimir_testing_full_system():
    lpc = LicensePlateCleaner()
    cd = CarDetector()
    lpnd = LicensePlateNumbersDetector()
    training_numbers = os.listdir("training_ocr")
    lista_digitos_entrenamiento = []
    file_opened = open("testing_full_system.txt", "a")
    for t_n in training_numbers:
        n = cv.imread("training_ocr/" + t_n, 0)
        lista_digitos_entrenamiento.append(n)
    lpnd.train(lista_digitos_entrenamiento)
    imagenes = os.listdir("testing_full_system")
    for i in imagenes:
        info_escritura = []
        img = cv.imread("testing_full_system/" + i, 0)
        lista_coches = cd.detectar_coche(img)
        for c in lista_coches:
            # plt.imshow(c)
            # plt.show()
            texto_matricula = ""
            resultado, centros_largos = lpc.obtener_num_matriculas(coche=c)

            numbers = []
            for r in resultado:
                numbers.append(r)
            for n in numbers:
                predict = lpnd.predict_numbers([n])
                texto_matricula = texto_matricula + predict[0]
            if (len(centros_largos) > 0):
                for c_l in centros_largos:
                    info_escritura.append((i, c_l[0], c_l[1], texto_matricula, c_l[2]))
        for i in info_escritura:
             file_opened.write("<" + str(i[0]) + ">" + "<" + str(i[1]) + ">" + "<" + str(i[2]) + ">" + "<" + str(i[3]) + ">" + "<" + str(i[4]) + ">" + "\n")
             file_opened.flush()
    file_opened.close()
imprimir_testing_full_system()
#imprimir_testing_ocr()

