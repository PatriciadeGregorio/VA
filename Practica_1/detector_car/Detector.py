import cv2 as cv
import numpy as np
import math
""" Clase Detector. Clase utilidad que proporciona una serie de metodos para comprobar donde se encuentra un objeto. Las imagenes proporcionadas
 tienen que tener objetos centrados."""
class Detector:
    __orb = None
    __flann = None
    __imagenes = []
    __DIVISION_MATRIZ = 10
    descriptorsKP = []
    keyPoints = []


    def __init__(self, orb, flann, imagenes):
        self.__orb = orb
        self.__flann = flann
        self.__imagenes = imagenes

    def training(self, imagenes):
        """
        training(imagenes) -> FlannBasedMatcher object
        . @brief Metodo que, dado un array de imagenes, se encarga de entrenar al sistema.
        . @param imagenes Array de imagenes para entrenar al sistema
        """
        for imagen in imagenes:
            kps = self.__orb.detect(imagen, None)
            kps, des = self.__orb.compute(imagen, kps)

            self.descriptorsKP.append(des)
            self.keyPoints.append(kps)
        self.__flann.add(self.descriptorsKP)

    def test(self, imgTest):
        """
        test (imgTest) -> posCoche
        . @brief Metodo que, dado una imagen, se encarga de devolver la posicion donde se encuentra el objeto
        . @param imgTest Imagen a testear por el sistema
        """
        # Testing
        kpsTest = self.__orb.detect(imgTest, None)
        kpsTest, desTest = self.__orb.compute(imgTest, kpsTest)
        tuplaKD = zip(kpsTest, desTest)
        matrizVotacion = np.zeros((np.int(imgTest.shape[0] / self.__DIVISION_MATRIZ), np.int(imgTest.shape[1] / self.__DIVISION_MATRIZ)), dtype=int)
        for t in tuplaKD:
            listaParecidos = self.__flann.knnMatch(t[1], k=6)
            for parecido in listaParecidos:
                for p in parecido:
                    vector_votacion = self.__get_vector_votacion(self.keyPoints[p.imgIdx][p.trainIdx], self.descriptorsKP[p.imgIdx][p.trainIdx], t[0], t[1], p.imgIdx)
                    if (vector_votacion[0] >= 0) & (vector_votacion[1] >= 0) & (
                            vector_votacion[0] < (imgTest.shape[0] / self.__DIVISION_MATRIZ) - 1) & (
                            vector_votacion[1] < (imgTest.shape[1] / self.__DIVISION_MATRIZ) - 1):
                        matrizVotacion[vector_votacion[0], vector_votacion[1]] += 1
        pos_coche = np.unravel_index(matrizVotacion.argmax(), matrizVotacion.shape)
        return pos_coche

    def __get_vector_votacion(self, kpParecido, desParecido, kp, des, imgIdx):
        """
        __get_vector_votacion(kpParecido, kp, imgIdx) -> vectorVotacion
        . @brief Funcion privada encargada de devolver el vector encargado de votar en la matriz de votacion (imagen destino)
        . @param kpParecido Keypoint que se encuentra en la imagen de training, listo para calcular el vector hacia el centro de la imagen
        . @param kp Keypoint de la imagen de test, listo para colocar el vector kpParecidoCentro en el y realizar la votacion

        """
        img = self.__imagenes[imgIdx]
        dimensiones = img.shape
        centro = (dimensiones[0] / 2, dimensiones[1] / 2)
        # listKpParecido = list(kpParecido.pt)
        # listKpParecido[0] = listKpParecido[0] * (kp.size / 2) / (kpParecido.size / 2)
        # listKpParecido[1] = listKpParecido[1] * (kp.size / 2) / (kpParecido.size / 2)
        # kpParecido.pt = tuple(listKpParecido)


        vector_v = (centro[0] - kpParecido.pt[0], centro[1] - kpParecido.pt[1])


        listV = list(vector_v)
        listV[0] = (listV[0] / kpParecido.size * kp.size)
        listV[1] = (listV[1] / kpParecido.size * kp.size)

        modulo = np.sqrt(listV[0]**2 + listV[1]**2)
        anguloCentro = np.arctan2(listV[1], listV[0])
        anguloCentro = (anguloCentro - math.radians(kpParecido.angle)) + math.radians(kp.angle)
        listV[0] = modulo * np.cos(anguloCentro)
        listV[1] = modulo * np.sin(anguloCentro)
        vector_v = tuple(listV)

        vectorColocado = np.int((kp.pt[0] + vector_v[0]) / self.__DIVISION_MATRIZ), np.int(
            (kp.pt[1] + vector_v[1]) / self.__DIVISION_MATRIZ)
        return vectorColocado

