import cv2 as cv
import numpy as np
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
        for imagen in imagenes:
            kps = self.__orb.detect(imagen, None)
            kps, des = self.__orb.compute(imagen, kps)

            self.descriptorsKP.append(des)
            self.keyPoints.append(kps)
        self.__flann.add(self.descriptorsKP)

    def test(self, imgTest):
        # Testing
        kpsTest = self.__orb.detect(imgTest, None)
        kpsTest, desTest = self.__orb.compute(imgTest, kpsTest)
        tuplaKD = zip(kpsTest, desTest)
        matrizVotacion = np.zeros((np.int(imgTest.shape[0] / self.__DIVISION_MATRIZ), np.int(imgTest.shape[1] / self.__DIVISION_MATRIZ)), dtype=int)
        for t in tuplaKD:
            listaParecidos = self.__flann.knnMatch(t[1], k=3)
            for parecido in listaParecidos:
                for p in parecido:
                    vector_votacion = self.__get_vector_votacion(self.keyPoints[p.imgIdx][p.trainIdx], t[0], p.imgIdx)
                    if (vector_votacion[0] >= 0) & (vector_votacion[1] >= 0) & (
                            vector_votacion[0] < (imgTest.shape[0] / self.__DIVISION_MATRIZ) - 1) & (
                            vector_votacion[1] < (imgTest.shape[1] / self.__DIVISION_MATRIZ) - 1):
                        matrizVotacion[vector_votacion[0], vector_votacion[1]] += 1
        pos_coche = np.unravel_index(matrizVotacion.argmax(), matrizVotacion.shape)
        return pos_coche

    def __get_vector_votacion(self, kpParecido, kp, imgIdx):
        img = self.__imagenes[imgIdx]
        dimensiones = img.shape
        centro = (dimensiones[0] / 2, dimensiones[1] / 2)
        vector_v = (centro[0] - kpParecido.pt[0], centro[1] - kpParecido.pt[1])
        return (np.int((kp.pt[0] + vector_v[0]) / self.__DIVISION_MATRIZ), np.int((kp.pt[1] + vector_v[1])/ self.__DIVISION_MATRIZ))

