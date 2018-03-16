import cv2 as cv
import numpy as np
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2


imagenes = []
rutaTraining = "training/frontal_"
rutaTest = "testing/test"
formato = ".jpg"
#Training
for i in range(48):
    i = i + 1
    m = cv.imread(rutaTraining + str(i) + formato, 0)
    imagenes.append(m)

#Keypoints
orb = cv.ORB_create(nfeatures=300, nlevels=4, scaleFactor=1.3)
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50) # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)

descriptorsKP = []
keyPoints = []
for imagen in imagenes:
    kps = orb.detect(imagen, None)
    kps, des = orb.compute(imagen, kps)

    descriptorsKP.append(des)
    keyPoints.append(kps)
flann.add(descriptorsKP)

def get_vector_votacion(kpParecido, kp, imgIdx):
    img = imagenes[imgIdx]
    dimensiones = img.shape
    centro =(dimensiones[0] / 2, dimensiones[1] / 2)
    vector_v = (centro[0] - kpParecido.pt[0], centro[1] - kpParecido.pt[1])
    return (np.int(kp.pt[0] + vector_v[0]), np.int(kp.pt[1] + vector_v[1]))

#Testing
imgTest = cv.imread(rutaTest + "13" + formato, 0)
kpsTest = orb.detect(imgTest, None)
kpsTest, desTest = orb.compute(imgTest, kpsTest)



tuplaKD = zip(kpsTest, desTest)
matrizVotacion = np.zeros(imgTest.shape, dtype=int)
for t in tuplaKD:
    listaParecidos = flann.knnMatch(t[1], k=3)
    for parecido in listaParecidos:
        #print(parecido)
        for p in parecido:
            vector_votacion = get_vector_votacion(keyPoints[p.imgIdx][p.trainIdx], t[0], p.imgIdx)
            if (vector_votacion[0] >= 0) & (vector_votacion[1] >= 0) & (vector_votacion[0] < imgTest.shape[0]) & (vector_votacion[1] < imgTest.shape[1]):
                matrizVotacion[vector_votacion[0], vector_votacion[1]] += 1
print(np.unravel_index(matrizVotacion.argmax(), matrizVotacion.shape))
posCoche = np.unravel_index(matrizVotacion.argmax(), matrizVotacion.shape)

for x in range(imgTest.shape[0]):
    for y in range(imgTest.shape[1]):
        imgTest[posCoche[0]][y] = 0
        imgTest[x][posCoche[1]] = 0

cv.imshow("", imgTest)
cv.waitKey()