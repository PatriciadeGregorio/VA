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
for i in range(5):
    i = i + 1
    m = cv.imread(rutaTraining + str(i) + formato, 0)
    imagenes.append(m)
#Keypoints
orb = cv.ORB_create(nfeatures=300, nlevels=4, scaleFactor=1.3)
# # FLANN parameters
# FLANN_INDEX_LSH = 6
# index_params= dict(algorithm = FLANN_INDEX_LSH,
#                    table_number = 6, # 12
#                    key_size = 12,     # 20
#                    multi_probe_level = 1) #2
# search_params = dict(checks=50) # or pass empty dictionary
# flann = cv.FlannBasedMatcher(index_params, search_params)

distancesKP = []
keyPoints = []
for imagen in imagenes:
    kps = orb.detect(imagen, None)
    kps, des = orb.compute(imagen, kps)

    distancesKP.append(des)
    keyPoints.append(kps)
# flann.add(distancesKP)


imgTest = cv.imread(rutaTest + "2" + formato, 0)
kpTest = orb.detect(imgTest, None)
kpsTest, desTest = orb.compute(imgTest, kpTest)

for i in range(5):
    flann_params = dict(algorithm=1, trees=4)
    flannIndex = cv.flann_Index(distancesKP[i], params=flann_params, distType=None)
    idx, dist = flannIndex.knnSearch(desTest, 1, params={})




