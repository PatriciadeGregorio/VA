import cv2 as cv
import numpy as np
imagenes = []
rutaTraining = "training/frontal_"
rutaTest = "testing/test"
formato = ".jpg"
#Training
for i in range(2):
    i = i + 1
    m = cv.imread(rutaTraining + str(i) + formato, 0)
    imagenes.append(m)
#Keypoints
orb = cv.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv.FlannBasedMatcher(index_params, search_params)

distancesKP = []
keyPoints = []
for imagen in imagenes:
    kps = orb.detect(imagen, None)
    kps, des = orb.compute(imagen, kps)
    distancesKP.append(des)
    keyPoints.append(kps)
    flann.add(descriptors=des)
    s = cv.drawKeypoints(imagen, kps, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("keypoints", s)
    cv.waitKey()

#Testing
orbTest = cv.ORB_create(nfeatures=100, nlevels=4, scaleFactor=1.3)
for i in range(2):
    m = cv.imread(rutaTest + "1" + formato, 0)
    kpTest = orbTest.detect(m, None)
    kpTest, desTest = orbTest.compute(m, kpTest)

    matches = cv.flann_Index
    # Need to draw only good matches, so create a mask
    #matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    # ratio test as per Lowe's paper
    # for j, (m, n) in enumerate(matches):
    #     if m.distance < 0.7 * n.distance:
    #         matchesMask[j] = [1, 0]
    #
    # draw_params = dict(matchColor=(0, 255, 0),
    #                    singlePointColor=(255, 0, 0),
    #                    matchesMask=matchesMask,
    #                    flags=0)
    # img3 = cv.drawMatchesKnn(m, kpTest, imagenes[i], keyPoints[i], matches, None, **draw_params)
    # cv.imshow("LA IMAGEN", img3)
    # cv.waitKey()







