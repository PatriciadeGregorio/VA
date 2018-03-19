import cv2 as cv
def face_detect(img):
    listFaces= []
    face_cascade = cv.CascadeClassifier('haar/coches.xml')

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    x = -1
    y = -1
    w = -1
    h = -1

    for (x_,y_,w_,h_) in faces:
        x = x_
        y = y_
        w = w_
        h = h_
        listFaces.append((x, y, w, h))
    return listFaces

rutaTesting = "testing/test"
formato = ".jpg"
for i in range(33):
    i = i + 1
    m = cv.imread(rutaTesting + str(i) + formato, 0)
    listFaces = face_detect(m)
    for face in listFaces:
        cv.rectangle(m, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)
    cv.imshow('img', m)
    cv.waitKey(0)
    cv.destroyAllWindows()
