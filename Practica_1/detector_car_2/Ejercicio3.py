import cv2 as cv
from detector_car_2.Detector import DetectorCascade

#Cargamos el archivo para cargar el clasificador
face_cascade_coches = cv.CascadeClassifier('../haar/coches.xml')
face_cascade_matriculas = cv.CascadeClassifier('../haar/matriculas.xml')
#Se procede a cargar las imagenes de test para dibujar las caras en el
rutaTesting = "../testing/test"
formato = ".jpg"
for i in range(33):
    i= i + 1
    img = cv.imread(rutaTesting + str(i) + formato, 0)
    #Sacamos las caras correspondientes a los coches y a las matriculas
    facesCoches = face_cascade_coches.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30),
    flags=cv.CASCADE_SCALE_IMAGE)

    facesMatricula = face_cascade_matriculas.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30),
    flags=cv.CASCADE_SCALE_IMAGE)

    #Cargamos el objeto para dibujar dichas caras y posteriormente se muestran
    dc = DetectorCascade(facesCoches, facesMatricula)
    dc.show_faces(img)
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()