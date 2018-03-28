import cv2 as cv
class DetectorCascade:
    __faces_coches = None
    __faces_matricula = None

    def __init__(self):
        faces_coches = cv.CascadeClassifier('../haar/coches.xml')
        faces_matricula = cv.CascadeClassifier('../haar/matriculas.xml')

        self.__faces_coches = faces_coches.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30),
                                                           flags=cv.CASCADE_SCALE_IMAGE)
        self.__faces_matricula = faces_matricula.detectMultiScale(img, scaleFactor=1.05, minNeighbors=6,
                                                                  minSize=(30, 30),
                                                                  flags=cv.CASCADE_SCALE_IMAGE)

    def __init__(self, faces_coches, faces_matricula):
        self.__faces_coches = faces_coches
        self.__faces_matricula = faces_matricula


    def __get_faces(self):
        """
        __get_faces() -> tupla
        . @brief Funcion privada que empaqueta las "caras"

        """
        listFacesCoches = []
        listFacesMatricula = []
        x = -1
        y = -1
        w = -1
        h = -1
        for (x_,y_,w_,h_) in self.__faces_coches:
            x = x_
            y = y_
            w = w_
            h = h_
            listFacesCoches.append((x, y, w, h))
        for (x_,y_,w_,h_) in self.__faces_matricula:
            x = x_
            y = y_
            w = w_
            h = h_
            listFacesMatricula.append((x, y, w, h))

        return (listFacesCoches, listFacesMatricula)

    def show_faces(self, imgTest):
        """
        show_faces(imgTest)
        . @brief Funcion que se encarga de pintar las caras en la imagen de test
        . @param imgTest Imagen a la que se le quiere pintar las caras
        """
        listFaces = self.__get_faces()
        for face in listFaces[0]:
            cv.rectangle(imgTest, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
        for face in listFaces[1]:
            cv.rectangle(imgTest, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)