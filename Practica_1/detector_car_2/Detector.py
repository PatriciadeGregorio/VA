import cv2 as cv
class DetectorCascade:
    __faces_coches = None
    __faces_matricula = None
    def __init__(self, faces_coches, faces_matricula):
        self.__faces_coches = faces_coches
        self.__faces_matricula = faces_matricula


    def __get_faces(self):
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
        listFaces = self.__get_faces()
        for face in listFaces[0]:
            cv.rectangle(imgTest, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)
        for face in listFaces[1]:
            cv.rectangle(imgTest, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)