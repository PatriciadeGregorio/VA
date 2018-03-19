import cv2 as cv
from Practica_1.detector_car_2.Detector import DetectorCascade

cap = cv.VideoCapture('Videos/video2.wmv')

face_cascade = cv.CascadeClassifier('haar/coches.xml')
face_cascade_matricula = cv.CascadeClassifier('haar/matriculas.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    faces_matricula = face_cascade.detectMultiScale(gray, 1.4, 10)
    #faces_coches = face_cascade.detectMultiScale(gray, 1.1, 5)
    dc = DetectorCascade([], faces_matricula)
    dc.show_faces(gray)
    # Display the resulting frame
    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()