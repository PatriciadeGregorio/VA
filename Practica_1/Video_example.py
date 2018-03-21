import cv2 as cv
from detector_car_2.Detector import DetectorCascade

cap = cv.VideoCapture('Videos/video1.wmv')

face_cascade_coche = cv.CascadeClassifier('haar/coches.xml')
face_cascade_matricula = cv.CascadeClassifier('haar/matriculas.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    faces_matricula = face_cascade_matricula.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30),
    flags=cv.CASCADE_SCALE_IMAGE)
    faces_coches = face_cascade_coche.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30),
    flags=cv.CASCADE_SCALE_IMAGE)
    dc = DetectorCascade(faces_coches, faces_matricula)
    dc.show_faces(gray)
    # Display the resulting frame
    cv.imshow('frame',gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()