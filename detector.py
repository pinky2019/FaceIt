import cv2
import numpy as np

cascadePath='D:\\facedetection\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml'
faceDetect=cv2.CascadeClassifier(cascadePath)
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("D:\\facedetection\\recognizer\\trainingData.yml")
id=0
font=cv2.FONT_HERSHEY_COMPLEX_SMALL
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id, conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="Pinky"
        cv2.putText((img),str(id),(x,y+h),font,3,(0,0,255),1)

    cv2.imshow('Face',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

