import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(1)
fps = cam.get(cv2.CAP_PROP_FPS)
print "FPS: {0}".format(fps)
font = cv2.FONT_HERSHEY_SIMPLEX

with open("names.txt") as f:
    content = f.readlines()
content = [x.strip('\n') for x in content] 
while True:
	
    ret, im = cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<65):
            p = content[id-1]
        else:
            p = "Unknown"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im,p,(x,y+h), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.imshow('Face',im) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
