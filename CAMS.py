import numpy as np
import cv2, os, os.path
from PIL import Image

# WRITE NAMES TO FILE and RECOGNIZE AND CHECK NAME WITH LINE NUMBER
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    ids=[]
    Id_old = -1
    files = open("names.txt","w+")
    #now looping through all the image paths and loading the ids and the images
    for imagePath in imagePaths:
        # Updates in Code
        # ignore if the file does not have jpg extension :
        if(os.path.split(imagePath)[-1].split(".")[-1]!='jpg'):
            continue

        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[0])
        name = str(os.path.split(imagePath)[-1].split(".")[1])
        if Id_old != Id:
            files.write(str(name) + "\n")
        Id_old = Id
        # extract the face from the training image sample
        faces = detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            ids.append(Id)
    files.close()
    return faceSamples,ids

def train():
	faces,ids = getImagesAndLabels('dataSet')
	recognizer.train(faces, np.array(ids))
	recognizer.write('recognizer/trainer.yml')

def dataset():
	cap = cv2.VideoCapture(camSelect)
	ID = 1+((len(os.walk("dataSet/").next()[2]))/50)
	name = raw_input("\nEnter Name\n\n---> ")
	SampleNum = 0

	while(True):
		ret, img = cap.read()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector.detectMultiScale(gray, 1.3, 5)
		for (x,y,w,h) in faces:
			SampleNum += 1
			cv2.imwrite("dataSet/" + str(ID) + "." + str(name) + "."  + str(SampleNum) + ".jpg", gray[y:y+h,x:x+w])
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			cv2.waitKey(100)
		cv2.imshow('frame',img)
		cv2.waitKey(1)
		if(SampleNum>49):
			cap.release()
			cv2.destroyAllWindows()
			break

def recog():
	recognizer.read('recognizer/trainer.yml')
	cascadePath = "haarcascade_frontalface_default.xml"
	faceCascade = cv2.CascadeClassifier(cascadePath);


	cam = cv2.VideoCapture(camSelect)
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

def UI():
	camSelect = (input("Select a task:\n1. Laptop\n2. Webcam\n\n---> "))-1
	while (True):
		x = input("Select a task:\n1. Add People to Database\n2. Train Recognizer\n3. Recognizer\n4. Quit\n\n---> ")
		if (x==1):
			dataset()
		elif(x==2):
			train()
		elif(x==3):
			recog()
		else:
			break

camSelect = 0
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
UI()