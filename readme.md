# CAMS
## Classroom Attendance Monitoring System

### **The Problem**

Instructors take up valuable class time in order to take attendance. However, even with this process there is no guarantee that at any given moment, the instructor is aware of who is present in the classroom.
      
### **Innovation description**

The CAMS (Classroom Attendance Monitoring System) seeks to solve this problem by processing information from strategically placed cameras at entryways of the classroom. The system is able to take attendance as students come in and out of the classroom and keeps a real time updated list of the student count. All of this information is available to administrators on a web app via a login.

### **Features**

CAMS provides real-time monitoring and keeps track of student's presence so that administrators are updated at all times, especially during emergency situations. 

### **Innovation operation**

CAMS uses cameras positioned at the entryways of the classroom to monitor traffic in and out of the class. Facial recognition is performed to achieve this task. The resulting information is updated in real time to a website, hosted on an online web server which only administrators can access and modify. 

### **Required technologies**

* HD Webcam Camera
* NVIDIA Jetson TK1 GPU Development Board
* OpenCV - Image Processing Library

~Abhishek Patel and Zarir Hamza, 2017
### Requirements
1. Python 2.7
2. OpenCV 3.3.1
3. PIL - Python Imaging Library, now Pillow
4. NumPy, latest version
5. Webcam *must* be **640 x 480** or better

## Usage
Run CAMS.py 
Selection Camera:
      
      Select a task:
            1. Laptop
            2. Webcam 
Select Options:

      Select a task:
            1. Add People to Database
            2. Train Recognizer
            3. Recognizer
            4. Quit  
In case of errors, refer back to **Requirements**, or email **abhi12.p@gmail.com** or **zarir1999@gmail.com**.

