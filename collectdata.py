import cv2
import random

cam=cv2.VideoCapture( 0 )

cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:

    flag,frame=cam.read()

    if flag:

      
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=cascade.detectMultiScale( gray,1.1,5)

        if len(faces)>0:


            x,y,w,h=faces[0]

            cv2.rectangle( frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imshow("Project",frame)
            k=cv2.waitKey(2)

            if k==ord('q'):
                break

            if k==ord('s'):

                roi=frame[y:y+h,x:x+w]
                roi=cv2.resize(roi, (300,300))

                n=random.randint(1,100)
                filename=f"./dataset/0/person-{n}.jpg"
                cv2.imwrite(filename,roi)

cam.release()