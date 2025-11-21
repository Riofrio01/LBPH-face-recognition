import cv2

cam=cv2.VideoCapture(0)

cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recog=cv2.face.LBPHFaceRecognizer_create()

recog.read("facemodel.yml")

while True:

    flag,frame=cam.read()

    if flag:


        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=cascade.detectMultiScale( gray,1.1,5)

        if len(faces)>0:

            x,y,w,h=faces[0]
            cv2.rectangle( frame,(x,y),(x+w,y+h),(0,255,0),2)


            roi=gray[y:y+h,x:x+w]

            roi=cv2.resize(roi,(300,300))

            id,confi=recog.predict(roi)

            print(f"id : {id}, confi : {confi}")

            cv2.imshow("Test Image",frame)
            k=cv2.waitKey(2)

            if k==ord('q'):
                break
       


cam.release()