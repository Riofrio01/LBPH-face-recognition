import cv2

from apidemo import insert_record

cam=cv2.VideoCapture(0)

cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recog=cv2.face.LBPHFaceRecognizer_create()

recog.read("facemodel.yml")

names={0:"Riofrio, Andrei", 1:"Maguire, Tobey", 2:"Garfield, Andrew"}

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


            if confi <30 :
                detected_name=names[id]
                ret=insert_record( detected_name)

                if ret=="0":
                    msg="Attendance recorded successfully"
                else:
                    msg="Attendance recorded for the day"
            else:
                detected_name="Unknown"

            print(f"id : {id}, confi : {confi}, detected_name : {detected_name}")

            text = detected_name
            org = (50,50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (0, 100, 0) # GREEN COLOR TEXT
            thickness = 2
            line_type = cv2.LINE_AA

            cv2.putText(frame, text, (50,100), font, font_scale, (255,0,0), thickness, line_type )

            if detected_name=="Unknown":
                pass
            else:
                cv2.putText(frame, msg, org, font, font_scale, color, thickness, line_type )
            cv2.imshow("Test Image",frame)
            k=cv2.waitKey(2)

            if k==ord('q'):
                break
       


cam.release()