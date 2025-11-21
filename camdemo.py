import cv2

cam=cv2.VideoCapture(0)

cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#   flag,frame=cam.read()

frame=cv2.imread("groupphoto.webp")
flag=True

if flag:


    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=cascade.detectMultiScale(gray,1.1,5)

    print(faces)

    for x,y,w,h in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Myimage",frame)
    cv2.waitKey(0)

else:
    print("Camera is not working !!!")

cam.release()
