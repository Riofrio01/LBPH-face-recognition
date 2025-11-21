
import cv2

recog=cv2.face.LBPHFaceRecognizer_create()

recog.read("facemodel.yml")

test_image=cv2.imread("./dataset/0/person-18.jpg",0)

cv2.imshow("test image",test_image)
cv2.waitKey()


id,confi=recog.predict(test_image)

print(f"id : {id}, confi : {confi}")
