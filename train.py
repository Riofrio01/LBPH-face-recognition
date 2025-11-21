from os import listdir
from pyexpat import features
import cv2

import numpy as np

root_dir="./dataset"

recog=cv2.face.LBPHFaceRecognizer_create()

features=[]
label=[]

i=0

for subfolder in listdir(root_dir):

    folder_path=f"{root_dir}/{subfolder}"

    print(f"------{folder_path}-------")

    for file in listdir(folder_path):

        file_path=f"{folder_path}/{file}"

        image=cv2.imread(file_path,0)

        features.append(image)
        label.append(i)
    
        # print(image)
        # cv2.imshow("project",image)
        # cv2.waitKey()

    i=i+1


    
print(f"features are \n {features}")
print(f"labels are \n {label}")

recog.train(features,np.array( label ))

recog.train(features,np.array( label ))

recog.save("facemodel.yml")
