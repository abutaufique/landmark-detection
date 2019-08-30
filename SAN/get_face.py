import cv2
import os
import pdb
old_gray = cv2.imread('/home/at7133/Research/landmark-detection/SAN/S010_006_00000001.png', 0)
haarcascades_path = '/home/at7133/Research/facial_symmetry/facial_symmetry/opencv_frontal_face/haarcascade_frontalface_alt.xml'
assert os.path.exists(haarcascades_path)
face_cascade = cv2.CascadeClassifier(haarcascades_path)
pdb.set_trace()
faces = face_cascade.detectMultiScale(old_gray, scaleFactor=1.1, minNeighbors=5)
(x,y,w,h) = faces[0]
print(x, y, x+w, y+h)
