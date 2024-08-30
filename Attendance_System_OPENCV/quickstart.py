import cv2
import numpy as np
import face_recognition
imgElon = face_recognition.load_image_file("images/Elon1.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("images/Elon2.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),2)

facelocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)
cv2.rectangle(imgTest,(facelocTest[3],facelocTest[0]),(facelocTest[1],facelocTest[2]),(0,255,0),2)
results = face_recognition.compare_faces(encodeElon,encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)

print(faceDis,results)
cv2.putText(imgTest,f"{results} {round(faceDis[0],2)}",(50,50),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
cv2.imshow("Elon Musk",imgElon)
cv2.imshow("Elon Musk 2",imgTest)
cv2.waitKey(0)
