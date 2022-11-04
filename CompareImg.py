import cv2
import numpy as np
import face_recognition

# Memuat Gambar dari folder
imgSample = face_recognition.load_image_file('ImageSample/Aldy Sample.JPG')
imgTest = face_recognition.load_image_file('ImageSample/Foto_sample.jpg')
# Konversi warna BGR to RGB
imgSample = cv2.cvtColor(imgSample, cv2.COLOR_BGR2RGB)
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# Deteksi Wajah
faceLocSample = face_recognition.face_locations(imgSample)[0]
encodeSample = face_recognition.face_encodings(imgSample)[0]
cv2.rectangle(imgSample, (faceLocSample[3], faceLocSample[0]), (faceLocSample[1], faceLocSample[2]), (255, 0, 255), 2)
print(faceLocSample)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
print(faceLocTest)

# Membandingkan Wajah
results = face_recognition.compare_faces([encodeSample], encodeTest)
faceDis = face_recognition.face_distance([encodeSample], encodeTest)
cv2.putText(imgTest, f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
print(results,faceDis)
# Manampilkan Gambar
cv2.imshow('Sample Gambar', imgSample)
cv2.imshow('Gambar Testing', imgTest)
cv2.waitKey(0)