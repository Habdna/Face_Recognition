# Face_Recognition
face recognition master

# : كود كشف الوجه

import face_recognition
image = face_recognition.load_image_file("1.png")
face_locations = face_recognition.face_locations(image)
# النموذج للتعرف على مشاعر Sun Ge في هذه الصورة:
import face_recognition
import numpy as np
import cv2
from keras.models import load_model
emotion_dict = {"غاضب": 0 , "حزن": 5 , "محايد": 4 , "اشمئزاز": 1 , "مفاجأة": 6 , "خوف": 2 , "سعيد": 3}
image = face_recognition.load_image_file("1.png")
 # تحميل الصورة
face_locations = face_recognition.face_locations(image)
 # ابحث عن الوجه
top, right, bottom, left = face_locations[0]
 # تأطير الوجه

face_image = image[top:bottom, left:right]
face_image = cv2.resize(face_image, (48,48))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
 # اضبط الحجم الذي يمكن أن يدخل إدخال النموذج

model = load_model("./model_v6_23.hdf5")
 # تحميل النموذج

predicted_class = np.argmax(model.predict(face_image))
 # العواطف المصنفة
label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]
 # إخراج المشاعر وفقًا لجدول رسم المشاعر
print(predicted_label)



