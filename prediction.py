import cv2
import tensorflow as tf
import numpy as np
import time


model = tf.keras.models.load_model('best_model.v2')  


class_labels = ['MaskON', 'Mask Improper', 'No Mask']


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)  

while True:
    
    ret, frame = cap.read()

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

   
    for (x, y, w, h) in faces:
        
        face_region = frame[y:y + h, x:x + w]

        # Preprocess the face ROI
        face_roi = cv2.resize(face_region, (224, 224))
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)
       

        # Predict the face mask label
        prediction = model.predict(face_roi)
        label_index = np.argmax(prediction)
        label = class_labels[label_index]

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    
    cv2.imshow('Face Mask Detection', frame)

    

    
    key = cv2.waitKey(1)

    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
