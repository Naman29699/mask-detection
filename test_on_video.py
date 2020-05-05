import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2

categories = ['mask', 'no_mask']

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5', compile=False)

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

face_cascade = cv2.CascadeClassifier(r'C:\anaconda\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

#feed in the video you want to detect upon
cap = cv2.VideoCapture('Clip1.wmv')

while(cap.isOpened()):
    ret, image = cap.read()
    defaulters = []
    count = 0
    
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi = image[y:y+h, x:x+w]
        size = (224, 224)
        #roi = ImageOps.fit(roi, size, Image.ANTIALIAS)
        roi = cv2.resize(roi, (224, 224))
        roi_array = np.asarray(roi)
        normalized_image_array = (roi_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        text = categories[np.argmax(prediction[0])]
        cv2.putText(image, text, org=(x, y), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale= 1, color =(255, 0, 0), thickness = 2)
        if text == 'mask':
            defaulters.append(int(0))
        elif text == 'no_mask':
            defaulters.append(int(1))
    for element in defaulters:
        count+=element
    cv2.putText(image, 'defaulters={}'.format(str(count)), org=(0, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale= 1, color =(255, 255, 0), thickness = 2)
    #print(count)
    #print(defaulters) 
    
    
    
    cv2.imshow('img', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


