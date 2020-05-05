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

#Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()

    
    #an empty list
    defaulters = []
    count = 0
    
    #detect faces in a frame
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    
    #x, y, w, h are coordinates of the box where a face is found
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi = image[y:y+h, x:x+w]
        size = (224, 224)
        #resizing the roi
        roi = cv2.resize(roi, (224, 224))
        roi_array = np.asarray(roi)
        normalized_image_array = (roi_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        text = categories[np.argmax(prediction[0])]
        cv2.putText(image, text, org=(x, y), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale= 1, color =(255, 0, 0), thickness = 2)
        if text == 'mask':
            defaulters.append(int(0)) #0 is appended to defaulters list
        elif text == 'no_mask':
            defaulters.append(int(1)) #1 is appended to defaulters list
    for element in defaulters:
        count+=element #counting number of 1's (people not wearing masks)
    cv2.putText(image, 'defaulters={}'.format(str(count)), org=(0, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX,fontScale= 1, color =(255, 255, 0), thickness = 2)
    #print(count)
    #print(defaulters) 
    
    
    
    cv2.imshow('img', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


