#!/usr/bin/python

import cv2, os
import numpy as np
from PIL import Image

# library.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.createLBPHFaceRecognizer()

def get_images_and_labels(path):
  
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    images = []

    labels = []
    for image_path in image_paths:
        # grayscale
        image_pil = Image.open(image_path).convert('L')
        
	# numpy array
        image = np.array(image_pil, 'uint8')
      
	n = int(os.path.split(image_path)[1].split(".")[0])
        
	# Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        
	for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(n)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

path = './myfaces'
p='./f'

images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

# Append the images with the extension .sad into image_paths
image_paths = [os.path.join(p, f) for f in os.listdir(p) if f.endswith('.sad')]
for image_path in image_paths:
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
        nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        if nbr_actual == nbr_predicted:
            print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
        else:
            print "{} is Incorrect Recognized as {}".format(nbr_actual, conf)
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)
