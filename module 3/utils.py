#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow import keras

def predict(img_path,model,con_dest='converted_img.jpg',threshold=135):
    num=cv2.imread(img_path)
    num=cv2.cvtColor(num, cv2.COLOR_BGR2GRAY)
    #num.resize(28,28)
    (thresh, num) = cv2.threshold(num, threshold, 255, cv2.THRESH_BINARY_INV)
    
    
    cv2.imwrite(con_dest,num)
    
    img = image.load_img(con_dest,color_mode = "grayscale",target_size=(28,28))
    plt.imshow(img)
    plt.title("Converted Image for Prediction")
    plt.show()
    img = image.img_to_array(img)
    test_img = (img.reshape((1,784)))/255.0
    ans=model.predict(test_img)
    
    print('Predictions for each class', ans)
    print('Predicted value is '+ str(np.argmax(ans)))


# In[ ]:




