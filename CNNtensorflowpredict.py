import tensorflow
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.python.keras.layers.pooling import MaxPool2D
import cv2
import math


model = tensorflow.keras.models.load_model('cnn_model')
model.summary()

input_shape = (28, 28, 1)
custom_sample = []

for i in range(0, 10):  
    gray = cv2.imread(str(i)+".png", cv2.IMREAD_GRAYSCALE)
    custom_sample.append(gray)
custom_sample = np.array(custom_sample)
images = custom_sample.copy()

custom_sample = custom_sample.reshape(len(custom_sample), input_shape[0], input_shape[1], input_shape[2])

pred = model.predict(custom_sample)
print(pred)

plt.figure()
for i in range(10):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap='gray')
    plt.xlabel(str(np.argmax(pred[i])) + " " + str(np.max(pred[i])))
plt.show()

