import tensorflow
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.python.keras.layers.pooling import MaxPool2D
import cv2
import math

# TODO:Rather than having to deal with whatever a idx3-ubyte datatype is im just going to use googles copy of the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('X_train: ' + str(train_images.shape)) # The images
print('Y_train: ' + str(train_labels.shape)) # their classifications 0-9
print('X_test:  '  + str(test_images.shape)) 
print('Y_test:  '  + str(test_labels.shape))

classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
plt.figure()
for i in range(30):
    plt.subplot(5,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(classes[train_labels[i]])


# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Data has to be reshaped to use tensorflows conv2d function (it need a three dimensional array currently the data is only two dimensional)
input_shape = (28, 28, 1)
train_images = train_images.reshape(len(train_images), input_shape[0], input_shape[1], input_shape[2])
test_images = test_images.reshape(len(test_images), input_shape[0], input_shape[1], input_shape[2])

model = models.Sequential() # indicating to keras that I wish to create my model layer by layer
model.add(layers.Conv2D(32, (5,5), padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size = (1, 1), strides = 1))
model.add(layers.Conv2D(32, (3,3), padding='valid', activation='relu'))
model.add(MaxPool2D(pool_size = (1, 1), strides = 1))
model.add(layers.Conv2D(16, (3,3), padding='valid', activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

gdoptimizer = tensorflow.keras.optimizers.SGD(learning_rate=0.1, decay = 0.005)

model.compile(optimizer=gdoptimizer, loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size = 256, epochs=10, validation_data=(test_images, test_labels))

model.save('cnn_model')

plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

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