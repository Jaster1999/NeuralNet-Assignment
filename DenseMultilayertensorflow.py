import tensorflow
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.python.keras.layers.pooling import MaxPool2D
import os
import cv2

# TODO:Rather than having to deal with whatever a idx3-ubyte datatype is im just going to use googles copy of the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('X_train: ' + str(train_images.shape)) # The images
print('Y_train: ' + str(train_labels.shape)) # their classifications 0-9
print('X_test:  '  + str(test_images.shape)) 
print('Y_test:  '  + str(test_labels.shape))

classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
plt.figure()
for i in range(40):
    plt.subplot(8,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap='gray')
    plt.xlabel(classes[train_labels[i]])


# Normalize pixel values to be between 0 and 1
train_images = train_images.reshape(-1, 28 * 28) / 255.0
test_images = test_images.reshape(-1, 28 * 28) / 255.0

#train_images, test_images = train_images / 255.0, test_images / 255.0

train_labels, test_labels = tensorflow.keras.utils.to_categorical(train_labels, 10), tensorflow.keras.utils.to_categorical(test_labels, 10)

model = models.Sequential() # indicating to keras that I wish to create my model layer by layer
#model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(784, activation='relu', input_shape=(784,)))
model.add(layers.Dense(392, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

gdoptimizer = tensorflow.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5, nesterov=True)

model.compile(optimizer=gdoptimizer, loss=tensorflow.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=4096 ,epochs=20, validation_data=(test_images, test_labels))
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
    img = cv2.imread(str(i)+".png", cv2.IMREAD_GRAYSCALE)
    custom_sample.append(img)
custom_sample = np.array(custom_sample)
images = custom_sample.copy()
'''
for x in range(len(custom_sample)):
    custom_sample[x] = custom_sample[x].astype(float) / 255.0
plt.imshow(custom_sample[0])
plt.show()'''

custom_sample = custom_sample.reshape(-1, 28 * 28) / 255.0

pred = model.predict(custom_sample)
print(pred)
for pre in pred[9]:
    print(pre)
plt.figure()
for i in range(10):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap='gray')
    plt.xlabel(str(np.argmax(pred[i])) + " " + str(np.max(pred[i])))
plt.show()



