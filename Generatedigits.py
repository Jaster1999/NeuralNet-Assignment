import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import ndimage

# This file is to convert a bunch of rgb photos of hand written digits to a format type for the mnist neural net



def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

custom_sample = []

for i in range(0, 10):  
    gray = cv2.imread("Testing/original Images/" + str(i)+".png", cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(255-gray, (28, 28))
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    gray = gray.astype(int) + 128
    gray[gray == 128] = 0
    gray = np.clip(gray, 0, 255)
    gray = gray.astype(np.uint8)
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted

    custom_sample.append(gray)

custom_sample = np.array(custom_sample)
images = custom_sample.copy()


if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))

for y in range(len(custom_sample)):
    cv2.imwrite(str(y)+".png", custom_sample[y])

plt.figure()
for i in range(10):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(images[i], cmap='gray')
    plt.xlabel("test")
plt.show()