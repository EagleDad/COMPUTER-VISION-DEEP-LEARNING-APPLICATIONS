# Import relevant libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# Get the path of the common directory to import common modules
directory = os.path.dirname( os.path.abspath(__file__))
#print("directory: " + directory)
parentDirectory = os.path.abspath(os.path.join(directory, os.pardir))
#print("parentDirectory: " + parentDirectory)
commonDirectory = os.path.abspath(os.path.join(parentDirectory, "common"))
#print("commonDirectory: " + commonDirectory)

# Extend path to import common modules
sys.path.append(commonDirectory)

# Import the common modukes
import dataPath

matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Read images
src = cv2.imread(dataPath.IMAGES_ROOT + "/airplane.jpg")
dst = cv2.imread(dataPath.IMAGES_ROOT + "/sky.jpg")

# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)
poly = np.array([ [4,80], [30,54], [151,63], [254,37], [298,90], [272,134], [43,122] ], np.int32)
src_mask = cv2.fillPoly(src_mask, [poly], (255, 255, 255))

# This is where the CENTER of the airplane will be placed
center = (800,100)

# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

plt.imshow(output[:,:,::-1])
plt.show()

# Read images : src image will be cloned into dst
im = cv2.imread(dataPath.IMAGES_ROOT + "/wood-texture.jpg")
obj= cv2.imread(dataPath.IMAGES_ROOT + "/iloveyouticket.jpg")

# Create an all white mask
mask = 255 * np.ones(obj.shape, obj.dtype)

# The location of the center of the src in the dst
width, height, channels = im.shape
center = (height//2, width//2)

# Seamlessly clone src into dst and put the results in output
normal_clone = cv2.seamlessClone(obj, im, mask, center, cv2.NORMAL_CLONE)
mixed_clone = cv2.seamlessClone(obj, im, mask, center, cv2.MIXED_CLONE)

plt.figure(figsize=[20,10])
plt.subplot(1,2,1)
plt.title("Normal Clone Result")
plt.imshow(normal_clone[:,:,::-1])
plt.subplot(1,2,2)
plt.title("Mixed Clone Result")
plt.imshow(mixed_clone[:,:,::-1])
plt.show()