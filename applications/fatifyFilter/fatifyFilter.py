# Import relevant libraries
import math
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os
import time


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
import renderFace
import dataPath
import faceBlendCommon
import mls 

matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

mls.GRID = 80

# Function to add boundary points of the image to the given 
# set of points
def addBoundaryPoints(cols, rows, points):
  # include the points on the boundaries
  points = np.append(points,[[0, 0]],axis=0)
  points = np.append(points,[[0, cols-1]],axis=0)
  points = np.append(points,[[rows-1, 0]],axis=0)
  points = np.append(points,[[rows-1, cols-1]],axis=0)
  points = np.append(points,[[0, cols/2]],axis=0)
  points = np.append(points,[[rows/2, 0]],axis=0)
  points = np.append(points,[[rows-1, cols/2]],axis=0)
  points = np.append(points,[[rows/2, cols-1]],axis=0)
  return points

# Variables for resizing to a standard height
RESIZE_HEIGHT = 360
FACE_DOWNSAMPLE_RATIO = 1.5

# Varibales for Dlib 
modelPath = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# Amount of bulge to be given for fatify
offset = 1.5

# Points that should not move
anchorPoints = [1, 15, 30]

# Points that will be deformed
deformedPoints = [ 5, 6, 8, 10, 11]

t = time.time()

# Read an image and get the landmark points
filename = dataPath.IMAGES_ROOT + '/hillary_clinton.jpg'

src = cv2.imread(filename)
height, width = src.shape[:2]

IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT

src = cv2.resize(src,None,
                   fx=1.0/IMAGE_RESIZE, 
                   fy=1.0/IMAGE_RESIZE, 
                   interpolation = cv2.INTER_LINEAR)

landmarks = faceBlendCommon.getLandmarks(detector, predictor, 
                             src, FACE_DOWNSAMPLE_RATIO)

print("Landmarks calculated in {}".format(time.time() - t))

# Set the center of face to be the nose tip
centerx, centery = landmarks[30][0], landmarks[30][1]

# Variables for storing the original and deformed points
srcPoints = []
dstPoints=[]

# Adding the original and deformed points using the landmark points
for idx in anchorPoints:
  srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
  dstPoints.append([landmarks[idx][0], landmarks[idx][1]])

for idx in deformedPoints:
  srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
  dstPoints.append([offset*(landmarks[idx][0] - centerx) + centerx, 
                    offset*(landmarks[idx][1] - centery) + centery])

# Converting them to numpy arrays
srcPoints = np.array(srcPoints)
dstPoints = np.array(dstPoints)

# Adding the boundary points to keep the image stable globally
srcPoints = addBoundaryPoints(src.shape[0],src.shape[1],srcPoints)
dstPoints = addBoundaryPoints(src.shape[0],src.shape[1],dstPoints)

print("Points gathered {}".format(time.time() - t))

# Performing moving least squares deformation on the image using the 
# points gathered above
dst = mls.MLSWarpImage(src, srcPoints, dstPoints)

print("Warping done {}".format(time.time() - t))

# Display and save the images
combined = np.hstack([src, dst])

plt.figure(figsize = (15,15))
plt.imshow(combined[:,:,::-1])
plt.title('Fatify Filter')
plt.axis('off')

print("Total time {}".format(time.time() - t))

plt.show()