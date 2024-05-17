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

# Function to add boundary points of the image to the given set of
#  points
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

# MLS grid size
mls.GRID = 30

# Varibales for Dlib 
modelPath = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"
faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(modelPath)

# Parameters for controlling the amount of deformation 
offset1 = 1.5
offset2 = 1.5

# Points that should not move
anchorPoints = [8, 30]

# Points that will be deformed
# For lips
deformedPoints1 = [48, 57, 54]
# For eyes
deformedPoints2 = [21, 22, 36, 45]

t = time.time()
# Read an image and get the landmark points
filename = dataPath.IMAGES_ROOT + '/ted_cruz.jpg'
src = cv2.imread(filename)
height, width = src.shape[:2]
IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
src = cv2.resize(src,None,
                   fx=1.0/IMAGE_RESIZE, 
                   fy=1.0/IMAGE_RESIZE, 
                   interpolation = cv2.INTER_LINEAR)

landmarks = faceBlendCommon.getLandmarks(faceDetector, landmarkDetector, src, 
                             FACE_DOWNSAMPLE_RATIO)

print("Landmarks calculated in {}".format(time.time() - t))

# Set the center to tip of chin
center1x, center1y = landmarks[8][0], landmarks[8][1]
# Set the center to point on nose
center2x, center2y = landmarks[28][0], landmarks[28][1]

# Variables for storing the original and deformed points
srcPoints = []
dstPoints = []

# Adding the original and deformed points using the landmark points
for idx in anchorPoints:
  srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
  dstPoints.append([landmarks[idx][0], landmarks[idx][1]])


for idx in deformedPoints1:
  srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
  dstPoints.append([offset1*(landmarks[idx][0] - center1x) + 
                    center1x, offset1*(landmarks[idx][1] - center1y)
                    + center1y])

for idx in deformedPoints2:
  srcPoints.append([landmarks[idx][0], landmarks[idx][1]])
  dstPoints.append([offset2*(landmarks[idx][0] - center2x) + 
                    center2x, offset2*(landmarks[idx][1] - center2y)
                    + center2y])

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
plt.title("Happify Filter")
plt.axis('off')
print("Total time {}".format(time.time() - t))
plt.show()