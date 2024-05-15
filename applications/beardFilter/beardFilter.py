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

matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

FACE_DOWNSAMPLE_RATIO = 1
RESIZE_HEIGHT = 480

# Points corresponding to Dlib which have been marked on the beard
selectedIndex = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                 31, 32, 33, 34, 35, 55, 56, 57, 58, 59]

# Read points corresponding to beard, stored in text files
def getSavedPoints(beardPointsFile):
  points = []
  lines = np.loadtxt(beardPointsFile, dtype='uint16')
  
  for p in lines:
    points.append((p[0], p[1]))
  
  return points 

# Load face detection and pose estimation models.
modelPath = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

# Load the beard image with alpha mask 
overlayFile = dataPath.IMAGES_ROOT + "/beard1.png"
imgWithMask = cv2.imread(overlayFile,cv2.IMREAD_UNCHANGED)

# split the 4 channels
b,g,r,a = cv2.split(imgWithMask)

# Take the first 3 channels and create the bgr image to be warped
beard = cv2.merge((b,g,r))
beard = np.float32(beard)/255

# Take the 4th channel and create the alpha mask used for blending
beardAlphaMask = cv2.merge((a,a,a))
beardAlphaMask = np.float32(beardAlphaMask)
plt.figure(figsize=[15,10])
plt.subplot(121);plt.imshow(beard[...,::-1])
plt.subplot(122);plt.imshow(np.uint8(beardAlphaMask[...,::-1]))
plt.show()

# Read the points marked on the beard
featurePoints1 = getSavedPoints( overlayFile + ".txt")

# Find delanauy traingulation for convex hull points
sizeImg1 = beard.shape    
rect = (0, 0, sizeImg1[1], sizeImg1[0])
dt = faceBlendCommon.calculateDelaunayTriangles(rect, featurePoints1)

if len(dt) == 0:
    print("No delaunay triangle found")

imageFile = dataPath.IMAGES_ROOT + "/ted_cruz.jpg"
targetImage = cv2.imread(imageFile)
height, width = targetImage.shape[:2]
IMAGE_RESIZE = np.float32(height)/RESIZE_HEIGHT
targetImage = cv2.resize(targetImage,None,
                 fx=1.0/IMAGE_RESIZE,
                 fy=1.0/IMAGE_RESIZE,
                 interpolation = cv2.INTER_LINEAR)

points2 = faceBlendCommon.getLandmarks(detector, predictor, cv2.cvtColor(targetImage, cv2.COLOR_BGR2RGB), FACE_DOWNSAMPLE_RATIO)
featurePoints2 = []
for p in selectedIndex:
    pt = points2[p]
    pt = faceBlendCommon.constrainPoint(pt, width, height)
    featurePoints2.append(pt)

targetImage = np.float32(targetImage)/255

beardWarped = np.zeros(targetImage.shape)
beardAlphaWarped = np.zeros(targetImage.shape)

# Apply affine transformation to Delaunay triangles
for i in range(0, len(dt)):
    t1 = []
    t2 = []

    #get points for img1, img2 corresponding to the triangles
    for j in range(0, 3):
      t1.append(featurePoints1[dt[i][j]])
      t2.append(featurePoints2[dt[i][j]])

    faceBlendCommon.warpTriangle(beard, beardWarped, t1, t2)
    faceBlendCommon.warpTriangle(beardAlphaMask, beardAlphaWarped, t1, t2)

beardWarpedMask = beardAlphaWarped/255
temp1 = np.multiply(targetImage, 1.0 - beardWarpedMask)
temp2 = np.multiply(beardWarped, beardWarpedMask)

out = temp1 + temp2
plt.figure(figsize=[15,10])
plt.subplot(121);plt.imshow(targetImage[...,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(out[...,::-1]);plt.title("Image with Beard")
cv2.imwrite(dataPath.RESULTS_ROOT + "/beardify.jpg", np.uint8(255*out))
plt.show()