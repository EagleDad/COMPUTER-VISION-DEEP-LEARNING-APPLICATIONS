# Import relevant libraries
import math
import dlib
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
import renderFace
import dataPath
import faceBlendCommon 

from faceBlendCommon import constrainPoint, warpTriangle

matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Landmark model location
PREDICTOR_PATH = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# Read two images
im1 = cv2.imread(dataPath.IMAGES_ROOT + "/hillary-clinton.jpg")
im2 = cv2.imread(dataPath.IMAGES_ROOT + "/presidents/bill-clinton.jpg")

# Detect landmarks in both images.
points1 = faceBlendCommon.getLandmarks(faceDetector, landmarkDetector, im1)
points2 = faceBlendCommon.getLandmarks(faceDetector, landmarkDetector, im2)

points1 = np.array(points1)
points2 = np.array(points2)

# Convert image to floating point in the range 0 to 1
im1 = np.float32(im1)/255.0
im2 = np.float32(im2)/255.0

# Dimensions of output image
h = 600
w = 600

# Normalize image to output coordinates.
imNorm1, points1 = faceBlendCommon.normalizeImagesAndLandmarks((h, w), 
                                                  im1, points1)
imNorm2, points2 = faceBlendCommon.normalizeImagesAndLandmarks((h, w), 
                                                  im2, points2)

# Calculate average points. Will be used for Delaunay triangulation.
pointsAvg = (points1 + points2)/2.0

# 8 Boundary points for Delaunay Triangulation
boundaryPoints = faceBlendCommon.getEightBoundaryPoints(h, w)
points1 = np.concatenate((points1, boundaryPoints), axis=0)
points2 = np.concatenate((points2, boundaryPoints), axis=0)
pointsAvg = np.concatenate((pointsAvg, boundaryPoints), axis=0)

# Calculate Delaunay triangulation.
rect = (0, 0, w, h)
dt = faceBlendCommon.calculateDelaunayTriangles(rect, pointsAvg)

# Start animation.
alpha = 0
increaseAlpha = True

while True:
    # Compute landmark points based on morphing parameter alpha
    pointsMorph = (1 - alpha) * points1 + alpha * points2

    # Warp images such that normalized points line up 
    # with morphed points.
    imOut1 = faceBlendCommon.warpImage(imNorm1, points1, pointsMorph.tolist(), dt)
    imOut2 = faceBlendCommon.warpImage(imNorm2, points2, pointsMorph.tolist(), dt)

    # Blend warped images based on morphing parameter alpha
    imMorph = (1 - alpha) * imOut1 + alpha * imOut2
    
    # Keep animating by ensuring alpha stays between 0 and 1.
    if (alpha <= 0 and not increaseAlpha):
        increaseAlpha = True
    if (alpha >= 1 and increaseAlpha):
        break
        increaseAlpha = False

    if increaseAlpha:
        alpha += 0.075
    else:
        alpha -= 0.075

    plt.imshow(imMorph[:,:,::-1])
    plt.show()