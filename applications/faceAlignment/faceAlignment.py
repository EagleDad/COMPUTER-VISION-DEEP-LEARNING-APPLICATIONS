# Import relevant libraries
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

print("Face Alignmnet")

# Set some params
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Landmark model location
PREDICTOR_PATH = dataPath.MODELS_ROOT + "/shape_predictor_5_face_landmarks.dat"

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# Read image
im = cv2.imread(dataPath.IMAGES_ROOT + "/face2.png")

plt.imshow(im[:,:,::-1])
plt.title("Image")
plt.show()

# Detect faces in the image
faceRects = faceDetector(im, 0)
print("Number of faces detected: ", len(faceRects))

# List to store landmarks of all detected faces
landmarksAll = []

# Loop over all detected face rectangles
for i in range(0, len(faceRects)):
  newRect = dlib.rectangle(int(faceRects[i].left()),
                          int(faceRects[i].top()),
                          int(faceRects[i].right()),
                          int(faceRects[i].bottom()))
  # For every face rectangle, run landmarkDetector
  landmarks = landmarkDetector(im, newRect)
  # Print number of landmarks
  if i==0:
    print("Number of landmarks",len(landmarks.parts()))

  # Store landmarks for current face
  landmarksAll.append(landmarks)

  # Next, we render the outline of the face using
  # detected landmarks.
  renderFace.renderFace2(im, landmarks)

plt.figure(figsize=(15,15))
plt.imshow(im[:,:,::-1])
plt.title("Facial Landmark detector")
plt.show()

# Detect landmarks.
points = faceBlendCommon.getLandmarks(faceDetector, landmarkDetector, im)

points = np.array(points)

# Convert image to floating point in the range 0 to 1
im = np.float32(im)/255.0

# Dimensions of output image
h = 600
w = 600

# Normalize image to output coordinates.
imNorm, points = faceBlendCommon.normalizeImagesAndLandmarks((h, w), im, points)

imNorm = np.uint8(imNorm*255)

# Display the results
plt.imshow(imNorm[:,:,::-1])
plt.title("Aligned Image")
plt.show()