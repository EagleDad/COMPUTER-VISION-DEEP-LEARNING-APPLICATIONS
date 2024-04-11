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

# Functions
def writeLandmarksToFile(landmarks, landmarksFileName):
  with open(landmarksFileName, 'w') as f:
    for p in landmarks.parts():
      f.write("%s %s\n" %(int(p.x),int(p.y)))

  f.close()

print("Facial Landmark Detector")

# Set some params
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Landmark model location
PREDICTOR_PATH = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"

# Get the face detector instance
faceDetector = dlib.get_frontal_face_detector()

# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# Read image
imageFilename = dataPath.IMAGES_ROOT + "/family.jpg"
im = cv2.imread(imageFilename)

# landmarks will be stored in results/family_i.txt
landmarksBasename = dataPath.RESULTS_ROOT + "/family"

plt.imshow(im[:,:,::-1])
plt.show()

# Detect faces in the image
faceRects = faceDetector(im, 0)
print("Number of faces detected: ",len(faceRects))

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

  # The code below saves the landmarks to 
  # results/family_0.txt … results/family_4.txt.
  landmarksFileName = landmarksBasename +"_"+ str(i)+ ".txt"
  print("Saving landmarks to", landmarksFileName)
  # Write landmarks to disk
  writeLandmarksToFile(landmarks, landmarksFileName)

outputFileName = dataPath.RESULTS_ROOT +  "/familyLandmarks.jpg"
print("Saving output image to", outputFileName)
cv2.imwrite(outputFileName, im)

plt.figure(figsize=(15,15))
plt.imshow(im[:,:,::-1])
plt.title("Facial Landmark detector")
plt.show()



