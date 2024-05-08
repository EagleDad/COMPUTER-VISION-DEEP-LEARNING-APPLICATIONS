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

# Initialize the dlib facial landmark detector variables
modelPath = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"

# initialize the dlib facial landmakr detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

t = time.time()

# Read images
filename1 = dataPath.IMAGES_ROOT + '/ted_cruz.jpg'
filename2 = dataPath.IMAGES_ROOT + '/donald_trump.jpg'

img1 = cv2.imread(filename1)
img2 = cv2.imread(filename2)
img1Warped = np.copy(img2)   

# Read array of corresponding points
points1 = faceBlendCommon.getLandmarks(detector, predictor, img1)
points2 = faceBlendCommon.getLandmarks(detector, predictor, img2)

# Find convex hull
hull1 = []
hull2 = []

hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

for i in range(0, len(hullIndex)):
    hull1.append(points1[hullIndex[i][0]])
    hull2.append(points2[hullIndex[i][0]])

# Find delanauy traingulation for convex hull points
sizeImg2 = img2.shape    
rect = (0, 0, sizeImg2[1], sizeImg2[0])

dt = faceBlendCommon.calculateDelaunayTriangles(rect, hull2)

if len(dt) == 0:
    print("No delanauy triangles calculated")

# Apply affine transformation to Delaunay triangles
for i in range(0, len(dt)):
    t1 = []
    t2 = []

    #get points for img1, img2 corresponding to the triangles
    for j in range(0, 3):
        t1.append(hull1[dt[i][j]])
        t2.append(hull2[dt[i][j]])

    faceBlendCommon.warpTriangle(img1, img1Warped, t1, t2)

print("Time taken for faceswap {:.3f} seconds".format(time.time()
    - t))
tClone = time.time()

# Calculate Mask for Seamless cloning
hull8U = []
for i in range(0, len(hull2)):
    hull8U.append((hull2[i][0], hull2[i][1]))

mask = np.zeros(img2.shape, dtype=img2.dtype)  

cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

plt.imshow(mask)
plt.title("Mask created from hull2 for seamless cloning")
plt.axis('off')
plt.show()

# find center of the mask to be cloned with the destination image
r = cv2.boundingRect(np.float32([hull2]))    

center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

# Clone seamlessly.
output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, 
                          cv2.NORMAL_CLONE)

print("Time taken for seamless cloning {:.3f} seconds".
      format(time.time() - tClone))

print("Total Time taken {:.3f} seconds ".format(time.time() - t))

plt.figure(figsize=(20,20))

plt.subplot(121)
plt.imshow(np.uint8(img1Warped)[:,:,::-1])
plt.title("Face Swapped before seamless cloning")
plt.axis('off')

plt.subplot(122)
plt.imshow(output[:,:,::-1])
plt.title("Face Swapped after seamless cloning")
plt.axis('off')

plt.show()