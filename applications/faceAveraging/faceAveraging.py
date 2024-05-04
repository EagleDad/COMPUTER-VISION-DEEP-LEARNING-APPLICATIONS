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

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.
def similarityTransform(inPoints, outPoints):
  s60 = math.sin(60*math.pi/180)
  c60 = math.cos(60*math.pi/180)

  inPts = np.copy(inPoints).tolist()
  outPts = np.copy(outPoints).tolist()

  # The third point is calculated so that the three points make an equilateral triangle
  xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
  yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]

  inPts.append([np.int(xin), np.int(yin)])

  xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
  yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]

  outPts.append([np.int(xout), np.int(yout)])

  # Now we can use estimateRigidTransform for calculating the similarity transform.
  tform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))
  return tform[0]

def normalizeImagesAndLandmarks(outSize, imIn, pointsIn):
  h, w = outSize

  # Corners of the eye in input image
  eyecornerSrc = [pointsIn[36], pointsIn[45]]

  # Corners of the eye in normalized image
  eyecornerDst = [(np.int(0.3 * w), np.int(h/3)), 
                  (np.int(0.7 * w), np.int(h/3))]

  # Calculate similarity transform
  tform = similarityTransform(eyecornerSrc, eyecornerDst)
  
  # Apply similarity transform to input image
  imOut = cv2.warpAffine(imIn, tform, (w, h))

  # reshape pointsIn from numLandmarks x 2 to numLandmarks x 1 x 2
  points2 = np.reshape(pointsIn, 
                      (pointsIn.shape[0], 1, pointsIn.shape[1]))
  
  # Apply similarity transform to landmarks
  pointsOut = cv2.transform(points2, tform)

  # reshape pointsOut to numLandmarks x 2
  pointsOut = np.reshape(pointsOut, 
                        (pointsIn.shape[0], pointsIn.shape[1]))

  return imOut, pointsOut

# Warps an image in a piecewise affine manner.
# The warp is defined by the movement of landmark points specified by
# pointsIn to a new location specified by pointsOut. 
# The triangulation beween points is specified by 
# their indices in delaunayTri.
def warpImage(imIn, pointsIn, pointsOut, delaunayTri):
  h, w, ch = imIn.shape
  # Output image
  imOut = np.zeros(imIn.shape, dtype=imIn.dtype)

  # Warp each input triangle to output triangle.
  # The triangulation is specified by delaunayTri
  for j in range(0, len(delaunayTri)):
    # Input and output points corresponding to jth triangle
    tin = []
    tout = []

    for k in range(0, 3):
      # Extract a vertex of input triangle
      pIn = pointsIn[delaunayTri[j][k]]
      # Make sure the vertex is inside the image.
      pIn = constrainPoint(pIn, w, h)

      # Extract a vertex of the output triangle
      pOut = pointsOut[delaunayTri[j][k]]
      # Make sure the vertex is inside the image.
      pOut = constrainPoint(pOut, w, h)

      # Push the input vertex into input triangle
      tin.append(pIn)
      # Push the output vertex into output triangle
      tout.append(pOut)

    # Warp pixels inside input triangle to output triangle.
    warpTriangle(imIn, imOut, tin, tout)
  return imOut

# Read all jpg image paths in folder.
def readImagePaths(path):
  # Create array of array of images.
  imagePaths = []
  # List all files in the directory and read points from text files 
  for filePath in sorted(os.listdir(path)):
    fileExt = os.path.splitext(filePath)[1]
    if fileExt in [".jpg", ".jpeg"]:
      print(filePath)

      # Add to array of images
      imagePaths.append(os.path.join(path, filePath))

  return imagePaths


# Landmark model location
PREDICTOR_PATH = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

dirName = dataPath.IMAGES_ROOT + "/presidents"

# Read all images
imagePaths = readImagePaths(dirName)

if len(imagePaths) == 0:
    print('No images found with extension jpg or jpeg')

# Read images and perform landmark detection.
images = []
allPoints = []

for imagePath in imagePaths:
    im = cv2.imread(imagePath)
    if im is None:
        print("image:{} not read properly".format(imagePath))
    else:
        points = faceBlendCommon.getLandmarks(faceDetector, landmarkDetector, im)
        if len(points) > 0:
            allPoints.append(points)

            im = np.float32(im)/255.0
            images.append(im)
        else:
            print("Couldn't detect face landmarks")

# Dimensions of output image
w = 600
h = 600

# 8 Boundary points for Delaunay Triangulation
boundaryPts = faceBlendCommon.getEightBoundaryPoints(h, w)

numImages = len(imagePaths)
numLandmarks = len(allPoints[0])

# Variables to store normalized images and points.
imagesNorm = []
pointsNorm = []

# Initialize location of average points to 0s
pointsAvg = np.zeros((numLandmarks, 2), dtype=np.float32)

# Warp images and trasnform landmarks to output coordinate system,
# and find average of transformed landmarks.
for i, img in enumerate(images):

    points = allPoints[i]
    points = np.array(points)

    img, points = faceBlendCommon.normalizeImagesAndLandmarks((h, w), img, points)

    # Calculate average landmark locations
    pointsAvg = pointsAvg + (points / (1.0*numImages))

    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.concatenate((points, boundaryPts), axis=0)

    pointsNorm.append(points)
    imagesNorm.append(img)

# Append boundary points to average points.
pointsAvg = np.concatenate((pointsAvg, boundaryPts), axis=0)

# Delaunay triangulation
rect = (0, 0, w, h)
dt = faceBlendCommon.calculateDelaunayTriangles(rect, pointsAvg)

# Output image
output = np.zeros((h, w, 3), dtype=np.float32)

# Warp input images to average image landmarks
for i in range(0, numImages):

    imWarp = faceBlendCommon.warpImage(imagesNorm[i], pointsNorm[i], 
                            pointsAvg.tolist(), dt)

    # Add image intensities for averaging
    output = output + imWarp

# Divide by numImages to get average
output = output / (1.0*numImages)

# Display result
plt.imshow(output[:,:,::-1])
plt.show()