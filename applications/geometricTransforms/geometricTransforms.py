# Standard imports
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

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# Create a black image of size 200x200
im = np.zeros((200,200,3), np.float32)

# Create a blue square in the center
im[50:150,50:150,0] = 1
im[50:150,50:150,1] = 0.6
im[50:150,50:150,2] = 0.2

# Display image
plt.imshow(im[:,:,::-1])
plt.show()

# Translate image
# Output dimension
outDim = im.shape[0:2]

# Translate by 25,25
warpMat = np.float32(
    [
        [1.0, 0.0, 25],
        [0,   1.0, 25]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Scale along x direction
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   1.0, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Scale image width
# Scale along x direction
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   1.0, 0]
    ])

result = cv2.warpAffine(im, warpMat, (2 * outDim[0], outDim[1]))

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Scale along both dimensions
# Scale along x and y directions
warpMat = np.float32(
    [
        [2.0, 0.0, 0],
        [0,   2.0, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, (2 * outDim[0], 2 * outDim[1]))

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Rotate Image about the origin (0,0)
# Rotate image 
angleInDegrees = 30
angleInRadians = 30 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

# Rotation matrix 
# https://en.wikipedia.org/wiki/Rotation_matrix
    
warpMat = np.float32(
    [
        [ cosTheta, sinTheta, 0],
        [ -sinTheta, cosTheta, 0]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Rotate image about a specific point (center)
# Rotate image 
angleInDegrees = 30
angleInRadians = 30 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

centerX = im.shape[0] / 2
centerY = im.shape[1] / 2

tx = (1-cosTheta) * centerX - sinTheta * centerY
ty =  sinTheta * centerX  + (1-cosTheta) * centerY

# Rotation matrix 
# https://en.wikipedia.org/wiki/Rotation_matrix
    
warpMat = np.float32(
    [
        [ cosTheta, sinTheta, tx],
        [ -sinTheta,  cosTheta, ty]
    ])

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Rotate image the easy way
# Get rotation matrix
rotationMatrix = cv2.getRotationMatrix2D((centerX, centerY), angleInDegrees, 1)

# Warp Image
result = cv2.warpAffine(im, rotationMatrix, outDim)

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Shear Transformation
shearAmount = 0.1

warpMat = np.float32(
    [
        [ 1, shearAmount, 0],
        [ 0, 1.0        , 0]
    ])


# Warp image
result = cv2.warpAffine(im, warpMat, outDim, None, flags=cv2.INTER_LINEAR)

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Complex Transformations
# Scale 
scaleAmount = 1.1
scaleMat = np.float32(
    [
        [ scaleAmount, 0.0,       ],
        [ 0,           scaleAmount]
    ])

# Shear 
shearAmount = -0.1 
shearMat = np.float32(
    [
        [ 1, shearAmount],
        [ 0, 1.0        ]
    ])

# Rotate by 10 degrees about (0,0)

angleInRadians = 10.0 * np.pi / 180.0

cosTheta = np.cos(angleInRadians)
sinTheta = np.sin(angleInRadians)

rotMat = np.float32(
    [
        [ cosTheta, sinTheta],
        [ -sinTheta, cosTheta]
    ])

translateVector = np.float32(
    [
        [10],
        [0]
    ])

# First scale is applied, followed by shear, followed by rotation. 
scaleShearRotate = rotMat @ shearMat @ scaleMat

# Add translation
warpMat = np.append(scaleShearRotate, translateVector, 1)
print(warpMat)
outPts = scaleShearRotate @ np.float32([[50, 50],[50, 149],[149, 50], [149, 149]]).T + translateVector
print(outPts)

# Warp image
result = cv2.warpAffine(im, warpMat, outDim)

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Complex Transformations using 3-Point Correspondences
srcPoints = np.float32([[50, 50],[50, 149],[149, 50]])
dstPoints = np.float32([[68, 45],[76, 155],[176, 27]])
estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]
print("True warp matrix:\n\n", warpMat)
print("\n\nEstimated warp matrix:\n\n", estimatedMat)

srcPoints = np.float32([[50, 50],[50, 149],[149, 149], [149, 50]])
dstPoints = np.float32([[68, 45],[76, 155],[183, 135], [176, 27]])

estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]

print("True warp matrix:\n\n", warpMat)
print("\n\nEstimated warp matrix:\n\n", estimatedMat)

# Warp image
result = cv2.warpAffine(im, estimatedMat, outDim)

# Display image
plt.imshow(result[...,::-1])
plt.show()

# Limitations of Affine Transform
# Transformed image
imT = np.zeros((200, 200, 3), dtype = np.float32)
dstPoints = np.float32([[75, 50],[50, 149], [149, 149], [124, 50]])
cv2.fillConvexPoly(imT, np.int32(dstPoints), (1.0, 0.6, 0.2), cv2.LINE_AA);

plt.figure(figsize=[10,10])
plt.subplot(121)
plt.imshow(im[:,:,::-1])
plt.title('Original Image')

plt.subplot(122)
plt.imshow(imT[:,:,::-1])
plt.title('Transformed Image')
plt.show()

estimatedMat = cv2.estimateAffine2D(srcPoints, dstPoints)[0]
print("\n\nEstimated warp matrix:\n\n", estimatedMat)

# Warp image
imA = cv2.warpAffine(im, estimatedMat, outDim)

# Display image
plt.figure(figsize=[10,10])
plt.subplot(121)
plt.imshow(imT[:,:,::-1])
plt.title('Transformed Image')

plt.subplot(122)
plt.imshow(imA[:,:,::-1])
plt.title('Image warped using estimated Affine Transform')
plt.show()

# Calculate homography
h, status = cv2.findHomography(srcPoints, dstPoints)
print(h)

# Warp source image to destination based on homography
imH = cv2.warpPerspective(im, h, outDim)

# Display image
plt.figure(figsize=[20,10])
plt.subplot(131)
plt.imshow(imT[:,:,::-1])
plt.title('Transformed Image')

plt.subplot(132)
plt.imshow(imA[:,:,::-1])
plt.title('Image warped using estimated Affine Transform')

plt.subplot(133)
plt.imshow(imH[:,:,::-1])
plt.title('Image warped using estimated Homography')
plt.show()

##
## Homography Example

# Read source image.
im_src = cv2.imread(dataPath.IMAGES_ROOT + '/book2.jpg')
# Four corners of the book in source image
pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]], dtype=float)


# Read destination image.
im_dst = cv2.imread(dataPath.IMAGES_ROOT + '/book1.jpg')
# Four corners of the book in destination image.
pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]], dtype=float)

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

# Display images
plt.figure(figsize=[20,10])
plt.subplot(131)
plt.imshow(im_src[...,::-1])

plt.subplot(132)
plt.imshow(im_dst[...,::-1])

plt.subplot(133)
plt.imshow(im_out[...,::-1])
plt.show()