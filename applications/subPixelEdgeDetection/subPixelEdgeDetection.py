# Import relevant libraries
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import pickle as cPickle

#
# Append directories
#

# Get the path of the common directory to import common modules
directory = os.path.dirname( os.path.abspath(__file__))
parentDirectory = os.path.abspath(os.path.join(directory, os.pardir))
commonDirectory = os.path.abspath(os.path.join(parentDirectory, "common"))

# Extend path to import common modules
sys.path.append(commonDirectory)

# Import the common modukes
import dataPath

#
# Helper fuinctions
#
def computeDerivative(img, sigmaX, sigmaY):
    # blurr the image
    gaussian = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigmaX, sigmaY=sigmaY)
    # create filter for derivative calulation
    dxFilter = np.array([[1],[0],[-1]])
    dyFilter = np.array([[1,0,-1]])
    dxxFilter = np.array([[1],[-2],[1]])
    dyyFilter = np.array([[1,-2,1]])
    dxyFilter = np.array([[1,-1],[-1,1]])
    # compute derivative
    dx = cv2.filter2D(gaussian,-1, dxFilter)
    dy = cv2.filter2D(gaussian,-1, dyFilter)
    dxx = cv2.filter2D(gaussian,-1, dxxFilter)
    dyy = cv2.filter2D(gaussian,-1, dyyFilter)
    dxy = cv2.filter2D(gaussian,-1, dxyFilter)
    return dx, dy, dxx, dyy, dxy

def computeMagnitude(dxx, dyy):
    # convert to float
    dxxFlt = dxx.astype(float)
    dyyFlt = dyy.astype(float)
    # calculate magnitude and angle
    mag = cv2.magnitude(dxxFlt, dyyFlt)
    phase = mag*180./np.pi
    return mag, phase

def computeHessian(dx, dy, dxx, dyy, dxy):
    # create empty list
    point=[]
    direction=[]
    value=[]
    hessianMaxImage = np.zeros(dx.shape)
    hessianMaxImage = hessianMaxImage.astype(float)
    # for the all image
    for x in range(0, dx.shape[1]): # column
        for y in range(0, dx.shape[0]): # line
            # if superior to certain threshold
            if dxy[y,x] > 0:
                # compute local hessian
                hessian = np.zeros((2,2))
                hessian[0,0] = dxx[y,x]
                hessian[0,1] = dxy[y,x]
                hessian[1,0] = dxy[y,x]
                hessian[1,1] = dyy[y,x]
                # compute eigen vector and eigne value
                ret, eigenVal, eigenVect = cv2.eigen(hessian)
                maxEigenVal = 0
                if np.abs(eigenVal[0,0]) >= np.abs(eigenVal[1,0]):
                    nx = eigenVect[0,0]
                    ny = eigenVect[0,1]
                    maxEigenVal = np.abs(eigenVal[0,0])
                else:
                    nx = eigenVect[1,0]
                    ny = eigenVect[1,1]
                    maxEigenVal = np.abs(eigenVal[1,0])
                # calculate denominator for the taylor polynomial expension
                denom = dxx[y,x]*nx*nx + dyy[y,x]*ny*ny + 2*dxy[y,x]*nx*ny
                # verify non zero denom
                if denom != 0:
                    T = -(dx[y,x]*nx + dy[y,x]*ny)/denom
                    # update point
                    if np.abs(T*nx) <= 0.5 and np.abs(T*ny) <= 0.5:
                        point.append((x,y))
                        direction.append((nx,ny))
                        value.append(np.abs(dxy[y,x]+dxy[y,x]))
                        #hessianMaxImage[y,x] = maxEigenVal
                        hessianMaxImage[y,x] = np.abs(dxy[y,x]+dxy[y,x])
    return point, direction, value, hessianMaxImage

def nonMaxSuppression(det, phase):
    # gradient max init
    gmax = np.zeros(det.shape)
    # thin-out evry edge for angle = [0, 45, 90, 135]
    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
            if phase[i][j] < 0:
                phase[i][j] += 360
            if ((j+1) < gmax.shape[1]) and ((j-1) >= 0) and ((i+1) < gmax.shape[0]) and ((i-1) >= 0):
                # 0 degrees
                if (phase[i][j] >= 337.5 or phase[i][j] < 22.5) or (phase[i][j] >= 157.5 and phase[i][j] < 202.5):
                    if det[i][j] >= det[i][j + 1] and det[i][j] >= det[i][j - 1]:
                        gmax[i][j] = det[i][j]
                # 45 degrees
                if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
                    if det[i][j] >= det[i - 1][j + 1] and det[i][j] >= det[i + 1][j - 1]:
                        gmax[i][j] = det[i][j]
                # 90 degrees
                if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
                    if det[i][j] >= det[i - 1][j] and det[i][j] >= det[i + 1][j]:
                        gmax[i][j] = det[i][j]
                # 135 degrees
                if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
                    if det[i][j] >= det[i - 1][j - 1] and det[i][j] >= det[i + 1][j + 1]:
                        gmax[i][j] = det[i][j]
    return gmax


#
# MAIN FUNCTION
#
# resize, grayscale and blurr
orgImage = cv2.imread(dataPath.IMAGES_ROOT +  "/TestImage.bmp")
plt.figure(figsize=(5,5))
plt.title("Original Input Image")
plt.imshow(orgImage)
grayImage = cv2.cvtColor(orgImage, cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(5,5))
plt.title("Original Gray Image")
plt.imshow(grayImage)
# compute derivative
dx, dy, dxx, dyy, dxy = computeDerivative(grayImage, 1.1, 1.1)

plt.figure(figsize=(5,5))
plt.title("DX")
plt.imshow(dx)

plt.figure(figsize=(5,5))
plt.title("DY")
plt.imshow(dy)

plt.figure(figsize=(5,5))
plt.title("DXX")
plt.imshow(dxx)

plt.figure(figsize=(5,5))
plt.title("DYY")
plt.imshow(dyy)

plt.figure(figsize=(5,5))
plt.title("DXY")
plt.imshow(dxy)

normal, phase = computeMagnitude(dxx, dyy)

plt.figure(figsize=(5,5))
plt.title("Normal")
plt.imshow(normal)

plt.figure(figsize=(5,5))
plt.title("Phase")
plt.imshow(phase)

dxyNms = nonMaxSuppression(normal, phase)

plt.figure(figsize=(5,5))
plt.title("dxyNms")
plt.imshow(dxyNms)



pt, dir, val, hessianMaxImage = computeHessian(dx, dy, dxx, dyy, dxy)

plt.figure(figsize=(5,5))
plt.title("Hessian")
plt.imshow(hessianMaxImage)

# take the first n max value
nMax = 5000
idx = np.argsort(val)
idx = idx[::-1][:nMax]
resultImage = orgImage.copy()
# plot resulting point
for i in range(0, len(idx)):
    resultImage = cv2.circle(resultImage, (pt[idx[i]][0], pt[idx[i]][1]), 1, (255, 0, 0), 1)

plt.figure(figsize=(5,5))
plt.title("Result")
plt.imshow(resultImage)

plt.show()