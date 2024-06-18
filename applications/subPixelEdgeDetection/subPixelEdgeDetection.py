# Import relevant libraries
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import pickle as cPickle
import scipy
import scipy.signal
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit 

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
    hessianMaxImage = np.zeros(dx.shape, np.float32)
    width = dx.shape[1]
    height = dx.shape[0]
    count = 0
    thresh = 5
    hessian = np.zeros((2, 2))

    for y in range(0, height):
        for x in range(0, width):
            #valid = np.abs(dxy[y, x]) > thresh

            # Potential line points have a high curvature 
            # An edge is characterized by high curvature in one direction and low curvature in orthogonal direction
            # Curvature can be estimated by the Hessian matrix
            # The eigenvalues of H are proportional to the curvature 
            Dxx = dxx[y, x]
            Dyy = dyy[y, x]
            Dxy = dxy[y, x]

            TraceH = Dxx + Dyy
            DetH = Dxx * Dyy - Dxy * Dxy

            # If the determinate of the hessian is negative, the point in question cannot be a extrmum and it is discarted
            #if DetH < 0:
            #    continue

            if np.abs(DetH) < 1:
                continue

            Ratio = TraceH * TraceH / DetH

            if Ratio < 5:
                continue

            valid = True
            if valid:
                
                # compute local hessian
                hessian[0, 0] = dxx[y, x]
                hessian[0, 1] = dxy[y, x]
                hessian[1, 0] = dxy[y, x]
                hessian[1, 1] = dyy[y, x]

                # compute eigen vector and eigne value
                ret, eigenVal, eigenVect = cv2.eigen(hessian)
                maxEigenVal = 0
                nx = 0
                ny = 0
                if np.abs(eigenVal[0,0]) >= np.abs(eigenVal[1,0]):
                    nx = eigenVect[0,0]
                    ny = eigenVect[0,1]
                    maxEigenVal = np.abs(eigenVal[0,0])
                else:
                    nx = eigenVect[1,0]
                    ny = eigenVect[1,1]
                    maxEigenVal = np.abs(eigenVal[1,0])

                hessianMaxImage[y,x] = maxEigenVal
                # calculate denominator for the taylor polynomial expension
                #denom = dxx[y,x]*nx*nx + dyy[y,x]*ny*ny + 2*dxy[y,x]*nx*ny
                # verify non zero denom
                #if denom != 0:
                #    T = -((dx[y,x]*nx + dy[y,x]*ny)/denom)
                #    # update point
                #    if np.abs(T*nx) <= 0.5 and np.abs(T*ny) <= 0.5:
                #        point.append((x,y))
                #        direction.append((nx,ny))
                #        value.append(np.abs(dxy[y,x]+dxy[y,x]))
                #        count = count + 1
                        #hessianMaxImage[y,x] = maxEigenVal
                        #hessianMaxImage[y,x] = np.abs(dxy[y,x]+dxy[y,x])
                #        hessianMaxImage[y, x] = 255#np.abs(dx[y,x]+dy[y,x])
                    
    print("Number points:", count)
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


def getSobelKernel(dx, dy, kernelSize):
    kx, ky = cv2.getDerivKernels(dx, dy, kernelSize )
    return kx.dot(ky.transpose())

def showSobelKernel(dx, dy, kernelSize, title="Kernel"):
    kernel = getSobelKernel(dx, dy, kernelSize)

    xx, yy = np.mgrid[0:kernel.shape[0], 0:kernel.shape[1]]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection = '3d')
    ax.plot_surface(xx, yy, kernel,rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=plt.cm.viridis)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

def showRequiredKernels():
    showSobelKernel(1, 0, 31, "Sx")
    showSobelKernel(0, 1, 31, "Sy")
    showSobelKernel(1, 1, 31, "Sxy")
    showSobelKernel(2, 0, 31, "Sxx")
    showSobelKernel(0, 2, 31, "Syy")
    plt.show()

def getDerivativeKernels(kernelSize=3):
    rx = getSobelKernel(1, 0, kernelSize)
    ry = getSobelKernel(0, 1, kernelSize)
    rxx = getSobelKernel(2, 0, kernelSize)
    ryy = getSobelKernel(0, 2, kernelSize)
    rxy = getSobelKernel(1, 1, kernelSize)

    return rx, ry, rxx, ryy, rxy


def edgesDeriche():
    print("Deriche")    

#
# MAIN FUNCTION
#

ym1 = 11
yo = 24
yp1 = 22

xData = [-1, 0, 1]
yData = [ym1, yo, yp1]

xData = np.asarray(xData) 
yData = np.asarray(yData) 

# Define the Gaussian function 
def Gauss(x, A, B): 
    y = A*np.exp(-1*B*x**2) 
    return y 
parameters, covariance = curve_fit(Gauss, xData, yData) 
  
fit_A = parameters[0] 
fit_B = parameters[1]


# plot
fit_y = Gauss(xData, fit_A, fit_B) 
plt.plot(xData, yData, 'o', label='data') 
plt.plot(xData, fit_y, '-', label='fit') 
plt.legend()
plt.show()

#http://www.learnpiv.org/subPixel/

denom = 2.0 * ( np.log(ym1) + np.log(yp1) - 2 * np.log(yo))

correction = (np.log(ym1) - np.log(yp1)) / denom



kernelSize = 3
blurSize = 5
for i in range(5):

    orgImage = cv2.imread(dataPath.IMAGES_ROOT +  "/TestImage.bmp")
    grayImage = cv2.cvtColor(orgImage, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.GaussianBlur(src=grayImage, ksize=(blurSize, blurSize), sigmaX=0, sigmaY=0)

    Sx = cv2.filter2D(src=grayImage, ddepth=cv2.CV_32F, kernel=getSobelKernel(1, 0, kernelSize))
    Sy = cv2.filter2D(src=grayImage, ddepth=cv2.CV_32F, kernel=getSobelKernel(0, 1, kernelSize))
    Sxy = cv2.filter2D(src=grayImage, ddepth=cv2.CV_32F, kernel=getSobelKernel(1, 1, kernelSize))
    Sxx = cv2.filter2D(src=grayImage, ddepth=cv2.CV_32F, kernel=getSobelKernel(2, 0, kernelSize))
    Syy = cv2.filter2D(src=grayImage, ddepth=cv2.CV_32F, kernel=getSobelKernel(0, 2, kernelSize))

    fig, axs = plt.subplots(3, 2)
    fig.suptitle('Working Images')

    axs[0, 0].set_title('Input Image')
    axs[0, 0].imshow(orgImage)

    axs[1, 0].set_title('Sx')
    axs[1, 0].imshow(Sx)

    axs[2, 0].set_title('Sy')
    axs[2, 0].imshow(Sy)


    axs[0, 1].set_title('Sxx')
    axs[0, 1].imshow(Sxx)

    axs[1, 1].set_title('Syy')
    axs[1, 1].imshow(Syy)

    axs[2, 1].set_title('Sxy')
    axs[2, 1].imshow(Sxy)


    pt, dir, val, hessianMaxImage = computeHessian(dx=Sx, dy=Sy, dxx=Sxx, dyy=Syy, dxy=Sxy)

    fig2, axs2 = plt.subplots()
    fig2.suptitle('Hessian Image')
    axs2.set_title('Hessian Image')
    axs2.imshow(hessianMaxImage)

    normal, phase = computeMagnitude(Sxx, Syy)

    fig3, axs3 = plt.subplots()
    fig3.suptitle('Mag Image')
    axs3.set_title('Mag Image')
    axs3.imshow(normal)

    plt.show()
    kernelSize = kernelSize + 2


#print("SOBEL X")
#kernelSize = 3
#kx, ky = cv2.getDerivKernels(1, 0, kernelSize )
#sobelx = kx.dot(ky.transpose())
#print(kx)
#print(ky)
#print(sobelx)

#print("SOBEL Y")
#kx, ky = cv2.getDerivKernels(0, 1, kernelSize )
#sobely = kx.dot(ky.transpose())
#print(kx)
#print(ky)
#print(sobely)

#img = cv2.filter2D(src=grayImage, ddepth=-1, kernel=sobely)
#plt.imshow(grayImage)
#plt.imshow(img)

#xx, yy = np.mgrid[0:sobely.shape[0], 0:sobely.shape[1]]

#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(projection = '3d')
#ax.plot_surface(xx, yy, sobely,rstride=3, cstride=3, linewidth=1, antialiased=True,
#            cmap=plt.cm.viridis)
#plt.xlabel("x")
#plt.ylabel("y")
#plt.title("sobely")


#plt.show()



# resize, grayscale and blurr
# orgImage = cv2.imread(dataPath.IMAGES_ROOT +  "/TestImage.bmp")
# plt.figure(figsize=(5,5))
# plt.title("Original Input Image")
# plt.imshow(orgImage)
# grayImage = cv2.cvtColor(orgImage, cv2.COLOR_BGR2GRAY)
# plt.figure(figsize=(5,5))
# plt.title("Original Gray Image")
# plt.imshow(grayImage)
# # compute derivative
# dx, dy, dxx, dyy, dxy = computeDerivative(grayImage, 1.1, 1.1)

# plt.figure(figsize=(5,5))
# plt.title("DX")
# plt.imshow(dx)

# plt.figure(figsize=(5,5))
# plt.title("DY")
# plt.imshow(dy)

# plt.figure(figsize=(5,5))
# plt.title("DXX")
# plt.imshow(dxx)

# plt.figure(figsize=(5,5))
# plt.title("DYY")
# plt.imshow(dyy)

# plt.figure(figsize=(5,5))
# plt.title("DXY")
# plt.imshow(dxy)

# normal, phase = computeMagnitude(dxx, dyy)

# plt.figure(figsize=(5,5))
# plt.title("Normal")
# plt.imshow(normal)

# plt.figure(figsize=(5,5))
# plt.title("Phase")
# plt.imshow(phase)

# dxyNms = nonMaxSuppression(normal, phase)

# plt.figure(figsize=(5,5))
# plt.title("dxyNms")
# plt.imshow(dxyNms)



# pt, dir, val, hessianMaxImage = computeHessian(dx, dy, dxx, dyy, dxy)

# plt.figure(figsize=(5,5))
# plt.title("Hessian")
# plt.imshow(hessianMaxImage)

# # take the first n max value
# nMax = 5000
# idx = np.argsort(val)
# idx = idx[::-1][:nMax]
# resultImage = orgImage.copy()
# # plot resulting point
# for i in range(0, len(idx)):
#     resultImage = cv2.circle(resultImage, (pt[idx[i]][0], pt[idx[i]][1]), 1, (255, 0, 0), 1)

# plt.figure(figsize=(5,5))
# plt.title("Result")
# plt.imshow(resultImage)

# plt.show()