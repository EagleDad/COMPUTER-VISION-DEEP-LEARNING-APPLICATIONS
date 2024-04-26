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

sourceImage = cv2.imread( dataPath.IMAGES_ROOT + "/times-square.jpg")

orgImage = sourceImage.copy()

targetImage = cv2.imread( dataPath.IMAGES_ROOT + "/times-square.jpg" )

replacementImage = cv2.imread( dataPath.IMAGES_ROOT + "/first-image.jpg" )

input_pts = []

windowNameSrc = "Source Image"
windowNameTgt = "Target Image"
windowNameAd = "Advertisement"

def updateView():
    cv2.imshow(windowNameSrc, sourceImage)
    cv2.imshow(windowNameTgt, targetImage)
    cv2.imshow(windowNameAd, replacementImage)

def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        loc = (int(x), int(y))

        cv2.circle(sourceImage, loc, 5, (0, 255, 0),5, -1)

        input_pts.append([x,y])

        if ( len(input_pts) == 4 ):
            #targetImage = orgImage.copy()
            #sourceImage = orgImage.copy()
            tgt = orgImage.copy()
           
            
            size = replacementImage.shape
            pts_dst = np.vstack(input_pts).astype(float)

            pts_src = np.array(
                       [
                        [0, 0],
                        [size[1] - 1, 0],
                        [size[1] - 1, size[0] -1],
                        [0, size[0] - 1 ]
                        ],dtype=float
                       )

            homography, status = cv2.findHomography( pts_src, pts_dst )

            tmp = cv2.warpPerspective( replacementImage, homography, (tgt.shape[1], tgt.shape[0]) )

            # Black out polygonal area in destination image.
            cv2.fillConvexPoly(tgt, pts_dst.astype(int), 0 , 16 )

            # Add warped source image to destination image.
            result = tgt + tmp

            globals()['targetImage'] = result.copy()
            globals()['sourceImage'] = orgImage.copy()
            globals()['input_pts'] = []

       



        updateView()

    


updateView()

cv2.setMouseCallback(windowNameSrc, onMouse )

key = 0

while (  key != 27 ):
    key = cv2.waitKey(20)

def mouse_handler(event, x, y, flags, data) :
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16)
        cv2.imshow("Image", data['im'])
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points(im):
    
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    
    return points    