import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib

# Get the path of the common directory to import common modules
directory = os.path.dirname( os.path.abspath(__file__))
#print("directory: " + directory)
parentDirectory = os.path.abspath(os.path.join(directory, os.pardir))
#print("parentDirectory: " + parentDirectory)
commonDirectory = os.path.abspath(os.path.join(parentDirectory, "common"))
#print("commonDirectory: " + commonDirectory)

# Extend path to import common modules
sys.path.append(commonDirectory)

import dataPath

print("Alpha Blending")

matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'

# Read the foreground image with alpha channel
foreGroundImage = cv2.imread(dataPath.IMAGES_ROOT + "/foreGroundAssetLarge.png", -1)

# Split png foreground image
b,g,r,a = cv2.split(foreGroundImage)

# Reverse B,G,R channels
plt.imshow(foreGroundImage[:,:,[2,1,0,3]])
plt.title("Foreground Image")
plt.show()

# Save the foregroung RGB content into a single object
foreground = cv2.merge((b,g,r))

# Save the alpha information into a single Mat
alpha = cv2.merge((a,a,a))

# Read background image
background = cv2.imread(dataPath.IMAGES_ROOT  + "/backGroundLarge.jpg")

plt.imshow(background[:,:,::-1])
plt.title("Background")
plt.show()

# Convert uint8 to float
foreground = foreground.astype(float)
background = background.astype(float)
alpha = alpha.astype(float)/255

# Perform alpha blending
foreground = cv2.multiply(alpha, foreground)
background = cv2.multiply(1.0 - alpha, background)
outImage = cv2.add(foreground, background)

plt.imshow(outImage[:,:,::-1]/255)
plt.title("Alpha Blended Image")
plt.show()