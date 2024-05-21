# Import relevant libraries
import dlib
import cv2
import numpy as np
import sys
import os

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
# END Append directories
#

#
# Define global variables here
#
windowName = "Virtual Makeup"
sourceImage = None
resultImage = None
overlayImage = None
combinedImage = None
helpImage = None
headerImage = None
helpStrings = []
headerStrings = []
textPixelHeight = 20
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontColor = ( 0, 255, 0 )
fontThickness = 2
lipColor = []
lipColors = []
colorPickerX1 = 0
colorPickerX2 = 0
colorPickerY1 = 0
colorPickerY2 = 0
scale = 0.75
colorPickerRoiDim = 50
#
# END Define global variables here
#

#
# Methode to detect the facial landmark points
#
def detectFacialLandmarks( imageIn ):
    modelPath = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"
    faceDetector = dlib.get_frontal_face_detector( )
    landmarkDetector = dlib.shape_predictor( modelPath )

    # DLib required the image to be gry scale
    # In this application we know that the input is color
    imDlib = cv2.cvtColor( imageIn, cv2.COLOR_BGR2RGB )

    # Detect faces
    faces = faceDetector( imDlib, 0 )

    # Get the landmark point. In this App we know that we only have one face
    landmarks = landmarkDetector( imDlib, faces[0] )

    return landmarks

#
# Methode to detect the facial landmark points in a defined range
#
def extractPoints(landmarks, start, end):
    points = []
    for i in range(start, end+1):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    return points
#
# END Methode to detect the facial landmark points in a defined range
#

#
# END Methode to detect the facial landmark points for a special location
#
def extractPointsForLocation(landmarks, location):

    if location == "jaw":
        return extractPoints( landmarks, 0, 16 )
    
    if location == "eyebrows":
        return extractPoints( landmarks, 17, 26 )
    
    if location == "nose":
        return extractPoints( landmarks, 27, 35 )
    
    if location == "eyes":
        return extractPoints( landmarks, 36, 47 )
    
    if location == "lips":
        return extractPoints( landmarks, 48, 67 )
    
    if location == "outerlip":
        return extractPoints( landmarks, 48, 59 )
    
    if location == "innerlip":
        return extractPoints( landmarks, 60, 67 )
    
    if location == "leftCheek":
        indieces = [1, 2, 3, 48, 49, 32, 41]
        points = []
        for i in indieces:
            point = [landmarks.part(i).x, landmarks.part(i).y]
            points.append(point)
        return points
    
    if location == "rightCheek":
        indieces = [15, 14, 13, 53, 54, 35, 46]
        points = []
        for i in indieces:
            point = [landmarks.part(i).x, landmarks.part(i).y]
            points.append(point)
        return points
    
    return

#
# Methode that exctrac special points from the landmarks
#

#
# Methode that draw points to an image
#
def drawPoint( image, point, color = (0, 255, 0), radius = 3, thickness=-1 ):
    cv2.circle(image, point, radius, color, thickness=thickness)

def drawPoints( image, points, color = (0, 255, 0), radius = 3, thickness=-1 ):
    for p in points:
        drawPoint(image, p, radius = radius, color=color, thickness= thickness)

#
#  END Methode that draw points to an image
#

#
# Update current view
#
def updateView():
    combinedImage = cv2.hconcat( [sourceImage, overlayImage] )
    combinedImage = cv2.hconcat( [combinedImage, resultImage] )

    fontScale = cv2.getFontScaleFromHeight(
            fontFace, textPixelHeight, fontThickness )

    height, width, channels = combinedImage.shape
    srcHeight, srcWidth, srcChannels = sourceImage.shape
    
    if globals()['helpImage'] is None: 
        helpImageHeight = textPixelHeight / 2

        for text in helpStrings: 
            helpImageHeight += textPixelHeight
            helpImageHeight += textPixelHeight / 2
        
        helpImageHeight += textPixelHeight / 2

        tmpHelpImage = np.zeros( ( int(helpImageHeight), width, 3 ), np.uint8 )

        textPosY = textPixelHeight + textPixelHeight / 2
        textPosX = textPixelHeight

        for text in helpStrings: 
            tmpHelpImage = cv2.putText( tmpHelpImage,
                                    text,
                                    ( int(textPosX), int(textPosY) ),
                                    fontFace,
                                    fontScale,
                                    fontColor,
                                    fontThickness,
                                    cv2.LINE_AA )

            textPosY += textPixelHeight + textPixelHeight / 2

        offsetX = srcWidth
        offsetY = 0
        
        for color in lipColors:
            roi = tmpHelpImage[offsetY : offsetY + colorPickerRoiDim, offsetX : offsetX + colorPickerRoiDim]
            roi[:] = color
            offsetX += colorPickerRoiDim


        globals()['colorPickerX1'] = srcWidth
        globals()['colorPickerX2'] = offsetX
        globals()['helpImage'] = tmpHelpImage.copy()

    

    if globals()['headerImage'] is None:
        headerImageHeight = 2 * textPixelHeight

        tmpHeaderImage = np.zeros( ( int(headerImageHeight), width, 3 ), np.uint8 )

        xOffset = textPixelHeight
        yPos = textPixelHeight + textPixelHeight / 2

        for text in headerStrings: 
            tmpHeaderImage = cv2.putText( tmpHeaderImage,
                                    text,
                                    ( int(xOffset), int(yPos) ),
                                    fontFace,
                                    fontScale,
                                    fontColor,
                                    fontThickness,
                                    cv2.LINE_AA )

            xOffset += srcWidth

        globals()['colorPickerY1'] = headerImageHeight + srcHeight
        globals()['colorPickerY2'] = headerImageHeight + srcHeight + colorPickerRoiDim
        globals()['headerImage'] = tmpHeaderImage.copy()

    combinedImage = cv2.vconcat( [headerImage, combinedImage] ) 
    combinedImage = cv2.vconcat( [combinedImage, helpImage] ) 

    roi = combinedImage[colorPickerY2 : colorPickerY2 + colorPickerRoiDim, colorPickerX1 : colorPickerX1 + colorPickerRoiDim]
    roi[:] = lipColor

    globals()['combinedImage'] = combinedImage.copy()
    cv2.imshow(windowName, combinedImage)   
#
# END
#


#
# Reset images to original state
#
def resetImages( image ):
    globals()['resultImage'] = image.copy()
    globals()['overlayImage'] = image.copy()
#
# END Reset images to original state
#

#
# Function that returs the U8C3 3channel lip mask mask 
#
def getLipMask( src, overlay ):
    # Get the landmark points corresponding for the lips
    landmarkPoints = extractPointsForLocation( facialLandmarks, "lips" )
    drawPoints( overlay, landmarkPoints )

    outerLip = extractPointsForLocation( facialLandmarks, "outerlip" )
    innerLip = extractPointsForLocation( facialLandmarks, "innerlip" )

    # Create mask image for inner and outer
    maskImageOuter = np.zeros_like( src )
    maskImageInner = np.zeros_like( src )

    # Fill all points in the lipp points white to get a valid lip mask 
    maskImageOuter = cv2.fillPoly( maskImageOuter, [outerLip], (255, 255, 255) )
    maskImageInner = cv2.fillPoly( maskImageInner, [innerLip], (255, 255, 255) )
    maskImage = maskImageOuter - maskImageInner
    return maskImage
#
#
#

#
# Calculate lipstick filter
#
def lipStickFilter( src, overlay, lipStickColor ):
    # Get the mask of the lips as 3 thannel 8 bit image
    maskImage = getLipMask( src, overlay )

    # Split channesl to get a single channel 8 biat mask for OpenCV operations
    c1, c2, c3 = cv2.split( maskImage )

    # Convert mask to float in range [ 0 ..1 ]
    maskImage = np.float32( maskImage ) / 255

    # Create a image for the lip color mask
    colorLipImage = np.zeros_like( src )
    # Fill the image with the desired color
    colorLipImage[:] = lipStickColor 

    # Convert the desired color image to float
    colorLipImage = np.float32( colorLipImage ) / 255

    # Blur mask to get better results
    kSize = 35
    maskImage = cv2.GaussianBlur( maskImage, (kSize, kSize), sigmaX = 0, sigmaY = 0, borderType = cv2.BORDER_DEFAULT )

    # Convert src to float in range [ 0 ..1 ] 
    srcFlt = np.float32( src ) / 255

    # Get the avereag color of the lips
    meanLipColor = cv2.mean( srcFlt, c1 )[0:3]
    meanLipColorImage = np.zeros_like( srcFlt )
    meanLipColorImage[:] = meanLipColor

    # Get the lips part but subtrats mean coler first to add lipstick on top of it
    lipsPart = (srcFlt - meanLipColor) * maskImage

    # Get colored lips
    coloredLips = (lipsPart + colorLipImage) * maskImage

    # Get the image without the lips part
    withOutLipsImage = srcFlt  * (1 - maskImage)

    # Get the result image by suming up both parts
    result = coloredLips + withOutLipsImage
    # Clip to valid range [0 .. 1]
    result = np.clip(result, 0, 1)
    result = np.uint8(255 * result)
    return result
#
# END Calculate lipstick filter
#

# Check if a point is inside a rectangle
def rectContains(rect, point):
  if point[0] < rect[0]:
    return False
  elif point[1] < rect[1]:
    return False
  elif point[0] > rect[0] + rect[2]:
    return False
  elif point[1] > rect[1] + rect[3]:
    return False
  return True

#
# Function that calculates the max inner circle using voronoi diagramm
# https://www.researchgate.net/publication/327419406_A_maximum_inscribed_circle_algorithm_based_on_Voronoi_diagrams_and_geometry_equations/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19
#
def maxInnerCircle( points ):
    points = np.array(points, dtype=np.float32)
    bRect = cv2.boundingRect(points)
    
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(bRect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # Get Delaunay triangulation
    (facets, centers) = subdiv.getVoronoiFacetList([])

    minDist = 0.0
    center = None

    for i in range(0, len(facets) ):
        for f in facets[i] :
            if rectContains(bRect, f):
                dist = cv2.norm(centers[i] - f)

                if dist > 0 and dist > minDist:
                    minDist = dist
                    center = f

    return [center, dist]

#
#
#

#
# Calculate blush filter
#
def blushFilter( src, overlay, blushColor ):

    # Extract the landmark point for left and right cheek
    landmarkPointsRight = extractPointsForLocation( facialLandmarks, "rightCheek" )
    landmarkPointsLeft = extractPointsForLocation( facialLandmarks, "leftCheek" )
    drawPoints( overlay, landmarkPointsRight, color=(0, 255, 0))
    drawPoints( overlay, landmarkPointsLeft, color=(0, 0, 255) )

    # The max inner circle descibes the region where bush would be applied
    [cr, rr] = maxInnerCircle(landmarkPointsRight)
    [cl, rl] = maxInnerCircle(landmarkPointsLeft)
    drawPoint(overlay, (int(cr[0]), int(cr[1])), radius=int(rr), color=(0, 255, 0), thickness=1)
    drawPoint(overlay, (int(cl[0]), int(cl[1])), radius=int(rl), color=(0, 0, 255), thickness=1)

    # Create mask images for left and right cheek
    height, width, channels = src.shape
    
    maskRightImage = np.zeros( (height, width, 1 ), np.uint8 )
    maskLeftImage = np.zeros( (height, width, 1 ), np.uint8 )

    cv2.circle(maskRightImage, (int(cr[0]), int(cr[1])), radius=int(rr), color=(255), thickness=-1)
    cv2.circle(maskLeftImage, (int(cl[0]), int(cl[1])), radius=int(rl), color=(255), thickness=-1)

    # The blush is more dominant in the center so we do a distance transformation to get a wheighted mask
    distTransformRight = cv2.distanceTransform(maskRightImage, cv2.DIST_L2, 5) 
    distTransformLeft = cv2.distanceTransform(maskLeftImage, cv2.DIST_L2, 3) 

    distTransformRight = cv2.normalize(distTransformRight, None, 0, 1.0, cv2.NORM_MINMAX) 
    distTransformLeft = cv2.normalize(distTransformLeft, None, 0, 1.0, cv2.NORM_MINMAX) 

    # Blur mask to get better results
    kSize = 35
    distTransformRight = cv2.GaussianBlur( distTransformRight, (kSize, kSize), sigmaX = 0, sigmaY = 0, borderType = cv2.BORDER_DEFAULT )
    distTransformLeft = cv2.GaussianBlur( distTransformLeft, (kSize, kSize), sigmaX = 0, sigmaY = 0, borderType = cv2.BORDER_DEFAULT )

    # Create a image for the lip color mask
    colorCheekImage = np.zeros_like( src )
    # Fill the image with the desired color
    colorCheekImage[:] = blushColor

    # Convert the desired color image to float
    colorCheekImage = np.float32( colorCheekImage ) / 255

    # Convert src to float in range [ 0 .. 1 ] 
    srcFlt = np.float32( src ) / 255

    # Get the avereag color of left and right cheek
    meanColorRight = cv2.mean( srcFlt, maskRightImage )[0:3]
    meanColorLeft = cv2.mean( srcFlt, maskLeftImage )[0:3]

    meanRightCheekColorImage = np.zeros_like( srcFlt )
    meanRightCheekColorImage[:] = meanColorRight

    meanLeftCheekColorImage = np.zeros_like( srcFlt )
    meanLeftCheekColorImage[:] = meanColorLeft

    maskImageRight = cv2.merge([distTransformRight, distTransformRight, distTransformRight])
    maskImageLeft = cv2.merge([distTransformLeft, distTransformLeft, distTransformLeft]) 

    # Get the cheek parts but subtract mean coler first to add blush on top of it
    rightCheekPart = (srcFlt - meanColorRight) * maskImageRight
    leftCheekPart = (srcFlt - meanColorLeft) * maskImageLeft


    # Get colored cheek part
    coloredCheekRight = (rightCheekPart + colorCheekImage) * maskImageRight
    coloredCheekLeft = (leftCheekPart + colorCheekImage) * maskImageLeft
    coloredCheek = coloredCheekRight + coloredCheekLeft

    # Get the image without the cheeks part
    withOutCheekRightImage = srcFlt  * (1 - maskImageRight)
    withOutCheek = withOutCheekRightImage  * (1 - maskImageLeft)

    # Get the result image by suming up both parts
    result = coloredCheek + withOutCheek

    # Clip to valid range [0 .. 1]
    result = np.clip(result, 0, 1)
    result = np.uint8(255 * result)
    return result
#
# END Calculate lipstick filter
#

#
# OpenCV mouse handler
#
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN :
        if x > colorPickerX1 and x < colorPickerX2 and y > colorPickerY1 and y < colorPickerY2:
            globals()['lipColor']  = combinedImage[y, x]
            updateView()
#
# END OpenCV mouse handler
#

#
# Run main loop
#

# Read in the main image
sourceImage = cv2.imread( dataPath.IMAGES_ROOT + "/girl-no-makeup.jpg")

height, width, srcChannels = sourceImage.shape
downWidth = scale * width
downHeight = scale * height
downSize = ( int(downWidth), int(downHeight) )
sourceImage = cv2.resize(sourceImage, downSize, interpolation= cv2.INTER_LINEAR)

resultImage = sourceImage.copy()
overlayImage = sourceImage.copy()

helpStrings.append("Press 'l' for lipstick. Select color ->")
helpStrings.append("Press 'b' for blush. Select color ->")
helpStrings.append("Press 'r' for reset")
helpStrings.append("Press 'ESC' for exit")

headerStrings.append("Original Image")
headerStrings.append("Overlay Image")
headerStrings.append("Result Image")

# Red
lipColors.append((170, 130, 255))
lipColors.append((35, 135, 249))
lipColors.append((106, 106, 231))
lipColors.append((91, 91, 214))
lipColors.append((75, 75, 193))
lipColors.append((63, 63, 184))
lipColors.append((34, 32, 185))

# Yellow
lipColors.append((81, 255, 254))

lipColor = lipColors[0]

facialLandmarks = detectFacialLandmarks(sourceImage)

updateView()

cv2.setMouseCallback( windowName, onMouse )

key = 0
while (  key != 27 ):

    if key == ord('l'):
        resetImages( sourceImage )
        resultImage = lipStickFilter(sourceImage, overlayImage, lipColor )
        updateView()

    if key == ord('b'):
        resetImages( sourceImage )
        resultImage = blushFilter(sourceImage, overlayImage, lipColor )
        updateView()

    if key == ord('r'):
        resetImages( sourceImage )
        updateView()

    key = cv2.waitKey(20)