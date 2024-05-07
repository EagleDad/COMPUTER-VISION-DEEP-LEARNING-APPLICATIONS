# Import relevant libraries
import math
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

# https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/

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

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# Dlib shape predictor model path
MODEL_PATH = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"
# Load model
shape_predictor = dlib.shape_predictor(MODEL_PATH)

def extractPoints(landmarks, start, end):
    points = []
    for i in range(start, end+1):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    return points

def smile_detector(imDlib, colorFrame):
    # Detect faces
    faces = detector(imDlib, 0)
    
    if len(faces):
        landmarks = shape_predictor(imDlib, faces[0])
    else:
        return False
    
    isSmiling = False
    ###
    ### YOUR CODE HERE
    ###
    # Return True if smile is detected

    # Based on the description of the facial ladmark points, the following is defined:
    # Mouth:    Points 48 - 67
    # Jaw:      Points 0 - 16

    # First we need to extract the corresponding points for mouth and jaw
    mouthPoints = extractPoints(landmarks, 48, 67)
    #for p in mouthPoints:
    #    cv2.circle(colorFrame, (p[0], p[1]), 1, (0,255,0), -1)

    jawPoints = extractPoints(landmarks, 0, 16)
    #for p in jawPoints:
    #    cv2.circle(colorFrame, (p[0], p[1]), 1, (0,0,255), -1)

    # If a person smiles can be idendified by the ration between width of the lip
    # and width of the jaw. To be able to calculate teh ration, we first need to get the with.
    # We do that by calculation the min enclosinf rectangle around the extracted points 
    mouthRect = cv2.minAreaRect(mouthPoints)
    jawRect = cv2.minAreaRect(jawPoints)

    (cxm, cym), (mouseWidth, mouseHeight), mouthAngle = mouthRect
    (cxj, cyj), (jawWidth, jawHeight), jawAngle = jawRect

    if mouthAngle > 0:
        mouseWidth, mouseHeight = mouseHeight, mouseWidth

    if jawAngle > 0:
        jawWidth, jawHeight = jawHeight, jawWidth

    boxMouth = cv2.boxPoints(mouthRect) 
    boxMouth = np.int0(boxMouth)
    #cv2.drawContours(colorFrame,[boxMouth],0,(0,255,0),2)

    boxJaw = cv2.boxPoints(jawRect) 
    boxJaw = np.int0(boxJaw)
    #cv2.drawContours(colorFrame,[boxJaw],0,(0,0,255),2)

    lipJawRatio = mouseWidth / jawWidth

    if lipJawRatio > 0.44:
        isSmiling = True
    else:
        isSmiling = False

    return isSmiling

# Initializing video capture object.
capture = cv2.VideoCapture(dataPath.IMAGES_ROOT + "/smile.mp4")
if(False == capture.isOpened()):
    print("[ERROR] Video not opened properly")    

# Create a VideoWriter object
smileDetectionOut = cv2.VideoWriter(dataPath.IMAGES_ROOT + "/smileDetectionOutput.avi",
                                   cv2.VideoWriter_fourcc('M','J','P','G'),
                                   15,(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                       int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
frame_number = 0
smile_frames = []
while (True):
    # grab the next frame
    isGrabbed, frame = capture.read()
    if not isGrabbed:
        break
        
    imDlib = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_has_smile = smile_detector(imDlib, frame)
    if (True == frame_has_smile):
        cv2.putText(frame, "Smiling :)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        smile_frames.append(frame_number)
#         print("Smile detected in Frame# {}".format(frame_number))
    if frame_number % 50 == 0:
        print('\nProcessed {} frames'.format(frame_number))
        print("Smile detected in Frames: {}".format(smile_frames))
    # Write to VideoWriter
    smileDetectionOut.write(frame)

    cv2.imshow("Frame", frame)
    cv2.waitKey(25) 
    
    frame_number += 1

capture.release()
smileDetectionOut.release()