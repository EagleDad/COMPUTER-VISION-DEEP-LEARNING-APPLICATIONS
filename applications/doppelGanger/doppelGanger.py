# Import relevant libraries
import dlib
import cv2
import numpy as np
import sys
import os
import glob
import random
import time
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

matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# Path to landmarks and face recognition model files
PREDICTOR_PATH = dataPath.MODELS_ROOT + '/shape_predictor_68_face_landmarks.dat'
FACE_RECOGNITION_MODEL_PATH = dataPath.MODELS_ROOT + '/dlib_face_recognition_resnet_model_v1.dat'

# Initialize face detector, facial landmarks detector 
# and face recognizer
faceDetector = dlib.get_frontal_face_detector()
shapePredictor = dlib.shape_predictor(PREDICTOR_PATH)
faceRecognizer = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)

# Root folder of the dataset
faceDatasetFolder = dataPath.IMAGES_ROOT + '/celeb_mini'
# Label -> Name Mapping file
labelMap = np.load(dataPath.MODELS_ROOT + "/celeb_mapping.npy", allow_pickle=True).item()

for key, value in labelMap.items():
    print('labelMap["{}"] = "{}";'.format(key,value))

print(labelMap)

# Each subfolder has images of a particular celeb
subfolders = os.listdir(faceDatasetFolder)

# Let us choose a random folder and display all images
random_folder = random.choice(subfolders)

# Also find out the name of the celeb from the folder name and folder-> name mapping dictionary loaded earlier
celebname = labelMap[random_folder]

# Load all images in the subfolder
imagefiles = os.listdir(os.path.join(faceDatasetFolder, random_folder))

# Read each image and display along with the filename and celeb name
for file in imagefiles:
#     Get full path of each image file
    fullPath = os.path.join(faceDatasetFolder,random_folder,file)
    im = cv2.imread(fullPath)
    plt.imshow(im[:,:,::-1])
    plt.show()
#     Also print the filename and celeb name
    print("File path = {}".format(fullPath))
    print("Celeb Name: {}".format(celebname))

#
# Enrollment
#

# nameLabelMap is dict with keys as person's name
# and values as integer label assigned to this person
# labels contain integer labels for corresponding image in imagePaths
nameLabelMap = {}
labels = []
imagePaths = []
for i, subfolder in enumerate(subfolders):
  currentSubFolder = os.path.join(faceDatasetFolder, subfolder)
  for x in os.listdir(currentSubFolder):
    xpath = os.path.join(currentSubFolder, x)
    if x.endswith('jpg') or x.endswith('JPEG'):
      imagePaths.append(xpath)
      labels.append(i)
      nameLabelMap[xpath] = labelMap[subfolder]

# Process images one by one
# We will store face descriptors in an ndarray (faceDescriptors)
# and their corresponding labels in dictionary (index)

ENROLL = False

if ENROLL:
    index = {}
    i = 0
    faceDescriptors = None

    for imagePath in imagePaths:
        print("processing: {}".format(imagePath))
        # read image and convert it to RGB
        img = cv2.imread(imagePath)

        # detect faces in image
        faces = faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        print("{} Face(s) found".format(len(faces)))
        # Now process each face we found
        for k, face in enumerate(faces):

            # Find facial landmarks for each detected face
            shape = shapePredictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)

            # convert landmarks from Dlib's format to list of (x, y) points
            landmarks = [(p.x, p.y) for p in shape.parts()]

            # Compute face descriptor using neural network defined in Dlib.
            # It is a 128D vector that describes the face in img identified by shape.
            faceDescriptor = faceRecognizer.compute_face_descriptor(img, shape)

            # Convert face descriptor from Dlib's format to list, then a NumPy array
            faceDescriptorList = [x for x in faceDescriptor]
            faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
            faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

            # Stack face descriptors (1x128) for each face in images, as rows
            if faceDescriptors is None:
                faceDescriptors = faceDescriptorNdarray
            else:
                faceDescriptors = np.concatenate((faceDescriptors, faceDescriptorNdarray), axis=0)

            # save the label for this face in index. We will use it later to identify
            # person name corresponding to face descriptors stored in NumPy Array
            index[i] = nameLabelMap[imagePath]
            i += 1

    # Write descriors and index to disk
    np.save(dataPath.RESULTS_ROOT + '/doppelganger_descriptors.npy', faceDescriptors)
    # index has image paths in same order as descriptors in faceDescriptors
    with open(dataPath.RESULTS_ROOT + '/doppelganger_index.pkl', 'wb') as f:
        cPickle.dump(index, f)

#
# TEST
#

THRESHOLD = 0.6

# load descriptors and index file generated during enrollment
index = np.load(dataPath.RESULTS_ROOT + '/doppelganger_index.pkl', allow_pickle=True)
faceDescriptorsEnrolled = np.load(dataPath.RESULTS_ROOT + '/doppelganger_descriptors.npy')

# read image
testImages = glob.glob(dataPath.IMAGES_ROOT + '/test-images/*.jpg')

for test in testImages:
    im = cv2.imread(test)
    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    #####################
    #  YOUR CODE HERE
    # detect faces in image
    faces = faceDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # Now process each face we found
    for face in faces:

        # Find facial landmarks for each detected face
        shape = shapePredictor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), face)

        # find coordinates of face rectangle
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Compute face descriptor using neural network defined in Dlib
        # using facial landmark shape
        faceDescriptor = faceRecognizer.compute_face_descriptor(im, shape)

        # Convert face descriptor from Dlib's format to list, then a NumPy array
        faceDescriptorList = [m for m in faceDescriptor]
        faceDescriptorNdarray = np.asarray(faceDescriptorList, dtype=np.float64)
        faceDescriptorNdarray = faceDescriptorNdarray[np.newaxis, :]

        # Calculate Euclidean distances between face descriptor calculated on face dectected
        # in current frame with all the face descriptors we calculated while enrolling faces
        distances = np.linalg.norm(faceDescriptorsEnrolled - faceDescriptorNdarray, axis=1)
        # Calculate minimum distance and index of this face
        argmin = np.argmin(distances)  # index
        minDistance = distances[argmin]  # minimum distance

        # Dlib specifies that in general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people.

        # This threshold will vary depending upon number of images enrolled
        # and various variations (illuminaton, camera quality) between
        # enrolled images and query image
        # We are using a threshold of 0.5

        # If minimum distance if less than threshold
        # find the name of person from index
        # else the person in query image is unknown
        if minDistance <= THRESHOLD:
            celeb_name = index[argmin]
        else:
            celeb_name = 'unknown'    
 
    ####################
    
    plt.subplot(121)
    plt.imshow(imDlib)
    plt.title("test img")
    
    #TODO - display celeb image which looks like the test image instead of the black image. 
    celebFolder = list(labelMap.keys())[list(labelMap.values()).index(celeb_name)]
    print(celebFolder)
    celebImagefiles = os.listdir(os.path.join(faceDatasetFolder, celebFolder))
    celebFullPath = os.path.join(faceDatasetFolder, celebFolder, celebImagefiles[0])
    im = cv2.imread(celebFullPath)
    plt.subplot(122)
    plt.imshow(im[:,:,::-1])
    plt.title("Celeb Look-Alike={}".format(celeb_name))
    plt.show()
