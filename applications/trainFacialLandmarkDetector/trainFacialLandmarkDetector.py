# Import relevant libraries
import cv2
import sys
import os
import dlib

import numpy as np
import matplotlib.pyplot as plt
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

# Import the common modukes
import renderFace
import dataPath

fldDatadir = dataPath.IMAGES_ROOT +  "/facial_landmark_data/33_points/"
numPoints = 33
modelName = 'shape_predictor_' + str(numPoints) + '_face_landmarks.dat'

# Set parameters of shape_predictor_trainer
options = dlib.shape_predictor_training_options()
options.cascade_depth = 10
options.num_trees_per_cascade_level = 500
options.tree_depth = 4
options.nu = 0.1
options.oversampling_amount = 20
options.feature_pool_size = 400
options.feature_pool_region_padding = 0
options.lambda_param = 0.1
options.num_test_splits = 20

# Tell the trainer to print status messages to the console so we can
# see training options and how long the training will take.
options.be_verbose = True

# Check if train and test XML files are present in facial_landmark_data folder
trainingXmlPath = os.path.join(fldDatadir, 
                                "training_with_face_landmarks.xml")
testingXmlPath = os.path.join(fldDatadir, 
                                "testing_with_face_landmarks.xml")
outputModelPath = os.path.join(fldDatadir, modelName)

# check whether path to XML files is correct
if os.path.exists(trainingXmlPath) and os.path.exists(testingXmlPath):
  # Train and test the model
  # dlib.train_shape_predictor() does the actual training. 
  # It will save the final predictor to predictor.dat.
  # The input is an XML file that lists the images in 
  # the training dataset and 
  # also contains the positions of the face parts.
  dlib.train_shape_predictor(trainingXmlPath, outputModelPath, options)

  # Now that we have a model we can test it.  
  # dlib.test_shape_predictor() measures the average distance 
  # between a face landmark output by the shape_predictor and 
  # ground truth data.

  print("\nTraining error: {}".format(
    dlib.test_shape_predictor(trainingXmlPath, outputModelPath)))

  # The real test is to see how well it does 
  # on data it wasn't trained on.
  print("Testing error: {}".format(
    dlib.test_shape_predictor(testingXmlPath, outputModelPath)))
# Print an error message if XML files are not 
# present in facial_landmark_data folder
else:
  print('training and test XML files not found.')
  print('Please check paths:')
  print('train: {}'.format(trainingXmlPath))
  print('test: {}'.format(testingXmlPath))  

