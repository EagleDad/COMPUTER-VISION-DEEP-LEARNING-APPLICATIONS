# Import relevant libraries
import dlib
import cv2
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

# Landmark model location
PREDICTOR_PATH = dataPath.MODELS_ROOT + "/shape_predictor_68_face_landmarks.dat"

# Set parameters for resizing and skipping frames
RESIZE_HEIGHT = 480
SKIP_FRAMES = 2

# Initialize the video capture device

# Create an imshow window
winName = "Fast Facial Landmark Detector"

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if OpenCV is able to read feed from camera
if (cap.isOpened() is False):
    print("Unable to connect to camera")
    sys.exit()

# Just a place holder. Actual value calculated after 100 frames.
fps = 30.0

# Get first frame
ret, im = cap.read()

# Resize the input frame
# We will use a fixed height image as input to face detector
if ret == True:
    height = im.shape[0]
    # calculate resize scale
    RESIZE_SCALE = float(height)/RESIZE_HEIGHT
    size = im.shape[0:2]
else:
    print("Unable to read frame")
    sys.exit()

# Set up face detector and landmark detector 
# Load face detection and pose estimation models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# initiate the tickCounter
t = cv2.getTickCount()
count = 0

# Loop over the video and display the result 
# Grab and process frames until the main window is closed by the user.
while(True):
    if count==0:
      t = cv2.getTickCount()
    # Grab a frame
    ret, im = cap.read()
    imDlib = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # create imSmall by resizing image by resize scale
    imSmall= cv2.resize(im, None, fx = 1.0/RESIZE_SCALE, fy = 1.0/RESIZE_SCALE, interpolation = cv2.INTER_LINEAR)
    imSmallDlib = cv2.cvtColor(imSmall, cv2.COLOR_BGR2RGB)
    
    # Process frames at an interval of SKIP_FRAMES.
    # This value should be set depending on your system hardware
    # and camera fps.
    # To reduce computations, this value should be increased
    if (count % SKIP_FRAMES == 0):
      # Detect faces
      faces = detector(imSmallDlib,0)

    # Iterate over faces
    for face in faces:
      # Since we ran face detection on a resized image,
      # we will scale up coordinates of face rectangle
      newRect = dlib.rectangle(int(face.left() * RESIZE_SCALE),
                               int(face.top() * RESIZE_SCALE),
                               int(face.right() * RESIZE_SCALE),
                               int(face.bottom() * RESIZE_SCALE))

      # Find face landmarks by providing reactangle for each face
      shape = predictor(imDlib, newRect)
      # Draw facial landmarks
      renderFace.renderFace(im, shape)

    # Put fps at which we are processinf camera feed on frame
    cv2.putText(im, "{0:.2f}-fps".format(fps), (50, size[0]-50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)
    # Display it all on the screen
    cv2.imshow(winName, im)
    # Wait for keypress
    key = cv2.waitKey(1) & 0xFF

    # Stop the program.
    if key==27:  # ESC
      # If ESC is pressed, exit.
      sys.exit()

    # increment frame counter
    count = count + 1
    # calculate fps at an interval of 100 frames
    if (count == 100):
      t = (cv2.getTickCount() - t)/cv2.getTickFrequency()
      fps = 100.0/t
      count = 0

cv2.destroyAllWindows()
cap.release()