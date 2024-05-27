# Import relevant libraries
import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
tf.keras.backend.clear_session()
from tensorflow.keras.layers import Input, Softmax, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


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

# Settting up batch size, random seed, and the dataset path

BATCH_SIZE = 64#8
SEED = 21
dataset_path = dataPath.IMAGES_ROOT +  '/RadiologyAI/chest_xray'

# Initialising ImageDataGenerator for data augmentation
# We use random horizontal flip for augmentation
# Pixels will be notmalised between 0 and 1
  # zca_epsilon: Epsilon for ZCA whitening. Default is 1e-6
  # Horizontal_flip: Boolean. Randomly flip inputs horizontally.
  # Rescale: Rescaling factor, defaults to None.
           # If None or 0, no rescaling is applied, otherwise it multiplied the data by the value provided
           # (after applying all other transformations)

train_val_gen = ImageDataGenerator(zca_epsilon = 0.0,
                                    horizontal_flip = True,
                                    rescale = 1./255,        # Do not change rescale
                                    #shear_range=0.2, 
                                    #zoom_range=0.2,
                                    #rotation_range=25,
                                    #width_shift_range=0.2, 
                                    #height_shift_range=0.2,
                                    #brightness_range=[0.7,1.3],
                                       )        

test_gen = ImageDataGenerator(zca_epsilon = 0.0,
                              horizontal_flip = False,
                              rescale = 1./255)             # Do not change rescale

# The evaluation on streamlit share assumes rescaling takes place,
# and it is 1./255 always

# Taking input of the train, validation, and test images using flow_from_directory() function
# Setting the image size to (224, 224) and setting the batch size

train_datagen = train_val_gen.flow_from_directory(directory = dataset_path + '/train',
                                                  target_size = (224, 224),
                                                  color_mode = "rgb",
                                                  classes = None,
                                                  class_mode = "categorical",
                                                  batch_size = BATCH_SIZE,
                                                  shuffle = True,
                                                  seed = SEED,
                                                  interpolation = "nearest")

val_datagen = train_val_gen.flow_from_directory(directory = dataset_path + '/val',
                                                target_size = (224, 224),
                                                color_mode = "rgb",
                                                classes = None,
                                                class_mode = "categorical",
                                                batch_size = BATCH_SIZE,
                                                shuffle = True,
                                                seed = SEED,
                                                interpolation = "nearest")


# For testing, we should take one input at a time. Hence, batch_size = 1

test_datagen = test_gen.flow_from_directory(directory = dataset_path + '/test',
                                            target_size = (224, 224),
                                            color_mode = "rgb",
                                            classes = None,
                                            class_mode = "categorical",
                                            batch_size = 1,
                                            shuffle = False,
                                            seed = SEED,
                                            interpolation = "nearest")

# Initialising MobileNet model and passing the imagenet weights
# We are specifying classes = 1000 because the model was trained on 1000 classes
# The classes will be changed afterwards according to our problem

#pretrained_model = tf.keras.applications.MobileNetV2(weights = 'imagenet',
#                                                   classes = 1000,
#                                                   input_shape = (224, 224, 3),
#                                                   include_top = False,
#                                                   pooling = 'max')
#tf.keras.applications.xception.Xception
# Load Xception model 
#pretrained_model = tf.keras.applications.Xception(weights = 'imagenet',
#                                                   classes = 1000,
#                                                   input_shape = (224, 224, 3),
#                                                   include_top = False,
#                                                   pooling = 'max') 

pretrained_model = tf.keras.applications.VGG16(weights = 'imagenet',
                                                   classes = 1000,
                                                   input_shape = (224, 224, 3),
                                                   include_top = False,
                                                   pooling = 'max') 

# We do not not have much images
# For transfered learning basd on 'https://cs231n.github.io/transfer-learning/' it is recommened to only train the classifier
# set base model trainable to false 
#for layers in pretrained_model.layers: 
#    layers.trainable=False
# Freeze all layers except the last 4
for layer in pretrained_model.layers[:-4]:
    layer.trainable = False

# Printing the model summary

print(pretrained_model.summary())

# Adding a prediction layer. It takes input from the last layer (global_max_pooling2d) of MobileNet
# It has 2 dense units, as it is a binary classification problem
#predictions = Dense(2, activation = 'softmax')(pretrained_model.output)

# Defining new model's input and output layers
# Input layer of the new model will be the same as MobileNet
# But the output of the new model will be the output of final dense layer, i.e., 2 units
#model = Model(inputs = pretrained_model.input, outputs = predictions)


# We use the SGD optimiser, with a very low learning rate, and loss function which is specific to two class classification
#model.compile(optimizer = tf.keras.optimizers.SGD(0.000001),
#              loss = "binary_crossentropy",
#              metrics = ["accuracy"])


#
#
#

# Classes of images in train dataset
classes = ['covid', 'normal', 'pneumonia']

# all class names
trainFolders = os.listdir(dataset_path)
print(trainFolders)

# See how many images you have per class, this helps you set the required percentage of validation data.
for currentFolder in trainFolders:
  classNames = os.listdir(os.path.join(dataset_path, currentFolder))
  print("Folder '{}' has following classes: {}".format( currentFolder, classNames))
  classRoot = os.path.join(dataset_path, currentFolder)

  for currentClass in classNames:
    print("Class: {}, has {} samples".format( currentClass, len(os.listdir(os.path.join(classRoot, currentClass )))))

# Preparing the Samples and Plot for displaying output

# Create figure
fig = plt.figure(figsize = (12, 12))
for i in range(9):
  plt.subplot(330 + 1 + i)
  img, label = next(train_datagen)

  label = label[0].astype('uint8')
  label = np.squeeze(label)
  label = np.argmax(label, axis = 0)

  plt.axis('off')
  plt.imshow((img[0]*255).astype(np.uint8))
  plt.title(classes[label])

plt.show()

#
# MobileNetV2 mit RMSprop
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8614951/#:~:text=Simple%20Summary,images%20which%20were%20then%20augmented.
#


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 80:
        lr *= 0.5e-3
    elif epoch > 60:
        lr *= 1e-3
    elif epoch > 40:
        lr *= 1e-2
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# Create the model https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9084577/
model = tf.keras.models.Sequential()
# Add the pre trained convolutional base model
model.add(pretrained_model)
# Add new layers
#model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(512, activation='relu'))
#model.add(Dense(64, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(16, activation='relu'))
model.add(Dense(len(classes), activation='softmax'))

# Adding Dense, BatchNorm and Droupout layers to base model
# to have output for 3 Class Classification
#x = Dense(1024, activation = 'relu')((pretrained_model.output))
#x = Dropout(0.5)(x)
#x = Dense(512, activation = 'relu')(x)
#x = Dense(64, activation = 'relu')(x)
#x = BatchNormalization()(x)
#x = Dense(16, activation='relu')(x)
#predictions = Dense(3, activation = 'softmax')(pretrained_model.output)

# Define the input and output layers of the model
#model = Model(inputs = pretrained_model.input, outputs = predictions)

#
# Configure the optimizer
# 
#optim = tf.keras.optimizers.SGD(0.000001)
optim = tf.keras.optimizers.Adam(0.0001)
#optim = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule(0))
#optim = tf.keras.optimizers.RMSprop(0.0001)

# Compile model and define Optimizer
model.compile(optimizer = optim,
              loss = "categorical_crossentropy",
              metrics = ["accuracy"])

# Printing Final Model Summary
model.summary()

# Assigning Checkpoint Path for Saved Model
filepath = dataPath.RESULTS_ROOT + '/chest-x-ray-best_model.h5'

# Defining ModelCheckpoint Callback
model_save = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                monitor = "val_accuracy",
                                                verbose = 0,
                                                save_best_only = True,
                                                save_weights_only = False,
                                                mode = "max",
                                                save_freq = "epoch")

# Defining Reduce lr callback
#ReduceLR reduces the learning rate by a factor (0.1), if validation loss remains same for 6 consecutive epochs.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                                                 factor = 0.1,
                                                 patience = 6,
                                                 verbose = 1,
                                                 min_delta = 5*1e-3,
                                                 min_lr = 5*1e-9,)



lr_scheduler = reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

callbacks = [reduce_lr, lr_scheduler, model_save]


# Training the model for 5 epochs
# Shuffle is set to false because the data is already shuffled in flow_from_directory() method
history = None
TRAIN = False

EPOCHS = 15

if TRAIN:
    history = model.fit(train_datagen,
                    epochs = EPOCHS,
                    steps_per_epoch = (len(train_datagen)),
                    validation_data = val_datagen,
                    validation_steps = (len(val_datagen)),
                    shuffle = False,
                    callbacks = callbacks)
    
    # Plotting the loss and accuracy graphs
    plt.figure(figsize = (15,7))

    tr_losses = history.history['loss']
    val_losses = history.history['val_loss']

    tr_accs = history.history['accuracy']
    val_accs = history.history['val_accuracy']

    plt.plot(tr_losses, label = "train_loss")
    plt.plot(val_losses, label = "val_loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Cost (J)")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(figsize = (15,7))

    plt.plot(tr_accs, label = "acc_train")
    plt.plot(val_accs, label = "acc_val")
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    plt.show()
    # Save the Model
    model.save(dataPath.RESULTS_ROOT + '/best_model.h5')
else:
    model.load_weights(filepath)
    

# Model prediction on test set

predictions = model.predict(test_datagen,
                            verbose = 1,
                            steps = (len(test_datagen)))

# Printing predicted classes on the test dataset

predictions.squeeze().argmax(axis = -1)

classification__report = classification_report(test_datagen.classes,
                                               predictions.squeeze().argmax(axis = 1))
print(classification__report)

# Generating confusion matrix to see where the model is misclassifying

confusion__matrix = confusion_matrix(test_datagen.classes,
                                     predictions.squeeze().argmax(axis = 1))
print(confusion__matrix)

# Defining a function to print a confusion matrix
# Code snippet referenced from: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html

import itertools
def plot_confusion_matrix(cm,
                          classes,
                          normalise = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Reds):

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    if normalise:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        cm = cm.round(2)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # Printing the confusion matrix (not normalised)

np.set_printoptions(precision = 2)
fig1 = plt.figure(figsize = (7, 6))
plot_confusion_matrix(confusion__matrix,
                      classes = np.unique(test_datagen.classes),
                      title = 'Confusion matrix without normalisation')
fig1.savefig(dataPath.RESULTS_ROOT + '/cm_wo_norm.jpg')
plt.show()

# Printing the confusion matrix (normalised)

np.set_printoptions(precision = 2)
fig2 = plt.figure(figsize = (7,6))
plot_confusion_matrix(confusion__matrix,
                      classes = np.unique(test_datagen.classes),
                      normalise = True,
                      title = 'Normalised Confusion matrix')
fig2.savefig(dataPath.RESULTS_ROOT + '/cm_norm.jpg')
plt.show()


def test(path):

  # Load the image using keras
  img = tf.keras.preprocessing.image.load_img(path,
                                              grayscale = False,
                                              color_mode = 'rgb',
                                              target_size = (224, 224, 3),
                                              interpolation = 'nearest')

  # Display the image
  plt.imshow(img)
  plt.axis('off')
  plt.show()

  # Convert image to array for feeding it to the model
  img_array = np.asarray(img)

  # Expand dimension of img array
  img_array = np.expand_dims(img_array, 0)

  # Take prediction
  predictions = model.predict(img_array)

  # Evaluate Score
  score = predictions[0]

  return print('This image is a {} with a {:.2f} % confidence.'.format(classes[np.argmax(score)], 100 * np.max(score)))


test ( dataset_path + '/test/covid/001.jpeg' )
test ( dataset_path + '/test/normal/00000002_000.png' )
test ( dataset_path + '/test/pneumonia/00027508_001.png' )

#
# Show errors
#
# Get the test_datagen filenames from the generator
fnames = test_datagen.filenames

# Get the ground truth from generator
ground_truth = test_datagen.classes

# Get the label to class mapping from the generator
label2index = test_datagen.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict(test_datagen, steps=test_datagen.samples/test_datagen.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
errorPercent = len(errors) / len(predicted_classes) * 100.0
print("No of errors = {}/{}. Error Rate: {}%".format(len(errors),test_datagen.samples, errorPercent))

test_dir = dataset_path + '/test'

# Show the errors
#for i in range(len(errors)):
#    pred_class = np.argmax(predictions[errors[i]])
#    pred_label = idx2label[pred_class]

#    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
#        fnames[errors[i]].split('/')[0],
#        pred_label,
#        predictions[errors[i]][pred_class])

#    original = load_img('{}/{}'.format(test_dir,fnames[errors[i]]))
#    plt.figure(figsize=[7,7])
#    plt.axis('off')
#    plt.title(title)
#    plt.imshow(original)
#    plt.show()