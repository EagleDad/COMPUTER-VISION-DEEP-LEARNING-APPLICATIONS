#
# 1. Import necessary modules
#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# set random seed
np.random.seed(2020)

#
# 2. Load Dataset
#
from tensorflow.keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Training data shape
train_images_shape = train_images.shape
# Training labels shape
train_labels_shape = train_labels.shape

print('Training data shape: ', train_images_shape, train_labels_shape)

# Testing data shape
test_images_shape = test_images.shape
# Testing labels shape
test_labels_shape = test_labels.shape

print('Testing data shape: ', test_images_shape, test_labels_shape)

#
# 3. Preprocessing
#
from tensorflow.keras.utils import to_categorical
# Change to float datatype
train_data = train_images.astype('float32')
test_data = test_images.astype('float32')

# Scale the data to lie between 0 to 1
train_data /= 255
test_data /= 255

# Change the labels from integer to categorical data
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#
# 4. TODO
# Currently, the model and optimizer is configured such that it gives very low accuracy ~10%. Y
# Your task is to explore options by modifying model and optimizer to get to more than 65% Training accuracy.
# Here are a few hints of what changes can help increase the accuracy in just 5 epochs:
# 1. Changing the Model parameters like activation type, droupout ratio etc.
# 2. Changing the optimizer
# 3. Changing optimizer parameters
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
    lr = 1e-3
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

from keras.callbacks import  LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [lr_reducer, lr_scheduler]

#
# 4.1. Model Architecture
#
def createModel():
    model = Sequential()

    # feature extraction part
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # classification part
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))
    
    return model

#
# 4.2. Configure Optimizer
#
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, RMSprop

#optim = SGD(lr=0.0001)
optim = RMSprop(learning_rate=lr_schedule(0))

#
# 5. Train Model
#
model = createModel()
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, epochs=5, batch_size=64,
                    validation_data=(test_data, test_labels_one_hot), verbose=1,
                    shuffle=False, callbacks=callbacks)

#
# 6. Plot Training and Validation Loss and Classification Accuracy
#
plt.figure(figsize=[20,6])
# plot loss
plt.subplot(121)
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='red', label='train')
plt.legend(['Training Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curve',fontsize=16)
# plot accuracy
plt.subplot(122)
plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='red', label='train')
plt.legend(['Training Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curve',fontsize=16)
plt.show()