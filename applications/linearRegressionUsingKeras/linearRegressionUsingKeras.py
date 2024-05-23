from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

print("Training set size: ", X_train.shape)
print("Test set size: ", X_test.shape)
print("Training example features: ", X_train[0,:])
print("Training example output: ", Y_train[0])

nFeatures = X_train.shape[1]

model = Sequential()
model.add(Dense(1, input_shape=(nFeatures,), activation='linear'))
 
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'mae'])

import tensorflow as tf
print(tf.__version__)

model.summary()

# To see detail output, change verbose to True
model.fit(X_train, Y_train, batch_size=4, epochs=1000, verbose=True)

# To see detail output, change verbose to True
# returns loss, metrics as speficfied in compilation step so it returns mse, mse and mae.
model.evaluate(X_test, Y_test, verbose=False)

Y_pred = model.predict(X_test)
 
print(Y_test[:5])
print(Y_pred[:5,0])