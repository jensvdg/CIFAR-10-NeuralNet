from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar10.load_data() 
num_train, height, width, depth = X_train.shape

X_train = X_train.astype('float32') 
X_test = X_test.astype('float32')

# Normalize the data to the 0-1 range
X_train /= np.max(X_train)
X_test /= np.max(X_test) 

# One-hot encode the labels
Y_train = np_utils.to_categorical(y_train, 10) 
Y_test = np_utils.to_categorical(y_test, 10) 

# Define the ConvNet Model
model = Sequential()
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(height, width, depth)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          verbose=1, validation_split=0.1)

score = model.evaluate(X_test, Y_test, verbose=0)
print score