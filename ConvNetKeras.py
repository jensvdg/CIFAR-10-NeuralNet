from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
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
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
]

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, depth)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=64, epochs=120,
          verbose=1, validation_split=0.1,
          callbacks=callbacks)

score = model.evaluate(X_test, Y_test, verbose=0)
print score
model.save()