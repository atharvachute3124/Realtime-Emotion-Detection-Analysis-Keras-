# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os, sys
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Average
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils

# Importing the dataset
df = pd.read_csv('fer2013.csv')

X_train, y_train, X_test, y_test = [], [], [], []

for index, row in df.iterrows():
    val=row['pixels'].split(" ")
    if 'Training' in row['Usage']:
        X_train.append(np.array(val,'float32'))
        y_train.append(row['emotion'])
    elif 'PublicTest' in row['Usage']:
        X_test.append(np.array(val,'float32'))
        y_test.append(row['emotion'])
    else:
        print(f"error occured at index :{index} and row:{row}")
        
num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
width, height = 48, 48

# Converting to array
X_train = np.array(X_train,'float32')
y_train = np.array(y_train,'float32')
X_test = np.array(X_test,'float32')
y_test = np.array(y_test,'float32')

y_train = np_utils.to_categorical(y_train, num_classes = num_labels)
y_test = np_utils.to_categorical(y_test, num_classes = num_labels)

X_train -= np.mean(X_train, axis=0)
X_train /= np.std(X_train, axis=0)
X_test -= np.mean(X_test, axis=0)
X_test /= np.std(X_test, axis=0)

X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

# Designing a CNN
model = Sequential()

# 1st hidden layer
model.add(Conv2D(num_features, kernel_size = (3,3), activation = 'relu', input_shape = (X_train.shape[1:])))
model.add(Conv2D(num_features, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.5))

# 2nd hidden layer
model.add(Conv2D(num_features, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(num_features, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.5))

# 3rd hidden layer
model.add(Conv2D(2*num_features, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(Conv2D(2*num_features, kernel_size = (3,3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

# Fully connected NN
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_labels, activation='softmax'))

model.compile(optimizer = Adam(), loss = categorical_crossentropy, metrics = ['accuracy'])

model.fit(X_train, y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1, 
          validation_data = (X_test, y_test),
          shuffle = True)

#Saving the  model to  use it later on
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("fer.h5")
