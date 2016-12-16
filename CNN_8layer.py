
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pylab
from PIL import Image
import os, sys
from sklearn.decomposition import PCA
import pandas as pd
import theano
from keras.callbacks import Callback
import keras.backend.tensorflow_backend as tfbe
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten
from keras.utils.visualize_util import plot

path = "/Users/aishwaryapatil/Downloads/keras_final"


Files = []
Xlist = []
labels =[]
names = []

imlist = os.listdir(path)  

imlist = imlist[1:]

im = np.array(Image.open(path+'/'+imlist[0])) #open one image to get the size
m,n = im.shape[0:2]


df_label = pd.DataFrame.from_csv('trainLabels.csv')

def get_labels(i):
    l = i.split(os.path.sep)[-1].split(".")[0]
    st1 =  df_label.loc[[l]]
    to_int = int(st1.values)
    return to_int



#getting the labels of the images in the data set and storing it in array
def imagevector_label():
    for file in os.listdir(path):
        Files.append(file)
    File = Files[1:]
    immatrix = np.array([np.array(Image.open(path+'/'+file)) for file in File],'f')    
    for file in File:   
        names.append(file)
        labels.append(get_labels(file))
    
    return immatrix, np.array(labels)        


img_vect, lab = imagevector_label()


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
 
# load data
(X_train, X_test, y_train, y_test) = train_test_split(
	img_vect, labels, test_size=0.25, random_state=42)
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 128, 128).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 128, 128).astype('float32')
 
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def cnnmodel():
	# creating model
     model = Sequential()
     model.add(Convolution2D(16, 5, 5, border_mode='valid', input_shape=(1, 128, 128), dim_ordering="th", activation='relu'))
     model.add(Convolution2D(16, 3, 3, activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),dim_ordering="th"))
     model.add(Convolution2D(32, 3, 3, activation='relu'))
     model.add(Convolution2D(32, 3, 3, activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),dim_ordering="th"))
     model.add(Convolution2D(64, 3, 3, activation='relu'))
     model.add(Convolution2D(64, 3, 3, activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),dim_ordering="th"))
     model.add(Convolution2D(96, 3, 3, activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),dim_ordering="th"))
     model.add(Convolution2D(96, 3, 3, activation='relu'))
     model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2),dim_ordering="th"))
     model.add(Dropout(0.5))
     model.add(Flatten())
     model.add(Dense(96, activation='relu'))     #Fully connected layer
     model.add(Dense(5, activation='relu'))
     model.add(Dense(num_classes, activation='softmax'))
     # Compiling model
     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
     return model
 
model = cnnmodel()
#Fitting the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=500, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)

print("CNN Error: %.2f%%" % (100-scores[1]*100)) #Error for CNN



