#This file is generated from downloading the Jupyter Notebook.

# coding: utf-8

# In[11]:

import csv
import cv2
import numpy as np


# In[12]:

def open_csv(filename) :
    lines = []
    with open(filename) as csvfile :
        reader = csv.reader(csvfile)
        for line in reader :
            lines.append(line)
    return lines
    


# In[13]:

lines = open_csv("./data/new_data/driving_log.csv")


# In[6]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)


# In[12]:

from sklearn.utils import shuffle
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #print("Entered")
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            measurements = []
            corrections = [0,0.3,-0.3]
            for line in batch_samples :
                for i in range(0,3) :
                    source_path = line[i]
                    correction  = corrections[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/new_data/IMG/' + filename
                    #print(current_path + '\n')
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3]) + correction
                    measurements.append(measurement)
                    images.append(cv2.flip(image,1))
                    measurement = float(line[3]) + correction
                    measurements.append(measurement * -1.0)
                    
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            #print(X_train.shape)
            yield shuffle(X_train, y_train)


# In[13]:

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


# In[14]:

from keras.models import Sequential
from keras.layers import Dense, Flatten,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

# In[15]:

# Architecture used by NVIDIA in the following paper.
#https://arxiv.org/pdf/1604.07316.pdf
model = Sequential()
model.add(Lambda(lambda x : x / 255.0,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
model.add(Dropout(p=0.93))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples)* 4, validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=1)


# In[16]:

model.save('model_temp.h5')


# In[ ]:



