#This file is generated from downloading the Jupyter Notebook.

import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def open_csv(filename) :
    lines = []
    with open(filename) as csvfile :
        reader = csv.reader(csvfile)
        for line in reader :
            lines.append(line)
    return lines


def get_data_stats(lines) :
    measurements = []
    for line in lines[1:] :
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = './data/data/IMG/' + filename
        measurement = float(line[3]) 
        measurements.append(measurement)
    df = pd.DataFrame(measurements)
    df.hist()
    plt.show()

        

#Read the lines from csv
lines = open_csv("./data/new_data/driving_log.csv")

#Uncomment to get the data stats
#get_data_stats(lines)

#Split into train and validation samples. 
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines[1:], test_size=0.2)

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

            #Corrections for center, right and left images
            corrections = [0,0.3,-0.3]

            for line in batch_samples :
                for i in range(0,3) :
                    source_path = line[i]
                    correction  = corrections[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/new_data/IMG/' + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    measurement = float(line[3]) + correction
                    measurements.append(measurement)
                    images.append(cv2.flip(image,1))
                    measurement = float(line[3]) + correction
                    measurements.append(measurement * -1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)




# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)




from keras.models import Sequential
from keras.layers import Dense, Flatten,Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout



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
model.add(Dropout(0.93))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=
            len(train_samples)* 4, validation_data=validation_generator,
                    nb_val_samples=len(validation_samples), nb_epoch=6)



#Saving the model
model.save('model.h5')






