# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:12:29 2019

@author: yexiaohan
"""

import keras
from keras.models import Model,load_model
from keras.layers import Reshape, Activation, Convolution2D,Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda,Dropout
from keras.layers.advanced_activations import LeakyReLU
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_file):
    train_list = open(data_file,'r')
    labels = []
    images = []
    for line in train_list:
        tmp = line.strip().split(' ')
        image = cv2.imread(tmp[0])
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        label = int(tmp[1])
        label = keras.utils.to_categorical(label,17)
        images.append(image)
        labels.append(label)
    return np.array(images),np.array(labels)

def vggnet():
    model = keras.Sequential()
    
    # Block 1, 2层
    
    model.add(Convolution2D(64, 3, 3,
    
                        border_mode='same', input_shape=(224, 224,3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(64, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
     
    
    # Block 2, 2层
    
    model.add(Convolution2D(128, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
     
    
    # Block 3, 3层
    
    model.add(Convolution2D(256, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(256, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(256, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
     
    
    # Block 4, 3层
    
    model.add(Convolution2D(512, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(512, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(512, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
     
    
    # Block 5, 3层
    
    model.add(Convolution2D(512, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(512, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Convolution2D(512, 3, 3,border_mode='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.load_weights('./logs/vgg16.h5')
    
    for layer in model.layers[:-10]:
        layer.trainable = False
        
    
    
    
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17,activation='softmax'))

    model.summary()
    #model = load_model('./logs/model.h5')
    return model

if __name__ == '__main__':
    
    train_data,train_label=load_data('train_list.txt')
    X_train,X_test, y_train, y_test = train_test_split(train_data,train_label,test_size=0.3, random_state=42)

    
    model = vggnet()
    """
    model = load_model('./logs1/models.h5')
    
    for layer in model.layers[:-16]:
        print(layer)
        layer.trainable = False
        """
    adam = keras.optimizers.Adam(lr=1e-4)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

    for i in range(40):
        
        model.fit(X_train,y_train,nb_epoch=1,batch_size=32)
        if i%2==0:
            loss,accuracy = model.evaluate(X_test,y_test)
            print('\nloss: ',loss)
            print('\naccuracy: ',accuracy)
        mp = "./logs1/models.h5"
        model.save(mp)

    
    
    
    
    
    