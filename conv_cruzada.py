# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:46:32 2018

@author: Pedro Henrique
"""

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(x, y),(x_teste, y_teste) = mnist.load_data()

prev = x.reshape(x.shape[0],28,28,1)

prev = prev.astype('float32')
prev /= 255

classe = np_utils.to_categorical(y,10)

kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=seed)

resultados = []

a = np.zeros(5)

b = np.zeros(shape = (classe.shape[0],1))

for i_treino, i_teste in kfold.split(prev, b):
    #print("Indice treinamento: {}".format(i_treino))
    #print("Indice teste: {}".format(i_teste))
    classificador = Sequential()
    classificador.add(Conv2D(32,(3,3),
                             input_shape=(28,28,1),
                             activation='relu'))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    
    classificador.add(Conv2D(32,(3,3), activation='relu'))
    classificador.add(BatchNormalization())
    classificador.add(MaxPooling2D(pool_size=(2,2)))
    # --*INICIO DA REDE NEURAL DENSA*--
    classificador.add(Flatten())
    # --* CAMADAS OCULTAS *--
    classificador.add(Dense(units=128,activation='relu'))
    classificador.add(Dropout(0.2))
    
    classificador.add(Dense(units=128,activation='relu'))
    classificador.add(Dropout(0.2))
    # --* SAIDA *--
    classificador.add(Dense(units=10,activation='softmax'))
    
    classificador.compile(loss='categorical_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
    
    classificador.fit(prev[i_treino], classe[i_treino], batch_size=128,
                      epochs=5, validation_data=(prev[i_teste], classe[i_teste]))

    r = classificador.evaluate(prev[i_teste],classe[i_teste])
    resultados.append(r[1])
    








