# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

(x_treino, y_treino), (x_teste, y_teste) = \
mnist.load_data()

plt.imshow(x_treino[5], cmap='gray')
plt.title('classe ' + str(y_treino[5]))


prev_treino = x_treino.reshape(x_treino.shape[0],
                               28,28, 1)

prev_teste = x_teste.reshape(x_teste.shape[0],
                               28,28, 1)

prev_treino = prev_treino.astype('float32')
prev_teste = prev_teste.astype('float32')

prev_treino /= 255
prev_teste /= 255

prev_treino[0]

classe_treino = np_utils.to_categorical(y_treino, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)


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

classificador.fit(prev_treino, classe_treino, batch_size=100,
                  epochs=10, validation_data=(prev_teste, classe_teste))

result = classificador.evaluate(prev_teste,classe_teste)



















