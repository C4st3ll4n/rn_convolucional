# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

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

#classificador.add(Dense(units=128,activation='relu'))
#classificador.add(Dropout(0.2))

classificador.add(Dense(units=128,activation='relu'))
classificador.add(Dropout(0.2))
# --* SAIDA *--
classificador.add(Dense(units=10,activation='softmax'))

classificador.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

g_treino = ImageDataGenerator(rotation_range=7,
                              horizontal_flip=True,
                              shear_range=0.2,
                              height_shift_range=0.07,
                              zoom_range=0.2)

g_teste = ImageDataGenerator()

base_treino = g_treino.flow(prev_treino, classe_treino, batch_size=128 )
base_teste = g_teste.flow(prev_teste, classe_teste, batch_size=128 )

classificador.fit_generator(base_treino, steps_per_epoch=468,
                  epochs=5, validation_data=base_teste, validation_steps=78)

result = classificador.evaluate_generator(base_teste)



















