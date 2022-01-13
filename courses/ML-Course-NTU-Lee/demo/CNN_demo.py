import time
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import fashion_mnist

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28*28)
    x_test = x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vector to binary class matries
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_train = x_train / 255
    x_test = x_test / 255
    #x_test = np.random.normal(x_test)
    return (x_train, y_train), (x_test, y_test)

if __name__ == '__main__':
    start = time.time()
    (x_train, y_train), (x_test, y_test) = load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) # 此為Tensorflow寫法
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    model2 = Sequential()
    model2.add(Conv2D(25, (3, 3), input_shape=(28, 28, 1)))
    model2.add(MaxPool2D(2, 2))
    model2.add(Conv2D(50, (3, 3)))
    model2.add(MaxPool2D(2, 2))
    model2.add(Flatten())
    model2.add(Dense(units=100, activation='relu'))
    model2.add(Dense(units=10, activation='softmax'))

    model2.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    model2.fit(x_train, y_train, batch_size=100, epochs=20)
    print("Time: ", time.time() - start)
    result = model2.evaluate(x_train, y_train)
    print('\nTrain Acc: ', result[1])

    result = model2.evaluate(x_test, y_test)
    print('\nTest Acc: ', result[1])

    model2.save('cnn_demo_model.h5')
