import time
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

def data_preprocessing():
    x_train = []
    y_train = []
    x_test = []

    with open('train.csv') as file:
        rows = csv.reader(file, delimiter=",")
        n_row = 0
        for row in rows:
            if n_row != 0:
                y_train.append(int(row[0]))
                row_x = row[1].split(' ')
                x_train.append(row_x)
            n_row += 1
    
    with open('test.csv') as file:
        rows = csv.reader(file, delimiter=",")
        n_row = 0
        for row in rows:
            if n_row != 0:
                row_x = row[1].split(' ')
                x_test.append(row_x)
            n_row += 1
    
    for i in range(len(x_train)):
        for j in range(len(x_train[i])):
            x_train[i][j] = int(x_train[i][j])
    
    for i in range(len(x_test)):
        for j in range(len(x_train[i])):
            x_test[i][j] = int(x_test[i][j])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_test = np.array(x_test)

    np.savez('hw3_data.npz', x_train = x_train, y_train = y_train, x_test=x_test)

def load_data():
    """ Loads the hw3 dataset """

    file = np.load('hw3_data.npz')
    x_train, y_train = file['x_train'], file['y_train']
    x_test = file['x_test']
    file.close()

    # add flip data
    new_x, new_y = data_augmentation(x_train, y_train)
    x_train = np.vstack((x_train, new_x))
    y_train = np.hstack((y_train, new_y))
    
    
    x_train = x_train.reshape(x_train.shape[0], 48*48)
    x_test = x_test.reshape(x_test.shape[0], 48*48)
    # convert class vector to binary class matries
    y_train = np_utils.to_categorical(y_train, 7)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = x_train / 255
    x_test = x_test / 255
    

    return (x_train, y_train), x_test

def data_augmentation(x, y):
    new_x, new_y = [], []  
    x = x.reshape(x.shape[0], 48, 48)

    for i in range(len(x)):
        new_x.append(cv2.flip(x[i], 1))
        new_y.append(y[i])

    new_x, new_y = np.array(new_x), np.array(new_y)
    new_x = new_x.reshape(new_x.shape[0], 48*48)

    return new_x, new_y


def gen_predict_data(data):
    output = []
    for i in range(len(data)):
        output.append([str(i)])
        y_predict = np.where(data[i] == np.max(data[i]))
        y_predict = int(y_predict[0])
        output[i].append(y_predict)
    
    with open('predict.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(["id","label"])
        for i in range(len(output)):
            writer.writerow(output[i])


if __name__ == '__main__':
    start = time.time()
    #data_preprocessing()
    (x_train, y_train), x_test = load_data()
    
    x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
    x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

    x_train_1 = x_train[0:20000]
    x_train_2 = x_train[28709:48709]
    x_validation_1 = x_train[20000:28709]
    x_validation_2 = x_train[48709:]
    y_train_1 = y_train[0:20000]
    y_train_2 = y_train[28709:48709]
    y_validation_1 = y_train[20000:28709]
    y_validation_2 = y_train[48709:]

    x_train = np.vstack((x_train_1, x_train_2))
    x_validation = np.vstack((x_validation_1, x_validation_2))
    y_train = np.vstack((y_train_1, y_train_2))
    y_validation = np.vstack((y_validation_1, y_validation_2))

    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))    
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=7, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='Adam',
                  metrics=['accuracy'])

    save_best = callbacks.ModelCheckpoint('hw3_best_model.h5', monitor='val_acc', 
                                        verbose=1, save_best_only=True, mode='max')

    early_stop = callbacks.EarlyStopping(monitor='val_acc', patience=5, verbose=0, 
                                         mode='auto')

    callbacks_list = [save_best, early_stop]

    model.fit(x_train, y_train, batch_size=32, epochs=20, 
              validation_data=(x_validation, y_validation), callbacks=callbacks_list)


    train_result = model.evaluate(x_train, y_train)
    print('\nTrain Acc: ', train_result[1])

    validation_result = model.evaluate(x_validation, y_validation)
    print('\nValidation Acc: ', validation_result[1])

    model.save('hw3_model.h5')
    '''
    
    model = load_model('hw3_best_model.h5')
    
    #train_result = model.evaluate(x_train, y_train)
    #print('\nTrain Acc: ', train_result[1])

    validation_result = model.evaluate(x_validation, y_validation)
    print('\nValidation Acc: ', validation_result[1])
 
    test_predict = model.predict(x_test)
    gen_predict_data(test_predict)   
    

    end = time.time()
    print('Time: ', end - start)