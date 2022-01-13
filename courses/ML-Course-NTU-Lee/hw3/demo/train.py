# This code is using tensorflow backend
#!/usr/bin/env python
# -- coding: utf-8 --
from cnn_simple import *
from utils import *
import os
import numpy as np
import argparse
import time

os.system('echo $CUDA_VISIBLE_DEVICES')
PATIENCE = 5 # The parameter is used for early stopping

def main():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='model/model-1')
    args = parser.parse_args()

    '''
    To begin with, you should first read your csv training file and 
    cut them into training set and validation set.
    Such as:
        with open(csvFile, 'r') as f:
            f.readline()
            for i, line in enumerate(f):
                data = line.split(',')
                label = data[0]
                pixel = data[1]
                ...
                ...
    In addition, we maintain it in array structure and save it in pickle
    '''

    # training data
    train_pixels = load_pickle('../train_pixels.pkl')
    train_labels = load_pickle('../train_labels.pkl')
    print ('# of training instances: ' + str(len(train_labels)))

    # validation data
    valid_pixels = load_pickle('../valid_pixels.pkl')
    valid_labels = load_pickle('../valid_labels.pkl')
    print ('# of validation instances: ' + str(len(valid_labels)))

    '''
    Modify the answer format so as to correspond with the output of keras model
    We can also do this to training data here, 
        but we choose to do it in "train" function
    '''

    for i in range(len(valid_labels)):
        valid_pixels[i] = np.fromstring(valid_pixels[i], dtype=float, sep=' ').reshape((48, 48, 1))
        onehot = np.zeros((7, ), dtype=np.float)
        onehot[int(valid_labels[i])] = 1.
        valid_labels[i] = onehot

    # start training
    train(args.batch, args.epoch, args.pretrain, args.save_every,
          train_pixels, train_labels,
          np.asarray(valid_pixels), np.asarray(valid_labels),
          args.model_name)

def train(batch_size, num_epoch, pretrain, save_every, train_pixels, train_labels, val_pixels, val_labels, model_name=None):

    if pretrain == False:
        model = build_model()
    else:
        model = load_model(model_name)

    '''
    "1 Epoch" means you have been looked all of the training data once already.
    Batch size B means you look B instances at once when updating your parameter.
    Thus, given 320 instances, batch size 32, you need 10 iterations in 1 epoch.
    '''

    num_instances = len(train_labels)
    iter_per_epoch = int(num_instances / batch_size) + 1
    batch_cutoff = [0]
    for i in range(iter_per_epoch - 1):
        batch_cutoff.append(batch_size * (i+1))
    batch_cutoff.append(num_instances)

    total_start_t = time.time()
    best_metrics = 0.0
    early_stop_counter = 0
    for e in range(num_epoch):
        #shuffle data in every epoch
        rand_idxs = np.random.permutation(num_instances)
        print ('#######')
        print ('Epoch ' + str(e+1))
        print ('#######')
        start_t = time.time()

        for i in range(iter_per_epoch):
            if i % 50 == 0:
                print ('Iteration ' + str(i+1))
            X_batch = []
            Y_batch = []
            ''' fill data into each batch '''
            for n in range(batch_cutoff[i], batch_cutoff[i+1]):
                X_batch.append(train_pixels[rand_idxs[n]])
                Y_batch.append(np.zeros((7, ), dtype=np.float))
                X_batch[-1] = np.fromstring(X_batch[-1], dtype=float, sep=' ').reshape((48, 48, 1))
                Y_batch[-1][int(train_labels[rand_idxs[n]])] = 1.

            ''' use these batch data to train your model '''
            model.train_on_batch(np.asarray(X_batch),np.asarray(Y_batch))

        '''
        The above process is one epoch, and then we can check the performance now.
        '''
        loss_and_metrics = model.evaluate(val_pixels, val_labels, batch_size)
        print ('\nloss & metrics:')
        print (loss_and_metrics)

        '''
        early stop is a mechanism to prevent your model from overfitting
        '''
        if loss_and_metrics[1] >= best_metrics:
            best_metrics = loss_and_metrics[1]
            print ("save best score!! "+str(loss_and_metrics[1]))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        '''
        Sample code to write result :

        if e == e:
            val_proba = model.predict(val_pixels)
            val_classes = val_proba.argmax(axis=-1)


            with open('result/simple%s.csv' % str(e), 'w') as f:
                f.write('acc = %s\n' % str(lossandmetrics[1]))
                f.write('id,label')
                for i in range(len(valclasses)):
                    f.write('\n' + str(i) + ',' + str(valclasses[i]))
        '''

        print ('Elapsed time in epoch ' + str(e+1) + ': ' + str(time.time() - startt))

        if (e+1) % saveevery == 0:
            model.save('model/model-%d.h5' %(e+1))
            print ('Saved model %s!' %str(e+1))

        if earlystopcounter >= PATIENCE:
            print ('Stop by early stopping')
            print ('Best score: '+str(best_metrics))
            break

    print ('Elapsed time in total: ' + str(time.time() - total_start_t))

if __name=='__main':
    main()
