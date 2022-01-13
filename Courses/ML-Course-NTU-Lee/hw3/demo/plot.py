#!/usr/bin/env python
# -- coding: utf-8 --

import argparse
from keras.utils.vis_utils import plot_model
from keras.models import load_model

def main():
    parser = argparse.ArgumentParser(prog='plot_model.py',
            description='Plot the model.')
    parser.add_argument('--model',type=str,default='model/model-7.h5')
    args = parser.parse_args()

    emotion_classifier = load_model(args.model)
    emotion_classifier.summary()
    plot_model(emotion_classifier,to_file='model.png')

if __name__ == '__main__':
    main()
