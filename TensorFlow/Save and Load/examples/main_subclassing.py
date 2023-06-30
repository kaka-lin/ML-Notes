import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf

from data.dataset import load_data
from model.model import CNN


CURR_PATH = os.getcwd()
MODEL_DIR = os.path.join(CURR_PATH, 'experiments/')
CHECKPOINT_PATH = MODEL_DIR + "checkpoints/"

# for this model
MODEL_NAME = 'subc_model'
MODEL_CHECKPOINTS_DIR = CHECKPOINT_PATH + MODEL_NAME + '/'


if __name__ == "__main__":
    # create a directory for the model if it doesn't already exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Loading dataset
    train_data, test_data = load_data()
    (x_train, y_train), (x_test, y_test) = train_data, test_data

    print("Training images: ", x_train.shape)
    print("Training lables: ", y_train.shape)
    print("Testing images: ", x_test.shape)
    print("Testing labels: ", y_test.shape)

    x_valid = x_train[54000:]
    y_valid = y_train[54000:]

    x_train = x_train[:54000]
    y_train = y_train[:54000]

    model = CNN()
    model.build((None, 28, 28, 1))
    # model.summary()

    model.compile(optimizer='Adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

    # Training model with callbacks.
    model.fit(x_train,
              y_train,
              epochs=1,
              batch_size=64,
              validation_data=(x_valid, y_valid))

    # Evaluate
    result = model.evaluate(x_train, y_train, verbose=2)
    print("Train Acc: ", result[1])
    result = model.evaluate(x_test, y_test, verbose=2)
    print("Test Acc: ", result[1])

    # Save Entire Model
    # In subclassing model:
    #   tf.saved_model.save(model, MODEL_DIR)
    #   == model.save(MODEL_DIR, save_format='tf')
    print("=" * 20, " Save Entire Model")
    # tf.saved_model.save(model, MODEL_DIR + MODEL_NAME)
    model.save(MODEL_DIR + MODEL_NAME, save_format='tf')

    # Save Weights (checkpoint)
    # save_format: Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
    #     '.keras' will default to HDF5 if `save_format` is `None`.
    #     Otherwise `None` defaults to 'tf'.
    print("=" * 20, " Save Weights (checkpoint)")
    model.save_weights(MODEL_CHECKPOINTS_DIR + MODEL_NAME)

    # Save Weights (h5)
    # Before we save or load model weight in h5 formae
    # we need created its variables: builds the model based on input shape.
    #   -> model.build(input_shape)
    print("=" * 20, " Save Weights (h5)")
    model.save_weights(MODEL_DIR + MODEL_NAME + '.h5')

    print("=" * 20, "Loading model and weights")

    # load model from SavedModel format
    model2 = tf.keras.models.load_model(MODEL_DIR + MODEL_NAME)
    result = model2.evaluate(x_test, y_test, verbose=2)
    print("[SavedModel] Test Acc: ", result[1])

    # Create a new model instance for load wights (chckpoint)
    new_model = CNN()
    new_model.compile(optimizer='Adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_accuracy'])
    new_model.load_weights(MODEL_CHECKPOINTS_DIR + MODEL_NAME)
    result = new_model.evaluate(x_test, y_test, verbose=2)
    print("[Load Weight, ckpt] Test Acc: ", result[1])

    # Create a new model instance for load wights (h5)
    new_model_2 = CNN()
    new_model_2.build((None, 28, 28, 1))
    new_model_2.compile(optimizer='Adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['sparse_categorical_accuracy'])
    new_model_2.load_weights(MODEL_DIR + MODEL_NAME + '.h5')
    result = new_model_2.evaluate(x_test, y_test, verbose=2)
    print("[Load Weight, h5] Test Acc: ", result[1])

    # Predict
    img_test = np.expand_dims(x_test[0], 0)
    scores = model2.predict(img_test)
    predict = str(np.argmax(scores))
    print("[SavedModel] gt: ", y_test[0], "predict: ", predict)

    img_test = np.expand_dims(x_test[0], 0)
    scores = new_model.predict(img_test)
    predict = str(np.argmax(scores))
    print("[Load Weight, ckpt] gt: ", y_test[0], "predict: ", predict)

    img_test = np.expand_dims(x_test[0], 0)
    scores = new_model_2.predict(img_test)
    predict = str(np.argmax(scores))
    print("[Load Weight, h5] gt: ", y_test[0], "predict: ", predict)
