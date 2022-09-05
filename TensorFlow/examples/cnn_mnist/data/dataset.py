import tensorflow as tf
from tensorflow.keras.datasets import mnist


def load_data():
    mnist_dataset = mnist.load_data()
    (x_train, y_train), (x_test, y_test) = mnist_dataset

    ### Preprocess the data
    # normalization
    # scale pixel value from 0:255 to 0:1
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)
