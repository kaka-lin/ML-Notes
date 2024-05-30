import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        # Define your layer here
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    # This line defines the inputs' sizes of the network call
    @tf.function(input_signature=[tf.TensorSpec([None, 28, 28, 1], tf.float32)])
    def call(self, x):
        # Define your forward pass here
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x
