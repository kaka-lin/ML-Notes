import tensorflow as tf
import numpy as np


def custom_conv2d_transpose(x, filter):
    # convert to tensor
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)
    filter_tf = tf.convert_to_tensor(filter, dtype=tf.float32)

    # padding with zeros and reshape
    x_tf = tf.reshape(tf.pad(x_tf, [[2, 2], [2, 2]]), [1, 6, 6, 1])
    # rotating filter clockwise by 180 degrees and reshape
    filter_tf = tf.image.rot90(tf.reshape(filter_tf, [3, 3, 1]), k=2)
    filter_tf = tf.reshape(filter_tf, [3, 3, 1, 1])

    # transposed convolution
    result = tf.nn.conv2d(x_tf, filter_tf, strides=[1, 1, 1, 1], padding="VALID")
    return result.numpy().reshape(4, 4)


if __name__ == "__main__":
    x = np.array([[1, 2], [4, 5]], dtype=np.float32)
    filter = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    # convert to tensor
    x_tf = tf.reshape(tf.convert_to_tensor(x, dtype=tf.float32), [1, 2, 2, 1])
    filter_tf = tf.reshape(tf.convert_to_tensor(filter, dtype=tf.float32), [3, 3, 1, 1])

    # transposed convolution
    result = tf.nn.conv2d_transpose(x_tf, filter_tf, output_shape=[1, 4, 4, 1], strides=[1, 1, 1, 1], padding="VALID")
    result_np = result.numpy().reshape(4, 4)

    # custom transposed convolution
    result2 = custom_conv2d_transpose(x, filter)

    print("input: \n", x)
    print("filter: \n", filter)
    print("result: \n", result_np)
    print("custom result: \n", result2)
