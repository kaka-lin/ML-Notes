import numpy as np
import tensorflow as tf


def softmax_1(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1)


def softmax_2(x):
    """
    Softmax implementation with Tensorflow2
    Args:
        - x [tensor]: 1xN tensors

    Returns:
        - soft_x: [tensor] softmax of x
    """
    exp = tf.exp(x)
    return exp / tf.reduce_sum(exp, 1, keepdims=True)


if __name__ == "__main__":
    x = np.array([[1, 4.2, 0.6, 1.23, 4.3, 1.2, 2.5]])
    x_tensor = tf.convert_to_tensor(x, np.float32)
    print("Input Array: ", x)
    print("Softmax Array: ", softmax_1(x))
    print("Softmax Array: ", softmax_2(x_tensor).numpy())
