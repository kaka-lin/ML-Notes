import numpy as np
import tensorflow as tf


def cross_entropy_np(y_true, y_pred, epsilon=1e-12):
    """
    Cross entropy loss implementation with Numpy
    Args:
        y_true [ndarray]: Ground truth values (NxC).
                          N batch size, C number of classe.
        y_pred [ndarray]: The predicted values (NxC).
                          N batch size, C number of classe.
        epsilon: Defaults to 1e-12.

    Returns:
        loss: cross entropy
    """
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    ce = -np.sum(y_true * np.log(y_pred)) / N
    return ce


if __name__ == "__main__":
    # y_true = np.array([[0, 0, 0, 1],
    #                    [0, 0, 0, 1]])
    # y_pred = np.array([[0.25, 0.25, 0.25, 0.25],
    #                    [0.01, 0.01, 0.01, 0.96]])

    y_true = np.array([[0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])

    cross_entropy_tf = tf.keras.losses.CategoricalCrossentropy()
    tf_ce = cross_entropy_tf(y_true, y_pred) # 1.177. Correct answer
    np_ce = cross_entropy_np(y_true, y_pred)

    print("Cross Entropy:")
    print("\tCorrect answer (tf): ", tf_ce.numpy())
    print("\tImplementation with Numpy: ", np_ce)

    ll_norm = tf.norm(tf_ce - np_ce, ord=1)
    assert ll_norm < 1e-5, "Cross Entropy calcuoation is wrong"
    print("Cross Entropy implementation is correct!")
