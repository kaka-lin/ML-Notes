import numpy as np
from scipy.special import softmax

np.set_printoptions(precision=6)


def k_softmax(x):
    return np.exp(x) / sum(np.exp(x))


if __name__ == "__main__":
    x = np.array([1, 4.2, 0.6, 1.23, 4.3, 1.2, 2.5])
    print("Input Array: ", x)
    print("Softmax Array: ", k_softmax(x))
    print("Softmax Array: ", softmax(x))
