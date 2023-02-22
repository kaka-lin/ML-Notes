from keras.models import Model
from keras.layers import Input, Lambda
import numpy as np

# 定義 Lambda 要執行的函數
def minus(inputs):
    x, y = inputs
    return (x-y)


if __name__ == "__main__":
    # Step 1. 定義模型
    a = Input(shape=(2,))
    b = Input(shape=(2,))

    # Lambda layer
    minus_layer = Lambda(minus, name='minus')([a, b])

    # Model
    model = Model(inputs=[a, b], outputs=minus_layer)

    # Step 2. 測試模型
    v0 = np.array([8, 4])
    v1 = np.array([5, 2])

    # output: [[3, 2]]
    print(model.predict([v0.reshape(1, 2), v1.reshape(1, 2)]))
