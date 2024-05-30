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
MODEL_NAME = 'subclassing_model'


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

    model = CNN()
    model.build((None, 28, 28, 1))
    # model.summary()

    # To define a forward pass, please override `Model.call()`.
    # To specify an input shape, either call `build(input_shape)` directly,
    # or call the model on actual data using `Model()`, `Model.fit()`, or `Model.predict()`.
    # If you have a custom training step,
    # please make sure to invoke the forward pass in train step through `Model.__call__`,
    # i.e. `model(inputs)`, as opposed to `model.call()`.
    output = model(x_test) # we need to call the model once before saving it

    # Save Entire Model
    # In subclassing model:
    #   tf.saved_model.save(model, MODEL_DIR)
    #   == model.save(MODEL_DIR, save_format='tf')
    print("=" * 20, " Save Entire Model")

    # signatures=model.call.get_concrete_function()
    # print(model.call.get_concrete_function())
    model.save(
        MODEL_DIR + MODEL_NAME,
        save_format='tf',
        signatures=model.call.get_concrete_function()
    )

    # load model from SavedModel format
    model2 = tf.keras.models.load_model(MODEL_DIR + MODEL_NAME)
    # print(list(model2.signatures.keys()))
    output2 = model(x_test)

    assert np.allclose(output, output2, atol=1e-3)
