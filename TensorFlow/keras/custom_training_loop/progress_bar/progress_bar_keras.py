import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


# Checking Device for Training (GPU or not)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)

# Parameters
NUM_EPOCHS = 10
BATCH_SIZE = 32

# Loading dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")
y_train, y_test = y_train.astype("int32"), y_test.astype("int32")
print("Training images: ", x_train.shape)
print("Training lables: ", y_train.shape)

# Use tf.data to batch and shuffle the dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(
    (x_test, y_test)).batch(BATCH_SIZE)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
# print(model.summary())

# Compile the model: optimizer and loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Select metrics to measure the loss and the accuracy of the model.
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Training
for epoch in range(NUM_EPOCHS):
    n_batches = x_train.shape[0] / BATCH_SIZE
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    bar = tf.keras.utils.Progbar(target=n_batches,
                                 stateful_metrics=["loss", "accuracy"])
    for idx, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # forward
            pred = model(x, training=True)
            loss = loss_fn(y, pred)
        # backward
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # get the index of the max log-probability
        # pred_max = tf.argmax(pred, axis=1, output_type=tf.int32)
        # correct = tf.cast(tf.equal(pred_max, y), tf.float32)
        # accuracy = tf.reduce_mean(correct).numpy()

        # Update training metric after batch
        train_loss.update_state(loss)
        train_accuracy.update_state(y, pred)

        bar.update(idx+1,
            values=[("loss", train_loss.result()), ("accuracy", train_accuracy.result())])

    train_loss.reset_states()
    train_accuracy.reset_states()
