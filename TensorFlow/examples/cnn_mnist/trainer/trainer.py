import tensorflow as tf
from absl import logging


@tf.function
def train_step(model, x_batch, y_batch, optimizer, loss_fn,
               train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # forward pass
        # `training=True` is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)

    # backward
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_accuracy.update_state(y_batch, predictions)


@tf.function
def test_step(model, x_batch, y_batch, loss_fn,
              test_loss, test_accuracy):
    # forward
    predictions = model(x_batch, training=False)
    t_loss = loss_fn(y_batch, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(y_batch, predictions)
