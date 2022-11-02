import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tqdm import tqdm
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS

from data.dataset import load_data
from model.model import CNN
from trainer.trainer import train_step, test_step

FLAGS = flags.FLAGS
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 10, 'number of epochs')
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_float('lr', 0.0001, 'learning rate')
flags.DEFINE_boolean('save_model', True, 'Save model for fine-tune or retrain')
flags.DEFINE_string('save_model_dir', 'checkpoints',
                    'path to save model')
flags.DEFINE_boolean('test_mode', False, 'Inference mode')


def main(argv):
    # Checking Device for Training (GPU or not)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # Loading dataset
    # Use `tf.data`` to batch and shuffle the dataset:
    train_data, test_data = load_data()
    (x_train, y_train), (x_test, y_test) = train_data, test_data
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(FLAGS.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(FLAGS.batch_size)

    if not FLAGS.test_mode:
        # Build the model
        model = CNN()

        # Compile the model: optimizer and loss
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        # Strating training
        n_batches = len(train_dataset)
        for epoch in range(FLAGS.epochs):
            print(f'Epoch {epoch+1}/{FLAGS.epochs}')
            with tqdm(train_dataset, total=n_batches,
                      bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:36}{r_bar}') as pbar:
                for batch, (x_train, y_train) in enumerate(pbar):
                    train_step(model, x_train, y_train,
                               optimizer, loss_fn,
                               train_loss, train_accuracy)
                    pbar.set_postfix({
                        'loss': train_loss.result().numpy(),
                        'accuracy': train_accuracy.result().numpy()})

            # Reset the metrics at the start of the next epoch
            train_loss.reset_states()
            train_accuracy.reset_states()

            # Save model
            if FLAGS.save_model:
                # save weights
                model.save_weights(f"{FLAGS.save_model_dir}/train_{epoch+1}.tf")

                # save the whole model
                # tf.saved_model.save(model, save_model_dir)

        # TODO: add the validation part
    else:
        # load the model
        model = CNN()
        model.load_weights('checkpoints/train_10.tf')
        # model = tf.saved_model.load(save_model_dir)

        # Compile the model: loss and metrics
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        # Testing
        n_batches_test = len(test_dataset)
        print(f'Test:')
        with tqdm(test_dataset, total=n_batches_test,
                  bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:36}{r_bar}') as pbar:
            for batch, (x_test, y_test) in enumerate(pbar):
                test_step(model, x_test, y_test, loss_fn,
                          test_loss, test_accuracy)
                pbar.set_postfix({
                    'loss': test_loss.result().numpy(),
                    'accuracy': test_accuracy.result().numpy()})

        test_acc = test_accuracy.result() * 100
        test_loss.reset_states()
        test_accuracy.reset_states()
        print(f"Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    app.run(main)
