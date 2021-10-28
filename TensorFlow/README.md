# Tensorflow 2.0 Learning Notes

This is my `TensorFlow 2.0`'s learning notes and some examples.

## Get Start

You can just install TensorFlow 2.0 with `pip` (as shown below ), or you can use a [docker image](./document/docker_image.md) that we already made.

```bash
$ pip install -r requirements.txt

# Clone this repository
$ git clone https://github.com/kaka-lin/tensorflow2-tutorials.git

# Run Jupyter Notebook in your local directory
$ jupyter notebook
```
### [Option] Tensorflow GPU Version

If you want to install `tensorflow-gpu`, you need to have `NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher`.

- Install [NVIDIA driver, CUDA Toolkit and cuDNN SDK](./document/nvidia.md).

## Examples

- [kaka-lin/yolov2-tf2](https://github.com/kaka-lin/yolov2-tf2)
- [kaka-lin/yolov3-tf2](https://github.com/kaka-lin/yolov3-tf2)

## Learning Notes

### 1. Beginner quickstart

- [The Basic - Training Your First Model](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/00_the_basics_training_first_model.ipynb)

### 2. Keras (tf.Keras)

- [Image Classification - MNIST](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/01_classification_mnist.ipynb)

- [Text Classification - IMDB](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/02_classification_imdb.ipynb)

- [Build models with `subclassing`](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/01_classification_mnist_model_subclassing.ipynb)

### 3. Data input pipeline (tf.data)

- Load and preprocess data

    - [TFRecord and tf.Example](https://github.com/kaka-lin/ML-Notes/tree/master/TensorFlow/data/load_and_preprocess_data/tfrecords)
