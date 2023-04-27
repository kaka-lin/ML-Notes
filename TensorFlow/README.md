# Tensorflow 2.0 Learning Notes

This is my `TensorFlow 2.0`'s learning notes and some examples.

## Get Start

You can just install TensorFlow 2.0 with `pip` (as shown below ), or you can use a [docker image](./document/docker_image.md) that we already made.

For Mac M1/M2 please see [TensorFlow on Mac M1/M2 with GPU support.](./document/macos_m1_m2.md)

```bash
$ pip install -r requirements.txt

# Clone this repository
$ git clone https://github.com/kaka-lin/tensorflow2-tutorials.git

# Run Jupyter Notebook in your local directory
$ jupyter notebook
```
### [Option] Tensorflow GPU Version

If you want to install `tensorflow-gpu`, you need to have `NVIDIA® GPU card with CUDA® Compute Capability 3.5 or higher`.

- Install [NVIDIA driver, CUDA Toolkit and cuDNN SDK](../Nvidia/nvidia-driver.md).

## Examples

Except for the examples we mention below, others examples please see [here](https://github.com/kaka-lin/ML-Notes/tree/master/TensorFlow/examples).

#### YOLO Series

- [kaka-lin/yolov2-tf2](https://github.com/kaka-lin/yolov2-tf2)
- [kaka-lin/yolov3-tf2](https://github.com/kaka-lin/yolov3-tf2)

## Learning Notes

All the files of notes are using `jupyter file (.ipynb)`, if you want to view and use `.py` file structure, please see all projects in the [examples folder](https://github.com/kaka-lin/ML-Notes/tree/master/TensorFlow/examples).

### 1. Beginner quickstart

- [The Basic - Training Your First Model](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/00_the_basics_training_first_model.ipynb)

### 2. Keras (tf.Keras)

#### Image Classification

- MNIST
  - Sequential model: [Image Classification - MNIST](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/01_classification_mnist.ipynb)
  - Subclassing: - [Build models with `subclassing`](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/01_classification_mnist_model_subclassing.ipynb)

#### Text Classification

- IMDB: [Text Classification - IMDB](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/keras/02_classification_imdb.ipynb)

#### Custom training

- [Writing a training loop from scratch](https://github.com/kaka-lin/ML-Notes/tree/master/TensorFlow/keras/custom_training_loop)
  - [Progress Bar](https://github.com/kaka-lin/ML-Notes/tree/master/TensorFlow/keras/custom_training_loop/progress_bar)

### 3. Data input pipeline (tf.data)

- Load and preprocess data

    - [TFRecord and tf.Example](https://github.com/kaka-lin/ML-Notes/tree/master/TensorFlow/data/load_and_preprocess_data/tfrecords)

### 4. Automatic Differentiation (tf.GradientTape)

- [Introduction](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/gradientTape/01_introduction.md)
