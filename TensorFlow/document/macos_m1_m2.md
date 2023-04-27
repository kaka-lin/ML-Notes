# TensorFlow on Mac M1/M2 with GPU support

Reference: [Get started with tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)

## Requirements

- macOS 12.0 or later (Get the latest beta)
- Python 3.8 or later

    Install the M1 [Miniconda](https://docs.conda.io/en/latest/miniconda.html#macos-installers) Version.

- Xcode command-line tools

    ```bash
    $ xcode-select --install
    ```

## Get Started

### 1. Install the Tensorflow dependencies:

```bash
$ conda install -c apple tensorflow-deps
```

### 2. Install base TensorFlow

```bash
$ pip2 install tensorflow-macos
```

### 3. Install tensorflow-metal plug-in

```bash
$ pip2 install tensorflow-metal
```

### 4. Verify

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0:], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
```

Output

```bash
Metal device set to: Apple M2

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

1 Physical GPUs, 1 Logical GPU
```
