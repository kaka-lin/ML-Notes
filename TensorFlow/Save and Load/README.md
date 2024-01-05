# Saving and Loading a TensorFlow model

詳細 code 請看: [examples/main_xxx.py](https://github.com/kaka-lin/ML-Notes/blob/master/TensorFlow/Save%20and%20Load/examples)

以下主要針對 TensorFlow 2.x 進行說明，也會比較與 TensorFlow 1.x 的差異。

在 TensorFlow 中有分成:

- 保存模型 (save model)
- 保存權重 (save weights)

## Saving and Loading Model

The entire model can be saved to a file that contains the weight values, the model's configuration, and even optimizer's configuration.

This allows you to checkpoint a model and resume training later: from the exact same state, without access to the original code.

### 1. Keras model (HDF5 format)

將模型保存成 HDF5 format 僅支援:

- Sequential model
- Functional model

```python
model.save('model.h5')
```

Loading model:

```python
new_model = tf.keras.models.load_model('model.h5')
```

#### 而這方法不支援 *subclassed models*。

因為 `Subclassed models` 是通過 overriding `tf.keras.Model` class 的方法來定義的，
這使得它們`不太容易被序列化 (serializable)`。

要解決這個問題，有下面幾個方法:

1. `使用 TensorFlow SavedModel 格式保存模型`:

    不使用 HDF5 格式，你可以使用 SavedModel 格式保存模型，這是 Subclassing model 的推薦方式。在調用`model.save()` 方法時，指定 `save_format='tf'`。

    ```python
    model.save('model_directory', save_format='tf')
    ```

    這將以 `SavedModel 格式` 保存你的模型，其中包含一個 TensorFlow protobuf 文件和一組 checkpoint files，如下:

    ```
    saved_model/
    ├── assets/
    ├── variables/
    │   ├── variables.data-00000-of-00001
    │   └── variables.index
    ├── fingerprint.pb
    ├── keras_metadata.pb
    └── saved_model.pb
    ```

    Loading model:

    ```python
    new_model = tf.keras.models.load_model('model_directory')
    ```

2. `只保存模型權重`:

    如果你只需要保存模型的權重而不保存整個模型架構，另一個選項是使用 `model.save_weights()` 方法只保存模型的權重。

    ```python
    model.save_weights('my_model.h5')
    ```

    如果直接加載模型會出錯，如下:

    ```python
    model.load_weights('my_model.h5')
    ```

    Error message:

    > ValueError: Unable to load weights saved in HDF5 format into a subclassed Model which has not created its variables yet. Call the Model first, then load the weights.

    所以再加載模型前，需要先 build，如下:

    ```python
    model.build(input_shape = <INPUT_SHAPE>)
    model.load_weights('my_model.h5')
    ```

### 2. SavedModel format

```python
tf.saved_model.save(model, path_to_dir)
```

Loading model:

```python
new_model = tf.saved_model.load(path_to_dir)
```

## Saving and Loading Weights

```python
model.save_weights('./checkpoints/my_checkpoint')
```

Loading weights:

```python
# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')
```

文件夾結構如下:

```sh
checkpoints/
├── checkpoint
├── my_checkpoint.data-00000-of-00001
├── my_checkpoint.index
```

##### Issue:

如果在 Load Weights 時遇到下面問題:

```
Value in checkpoint could not be found in the restored object
```

可以加上 `expect_partial()` 來解決

- `expect_partial()`: Silence warnings about incomplete checkpoint restores. Warnings are otherwise printed for unused parts of the checkpoint file or object when the Checkpoint object is deleted (often at program shutdown).

如下:

```python
model.load_weights('./checkpoints/my_checkpoint').expect_partial()
```
