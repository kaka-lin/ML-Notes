# 訓練後量化 (Post Training Quantization, PTQ)

`訓練後量化 (Post Training Quantization)` 是一種轉換技術，可以減少模型大小，同時還可以改善 CPU 和硬體加速器的延遲，且模型精度幾乎沒有下降。

## Optimization Methods

There are several post-training quantization options to choose from. Here is a summary table of the choices and the benefits they provide:

| 技術 | 好處 | 硬體 |
| :-: | :-: | :-: |
| Dynamic range quantization | 4x smaller, 2x-3x speedup | CPU |
| Full integer quantization | 4x smaller, 3x+ speedup | CPU, Edge TPU, Microcontrollers |
| Float16 quantization  | 2x smaller, GPU acceleration | CPU, GPU |

Tensorflow 提供了以下 decision tree 幫助我們判斷哪種量化方案最適合我們，如下:

![](images/PTQ.png)

### Dynamic range quantization

- 轉換時:

    將`權重(weights)`從浮點靜態量化 (statucally quantizes) 為整數 (8-bits of precision)`

- 推理時:

    會根據其範圍動態量化 `activation functions` 為 8-bits，並且用 8-bits 的權重和激活函數進行計算。

此種量化方法:

- 減少記憶體使用並加快計算速度，而無需提供有代表性的數據集進行校正。
- 提供了接近 `fully fixed-point` inferences 的 latencies。

    然而，輸出仍然使用浮點數儲存，因此此法運算速度的提升仍小於 `full fixed-point` 所提升的。

###### TFLite Example:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

###### PyTorch Example:

```python
quantized_model = torch.quantization.quantize_dynamic(model, {需要量化的layer}, dtype=torch.qint8)
```

### Full integer quantization

通過確保所有模型數學 (model math) 都是整數量化，可以進一步改善延遲，減少 peak memory 的使用量以及兼容僅支持整數的硬體設備或加速器 (accelerators)。

對於 `full integer quantization`:

- 需要校正或估算模型中所有浮點數 tensor 的範圍，即 (min, max)
- 需要有一個代表性的 dataset 來進行校正

    因為 `model input`, `activations (outputs of intermediate layers)` 和 `model output` 等 `variable tensors` 相對於 `weights`, `biases` 等 `constant tensors` 來說無法被校正，除非我們運行幾個 inference cycles。

    > 該數據集可以是訓練或驗證數據集的一小部分（大約 100-500 個樣本）

    From TensorFlow 2.7 version, you can specify the representative dataset through a [signature](https://www.tensorflow.org/lite/guide/signatures) as the following example:

    ```python
    def representative_dataset():
        for data in dataset:
            yield {
                "image": data.image,
                "bias": data.bias,
            }
    ```

    其他方法請參考: [TensorFlow 官網](https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization)

在 Tensorflow 中有兩種轉換模式

#### 1. Integer with float fallback (using default float input/output)

In order to fully integer quantize a model, but use float operators when they don't have an integer implementation (to `ensure conversion occurs smoothly`), use the following steps:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_quant_model = converter.convert()
```

> Note: This *tflite_quant_model* won't be compatible with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU) because `the input and output still remain float` in order to have the same interface as the original float only model.

#### 2. Integer only

Additionally, to ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the Coral Edge TPU), you can enforce full integer quantization for all ops including the input and output, by using the following steps:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
```

> Creating integer only models is a common use case for [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) and [Coral Edge TPUs](https://coral.ai/).

### Float16 quantization

## Reference

- [Tensorflow/Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
