# 訓練後量化 (Post Training Quantization, PTQ)

`訓練後量化 (Post Training Quantization)` 是一種轉換技術，可以減少模型大小，同時還可以改善 CPU 和硬體加速器的延遲，且模型精度幾乎沒有下降。

## Optimization Methods

There are several post-training quantization options to choose from. Here is a summary table of the choices and the benefits they provide:

| 技術 | 好處 | 硬體 |
| :-: | :-: | :-: |
| [Dynamic range quantization](#dynamic-range-quantization) | 4x smaller, 2x-3x speedup | CPU |
| [Full integer quantization](#full-integer-quantization) | 4x smaller, 3x+ speedup | CPU, Edge TPU, Microcontrollers |
| [Float16 quantization](#float16-quantization)  | 2x smaller, GPU acceleration | CPU, GPU |

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

在 Tensorflow 中，對於要量化模型的程度，有多種選擇，如下所示。

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

Additionally, to ensure compatibility with integer only devices (such as 8-bit microcontrollers) and accelerators (such as the `Coral Edge TPU`), you can enforce `full integer quantization` for all ops including the input and output, by using the following steps:

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

您可以通過將`權重 (weights)`量化為 `float16` 來減小 model 的大小。

優點:

- 模型大小縮減一半 (因為所有權重都變成了原始大小的一半)
- 準確率的損失最小
- 支持一些可以直接對 `float16` data 進行運算的 delegates (例如: GPU delegate)，從而使執行速度更快 (比 float32 計算更快)

缺點:

- It does not reduce latency as much as a quantization to fixed point math.

- By default, a float16 quantized model will "dequantize" the weights values to float32 when run on the CPU.

    > Note that the GPU delegate will not perform this dequantization, since it can operate on float16 data.)

如下步驟:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()
```

## Model accuracy

由於`權重是在訓練後量化的`，因此`可能會降低準確性`，尤其是對於較小的 networks。

另外如果準確度下降太多，請考慮使用[量化感知訓練 (quantization aware training)](https://www.tensorflow.org/model_optimization/guide/quantization/training)，但是這樣做需要在模型訓練期間進行修改以添加 `fake quantization nodes`。

> - Pre-trained fully quantized models are provided for specific networks on [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite&q=quantized).
> - It is important to check the accuracy of the quantized model to verify that any degradation in accuracy is within acceptable limits. There are tools to evaluate [TensorFlow Lite model accuracy](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks).

## Representation for quantized tensors

8-bit quantization approximates floating point values using the following formula.

$$real\_value = (int8\_value - zero\_point) \times scale$$

The representation has two main parts:

- Per-axis (aka per-channel) or per-tensor weights represented by int8 two’s complement values in the range [-127, 127] with zero-point equal to 0.
- Per-tensor activations/inputs represented by int8 two’s complement values in the range [-128, 127], with a zero-point in range [-128, 127].

For a detailed view of our quantization scheme, please see our [quantization spec](https://www.tensorflow.org/lite/performance/quantization_spec). Hardware vendors who want to plug into TensorFlow Lite's delegate interface are encouraged to implement the quantization scheme described there.

## Reference

- [Tensorflow/Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
