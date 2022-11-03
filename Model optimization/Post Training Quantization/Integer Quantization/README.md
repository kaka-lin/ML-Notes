# 整數量化 (Integer Quantization)

`Integer quantization` 是一種優化策略，將 32-bit 浮點數（例如: 權重和 activations）轉換為最接近的 8-bit fixed-point。這樣可以減小模型大小且加快的推理速度，這對於微控制器等低功耗設備很有價值。

```
Edge TPU 等僅支持整數的加速器也需要使用此種數據格式。
```

## Integer Quantization in TFLite

In TensorFlow Lite, You actually have several options as to how much you want to quantize a model, as below:

- `Integer with float fallback (using default float input/output)`
- `Integer only (full integer quantization)`

Code:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
tflite_quant_model = converter.convert()
```

### Example

In this example, you'll train an MNIST model from scratch, convert it into a Tensorflow Lite file, and  you'll perform `"full integer quantization"`, which converts all weights and activation outputs into 8-bit integer data.

Finally, you'll check the accuracy of the converted model and compare it to the original float model.

結果如下:

```bash
Model Size:
{'baselin model (TF2)': 11149205,
 'non quantized tflite': 11084920,
 'ptq dynamic tflite': 2774512,
 'ptq float fullback tflite': 2775032,
 'ptq int only tflite': 2775056}
Model Accuracy:
{'baseline model': 0.9815,
 'non quantized tflite': 0.9815,
 'ptq dynamic tflite': 0.9815,
 'ptq float fullback tflite': 0.9812,
 'ptq int only tflite': 0.9814}
```

詳細程式步驟請看:

- [example/full_integer_quantization.ipynb](https://github.com/kaka-lin/ML-Notes/blob/master/Model%20optimization/Post%20Training%20Quantization/Integer20Quantization/example/full_integer_quantization.ipynb)
  - Colab: <a href="https://colab.research.google.com/github/kaka-lin/ML-Notes/blob/master/Model%20optimization/Post%20Training%20Quantization/Integer20Quantization/example/full_integer_quantization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Reference

- [Post-training integer quantization](https://www.tensorflow.org/lite/performance/post_training_integer_quant)
