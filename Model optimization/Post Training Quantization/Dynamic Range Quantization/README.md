# 動態範圍量化 (Dynamic Range Quantization)

- 轉換時:

    將`權重(weights)`從浮點靜態量化 (statucally quantizes) 為整數 (8-bits of precision)`

- 推理時:

    會根據其範圍動態量化 `activation functions` 為 8-bits，並且用 8-bits 的權重和激活函數進行計算。

此種量化方法:

- 減少記憶體使用並加快計算速度，而無需提供有代表性的數據集進行校正。
- 提供了接近 `fully fixed-point` inferences 的 latencies。

    然而，輸出仍然使用浮點數儲存，因此此法運算速度的提升仍小於 `full fixed-point` 所提升的。

## Dynamic Range Quantization in TFLite

TensorFlow Lite 現在支持將權重轉換為 8 bits precision，作為從 `TensorFlow GraphDef` 到 `TensorFlow Lite's FlatBuffer` 格式的模型轉換的一部分。

Dynamic range quantization 可將 model size 縮小 4 倍。
此外，TFLite 支持對 activations 進行 fly quantization 和 dequantization，以實現以下效果:

1. Using quantized kernels for faster implementation when available.
2. Mixing of floating-point kernels with quantized kernels for different parts of the graph.

The activations are always stored in floating point. For ops that support quantized kernels, `the activations are quantized to 8 bits of precision dynamically prior to processing and are de-quantized to float precision after processing`. Depending on the model being converted, this can give a speedup over pure floating point computation.

- Activations: 總是以浮點數儲存
- Activations: 在處理之前被動態量化為 8 bits，並在處理後被反量化為 float precision。

與量化感知訓練 (QAT) 相比，此方法為:

- 權重:在訓練後量化
- Activations: 在推理時動態量化

因此，模型權重不會被重新訓練以補償量化引起的誤差。

> 請務必檢查量化模型的準確率，以確保下降程度是可以接受的。


Code:

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```

### Example

This example trains an MNIST model from scratch, checks its accuracy in TensorFlow, and then converts the model into a Tensorflow Lite flatbuffer with dynamic range quantization. Finally, it checks the accuracy of the converted model and compare it to the original float model.

結果如下:

```bash
Model Size:
{'baselin model (TF2)': 11149205,
 'non quantized tflite': 11084920,
 'ptq dynamic tflite': 2774512}
Model Accuracy:
{'baseline model': 0.9788,
 'non quantized tflitel': 0.9788,
 'ptq dynamic tflite': 0.9788}
```

詳細程式步驟請看:

- [example/dynamic_range_quantization.ipynb](https://github.com/kaka-lin/ML-Notes/blob/master/Model%20optimization/Post%20Training%20Quantization/Dynamic%20Range%20Quantization/example/dynamic_range_quantization.ipynb)
  - Colab: <a href="https://colab.research.google.com/github/kaka-lin/ML-Notes/blob/master/Model%20optimization/Post%20Training%20Quantization/Dynamic%20Range%20Quantization/example/dynamic_range_quantization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Reference

- [Post-training dynamic range quantization](https://www.tensorflow.org/lite/performance/post_training_quant)
