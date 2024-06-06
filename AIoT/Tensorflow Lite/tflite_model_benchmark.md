# TFLite Model Benchmark Tool

Perform TFLite model on Desktop/Android device to get the inference time

## Run on Android

### 0. Prepare

- `ADB Tool (command line tool for android)`

    We can use adb to check the device connect status.

    ```sh
    $ adb devices
    ```


### 1. Build the `benchmark_model` for specific platform.

For example:

- device: *android_arm64*:

```sh
$ bazel build -c opt \
  --config=android_arm64 \
  tensorflow/lite/tools/benchmark:benchmark_model
```

### 2. Connect your phone. Push the binary to your phone with adb push (make the directory if required):

```sh
$ adb push benchmark_model /data/local/tmp
```

#### More Info about connecting your phone:

> Use ADB to discover IP, and connect your phone.
>
> ```sh
> $ adb shell ip -f inet addr show wlan0
> $ adb tcpip 5555
> $ adb connect XXX.XXX.XXX.XXX(mobile ip)
> ```

### 3. Make the binary executable.

```sh
$ adb shell chmod +x /data/local/tmp/benchmark_model
```

### 4. Push the compute graph (tflite model) that you need to test

For example:

```sh
$ adb push mobilenet_quant_v1_224.tflite /data/local/tmp
```

### 5. Run the benchmark

For example:

```sh
$ adb shell /data/local/tmp/benchmark_model \
    --graph=/data/local/tmp/mobilenet_quant_v1_224.tflite \
    --num_threads=4 \
    --warmup_runs=1 \
    --num_runs=50 \
    --use_gpu=true
```

> can change `--use_gpu` to `--use_xnnpack`


### 6. The Result

Example as below:

```sh
STARTING!
Log parameter values verbosely: [0]
Min num runs: [50]
Num threads: [4]
Min warmup runs: [1]
Graph: [/data/local/tmp/xxx.tflite]
#threads used for CPU inference: [4]
Use gpu: [1]
Loaded model /data/local/tmp/xxx.tflite
INFO: Initialized TensorFlow Lite runtime.
...
Inference timings in us: Init: 5685, First inference: 18535, Warmup (avg): 14462.3,, Inference (avg): 14575.2
...
```
  - Inference time / 1000 = ms

## Reference

- [TFLite Model Benchmark Tool with C++ Binary](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
