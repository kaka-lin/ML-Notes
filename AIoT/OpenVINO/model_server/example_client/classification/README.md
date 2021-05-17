# Example client for classification

## 1. Download OpenVINO Model Server Docker image

```bash
$ docker pull openvino/model_server:latest
```

## 2. Download Model

In this example, we use [resnet18-xnor-binary-onnx-0001](https://docs.openvinotoolkit.org/latest/omz_models_model_resnet18_xnor_binary_onnx_0001.html) that is Intel's Pre-trained model

```bash
# download *.xml
$ curl --create-dirs https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/resnet18-xnor-binary-onnx-0001/FP32-INT1/resnet18-xnor-binary-onnx-0001.xml -o model/1/resnet18-xnor-binary-onnx-0001.xml

# download *.bin
$ curl --create-dirs https://download.01.org/opencv/2021/openvinotoolkit/2021.1/open_model_zoo/models_bin/1/resnet18-xnor-binary-onnx-0001/FP32-INT1/resnet18-xnor-binary-onnx-0001.bin -o model/1/resnet18-xnor-binary-onnx-0001.bin
```

## 3. Start the Model Server Container

Start the Model Server container:

```bash
$ docker run --rm -d -p 9000:9000 \
-v $(pwd)/model:/models/resnet18 \
openvino/model_server:latest \
--model_path /models/resnet18 \
--model_name resnet18 \
--port 9000
```

## 4. Run Inference

### 1. Install the dependencies

```bash
$ pip3 install -r client_requirements.txt
```

### 2. Run the client

```bash
$ python3 classification.py
```
