# Example of ResNet-50 with VART

## 1. Download the model

In this example, we use `resnet50`.

model detailed information:
- [Vitis-AI/models/AI-Model-Zoo/model-list/cf_resnet50_imagenet_224_224_7.7G_1.3/model.yaml](
https://github.com/Xilinx/Vitis-AI/blob/master/models/AI-Model-Zoo/model-list/cf_resnet50_imagenet_224_224_7.7G_1.3/model.yaml)

In the model.yaml, you will find the model's download links for different platforms.

Download the model for `ZCU104`

```bash
$ wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104-r1.3.0.tar.gz
```

## 2. Put model into Target

```bash
$ scp resnet50-zcu102_zcu104-r1.3.1.tar.gz root@IP_OF_BOARD:~/
```

## 3. Untar the model

```bash
$ mkdir -p /usr/share/vitis_ai_library/models

$ tar zxvf resnet50-zcu102_zcu104-r1.3.1.tar.gz
$ cp resnet50 /usr/share/vitis_ai_library/models -r
```

## 4. Run the example

```bash
$ python3 main.py

# or you can specify thread_number, the model path
$ python3 main.py <thread_number> <resnet50_xmodel_file>
```
