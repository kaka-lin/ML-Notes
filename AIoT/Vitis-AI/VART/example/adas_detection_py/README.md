# Example of ADAS Detection

This is the Python version of [Vitis-AI/demo/VART/adas_detection/](https://github.com/Xilinx/Vitis-AI/tree/master/demo/VART/adas_detection)


## 1. Download the model

In this example, we use `yolov3_adas_pruned_0_9-zcu102_zcu104-r1.3.0.tar`.

Download the model for `ZCU104`

```bash
$ mkdir -p model
$ wget https://www.xilinx.com/bin/public/openDownload?filename=yolov3_adas_pruned_0_9-zcu102_zcu104-r1.3.0.tar.gz -O model/yolov3_adas_pruned_0_9-zcu102_zcu104-r1.3.0.tar.gz
```

## 2. Put model into Target

```bash
$ scp yolov3_adas_pruned_0_9-zcu102_zcu104-r1.3.0.tar.gz root@IP_OF_BOARD:~/
```

## 3. Untar the model

```bash
$ mkdir -p /usr/share/vitis_ai_library/models

$ tar yolov3_adas_pruned_0_9-zcu102_zcu104-r1.3.0.tar.gz
$ cp yolov3_adas_pruned_0_9 /usr/share/vitis_ai_library/models -r
```

## 4. Run the example

```bash
$ python3 main.py
```
