# Tiny YOLOv4 TensorFlow Lite mode on Ubuntu

This repo is refer to [michhar/jetson-gpu-yolov4](https://github.com/michhar/jetson-gpu-yolov4)

## Build the docker container

```bash
$ docker build --rm -t kakalin/yolov4-tiny-tflite .
```

## Running and testing

```bash
$ docker run --rm -d -i \
--name my_yolo_container \
-p 5000:5000 \
kakalin/yolov4-tiny-tflite:latest
```

After started service, you can open [http://127.0.0.1:5000](http://127.0.0.1:5000), you would see "Hello from Tiny Yolov4 inferencing based on TensorFlow Lite"


### "/score"

To get a list of detected objects, use the following command.

```bash
$ curl -X POST http://127.0.0.1:5000/score -H "Content-Type: image/jpeg" --data-binary @<full_path_to_image_file_in_jpeg>

# example
$ curl -X POST http://127.0.0.1:5000/score -H "Content-Type: image/jpeg" --data-binary @images/dog.jpg
```

If successful, you will see JSON printed on your screen that looks something like this

```json
{
    "inferences": [
        {
            "type": "entity",
            "entity": {
                "tag": {
                    "value": "truck",
                    "confidence": "0.85265654"
                },
                "box": {
                    "l": "0.5947845",
                    "t": "0.1360384",
                    "w": "0.32280278",
                    "h": "0.15790814"
                }
            }
        },
        {
            "type": "entity",
            "entity": {
                "tag": {
                    "value": "dog",
                    "confidence": "0.74873686"
                },
                "box": {
                    "l": "0.16520005",
                    "t": "0.38729265",
                    "w": "0.26126826",
                    "h": "0.5455508"
                }
            }
        },
        {
            "type": "entity",
            "entity": {
                "tag": {
                    "value": "car",
                    "confidence": "0.3766461"
                },
                "box": {
                    "l": "0.5947845",
                    "t": "0.1360384",
                    "w": "0.32280278",
                    "h": "0.15790814"
                }
            }
        },
        {
            "type": "entity",
            "entity": {
                "tag": {
                    "value": "bicycle",
                    "confidence": "0.28965604"
                },
                "box": {
                    "l": "0.07310471",
                    "t": "0.14914557",
                    "w": "0.7133467",
                    "h": "0.7136574"
                }
            }
        }
    ]
}
```

## Acknowledgments

- [michhar/jetson-gpu-yolov4](https://github.com/michhar/jetson-gpu-yolov4)
