# YOLOv3 OVMS Service with REST API

If you only want to run a object detection server with `OVMS`, you can refer to [AIoT/OpenVINO/model_server/example_client/object_detection/](https://github.com/kaka-lin/ML-Notes/tree/master/AIoT/OpenVINO/model_server/example_client/object_detection)

## Download Model

Download `yolo_v3.xml` and `yolo_v3.bin`, and then move them into `model/1` folder

- [yolo_v3.xml](https://drive.google.com/file/d/1d6S7e-7XCeuSapGe0QfmJ9ZKxKJgl5S-/view?usp=sharing)
- [yolo_v3.bin](https://drive.google.com/file/d/1hQMYSRs8HygTvkHLl0Fsrb3KNavF1wC-/view?usp=sharing)

## Run with docker-compose

```bash
$ docker-compose up -d
```

## Rebuild docker container

```bash
$ docker-compose build
```

## Stop docker-compose

```bash
$ docker-compose down
```

## Testing

After started service, you can open [http://127.0.0.1:5000](http://127.0.0.1:5000), you would see "Hello from Yolov3 inferencing based OVMS"

### "/score"

To get a list of detected objects, use the following command.

```bash
$ curl -X POST http://127.0.0.1:5000/score -H "Content-Type: image/jpeg" --data-binary @<full_path_to_image_file_in_jpeg>

# example with pretty-print JSON
$ curl -X POST http://127.0.0.1:5000/score -H "Content-Type: image/jpeg" --data-binary @images/dog.jpg | python -m json.tool
```

If successful, you will see JSON printed on your screen that looks something like this

```json
{
    "inferences": [
        {
            "entity": {
                "box": {
                    "h": "0.5559119",
                    "l": "0.14300361",
                    "t": "0.22801572",
                    "w": "0.5901649"
                },
                "tag": {
                    "confidence": "0.53395486",
                    "value": "bicycle"
                }
            },
            "type": "entity"
        },
        {
            "entity": {
                "box": {
                    "h": "0.1446465",
                    "l": "0.600061",
                    "t": "0.13548738",
                    "w": "0.2928388"
                },
                "tag": {
                    "confidence": "0.5238078",
                    "value": "truck"
                }
            },
            "type": "entity"
        },
        {
            "entity": {
                "box": {
                    "h": "0.546983",
                    "l": "0.15526156",
                    "t": "0.39115456",
                    "w": "0.25584072"
                },
                "tag": {
                    "confidence": "0.5342895",
                    "value": "dog"
                }
            },
            "type": "entity"
        }
    ]
}
```

## Acknowledgments

- [michhar/jetson-gpu-yolov4](https://github.com/michhar/jetson-gpu-yolov4)
