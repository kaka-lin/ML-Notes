import json


def create_config():
    config_file = "config.json"
    config = {
        "model_config_list": [
            {
                "config": {
                    "name": "face_detection",
                    "base_path": "/workspace/face-detection-retail-0004/",
                    "shape": "(1,3,400,600)",
                    "layout": "NHWC"
                }
            },
        ]
    }

    configObj = json.dumps(config)
    with open(config_file, 'w') as f:
        f.write(configObj)


if __name__ == "__main__":
    create_config()
