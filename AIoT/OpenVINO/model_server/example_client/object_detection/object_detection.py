import os
import datetime

import cv2
import grpc
import numpy as np
from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from common import load_classes, generate_colors, draw_outputs
from yolo_utils import yolo_eval


def image_process(image_name):
    image = cv2.imread(image_name).astype(np.float32)
    image = cv2.resize(image, (416, 416))

    # switch from HWC to CHW
    # and reshape to (1, 3, size, size)
    # for model input requirements
    image = image.transpose(2, 0, 1).reshape(1, 3, 416, 416)

    return image


def run():
    model_name = "yolov3"
    input_layer = "inputs"
    input_lists = "input_images.txt"
    output_layers = [
        "detector/yolo-v3/Conv_14/BiasAdd/YoloRegion",
        "detector/yolo-v3/Conv_22/BiasAdd/YoloRegion",
        "detector/yolo-v3/Conv_6/BiasAdd/YoloRegion"
    ]
    class_names = load_classes("model_data/coco.names")
    output_folder = "./output"

    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)

    with open(input_lists, 'r') as f:
        lines = f.readlines()

    # output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("Start processing:")
    print(f"\tModel name: {model_name}")
    print(f"\tImages list file: {input_lists}")

    with grpc.insecure_channel('localhost:9000') as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        for idx, line in enumerate(lines):
            path = line.strip()
            image_name = path.split("/")[1][:-4].strip()
            image = image_process(path)

            request = predict_pb2.PredictRequest()
            request.model_spec.name = model_name
            request.inputs[input_layer].CopyFrom(
                make_tensor_proto(image, shape=(image.shape)))
            start_time = datetime.datetime.now()
            # result includes a dictionary with all model outputs
            result = stub.Predict(request, 10.0)
            end_time = datetime.datetime.now()

            print("Available outputs:")
            for output_layer in result.outputs:
                print(output_layer)

            yolo_outputs = [[], [], []]
            for output_layer in output_layers:
                output = make_ndarray(result.outputs[output_layer])
                output_numpy = np.array(output)
                anchor_size = output_numpy.shape[2]
                output_numpy = output_numpy.transpose(0, 2, 3, 1).reshape(
                    1, anchor_size, anchor_size, 3, 85)
                yolo_outputs[int((anchor_size / 13) / 2)] = output_numpy

            scores, boxes, classes = yolo_eval(
                yolo_outputs,
                classes=80,
                score_threshold=0.5,
                iou_threshold=0.3
            )

            # Draw bounding boxes on the image file
            image = cv2.imread(path)
            image = draw_outputs(
                image, (scores, boxes, classes), class_names, colors)

            # Save
            image_saved = f"{output_folder}/{image_name}_out.jpg"
            print(f"Saving image to {image_saved}")
            cv2.imwrite(f"{image_saved}", image)


if __name__ == "__main__":
    run()
