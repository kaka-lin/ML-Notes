import argparse
import datetime

import cv2
import grpc
import numpy as np

from tensorflow import make_tensor_proto, make_ndarray
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import classes


def image_process(image_name):
    image = cv2.imread(image_name).astype(np.float32)
    image = cv2.resize(image, (224, 224))

    # switch from HWC to CHW
    # and reshape to (1, 3, size, size)
    # for model input requirements
    image = image.transpose(2, 0, 1).reshape(1, 3, 224, 224)

    return image


def run():
    model_name = "resnet18"
    input_layer = "input.1"
    output_layer = "486"
    input_lists = "input_images.txt"

    matched = 0
    processing_times = np.zeros((0), int)

    with open(input_lists, 'r') as f:
        lines = f.readlines()

    print("Start processing:")
    print(f"\tModel name: {model_name}")
    print(f"\tImages list file: {input_lists}")

    with grpc.insecure_channel('localhost:9000') as channel:
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        for idx, line in enumerate(lines):
            path, label = line.strip().split(" ")
            image = image_process(path)

            request = predict_pb2.PredictRequest()
            request.model_spec.name = model_name
            request.inputs[input_layer].CopyFrom(make_tensor_proto(image, shape=(image.shape)))
            start_time = datetime.datetime.now()
            result = stub.Predict(request, 10.0) # result includes a dictionary with all model outputs
            end_time = datetime.datetime.now()

            if output_layer not in result.outputs:
                print("Invalid output name", output_layer)
                print("Available outputs:")
                for Y in result.outputs:
                    print(Y)
                exit(1)

            duration = (end_time - start_time).total_seconds() * 1000
            processing_times = np.append(processing_times, np.array([int(duration)]))
            output = make_ndarray(result.outputs[output_layer])
            nu = np.array(output)

            # for object classification models show imagenet class
            print('Processing time: {:.2f} ms; speed {:.2f} fps'.format(round(duration, 2), round(1000 / duration, 2)))
            ma = np.argmax(nu)
            mark_message = ""
            if int(label) == ma:
                matched += 1
                mark_message = "; Correct match."
            else:
                mark_message = "; Incorrect match. Should be {} {}".format(label, classes.imagenet_classes[int(label)])

            print("\t", idx, classes.imagenet_classes[ma], ma, mark_message)

    latency = np.average(processing_times)
    accuracy = matched / len(lines)

    print("Overall accuracy=", accuracy*100, "%")
    print("Average latency=", latency, "ms")


if __name__ == "__main__":
    run()
