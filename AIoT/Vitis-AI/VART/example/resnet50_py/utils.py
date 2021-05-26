import os
import math

import cv2
import numpy as np

_B_MEAN = 104.0
_G_MEAN = 107.0
_R_MEAN = 123.0
MEANS = [_B_MEAN, _G_MEAN, _R_MEAN]
SCALES = [1.0, 1.0, 1.0]


def preprocess_one_image_fn(image_path, width=224, height=224):
    """pre-process for resnet50 (caffe)"""
    means = MEANS
    scales = SCALES
    image = cv2.imread(image_path)
    image = cv2.resize(image, (width, height))
    B, G, R = cv2.split(image)
    B = (B - means[0]) * scales[0]
    G = (G - means[1]) * scales[1]
    R = (R - means[2]) * scales[2]
    return image


def CPUCalcSoftmax(data, size):
    sum = 0.0
    result = [0 for i in range(size)]
    for i in range(size):
        result[i] = math.exp(data[i])
        sum += result[i]
    for i in range(size):
        result[i] /= sum
    return result


def TopK(data, size, k=5, file_path="./model_data/imagenet1000.txt"):
    """Get Topk results

    Get topk results according to its probability

    Parameters:
      data: data result of softmax
      size: size of input data
      k:    calculation result (TopK)
      file_path: records the infotmation of kinds (class)
    """

    cnt = [i for i in range(size)]
    pair = zip(data, cnt)
    pair = sorted(pair, reverse=True)
    softmax_new, cnt_new = zip(*pair)

    with open(file_path, "r") as fp:
        classes = fp.readlines()

        for i in range(k):
            print("Top[{}] prob = {:.8f} name = {}".format(
                i, softmax_new[i], classes[cnt_new[i]].strip()))


if __name__ == "__main__":
    image_dir = "./images/"
    # input data
    list_images = os.listdir(image_dir)
    images = list(map(preprocess_one_image_fn,
                      [os.path.join(image_dir, image) for image in list_images]))
    images = np.array(images)
    n_of_images = len(images)

    input_ndim = (1, 224, 224, 3)  # (n, 224, 224, 3)
    output_ndim = (1, 1000)
    inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
    outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

    # Init input image to input buffers
    count = 0
    for j in range(1):
        imageRun = inputData[0]
        # example:
        #   count: 0, n_of_images: 10, batch: 2
        #   count += batch_size
        #   count 0: (0, 1) % 10 -> (0, 1)
        #   count 2: (2, 3) % 10 -> (2, 3)
        imageRun[j, ...] = images[(count + j) %
                                  n_of_images].reshape(input_ndim[1:])

    imageRun = np.array(imageRun)
    print(imageRun.shape)
