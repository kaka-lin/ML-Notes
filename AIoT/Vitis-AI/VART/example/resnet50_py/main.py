import os
import sys
import time
import math
import threading
from ctypes import *
from typing import List

import cv2
import numpy as np
import xir
import vart

from utils import preprocess_one_image_fn
from resnet_thread import ResNetThread

global THREAD_NUM
THREAD_NUM = 1


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    """"Obtain DPU subgrah."""

    assert graph is not None, "'graph' should not be None."

    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."

    if root_subgraph.is_leaf:
        return []

    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0

    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def main(argv):
    global THREAD_NUM
    if len(argv) >= 2:
        THREAD_NUM = int(argv[1])
        if len(argv) >= 3:
            model_file = argv[2]

    image_dir = "./images/"
    model_file = "/usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel"

    # deserialized to the Graph object.
    g = xir.Graph.deserialize(model_file)

    # Get the subgraph that run in dpu
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel

    # Create DPU runner
    all_dpu_runners = []
    for i in range(int(THREAD_NUM)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input data
    list_images = os.listdir(image_dir)
    images = list(map(preprocess_one_image_fn,
                      [os.path.join(image_dir, image) for image in list_images]))

    """cnt variable

    The cnt variable is used to control the number of times a single-thread DPU runs.
    Users can modify the value according to actual needs. It is not recommended to use
    too small number when there are few input images, for example:

    1. If users can only provide very few images, e.g. only 1 image, they should set
         a relatively large number such as 360 to measure the average performance;
    2. If users provide a huge dataset, e.g. 50000 images in the directory, they can
         use the variable to control the test time, and no need to run the whole dataset.
    """
    cnt = 360
    threads = []
    time_start = time.time()
    for i in range(THREAD_NUM):
        threads.append(ResNetThread(
            all_dpu_runners[i], images, cnt, f"thread_{i}"))
        threads[i].start()

    for thread in threads:
        thread.join()

    del all_dpu_runners

    time_end = time.time()
    time_total = time_end - time_start
    total_frames = cnt * THREAD_NUM
    fps = float(total_frames / time_total)
    print(f"FPS={fps:.2f}, \
            total frames={total_frames:.2f}, \
            time={time_total:.6f} seconds")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        thread_num = 1
        argv = []

        if len(sys.argv) == 2:
            thread_num = sys.argv[1]
            argv = sys.argv

        print("usage : python3 main.py <thread_number> <resnet50_xmodel_file>")
        print(f"use case: \
                    \n\tthread_number={thread_num}, \
                    \n\tresnet50_xmodel_file=/usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel")
        main(argv)
    else:
        print(f"use case: \
                    \n\tthread_number={sys.argv[1]}, \
                    \n\tresnet50_xmodel_file={sys.argv[2]}")
        main(sys.argv)
