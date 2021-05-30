import os
import sys
import time
import math
import threading
from ctypes import *
from typing import List
from collections import deque

import cv2
import numpy as np
import xir
import vart

from common import get_args
from threads import VideoThread, DisplayThread, YOLOv3Thread


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


def main(args):
    image_dir = "./images/"
    model_file = args.model_file

    # xmodel deserialized to the Graph object.
    graph = xir.Graph.deserialize(model_file)

    # Get the subgraph that run in dpu
    subgraphs = get_child_subgraph_dpu(graph)
    # yolov3 should have one and only one dpu subgraph.";
    assert len(subgraphs) == 1  # only one DPU kernel
    print(f"create running for subgraph: {subgraphs[0].get_name()}")

    """Create 4 DPU Tasks for YOLO-v3 network model

    Spawn 6 threads:
      - 1 thread for reading video frame
      - 4 identical threads for running YOLO-v3 network model
      - 1 thread for displaying frame in monitor
    """
    # Create 4 DPU runner
    all_dpu_runners = []
    for i in range(args.yolo_runner):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # Create 6 threads and deque/lock for data flow
    threads = []
    deque_input = deque()
    deque_output = deque()
    lock_input = threading.Lock()
    lock_output = threading.Lock()

    # Video and Display thread
    # args.video_file
    video_thread = VideoThread(args.video_file, deque_input, lock_input, 'video_thread')
    display_thread = DisplayThread(deque_output, lock_output, 'display_thread')
    threads.append(display_thread)
    threads.append(video_thread)

    # Yolo Thread
    for i in range(args.yolo_runner):
        threads.append(YOLOv3Thread(
            all_dpu_runners[i],
            deque_input, lock_input,
            deque_output, lock_output,
            f"yolo_thread_{i}"))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    args = get_args().parse_args()
    main(args)
