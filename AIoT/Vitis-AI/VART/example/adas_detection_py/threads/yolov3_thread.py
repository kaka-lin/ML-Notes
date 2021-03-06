import time
import threading

import numpy as np

from common import preprocess_one_image_fn, draw_outputs, load_classes, generate_colors
from yolo_utils import yolo_eval
from priority_queue import PriorityQueue


class YOLOv3Thread(threading.Thread):
    def __init__(self, runner: "Runner", deque_input, lock_input,
                 deque_output, lock_output, thread_name):
        super(YOLOv3Thread, self).__init__(name=thread_name)

        self.runner = runner
        self.deque_input = deque_input
        self.lock_input = lock_input
        self.deque_output = deque_output
        self.lock_output = lock_output

        self.class_names = load_classes('./model_data/adas_classes.txt')
        self.colors = generate_colors(self.class_names)

    def set_input_image(self, input_run, frame, size):
        w, h = size
        img = preprocess_one_image_fn(frame, w, h)
        input_run[0, ...] = img.reshape((h, w, 3))

    def run(self):
        # Get input/output tensors and dims
        inputTensors = self.runner.get_input_tensors()
        outputTensors = self.runner.get_output_tensors()
        input_ndim = tuple(inputTensors[0].dims)  # (1, 256, 512, 3)
        result0_ndim = tuple(outputTensors[0].dims)  # (1, 8, 16, 40)
        result1_ndim = tuple(outputTensors[1].dims)  # (1, 16, 32, 40)
        result2_ndim = tuple(outputTensors[2].dims)  # (1, 32, 64, 40)
        result3_ndim = tuple(outputTensors[3].dims)  # (1, 64, 126, 40)

        # input/output data define
        input_data = [np.empty(input_ndim, dtype=np.float32, order="C")]
        result0 = np.empty(result0_ndim, dtype=np.float32, order="C")
        result1 = np.empty(result1_ndim, dtype=np.float32, order="C")
        result2 = np.empty(result2_ndim, dtype=np.float32, order="C")
        result3 = np.empty(result3_ndim, dtype=np.float32, order="C")
        results = [result0, result1, result2, result3]

        # get input width, height for preprocess
        input_shape = (input_ndim[2], input_ndim[1])

        while True:
            self.lock_input.acquire()

            # empy
            if not self.deque_input:
                self.lock_input.release()
                continue
            else:
                # get input frame from input frames queue
                data_from_deque = self.deque_input[0]
                self.deque_input.popleft()
                self.lock_input.release()

            # Init input image to input buffers
            img = data_from_deque['img']
            idx = data_from_deque['idx']
            start_time = data_from_deque['time']
            self.set_input_image(input_data[0], img, input_shape)

            # invoke the running of DPU for yolov3
            """Benchmark DPU FPS performance over Vitis AI APIs `execute_async()` and `wait()`"""
            # (self: vart.Runner, arg0: List[buffer], arg1: List[buffer]) -> Tuple[int, int]
            job_id = self.runner.execute_async(input_data, results)
            self.runner.wait(job_id)

            self.post_process(img, results, input_shape)

            self.lock_output.acquire()
            img_info = PriorityQueue(idx, img, start_time)
            self.deque_output.append(img_info)
            self.deque_output.sort()
            self.lock_output.release()

    def post_process(self, image, results, input_ndim):
        """Xilinx ADAS detction model: YOLOv3

        Name: yolov3_adas_pruned_0_9
        Input shape: (256, 512, 3)
        Classe: 3
        Anchor: 5, for detail please see `yolo_utils.py`
        Outputs: 4
            outputs_node: {
                "layer81_conv",
                "layer93_conv",
                "layer105_conv",
                "layer117_conv",
            }
        """

        image_shape = (image.shape[1], image.shape[0]) # (w, h)
        scores, boxes, classes = yolo_eval(
            results,
            image_shape=image_shape,
            input_ndim=input_ndim,
            classes=3,
            score_threshold=0.5,
            iou_threshold=0.7)

        # print("detection:")
        # for i in range(scores.shape[0]):
        #     print("\t{}, {}, {}".format(
        #         self.class_names[int(classes[i])], scores[i], boxes[i]
        #     ))

        image = draw_outputs(image, (scores, boxes, classes),
                             self.class_names, self.colors)
