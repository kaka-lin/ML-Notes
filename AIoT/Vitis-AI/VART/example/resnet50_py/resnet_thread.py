import time
import threading

import numpy as np

from utils import CPUCalcSoftmax, TopK


class ResNetThread(threading.Thread):
    def __init__(self, runner: "Runner", img, cnt, thread_name):
        super(ResNetThread, self).__init__(name=thread_name)

        self.runner = runner
        self.img = img
        self.cnt = cnt

    def run(self):
        """重寫父類run方法，在執行緒啟動後執行該方法內的程式"""

        # Get input/output tensors and dims
        inputTensors = self.runner.get_input_tensors()
        outputTensors = self.runner.get_output_tensors()
        input_ndim = tuple(inputTensors[0].dims) # (n, 224, 224, 3)
        output_ndim = tuple(outputTensors[0].dims) # (n, 1000)

        # resnet50: 1000 classes
        batch_size = input_ndim[0] # 1
        classes = outputTensors[0].get_data_size() # 1000
        pre_output_size = int(classes / batch_size)
        n_of_images = len(self.img)

        count = 0
        while count < self.cnt:
            # Prepare batch input/output numpy array
            inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
            outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

            # Init input image to input buffers
            for i in range(batch_size):
                imageRun = inputData[0]
                # example:
                #   count: 0, n_of_images: 10, batch: 2
                #   count += batch_size
                #   count 0: (0, 1) % 10 -> (0, 1)
                #   count 2: (2, 3) % 10 -> (2, 3)
                imageRun[i, ...] = self.img[(count + i) % n_of_images].reshape(input_ndim[1:])

            # Run with batch
            """Benchmark DPU FPS performance over Vitis AI APIs `execute_async()` and `wait()`"""
            job_id = self.runner.execute_async(inputData, outputData)
            self.runner.wait(job_id)

            # Softmax & TopK calculate with batch
            for i in range(batch_size):
                # Calculate softmax on CPU
                # and display TOP-5 classification results
                softmax = CPUCalcSoftmax(outputData[0][i], pre_output_size)
                TopK(softmax, pre_output_size, 5, "./model_data/imagenet1000.txt")

            count = count + batch_size

