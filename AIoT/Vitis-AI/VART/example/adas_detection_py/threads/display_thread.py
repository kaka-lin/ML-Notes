import time
import threading

import cv2
import numpy as np


class DisplayThread(threading.Thread):
    def __init__(self, deque_output, lock_output, thread_name):
        super(DisplayThread, self).__init__(name=thread_name)

        self.deque_output = deque_output
        self.lock_output = lock_output
        self.running = False

    def run(self):
        self.running = True
        self.output_img_idx = 1

        while self.running:
            self.lock_output.acquire()

            # empty
            if not self.deque_output:
                self.lock_output.release()
                time.sleep(10/1000000)
            elif self.output_img_idx == self.deque_output[0]['idx']:
                img = self.deque_output[0]['img']
                start_time = self.deque_output[0]['time']

                fps = "Fps: {:.2f}".format(1 / (time.time() - start_time))
                cv2.putText(img, fps, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("Object Detection@Xilinx DPU", img)
                self.output_img_idx += 1
                self.deque_output.pop(0)
                self.lock_output.release()

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                self.lock_output.release()

        cv2.destroyAllWindows()
