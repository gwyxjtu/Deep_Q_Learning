import numpy as np
import cv2


class FrameProcess:

    def __init__(self, frame_height=84, frame_width=84):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def process(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(
            frame, (self.frame_height, self.frame_width), interpolation=cv2.INTER_NEAREST)
        # 为了迎合TensorFlow的输入
        return frame[:, :, np.newaxis]
