from mss import mss
import cv2
import numpy as np


class FrameCapture:
    def __init__(self, monitor):
        if monitor is None:
            sct = mss()
            self.monitor = sct.monitors[1]
        else:
            self.monitor = monitor

    def capture_frame(self):
        sct = mss()
        # Takes screenshot and turns it into Numpy array
        img = np.array(sct.grab(self.monitor))
        img = cv2.resize(img, (1280, 720))
        return img
