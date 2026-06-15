from mss import mss
import cv2
import numpy as np


class FrameCapture:
    def __init__(self, monitor):
        if monitor is None:
            self.monitor = {"top": 0,  # Distance from upper screen border
                            "left": 0,  # Distance from left screen border
                            "width": 1280,
                            "height": 720
                            }
        else:
            self.monitor = monitor

    def capture_frame(self):
        # Create mss object by using sct (Screen Capture Task) as variable
        sct = mss()

        # Takes screenshot and turns it into Numpy array
        img = np.array(sct.grab(self.monitor))
        cv2.imshow("Trackmania", img)  # Shows taken screenshot

        return img


class FrameViewer:
    pass
