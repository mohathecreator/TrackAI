import pandas as pd
import cv2
from pathlib import Path


class DataRecorder:
    def __init__(self):
        self.data_dir = Path("dataset")
        self.image_dir = self.data_dir / "images"
        self.counter = 1
        self.data = {}

    def record(self, frame, inputs):
        frame_path = str(self.image_dir / f"frame_{self.counter:03d}.png")
        cv2.imwrite(frame_path, frame)
        self.data.update({frame_path: inputs})
        self.counter += 1

    def save_csv(self):
        df = pd.DataFrame(list(self.data.items()),
                          columns=["image_name", "inputs"])
        inputs_path = str(self.data_dir / "inputs.csv")
        df.to_csv(inputs_path)
