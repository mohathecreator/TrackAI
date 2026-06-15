import pandas as pd
import cv2
from pathlib import Path
from datetime import datetime


class DataRecorder:
    def __init__(self):
        self.data_dir = Path("dataset")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = self.data_dir / timestamp
        self.image_dir = self.session_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
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
        inputs_path = str(self.session_dir / "inputs.csv")
        df.to_csv(inputs_path)
