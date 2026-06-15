from pathlib import Path
import pandas as pd
import cv2
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

from model import TrackmaniaNet


class TrackmaniaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        self.frames = []
        self.labels = []

        for session_path in Path(dataset_dir).glob("track-*"):
            csv_path = session_path / "inputs.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            images_dir = session_path / "images"

            for idx, row in df.iterrows():
                image_name = Path(row["image_name"]).name
                inputs_str = row["inputs"]

                image_path = images_dir / image_name
                if not image_path.exists():
                    continue

                inputs_dict = ast.literal_eval(inputs_str)
                frame = cv2.imread(str(image_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frame = torch.from_numpy(frame).permute(2, 0, 1)

                label = torch.tensor([
                                    float(inputs_dict['w']),
                                    float(inputs_dict['a']),
                                    float(inputs_dict['s']),
                                    float(inputs_dict['d'])],
                                    dtype=torch.float32)

                self.frames.append(frame)
                self.labels.append(label)
