from pathlib import Path
import pandas as pd
import cv2
import ast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
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

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index], self.labels[index]


dataset = TrackmaniaDataset("dataset")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrackmaniaNet().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (batch_frames, batch_labels) in enumerate(dataloader):
        batch_frames = batch_frames.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_frames)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, "
                  f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "trackmania_model.pth")
print("Model saved!")
