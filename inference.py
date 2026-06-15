import cv2
import numpy as np
import torch
from mss import mss
from pynput.keyboard import Controller, Key
import time

from model import TrackmaniaNet


device = torch.device("cpu")
model = TrackmaniaNet().to(device)
model._load_state_dict(torch.load("trackmania_model.pth"))
model.eval()

keyboard = Controller()
monitor = mss().monitors[1]

key_map = {'w': 'w', 'a': 'a', 's': 's', 'd': 'd'}
pressed_keys = {'w': False, 'a': False, 's': False, 'd': False}

print("Starting inference. Press ESC to stop")
time.sleep(2)
