import cv2
import numpy as np
import torch
from mss import mss
from pynput.keyboard import Controller
import time

from model import TrackmaniaNet


device = torch.device("cpu")
model = TrackmaniaNet().to(device)
model.load_state_dict(torch.load("trackmania_model.pth"))
model.eval()

keyboard = Controller()
monitor = mss().monitors[1]

pressed_keys = {'w': False, 'a': False, 's': False, 'd': False}

print("Starting inference. Press Ctrl+C to stop")
time.sleep(2)

try:
    mss_instance = mss()
    while True:
        screenshot = np.array(mss_instance.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGB)
        frame = cv2.resize(frame, (640, 360))
        frame = frame.astype(np.float32) / 255.0
        frame = torch.from_numpy(frame).permute(2, 0, 1)
        frame = frame.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frame)
            predictions = torch.sigmoid(outputs).cpu().numpy()[0]

        actions = {'w': predictions[0], 'a': predictions[1],
                   's': predictions[2], 'd': predictions[3]}

        for key, pred in actions.items():
            should_press = pred > 0.5
            is_pressed = pressed_keys[key]

            if should_press and not is_pressed:
                keyboard.press(key)
                pressed_keys[key] = True
            elif not should_press and is_pressed:
                keyboard.release(key)
                pressed_keys[key] = False

            print(f"{key}: {pred:.2f}", end=" | ")
        print()

except KeyboardInterrupt:
    for key in ['w', 'a', 's', 'd']:
        if pressed_keys[key]:
            keyboard.release(key)
    print("\nInference stopped")
