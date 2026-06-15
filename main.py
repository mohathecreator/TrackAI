from frame_capture import FrameCapture
from input_capture import InputCapture
from data_recorder import DataRecorder
import keyboard
import json


frame_capturing = FrameCapture(None)
input_capturing = InputCapture()
data_recording = DataRecorder()

with open("config.json", "r") as f:
    config = json.load(f)
recording_trigger = config["recording_trigger"]

while True:
    frame = frame_capturing.capture_frame()
    inputs = input_capturing.capture_input()

    if keyboard.is_pressed(recording_trigger):
        data_recording.record(frame, inputs)

    if keyboard.is_pressed("esc"):
        print("Finished recording")
        break

data_recording.save_csv()
