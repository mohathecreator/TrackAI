from frame_capture import FrameCapture
from input_capture import InputCapture
from data_recorder import DataRecorder
import keyboard


frame_capturing = FrameCapture(None)
input_capturing = InputCapture(None)
data_recording = DataRecorder()

while True:
    frame = frame_capturing.capture_frame()
    inputs = input_capturing.capture_input()
    data_recording.record(frame, inputs)

    if keyboard.is_pressed("esc"):
        print("Finished recording")
        break

data_recording.save_csv()
