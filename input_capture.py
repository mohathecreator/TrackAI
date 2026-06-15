import keyboard
import json


class InputCapture:
    def __init__(self, keys=None):
        if keys is None:
            with open("config.json", "r") as f:
                config = json.load(f)
            self.keys = config["keys"]
        else:
            self.keys = keys
        self.inputs = {}

    def capture_input(self):
        self.inputs = {key: keyboard.is_pressed(key) for key in self.keys}

        return self.inputs
