import keyboard


class InputCapture:
    def __init__(self, keys):
        if keys is None:
            self.keys = ["left", "right", "up", "down"]
        else:
            self.keys = keys
        self.inputs = {}

    def capture_input(self):
        self.inputs = {key: keyboard.is_pressed(key) for key in self.keys}

        return self.inputs
