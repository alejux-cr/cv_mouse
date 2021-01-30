import pyautogui

class Cursor:
    def __init__(self):
        self.screenWidth, self.screenHeight = pyautogui.size()
        self.currentMouseX, self.currentMouseY = pyautogui.position()
        self.speed = 10

    def update_frame(self, frame):
        # this method can be used to set a mark of the tracked movement in the frame (finger, hand)
        return True

    def update_position(self, fg_mask):
        return True

    def on_click:
        pyautogui.click()