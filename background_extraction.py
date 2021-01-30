import cv2 as cv
import numpy as np
from collections import deque

class BackgroundExtraction:
    def __init__(self, width, height, scale, maxlen=10):
        self.maxlen = maxlen
        self.scale = scale
        self.width = width//scale
        self.height = height//scale
        self.buffer = deque(maxlen=maxlen)
        self.background = None

    def calculate_background(self):
        self.background = np.zeros((self.height, self.width), dtype='float32')
        for item in self.buffer:
            self.background += item
        self.background /= len(self.buffer)

    def update_background(self, old_frame, new_frame):
        self.background -= old_frame/self.maxlen
        self.background += new_frame/self.maxlen

    def update_frame(self, frame):
        if len(self.buffer) < self.maxlen:
            self.buffer.append(frame)
            self.calculate_background()
        else:
            old_frame = self.buffer.popleft()
            self.buffer.append(frame)
            self.update_background(old_frame, frame)

    def get_background(self):
        return self.background.astype('uint8')

    def apply(self, frame):
        down_scale = cv.resize(frame, (self.width, self.height))
        gray = cv.cvtColor(down_scale, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        self.update_frame(gray)
        abs_diff = cv.absdiff(self.get_background(), gray)
        _, ad_mask = cv.threshold(abs_diff, 15, 255, cv.THRESH_BINARY)
        return cv.resize(ad_mask, (self.width*self.scale, self.height*self.scale))