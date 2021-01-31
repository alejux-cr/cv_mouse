# Model to find a pre-set image/object as mouse (ex. smiley face, cross, doted image in finter/head)
import cv2 as cv
import numpy as np
from collections import deque

def __init__(self, maxlen=10, feature):
    self.buffer = deque(maxlen=maxlen)
    self.feature = feature

def match(self):

