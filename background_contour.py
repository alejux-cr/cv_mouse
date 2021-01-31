import cv2 as cv
import numpy as np
from collections import deque

class BackgroundContour:
    def __init__(self, maxlen=10):
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.background = None

    def apply(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 127, 255, 0)
        contours, hierarchy =  cv.findContours(thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        external_contours = np.zeros(gray.shape)
        internal_contours = np.zeros(gray.shape)
        for i in range(len(contours)):
            #External contour
            if hierarchy[0][i][3] == -1:
                cv.drawContours(external_contours, contours, i, 255, -1)
            else:
                cv.drawContours(internal_contours, contours, i, 255, -1)

                
        return external_contours, internal_contours
