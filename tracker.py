import cv2 as cv
import numpy as np

class Tracker:
    def __init__(self, prev_frame):
        # params for pyramid Lucas-Kanade method
        self.corner_track_params = dict(maxCorners = 10, qualityLevel= 0.4, minDistance = 1, blockSize = 7)
        # max level pyramid size, the more levels, the smaller the resolution
        # criteria param adjust
        self.lk_params = dict(winSize = (200, 200), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray_frame = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        # Points to track
        self.prev_points = cv.goodFeaturesToTrack(self.prev_gray_frame, mask=None, **self.corner_track_params)
        self.hsv_mask = np.zeros_like(prev_frame)
        self.hsv_mask[:,:,1] = 255 # fully saturated

    def lucas_kanade_track(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        mask = np.zeros_like(frame)
        next_points, status, err = cv.calcOpticalFlowPyrLK(self.prev_gray_frame, frame_gray, self.prev_points, None, **self.lk_params)
        good_new = next_points[status == 1]
        good_prev = self.prev_points[status == 1]

        for i, (new, prev) in enumerate(zip(good_new, good_prev)):
            # ravel method = reshape order [[1,2,3], [4,5,6]].ravel() = [1,2,3,4,5,6]
            x_new, y_new = new.ravel()
            x_prev, y_prev = prev.ravel()

            mask = cv.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 3)

            frame = cv.circle(frame, (x_new, y_new), 8, (0,0,255), -1)

        img = cv.add(frame, mask)
        self.prev_gray_frame = frame_gray.copy()
        self.prev_points = good_new.reshape(-1, 1, 2)
        return img

    # Better used to track the whole movement of the frame
    def gunner_farneback_track(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(self.prev_gray_frame, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[:,:,0], flow[:,:,1], angleInDegrees=True)
        self.hsv_mask[:,:,0] = ang/2
        self.hsv_mask[:,:,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(self.hsv_mask, cv.COLOR_HSV2BGR)
        self.prev_gray_frame = frame_gray.copy()
        return bgr

