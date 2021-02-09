import cv2 as cv
import numpy as np
import pdb

class Tracker:
    def __init__(self, frame):
        # params for pyramid Lucas-Kanade method
        self.corner_track_params = dict(maxCorners = 10, qualityLevel= 0.4, minDistance = 1, blockSize = 7)
        # max level pyramid size, the more levels, the smaller the resolution
        # criteria param adjust
        self.lk_params = dict(winSize = (200, 200), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Points to track
        self.prev_points = cv.goodFeaturesToTrack(self.prev_gray_frame, mask=None, **self.corner_track_params)
        self.hsv_mask = np.zeros_like(frame)
        self.hsv_mask[:,:,1] = 255 # fully saturated

        self.face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

        # For api tracker only
        self.roi = cv.selectROI(frame, False)
        self.api_tracker = None

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

    def cam_shift_track(self, frame):
        face_rects = self.face_cascade.detectMultiScale(frame)

        if len(face_rects) > 0:

            (face_x, face_y, w, h) = tuple(face_rects[0])
            track_window = (face_x, face_y, w, h)

            roi = frame[face_y:face_y+h, face_x:face_x+w]

            hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

            roi_hist = cv.calcHist([hsv_roi], [0], None, [180], [0, 180])

            cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

            term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
            
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            dst = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # using MeanShift
            #ret, track_window = cv.meanShift(dst, track_window, term_criteria)

            #x, y, w, h = track_window

            #img2 = cv.rectangle(frame, (x, y), (x+w, y+h), (0,0,255),5)
            #####

            # using CamShift to improve rectangle size

            ret, track_window = cv.CamShift(dst, track_window, term_criteria)

            pts = cv.boxPoints(ret)
            pts = np.int0(pts)
            img2 = cv.polylines(frame, [pts], True, (0,0,255), 5)
        else:
            img2 = frame

        return img2

    def api_tracker_create(self, type, frame):
        types = {
            1: cv.legacy.TrackerBoosting_create(),
            2: cv.TrackerMIL_create(),
            3: cv.TrackerKCF_create(),
            4: cv.legacy.TrackerTLD_create(),
            5: cv.legacy.TrackerMedianFlow_create()
        }
        self.api_tracker = types.get(type, "Invalid API Tracker")
        ret = self.api_tracker.init(frame, self.roi)      

    def api_tracker_update(self, frame):
        
        success, self.roi = self.api_tracker.update(frame)
        
        # roi var is a tuple of 4 floats
        # we need each value as integers
        (x,y,w,h) = tuple(map(int,self.roi))

        if success:
            p1 = (x, y)
            p2 = (x+w, y+h)
            cv.rectangle(frame, p1, p2, (0,255,0), 3)
        else:
            cv.putText(frame, "Fail to detect tracking", (100,200), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)


        
