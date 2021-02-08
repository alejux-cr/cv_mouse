import cv2 as cv
import numpy as np
from collections import deque
from background_extraction import BackgroundExtraction
from background_contour import BackgroundContour
from feature_matcher import FeatureMatcher
from detector import ObjectDetector
from tracker import Tracker

def main():

    cap = cv.VideoCapture(0)
    #width = 640
    #height = 480
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    scale = 2

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    bg_buffer = BackgroundExtraction(width, height, scale, maxlen=5)
    contour_extractor = BackgroundContour()
    matcher = FeatureMatcher()
    detector = ObjectDetector()

    # Read frame here too for tracker init ONLY 
    _, frame = cap.read()
    tracker = Tracker(frame)

    while True:
        # Reading, resizing, and flipping the frame
        _, frame = cap.read()
        frame = cv.resize(frame, (width, height))
        frame = cv.flip(frame, 1)

        # Processing the frame
        fg_mask = bg_buffer.apply(frame)

        # Process contours
        external_contour, internal_contours = contour_extractor.apply(frame)

        #cv.imshow("External contours", external_contour)
        #cv.imshow("Internal contours", internal_contours)

        # Match the set cursor in the frame with different methods
        # bf_matches = matcher.brute_force_orb_match(frame)
        # cv.imshow("BF ORB Matches", bf_matches)

        #sift_matches = matcher.brute_force_sift_match(frame)
        #cv.imshow("BF SIFT Matches", sift_matches)
        
        #flann_matches = matcher.flann_match(frame)
        # cv.imshow("FLANN Matches", flann_matches)

        #detector.detect(frame)
        #cv.imshow("FG Mask", fg_mask)

        #img = tracker.lucas_kanade_track(frame)
        #cv.imshow("Lucas-Kanade tracker", img)
        img = tracker.gunner_farneback_track(frame)
        cv.imshow("Gunner-Farneback tracker", img)
        cv.imshow("Headmouse", frame)

        key = cv.waitKey(1)
        if  key == ord('q'):
            cap.release()
            break
        elif key == ord('s'): # Same image of desired cursor (ex. something simple)
            cv.imwrite(filename='cursor.jpg', img=frame)
            img_new = cv.imread('cursor.jpg', cv.IMREAD_GRAYSCALE)
            img_new = cv.imshow("Captured Image", img_new)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
    
