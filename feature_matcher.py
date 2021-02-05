# Model to find a pre-set image/object as mouse (ex. smiley face, cross, doted image in finter/head)
import cv2 as cv
import numpy as np
from collections import deque

class FeatureMatcher:
    def __init__(self, maxlen=10):
        self.buffer = deque(maxlen=maxlen)
        # init cursor as pre set feature to any image that wants to be tracked or used as cursor
        self.feature = cv.cvtColor(cv.imread('cursor.jpg'), cv.COLOR_BGR2GRAY)

    # Brute-Force matching with ORB descriptors
    def brute_force_orb_match(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(self.feature, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x:x.distance)
        # use top 2 matches
        cursor_matches = cv.drawMatches(self.feature, kp1, gray, kp2, matches[:2], None, flags=2)
        return cursor_matches

    # Brute-Fore mathing with SIFT(Scale Invariant Feature Trans)form descriptors and Ratio Test
    def brute_force_sift_match(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.feature, None)
        kp2, des2 = sift.detectAndCompute(gray, None)
        bf = cv.BFMatcher()
        # use top 2 k matches
        matches = bf.knnMatch(des1, des2, k=2)
        # apply ratio test
        best_matches = []
        for match1, match2 in matches:
            # if the distance of m1 is less than 75% of m2 its a good match, may vary
            # Less distance == better match
            if match1.distance < 0.75*match2.distance:
                best_matches.append([match1])
        sift_matches = cv.drawMatchesKnn(self.feature, kp1, gray, kp2, best_matches, None, flags=2)
        return sift_matches 

    # FLANN (Fast library for aprox nearest neighbors) faster than brute force but finds general instead of more accurate as bf
    # better used for large images
    def flann_match(self, frame):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.feature, None)
        kp2, des2 = sift.detectAndCompute(gray, None)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        matches_mask = [[0,0] for i in range(len(matches))]
        best_matches = []
        for i, (match1, match2) in enumerate(matches):
            if match1.distance < 0.7*match2.distance:
                # label lines 
                matches_mask[i] = [1,0]

        draw_params = dict(matchColor=(0,255,0),
            singlePointColor=(255,0,0),
            matchesMask=matches_mask,
            flags=2)

        flann_matches = cv.drawMatchesKnn(self.feature, kp1, gray, kp2, matches, None, **draw_params)
        return flann_matches
