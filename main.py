import cv2 as cv
import numpy as np
from collections import deque
from background_extraction import BackgroundExtraction
from background_contour import BackgroundContour

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

    while True:
        # Reading, resizing, and flipping the frame
        _, frame = cap.read()
        frame = cv.resize(frame, (width, height))
        frame = cv.flip(frame, 1)

        # Processing the frame
        fg_mask = bg_buffer.apply(frame)

        # Process contours
        external_contour, internal_contours = contour_extractor.apply(frame)

        cv.imshow("External contours", external_contour)
        cv.imshow("Internal contours", internal_contours)
        
        cv.imshow("FG Mask", fg_mask)
        cv.imshow("Headmouse", frame)

        key = cv.waitKey(1)
        if  key == ord('q'):
            cap.release()
            break
        elif key == ord('s'): # Same image of desired cursor (ex. something simple)
            cv.imwrite(filename='saved_img.jpg', img=frame)
            img_new = cv.imread('saved_img.jpg', cv.IMREAD_GRAYSCALE)
            img_new = cv.imshow("Captured Image", img_new)


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
    
