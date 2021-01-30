import cv2 as cv
import numpy as np
from collections import deque
from background_extraction import BackgroundExtraction

def main():
    width = 640
    height = 480
    scale = 2

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    bg_buffer = BackgroundExtraction(width, height, scale, maxlen=5)

    while True:
        # Reading, resizing, and flipping the frame
        _, frame = cap.read()
        frame = cv.resize(frame, (width, height))
        frame = cv.flip(frame, 1)

        # Processing the frame
        fg_mask = bg_buffer.apply(frame)

        cv.imshow("FG Mask", fg_mask)
        cv.imshow("Headmouse", frame)

        if cv.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
    cv.destroyAllWindows()
    
