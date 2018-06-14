# author: Robert Krook (guskrooro@student.gu.se)
# You may use this code for educational purposes if you keep these three lines
# as they are currently.

import numpy as np
import scipy.misc
from imageproc import imageproc
from util import util
import cv2 as cv
from sklearn import linear_model

# only want to pick up lines from a certain region of the videostream
def setROI(img_full):
    mask = np.zeros(img_full.shape, dtype=np.uint8)
    roi_corners = np.array([[(0,360),
                             (0,230), 
                             (340,180),
                             (360,180), 
                             (640,240), 
                             (640,360)]], dtype=np.int32)
    channel_count = img_full.shape[2]
    ignore_mask_color =(255,)*channel_count
    cv.fillPoly(mask, roi_corners, ignore_mask_color)
    img_roi = cv.bitwise_and(img_full, mask)
    return img_roi

# filters out lines that don't match the slope_treshold, and
# splits them up in respective x and y lists.
def handleLines(lines, lower_slope_threshold, upper_slope_threshold):
    left_x  = []
    left_y  = []
    right_x = []
    right_y = []
    for line in lines:
        x1,y1,x2,y2 = line
        coef = float(x1-x2)/(y1-y2)

        if (abs(coef) >= abs(lower_slope_threshold) and 
            abs(coef) <= abs(upper_slope_threshold)):
            if coef < 0:
                left_x.append(x1)
                left_x.append(x2)
                left_y.append(y1)
                left_y.append(y2)
            else:
                right_x.append(x1)
                right_x.append(x2)
                right_y.append(y1)
                right_y.append(y2)
    return left_x, left_y, right_x, right_y

# create instances to the python library that calls futhark
ip = imageproc()
ut = util()

cap = cv.VideoCapture('green.mp4')
if cap.isOpened() == False:
    print('error opening video')

# if you don't like ugly mega while-loops, this is where i advice
# you to look at something else. This is not my proudest moment.
# viewer discretion is adviced
while(cap.isOpened()):
    ret,frame = cap.read()
    # make a copy to draw on
    final = np.array(frame)

    if ret == True:
        # set region of interest
        roi       = setROI(frame)

        # grayscale the image
        gray     = ut.grayscale(np.int32(roi)).get()
        
        # equalize the histogram. As calculating a histogram in a functional
        # setting with immutable data is fruitless, this is done in openCV.
        histEq   = cv.equalizeHist(np.array(gray, dtype=np.uint8))

        # binarize the image, keeping only those pixels that have a value
        # greater than 240. You might need to tweak this depending on what
        # setting your image is captured in.
        binarized = ut.binarization(np.int32(histEq), 240).get()

        # Now that hopefully only the lanes are left, perform canny edge
        # detection to keep only the edges of the lanes. Change the first
        # argument (std_dev) to change how well the image is blurred before
        # edges are extracted. The following two arguments are the lower and
        # upper threshold for the double thresholding. Should typically be 1:2
        # format, and might also need to be changed to suit your image.
        # Without all the preprocessing i do here, i need as low as 5 10 to get
        # good lines.
        edge      = ip.canny_edge_detection(binarized, 5, 20, 40).get()

        # The undesired data is (hopefully) gone from the image, now perform
        # the probabilistic hough transform to extract the data.  This is also
        # done in openCV as this would probably also be fruitless in a 
        # functional setting. The blob extraction alone would be very heavy in
        # memory traffic.
        minLineLength = 5
        maxLineGap = 20
        lines = cv.HoughLinesP(np.array(edge, dtype=np.uint8), 
                               1,
                               np.pi/180,
                               80,
                               minLineLength,
                               maxLineGap)

        if lines is None:
            lines = [[]]
            print("no lines")
        else:
            left_x, left_y, right_x, right_y = handleLines(lines[0], 0.3, 4)

            # basically, if we wish to draw a center line. We only do this if
            # both lines are present
            exists_both = False
            l_x1 = 0
            l_x2 = 0
            r_x1 = 0
            r_x2 = 0

            # fit a linear model to the left lines and draw it on the image
            if len(left_x) > 0 and len(left_y) > 0:
                exists_both = True
                regr_left = linear_model.LinearRegression()
                regr_left.fit(np.reshape(left_y, (-1, 1)), left_x)
                l_x1 = int(regr_left.predict(360)[0])
                l_x2 = int(regr_left.predict(180)[0])
                cv.line(final, (l_x1, 360), (l_x2, 180), (0,255,0), 2)

            # fit a linear model to the right lines and draw it on the image
            if len(right_x) > 0 and len(right_y) > 0:
                exists_both = exists_both and True
                regr_right = linear_model.LinearRegression()
                regr_right.fit(np.reshape(right_y, (-1,1)), right_x)
                r_x1 = int(regr_right.predict(360)[0])
                r_x2 = int(regr_right.predict(180)[0])
                cv.line(final, (r_x1, 360), (r_x2, 180), (2,255,0), 2)

            # if both lines exists, draw the center line
            if exists_both:
                cv.line(final, (l_x1, 360), (r_x1, 360), color=3)
                cv.line(final, (l_x2, 180), (r_x2, 180), color=3)
                mid_x_top = l_x1 + abs(l_x1 - r_x1)/2
                mid_x_bot = l_x2 + abs(l_x2 - r_x2)/2
                cv.line(final, (mid_x_top, 360), (mid_x_bot, 180), color=5)


        # change the dimensions of the images so they can all be
        # printed in the same frame without becoming too big
        frame     = cv.resize(np.array(frame, dtype=np.uint8), 
                             (0,0), None, .65, .65)
        roi       = cv.resize(roi, (0,0), None, .65, .65)
        histEq    = cv.resize(histEq, (0,0), None, .65, .65)
        binarized = cv.resize(np.array(binarized, dtype=np.uint8), 
                             (0,0), None, .65, .65)
        edge      = cv.resize(np.array(edge, dtype=np.uint8), 
                             (0,0), None, .65, .65)
        final     = cv.resize(final, (0,0), None, .65, .65)

        # make the 1 channel images have 3 channels, so we can stack
        # and concatenate them however we like
        histEq    = cv.cvtColor(histEq,    cv.COLOR_GRAY2BGR)
        binarized = cv.cvtColor(binarized, cv.COLOR_GRAY2BGR)
        edge      = cv.cvtColor(edge,      cv.COLOR_GRAY2BGR)

        # stack and concatenate pictures to achieve one mega-picture
        top_row = np.hstack((frame, roi, histEq))
        bot_row = np.hstack((binarized, edge, final))
        result = np.vstack((top_row, bot_row))
        

        cv.imshow('edge-detection', np.array(result, dtype = np.uint8 ))

        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
