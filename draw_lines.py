import pickle
import numpy as np
from HELPERS import helper_funs as hf

#open hlines
with open('C:/Users/martin_dev/Documents/UDACITY_ND/CarND-LaneLines-P1/hlines.pkl', 'rb') as f:
    hlines = pickle.load(f)







def draw_lines(img, hlines, lines_previous, color=[255, 0, 0], thickness=13, counter=0):

    x_size = img.shape[1]
    y_size = img.shape[0]
    horizon = y_size * 0.5

    left_hlines = []
    left_hlines_len = []
    right_hlines = []
    right_hlines_len = []
    left_hlines_kq = []
    righ_hlines_kq = []
    #sort left, right hlines
    for index, line in enumerate(hlines):
        for x1, y1, x2, y2 in line:
            slope =(y2 - y1)/(x2 - x1)
            angle = np.arctan2((y2 - y1), (x2 - x1))
            intercept = y1 - x1 * slope
            line_len = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if slope < 0:
                left_hlines.append([[x1,y1,x2,y2]])
                left_hlines_len.append(line_len)
            else:
                right_hlines.append([[x1, y1, x2, y2]])
                right_hlines_len.append(line_len)
    # numpy array
    left_hlines = np.array(left_hlines)
    right_hlines = np.array(right_hlines)

    left_hlines_len = np.array(left_hlines_len)
    right_hlines_len = np.array(right_hlines_len)

    # avg lines
    left_line_pts = hf.avg_lines_by_points(left_hlines, weights=left_hlines_len, weighted=True)
    right_line_pts = hf.avg_lines_by_points(right_hlines, weights=right_hlines_len, weighted=True)



    # result lines 0=left,1=right
    new_lines = np.zeros(shape=(2, 4), dtype=np.int32)

    #left
    # y = kx+q -> x=
    k,q = hf.line_kq_from_pts(left_line_pts)
    left_bottom_x = (y_size - q) / k
    left_top_x = (horizon - q) / k
    if (left_bottom_x >= 0):
        new_lines[0] = [left_bottom_x, y_size, left_top_x, horizon]
    #right
    k,q = hf.line_kq_from_pts(right_line_pts)
    right_bottom_x = (y_size - q) / k
    right_top_x = (horizon - q) / k
    if (right_bottom_x <= x_size):
        new_lines[1] = [right_bottom_x, y_size, right_top_x, horizon]

    lines_previous = np.array([])
    counter = 100
    if lines_previous.size == 0:
        x = 0
    else:

        if counter < 20:
            alfa = 0.99
        else:
            alfa = 0.15
        new_lines[0][0] = new_lines[0][0] * alfa + lines_previous[0][0] * (1 - alfa)
        new_lines[0][2] = new_lines[0][2] * alfa + lines_previous[0][2] * (1 - alfa)
        new_lines[1][0] = new_lines[1][0] * alfa + lines_previous[1][0] * (1 - alfa)
        new_lines[1][2] = new_lines[1][2] * alfa + lines_previous[1][2] * (1 - alfa)
        # print(alfa)
    for line in new_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return new_lines


