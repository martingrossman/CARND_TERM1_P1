"helper funs"

#imports
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from operator import itemgetter

# def create_fullpath_lst(dirpath):
#
#     fullpath_lst = []
#     print(os.listdir(dirpath))
#     # for file_name in os.listdir(dirpath):
#     #     fullpath_lst.append(os.path.join(dirpath, file_name))
    # return fullpath_lst

def process_video(video_in_path, video_out_path, pipeline, show_video=True):


    clip_in = VideoFileClip(video_in_path)
    clip_frame = clip_in.fl_image(pipeline)
    clip_frame.write_videofile(video_out_path, audio=False)
    if show_video :
        return(
        HTML("""
        <video width="960" height="540" controls>
          <source src="{0}">
        </video>
        """.format(video_out_path)))




def dir_content_lst(dir_path):
    return [image for image in os.listdir(dir_path)]

def dir_content_fullpath_lst(dir_path):
    return [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def select_rgb_yellow(image):
    # white color mask
    # yellow color mask
    lower = np.uint8([190, 190, 0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    # mask = cv2.bitwise_or(white_mask, yellow_mask)

    masked = cv2.bitwise_and(image, image, mask=yellow_mask)
    return masked

def convert_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def select_white_yellow_L(image):
    converted = convert_lab(image)

    gray_img = grayscale(image)
    subdued_gray = (gray_img / 2).astype('uint8')

    # white color mask
    d = 15
    lower = np.uint8([210, -d + 128, -d + 128])
    upper = np.uint8([255, d + 128, d + 128])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([128, -28 + 128, 30 + 128])
    upper = np.uint8([255, 28 + 128, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)

    # combine the mask
    # mask = cv2.bitwise_or(white_mask, yellow_mask)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    mask2 = cv2.bitwise_or(subdued_gray, mask)

    return cv2.bitwise_and(image, image, mask=mask)

def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with

    x = img.shape[1]
    y = img.shape[0]
    vertices = np.array([[(x * 0., y), (x * .47, y * .58), (x * .53, y * .58), (x, y)]], dtype=np.int32)
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


#plot fun
def plot_imfiles(imnames_lst,images_im,cmap=None,title=''):
    plt.figure(figsize=(7, 13),)
    plt.subplots_adjust(left=0.05 , bottom=0.05, right=0.99, top=0.92,
                    wspace=0.25, hspace=0.35)

    plt.suptitle(title)

    for idx, (im_name, image_im) in enumerate(zip(imnames_lst, images_im)):
        plt.subplot(len(imnames_lst),2,idx+1)
        plt.title(im_name)
        plt.imshow(image_im,cmap=cmap)
    #plt.tight_layout()

def process_filters(image_in, image_name_in='name', filters_dict=[]):
    # process image in by applying filters in filters_dict
    image_temp = image_in
    image_process_pipe_lst = [(image_in, image_name_in)]
    # if no filtering funs
    if filters_dict:
        for idx, (f, param) in enumerate(filters_dict.items()):
            if param == 'none':
                image_temp = f(image_temp)
                image_process_pipe_lst.append((image_temp, f.__name__))
            elif param == 'image':
                image_temp = f(image_temp, image_in)
                image_process_pipe_lst.append((image_temp, f.__name__))
            else:
                args = param
                image_temp = f(image_temp, *args)
                image_process_pipe_lst.append((image_temp, f.__name__))

    return image_process_pipe_lst

def plot_filters(image_pipe_lst, fgs=(20, 10)):
    # plot filter for one image
    colormap = None
    cols = len(image_pipe_lst)
    plt.figure(figsize=fgs)

    for idx, img_tuple in enumerate(image_pipe_lst):
        if len(img_tuple[0].shape) == 2:
            colormap = 'gray'
        plt.subplot(1, cols, idx + 1)
        plt.title(img_tuple[1])
        plt.imshow(img_tuple[0], cmap=colormap)
    plt.show()

def plot_pipes(processed_images_lst, fgs=(20, 15)):
    # plots filters on all images in processed_images_lst

    rows = len(processed_images_lst)
    cols = len(processed_images_lst[0])
    plt.figure(figsize=fgs)
    colormap = None
    for jx,image_pipe_lst in enumerate(processed_images_lst):
        for idx, img_tuple in enumerate(image_pipe_lst):
            if len(img_tuple[0].shape) == 2:
                colormap = 'gray'
            plt.subplot(rows, cols, (idx + 1)+(jx*cols))
            plt.title(img_tuple[1])
            plt.imshow(img_tuple[0], cmap=colormap)
    plt.tight_layout()
    plt.show()


def draw_lines(img, lines, lines_previous, color=[255, 0, 0], thickness=13, counter=0):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    x_size = img.shape[1]
    y_size = img.shape[0]
    lines_slope_intercept = np.zeros(shape=(len(lines), 2))
    lines_slope_angle = np.zeros(shape=(len(lines), 2))

    for index, line in enumerate(lines):
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            angle = np.arctan2((y2 - y1), (x2 - x1))
            intercept = y1 - x1 * slope
            lines_slope_intercept[index] = [slope, intercept]
            lines_slope_angle[index] = [angle]
    max_slope_line = lines_slope_intercept[lines_slope_intercept.argmax(axis=0)[0]]
    min_slope_line = lines_slope_intercept[lines_slope_intercept.argmin(axis=0)[0]]
    left_slopes = []
    left_intercepts = []
    right_slopes = []
    right_intercepts = []
    left_angles = []
    right_angles = []
    # this gets slopes and intercepts of lines similar to the lines with the max (immediate left) and min
    # (immediate right) slopes (i.e. slope and intercept within x%)
    for line, angle in zip(lines_slope_intercept, lines_slope_angle):
        if abs(line[0] - max_slope_line[0]) < 0.15 and abs(line[1] - max_slope_line[1]) < (0.15 * x_size):
            left_slopes.append(line[0])
            left_intercepts.append(line[1])
            left_angles.append(angle[0])

        elif abs(line[0] - min_slope_line[0]) < 0.15 and abs(line[1] - min_slope_line[1]) < (0.15 * x_size):
            right_slopes.append(line[0])
            right_intercepts.append(line[1])
            right_angles.append(angle[0])
    # print(np.rad2deg(right_angles),right_slopes)

    # left and right lines are averages of these slopes and intercepts, extrapolate lines to edges and center*
    # *roughly
    new_lines = np.zeros(shape=(1, 2, 4), dtype=np.int32)
    if len(left_slopes) > 0:
        left_line = [sum(left_slopes) / len(left_slopes), sum(left_intercepts) / len(left_intercepts)]
        left_line = [np.tan(sum(left_angles) / len(left_angles)), sum(left_intercepts) / len(left_intercepts)]  # MG
        left_bottom_x = (y_size - left_line[1]) / left_line[0]
        left_top_x = (y_size * .575 - left_line[1]) / left_line[0]
        if (left_bottom_x >= 0):
            new_lines[0][0] = [left_bottom_x, y_size, left_top_x, y_size * .575]
    if len(right_slopes) > 0:
        right_line = [sum(right_slopes) / len(right_slopes), sum(right_intercepts) / len(right_intercepts)]
        right_line = [np.tan(sum(right_angles) / len(right_angles)),
                      sum(right_intercepts) / len(right_intercepts)]  # MG
        right_bottom_x = (y_size - right_line[1]) / right_line[0]
        right_top_x = (y_size * .575 - right_line[1]) / right_line[0]
        if (right_bottom_x <= x_size):
            new_lines[0][1] = [right_bottom_x, y_size, right_top_x, y_size * .575]

    if lines_previous.size == 0:
        x = 0
    else:
        # print('left:',lines_previous[0][0][0])
        # print('left:',lines_previous[0][0][2])
        if counter < 20:
            alfa = 0.99
        else:
            alfa = 0.15
        new_lines[0][0][0] = new_lines[0][0][0] * alfa + lines_previous[0][0][0] * (1 - alfa)
        new_lines[0][0][2] = new_lines[0][0][2] * alfa + lines_previous[0][0][2] * (1 - alfa)
        new_lines[0][1][0] = new_lines[0][1][0] * alfa + lines_previous[0][1][0] * (1 - alfa)
        new_lines[0][1][2] = new_lines[0][1][2] * alfa + lines_previous[0][1][2] * (1 - alfa)
        # print(alfa)
    for line in new_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return new_lines


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, lines_previous):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines_raw = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    new_lines = draw_lines(line_img, lines_raw, lines_previous)
    return line_img#, lines_raw, new_lines


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)