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

def convert_to_lab(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def select_white_yellow_L(image):

    converted = convert_to_lab(image)

    # white mask
    d = 25
    lower = np.uint8([210, -d + 128, -d + 128])
    upper = np.uint8([255, d + 128, d + 128])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow  mask
    lower = np.uint8([128, -28 + 128, 30 + 128])
    upper = np.uint8([255, 28 + 128, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    mask = cv2.bitwise_or(yellow_mask, white_mask)

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
                #print(image_temp)
                image_temp = f(image_temp, image_in)
                image_process_pipe_lst.append((image_temp, f.__name__))
            # elif f.__name__=='hough_lines':
            #     args = param
            #     image_temp = f(image_temp, *args)
            #     image_process_pipe_lst.append((image_temp[0], f.__name__))
            else:
                args = param
                image_temp = f(image_temp, *args)
                if isinstance(image_temp, tuple):
                    image_temp = image_temp[0]
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


def avg_lines_by_points(lines, weights=[], weighted=True):

    if len(lines):
        start_x = lines[:, 0, 0]
        start_y = lines[:, 0, 1]
        end_x = lines[:, 0, 2]
        end_y = lines[:, 0, 3]

        if weighted & len(weights)>1:
            x1= np.average(start_x,weights=weights)
            y1 = np.average(start_y, weights=weights)
            x2 = np.average(end_x, weights=weights)
            y2 = np.average(end_y, weights=weights)
        else:
            x1 = start_x.mean()
            y1 = start_y.mean()
            x2 = end_x.mean()
            y2 = end_y.mean()
    else:
        x1 = []
        y1 = []
        x2 = []
        y2 = []

    return [x1,y1,x2,y2]


def line_kq_from_pts(line_pts):
    k =(line_pts[3]-line_pts[1])/(line_pts[2]-line_pts[0])
    q = line_pts[1] - k * line_pts[0]
    return(k,q)


def draw_lines(img, hlines, lines_previous, color=[255, 0, 0], thickness=13, counter=0):
    # hlines is result of hough function 3d array
    # img to draw on
    # lines_previous  - previously infered lines
    x_size = img.shape[1]
    y_size = img.shape[0]

    horizon_height = y_size * 0.6

    left_hlines = []
    left_hlines_len = []
    right_hlines = []
    right_hlines_len = []

    # sort left, right hlines
    for index, line in enumerate(hlines):
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            angle = np.arctan2((y2 - y1), (x2 - x1))
            intercept = y1 - x1 * slope
            line_len = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if (abs(slope) > 0.2) & (abs(slope) < 0.8):
                if slope < 0:
                    # print('left slope: {}'.format(slope))
                    left_hlines.append([[x1, y1, x2, y2]])
                    left_hlines_len.append(line_len)
                else:
                    # print('right slope: {}'.format(slope))
                    right_hlines.append([[x1, y1, x2, y2]])
                    right_hlines_len.append(line_len)

    # case of no lines check use previous
    if not len(left_hlines):
        left_hlines = np.array([[lines_previous[0]]])
    if not len(right_hlines):
        right_hlines = np.array([[lines_previous[1]]])

    # numpy arrays
    left_hlines = np.array(left_hlines)
    right_hlines = np.array(right_hlines)
    left_hlines_len = np.array(left_hlines_len)
    right_hlines_len = np.array(right_hlines_len)

    # averaging lines by points, weighted by length
    left_line_pts = avg_lines_by_points(left_hlines, weights=left_hlines_len, weighted=True)
    right_line_pts = avg_lines_by_points(right_hlines, weights=right_hlines_len, weighted=True)

    # Bottom and top stretching of line
    # Result lines  0=left, 1=right new lines
    new_lines = np.zeros(shape=(2, 4), dtype=np.int32)
    # left
    k, q = line_kq_from_pts(left_line_pts)
    left_bottom_x = (y_size - q) / k
    left_top_x = (horizon_height - q) / k
    if left_bottom_x >= 0:
        new_lines[0] = [left_bottom_x, y_size, left_top_x, horizon_height]
    # right
    k, q = line_kq_from_pts(right_line_pts)
    right_bottom_x = (y_size - q) / k
    right_top_x = (horizon_height - q) / k
    if right_bottom_x <= x_size:
        new_lines[1] = [right_bottom_x, y_size, right_top_x, horizon_height]

    # Low pass filtering
    if not lines_previous.size == 0:

        if counter < 5:
            # At the begining almost no filtering #TODO functional dependence
            alfa = 0.9
        else:
            # low pass filter on x values
            alfa = 0.12
        new_lines[0][0] = new_lines[0][0] * alfa + lines_previous[0][0] * (1 - alfa)
        new_lines[0][2] = new_lines[0][2] * alfa + lines_previous[0][2] * (1 - alfa)
        new_lines[1][0] = new_lines[1][0] * alfa + lines_previous[1][0] * (1 - alfa)
        new_lines[1][2] = new_lines[1][2] * alfa + lines_previous[1][2] * (1 - alfa)

    # Draw lines
    for line in new_lines:
        cv2.line(img, (line[0], line[1]), (line[2], line[3]), color, thickness)

    return new_lines

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, lines_previous,counter):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines_raw = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)

    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    new_lines = draw_lines(line_img, lines_raw, lines_previous,counter=counter)


    return line_img, lines_raw, new_lines


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    #print(type(img),type(initial_img))
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
	
def show_videos_html(video_path):
    r=HTML("""<video width="960" height="540" controls> <source src="{0}"> </video>""".format(video_path))
    return(r)