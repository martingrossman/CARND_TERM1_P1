"helper funs"

#imports
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from IPython.display import HTML

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
