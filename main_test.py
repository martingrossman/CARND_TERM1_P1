from HELPERS import helper_funs as hf

# importing some useful packages
import matplotlib.image as mpimg
from collections import OrderedDict
import numpy as np
from operator import itemgetter

# read data
test_images_dir = 'test_images/'
test_videos_dir = 'test_videos'
test_images_fullpath_list = hf.dir_content_fullpath_lst(test_images_dir)


# parameters
canny_low = 100
canny_high = 200
gbsize = 7
#rg_ll_pt = # TODO region mask

if gbsize%2 == 0:

    print('gbsize is',gbsize,', change to odd number!')
    gbsize = gbsize + 1

hough_rho = 1
hough_theta = np.pi / 90
hough_threshold = 15
hough_min_line_length = 70
hough_max_line_gap = 180
lines_previous = np.array([])


# add image processing filters to create ordered pipeline as ordered dict
# for each filter add parameters into dictionary as tuples
# needs to be ordered to fix the order (no need in 3.6 up)
filters_funs_d = OrderedDict()

filters_funs_d[hf.select_rgb_yellow] = 'none'
filters_funs_d[hf.grayscale] = 'none'
filters_funs_d[hf.gaussian_blur] = (gbsize,) #needs to be tuple ()
filters_funs_d[hf.canny] = (canny_low, canny_high)
filters_funs_d[hf.region_of_interest] = 'none'
# filters_funs_d[hf.hough_lines] = (hough_rho, hough_theta, hough_threshold,
#                                  hough_min_line_length, hough_max_line_gap, lines_previous)
# filters_funs_d[hf.weighted_img] = 'image'

# create image list
image_list = [mpimg.imread(test_images_fullpath_list[idx])
              for idx, image_file in enumerate(test_images_fullpath_list)]
# create image names list
image_name_list = [str(test_images_fullpath_list[idx]).split('/')[1]
                   for idx, image_file in enumerate(test_images_fullpath_list)]


# Processing
# For each image in image list run process_filters to apply pipeline
processed_images_lst=[hf.process_filters(image, image_name, filters_funs_d)
                      for image, image_name in zip(image_list, image_name_list)]


# Printing
b=(0, 1, 7) # select which filter to plot
#processed_images_lst_sel = [itemgetter(*b)(sublist) for sublist in processed_images_lst]
#hf.plot_pipes(processed_images_lst_sel, fgs=(20, 10))


counter = 0

def process_pipeline0(img):
    global counter
    counter = counter + 1
    print(counter)
    global lines_previous
    # image = mpimg.imread(img)
    image = img
    im = hf.process_filters(img, 'none', filters_funs_d)
    masked_img = im[-1][0]

    # hough
    hough_rho = 1
    hough_theta = np.pi / 180
    hough_threshold = 20
    hough_min_line_length = 40
    hough_max_line_gap = 200
    hough_img, lines_raw, lines_new = hf.hough_lines(masked_img, hough_rho, hough_theta, hough_threshold,
                                                     hough_min_line_length, hough_max_line_gap, lines_previous,counter)
    # result

    result = hf.weighted_img(hough_img, image)
    lines_previous = lines_new
    # result = np.dstack(3*[masked_img]).astype('uint8')

    return result



def process_video(video_in_path,video_out_path,pipeline,show_video=True):


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

lines_previous=np.array([])


from moviepy.editor import VideoFileClip
from IPython.display import HTML
#process_video('test_videos/solidYellowLeft.mp4',"test_videos_output/solidYellowLeft.mp4",process_pipeline0)
#process_video('test_videos/solidWhiteRight.mp4',"test_videos_output/solidWhiteRight.mp4",process_pipeline0)
process_video('test_videos/challenge.mp4',"test_videos_output/challenge.mp4",process_pipeline0)