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

if (gbsize%2==0):

    print('gbsize is',gbsize,', change to odd number!')
    gbsize = gbsize + 1

hough_rho = 1
hough_theta = np.pi / 180
hough_threshold = 20
hough_min_line_length = 40
hough_max_line_gap = 200
lines_previous =np.array([])


# ad image processing filters to create ordered pipeline as ordered dict
# for each filter add parameters
# needs to be ordered to fix the order (no need in 3.6 up)
filters_funs_d = OrderedDict()

filters_funs_d[hf.select_white_yellow_L] = 'none'
filters_funs_d[hf.grayscale] = 'none'
filters_funs_d[hf.gaussian_blur] = (gbsize,) #needs to be tuple ()
filters_funs_d[hf.canny] = (canny_low, canny_high)
filters_funs_d[hf.region_of_interest] = 'none'
filters_funs_d[hf.hough_lines] = (hough_rho, hough_theta, hough_threshold,
                                 hough_min_line_length, hough_max_line_gap, lines_previous)
filters_funs_d[hf.weighted_img] = 'image'
# result


# create image list
image_list = []
image_name_list = []
for idx, image_file in enumerate(test_images_fullpath_list):

    image_list.append(mpimg.imread(test_images_fullpath_list[idx]))
    image_name_list.append(str(test_images_fullpath_list[idx]).split('/')[1])

# Processing
processed_images_lst=[]
for image,image_name in zip(image_list, image_name_list):
    processed_images_lst.append(hf.process_filters(image, image_name, filters_funs_d))



print(len(processed_images_lst))
print(len(processed_images_lst[0]))


b=(0, 1, 2, 6)
processed_images_lst_sel = [itemgetter(*b)(sublist) for sublist in processed_images_lst]
hf.plot_pipes(processed_images_lst_sel, fgs=(20, 10))