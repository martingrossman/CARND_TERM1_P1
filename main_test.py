from HELPERS import helper_funs as hf

# importing some useful packages
import matplotlib.image as mpimg
from collections import OrderedDict
import cv2
import numpy as np

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






# ad image processing filters to create ordered pipeline as ordered dict
# for each filter add parameters
filters_funs_d = OrderedDict() #needs to be ordered to fix the order (no need in 3.6 up)
filters_funs_d[hf.select_white_yellow_L] ='none'
filters_funs_d[hf.grayscale] = 'none'
filters_funs_d[hf.gaussian_blur] = (9,) #needs to be tuple ()
filters_funs_d[hf.canny] = (100,200)
filters_funs_d[hf.region_of_interest] = 'none'

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


hf.plot_pipes(processed_images_lst, fgs=(20, 10))