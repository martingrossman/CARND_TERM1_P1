from HELPERS import helper_funs as hf

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import OrderedDict


#read data
test_images_dir = 'test_images/'
test_videos_dir = 'test_videos'
test_images_lst = hf.dir_content_lst(test_images_dir)
test_images_fullpath_list = hf.dir_content_fullpath_lst(test_images_dir)

#read images via imread
#convert to grayscale
#edge detect
images_im = [] # originals list
images_gs = [] # grayscaled list
images_ed = [] # edges list
images_gb = [] # gaussian blur list
images_ms = [] # masked images


#parameters
canny_low = 100
canny_high = 200
gbsize = 7
#rg_ll_pt = # TODO region mask

if (gbsize%2==0):

    print('gbsize is',gbsize,', change to odd number!')
    gbsize = gbsize + 1


def process_filters(image_in, image_name_in, filters_dict):

    image_temp = image_in
    image_process_pipe_lst = [(image_in, image_name_in)]
    # if no filtering funs
    if not filters_funs_d:
        return image_in
    else:
        for idx, (f, param) in enumerate(filters_dict.items()):

            if param == 'none':
                image_temp = f(image_temp)
                image_process_pipe_lst.append((image_temp, f.__name__))
            else:
                args = param
                image_temp = f(image_temp, *args)
                image_process_pipe_lst.append((image_temp, f.__name__))

    return image_process_pipe_lst


def plot_filters(image_pipe_lst, fgs=(20, 10)):

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

    rows = len(processed_images_lst)
    cols = len(processed_images_lst[0])
    plt.figure(figsize=fgs)
    colormap = None
    for jx,image_pipe_lst in enumerate(processed_images_lst):
        for idx, img_tuple in enumerate(image_pipe_lst):
            if len(img_tuple[0].shape) == 2:
                colormap = 'gray'
            plt.subplot(rows, cols, (idx + 1)+(jx*4))
            plt.title(img_tuple[1])
            plt.imshow(img_tuple[0], cmap=colormap)
    plt.tight_layout()
    plt.show()


# ad image processing filters to create ordered pipeline as ordered dict
# for each filter add parameters
filters_funs_d = OrderedDict() #needs to be ordered to fix the order (no need in 3.6 up)
filters_funs_d[hf.grayscale] = 'none'
filters_funs_d[hf.gaussian_blur] = (9,) #needs to be tuple ()
filters_funs_d[hf.canny] = (100,200)

# create image list
image_list = []
image_name_list = []
for idx,image_file in enumerate(test_images_fullpath_list):

    image_list.append(mpimg.imread(test_images_fullpath_list[idx]))
    image_name_list.append(str(test_images_fullpath_list[idx]).split('/')[1])


processed_images_lst=[]
for image,image_name in zip(image_list, image_name_list):
    processed_images_lst.append(process_filters(image,image_name,filters_funs_d))


print(len(processed_images_lst))
print(len(processed_images_lst[0]))


plot_pipes(processed_images_lst, fgs=(20, 10))