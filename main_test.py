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

# for idx, (im_name, im_path) in enumerate(zip(test_images_lst, test_images_fullpath_list)):
#     images_im.append(mpimg.imread(im_path))
#     images_gs.append(hf.grayscale(images_im[idx]))
#     images_ed.append(hf.canny(images_gs[idx],canny_low,canny_high))
#     images_gb.append(hf.gaussian_blur(images_ed[idx],gbsize))
#     #images_ms.append() # TODO region mask

#plot images


# hf.plot_imfiles(test_images_lst,images_im)
# hf.plot_imfiles(test_images_lst,images_gs,'gray')
# hf.plot_imfiles(test_images_lst,images_ed,'gray')
# hf.plot_imfiles(test_images_lst,images_gb,'gray','aaaa')



# plt.show()



# ad image processing filters to create ordered pipeline as ordered dict
# for each filter add parameters
filters_funs_d = OrderedDict() #needs to be ordered to fix the order (no need in 3.6 up)

filters_funs_d[hf.grayscale] = 'none'
filters_funs_d[hf.gaussian_blur] = (9,) #needs to be tuple ()
filters_funs_d[hf.canny] = (100,200)


# for f,v in filters_funs_d.items():
#     print(f,v)
#     print(type(v))
#     if v == '':
#         print('empty')

#########################
# def name_from_path(file_path: str)->str:
#     return str(file_path).split('\\')[1]
#
# def mysum(a: int, b:int)-> int:
#     sum = a+b
#     return sum
# test_images_fullpath_list[1]
# ################
#
# def plot_improcsess(image_file, img_name ='', filters_funs_lst=[], fgs=(20, 10)):
#
#     cmap=None
#     img= mpimg.imread(image_file)
#     cols= len(filters_funs_lst)+1
#
#     plt.figure(figsize=fgs)
#
#     plt.subplot(1, cols, 1)
#     plt.title(img_name)
#     plt.imshow(img)
#     image_temp=img
#
#
#     for idx,(f,v) in enumerate(filters_funs_d.items()):
#         # funs
#         print(image_temp.shape)
#         #print(idx,f)
#         plt.subplot(1, cols, idx+2)
#         plt.title(f.__name__)
#         cmap = 'gray' if len(image_temp.shape) == 2 else cmap
#
#
#         if v == 'none':
#             image_temp = f(image_temp)
#             plt.imshow(image_temp,cmap=cmap)
#         else:
#             #print(v)
#             #print(type(v))
#             args=v
#             image_temp = f(image_temp,*args)
#             plt.imshow(image_temp,cmap=cmap)
#
# def improcsess_filters(image_file, img_name ='', filters_funs_d=[]):
#     image_pipe_lst = []
#     cmap = None
#     img = mpimg.imread(image_file)
#     image_temp = img
#     image_pipe_lst.append(image_temp)
#
#     for idx,(f,v) in enumerate(filters_funs_d.items()):
#         if v == 'none':
#             image_temp = f(image_temp)
#             image_pipe_lst.append(image_temp)
#             args=v
#             image_temp = f(image_temp,*args)
#             image_pipe_lst.append(image_temp)
#     return image_pipe_lst


# process image through all filters funs in filters_dict
# input is image, output is lst of processed image with their names
# image_process_pipe_lst = [(processed image, 'fun name')]

def process_filters(image, image_name='name', filters_dict={}):

    image_temp = image
    image_process_pipe_lst = [(image, image_name)]
    # if no filtering funs
    if not filters_funs_d:
        return image
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


#TODO

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

plot_pipes(processed_images_lst, fgs=(20, 10))