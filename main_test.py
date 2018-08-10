from HELPERS import helper_funs as hf

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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

for idx, (im_name, im_path) in enumerate(zip(test_images_lst, test_images_fullpath_list)):
    images_im.append(mpimg.imread(im_path))
    images_gs.append(hf.grayscale(images_im[idx]))
    images_ed.append(hf.canny(images_gs[idx],canny_low,canny_high))
    images_gb.append(hf.gaussian_blur(images_ed[idx],gbsize))
    #images_ms.append() # TODO region mask

#plot images


hf.plot_imfiles(test_images_lst,images_im)
hf.plot_imfiles(test_images_lst,images_gs,'gray')
hf.plot_imfiles(test_images_lst,images_ed,'gray')
hf.plot_imfiles(test_images_lst,images_gb,'gray','aaaa')



plt.show()
