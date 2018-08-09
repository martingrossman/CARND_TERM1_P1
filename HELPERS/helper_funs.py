"helper funs"

#imports
import os
import cv2
import matplotlib.pyplot as plt

def print_hello():
    print('hello worlds')

# def create_fullpath_lst(dirpath):
#
#     fullpath_lst = []
#     print(os.listdir(dirpath))
#     # for file_name in os.listdir(dirpath):
#     #     fullpath_lst.append(os.path.join(dirpath, file_name))
    # return fullpath_lst

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


#plot fun
def plot_imfiles(imnames_lst,images_im,cmap=None):
    plt.figure(figsize=(7, 12))

    for idx, (im_name, image_im) in enumerate(zip(imnames_lst, images_im)):
        plt.subplot(len(imnames_lst),2,idx+1)
        plt.title(im_name)
        plt.imshow(image_im,cmap=cmap)
    plt.tight_layout()