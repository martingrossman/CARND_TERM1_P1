"module for image analysis"

# if __name__ == "__main__":

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

# Read Data
test_images_dir = 'test_images/'
test_videos_dir = 'test_videos'

test_images_lst = os.listdir(test_images_dir)

#Plot Images
os.
image = mpimg.imread('test_images/solidWhiteRight.jpg')