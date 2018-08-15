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

def plot_images(images_ndar_lst, name='', cmap=None, columns=2):
    #columns = 2
    rows = (len(images_ndar_lst) + 1) // columns

    plt.figure(figsize=[17, 30])
    for idx, image in enumerate(images_ndar_lst):
        plt.subplot(rows, columns, idx + 1)
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.title(name[idx])
        plt.axis('off')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)




fl = [myfun1, myfun1]
[f() for f in fl]
for f in fl:
    print(f())


line_map = {'left': [], 'right' : []}
m=-5
b=2
if m < 0:
    line_map['left'].append((m,b))
else:
    line_map['right'].append((m,b))



line_map[m > 0].append((m,b))

line_map[False]




def myfun(a,fl,*args):
    fl[0]
    print(a)


myfun(10,mysum,5,8)



img = mpimg.imread('test_images/solidWhiteRight.jpg')

plot_images([img],columns=1)

#masking
mask = np.zeros_like(img)
x = mask.shape[1]
y = mask.shape[0]
vertices = np.array([(x * 0., y), (x * .45, y * .58), (x * .55, y * .58), (x, y)], dtype=np.int32)
cv2.polylines(img, [vertices], True, (255, 0, 0))

plt.imshow(img)
if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
else:
    ignore_mask_color = 255

# filling pixels inside the polygon defined by "vertices" with the fill color
cv2.fillPoly(mask, [vertices], ignore_mask_color)
masked_image = cv2.bitwise_and(img, mask)


plt.figure()
plt.subplot(3,1,1)
plt.imshow(img)
plt.subplot(3,1,2)
plt.imshow(mask)
plt.subplot(3,1,3)
plt.imshow(masked_image)




pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
print(pts)
#pts = pts.reshape((-1,1,2))
print(pts)
cv2.polylines(img,[pts],True,(0,255,255))
plt.imshow(img)