import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg

fname = ('/home/gokul/CarND-Advanced-Lane-Lines-master/test_images/test5.jpg')
image = cv2.imread(fname)

def s_thresh(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_thresh_min = thresh[0]
    s_thresh_max = thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    return s_binary

s_binary = s_thresh(image, thresh=(125, 245))
plt.imshow(s_binary)
plt.show()
plt.imshow(image, cmap='gray')
plt.show()
r_channel = image[:, :, 0]
g_channel = image[:, :, 1]
b_channel = image[:, :, 2]
plt.imshow(r_channel)
plt.show()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY   )
plt.imshow(gray)
plt.show()
plt.imshow(g_channel)
plt.show()
plt.imshow(b_channel)
plt.show()
hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
h_channel = hls[:, :, 0]
l_channel = hls[:, :, 1]
s_channel = hls[:, :, 2]
plt.imshow(h_channel)
plt.show()
plt.imshow(l_channel)
plt.show()
plt.imshow(s_channel)
plt.show()

'''
def abs_sobel_thresh(img, orient, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    scaled_sobel = np.uint8((255*(np.absolute(sobel)))/np.max(np.absolute(sobel)))
    binary_image = np.zeros_like(scaled_sobel)
    binary_image[(scaled_sobel<thresh[1]) & (scaled_sobel>thresh[0])] = 1
    return binary_image

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = np.sqrt((sobelx**2) + (sobely**2))
    scaled_sobel = np.uint8((255 * sobel) / np.max(sobel))
    binary_image = np.zeros_like(scaled_sobel)
    binary_image[(scaled_sobel < thresh[1]) & (scaled_sobel > thresh[0])] = 1
    return binary_image

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    abs_grid = np.arctan2(abs_sobely, abs_sobelx)
    binary_image = np.zeros_like(abs_grid)
    binary_image[(abs_grid < thresh[1]) & (abs_grid > thresh[0])] = 1
    return binary_image

gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 100))
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100))
mag_binary = mag_thresh(image, sobel_kernel=3, thresh=(50, 100))
dir_binary = dir_threshold(image, sobel_kernel=3, thresh=(0.7, 1.3))
s_binary = s_thresh(image, thresh=(170, 255))

combined = np.zeros_like(dir_binary)
combined[(gradx == 1) & (grady == 1) | (dir_binary == 1) & (mag_binary == 1)] = 1
combined_binary = np.zeros_like(s_binary)
combined_binary[(s_binary == 1) | (combined == 1)] = 1
plt.imshow(combined_binary, cmap='gray')
plt.show()
'''