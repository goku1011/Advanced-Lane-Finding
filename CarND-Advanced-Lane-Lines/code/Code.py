import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pylab


'''
images = glob.glob('/home/gokul/CarND-Advanced-Lane-Lines-master/camera_cal/calibration*.jpg')

objpoints = []
imgpoints = []

objp = np.zeros((6*9, 3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
i = 0
for fname in images:
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    i = i + 1
    imgpoints.append(corners)
    objpoints.append(objp)
    objpoint = []
    imgpoint = []
    imgpoint.append(corners)
    objpoint.append(objp)
    if ret == True:
        img = cv2.drawChessboardCorners(image, (9,6), corners, ret)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoint, imgpoint, gray.shape[::-1], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)

ret, mtx, dist, rvecs, vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(dist)
print(mtx)
j = 0
for fname in images:
    j = j + 1
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    img = cv2.drawChessboardCorners(image, (9, 6), corners, ret)
    dst = cv2.undistort(image, mtx, dist, None, mtx)

    if j == 17:
        plt.imshow(dst)
        plt.show()
        plt.imshow(image)
        plt.show()
'''

''' Using calibrated 'dist' and 'mtx' '''
dist = np.array([[-2.41017956e-01,  -5.30721171e-02,  -1.15810354e-03,  -1.28318858e-04,  2.67125300e-02]])
mtx = np.array([[1.15396093e+03,   0.00000000e+00,   6.69705357e+02],
                [0.00000000e+00,   1.14802496e+03,   3.85656234e+02],
                [0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
fname = ('/home/gokul/CarND-Advanced-Lane-Lines-master/test_images/test6.jpg')
image = cv2.imread(fname)


def s_thresh(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]
    s_thresh_min = thresh[0]
    s_thresh_max = thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    return s_binary

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

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
undst = cv2.undistort(gray, mtx, dist, None, mtx)
img_size = (image.shape[1], image.shape[0])
'''
src = np.float32(
    [[573, 465.781],
     [710, 465.781],
     [1040, 680],
     [260, 680]]
)
dst = np.float32(
    [[260, 20],
     [1040, 20],
     [1040, 680],
     [260, 680]]
)
M = cv2.getPerspectiveTransform(src, dst)
'''
M = np.array([[-6.18513104e-01,  -1.52256609e+00,   1.03534494e+03],
              [-9.43689571e-15,  -1.95321702e+00,   9.07598663e+02],
              [-1.47451495e-17,  -2.38016633e-03,   1.00000000e+00]])
binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
plt.imshow(binary_warped, cmap='gray')
plt.show()



histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

nwindows = 20

window_height = np.int(binary_warped.shape[0]/nwindows)

nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

leftx_current = leftx_base
rightx_current = rightx_base

# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50

left_lane_inds = []
right_lane_inds = []

for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()
