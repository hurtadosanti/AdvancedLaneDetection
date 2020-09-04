import cv2
import numpy as np
import math


def abs_sobel_thresh(image: np.ndarray, orient='x', sobel_kernel=3, thresh=(0, 255)) -> np.ndarray:
    """ Define a function that takes an image, gradient orientation, and threshold min / max values."""
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output


def mag_thresh(image: np.ndarray, sobel_kernel: int = 3, thresh=(0, 255)) -> np.ndarray:
    """Take both Sobel x and y gradients and Calculate the gradient magnitude"""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output


def dir_threshold(image: np.ndarray, sobel_kernel: int = 3, thresh=(0, np.pi / 2)) -> np.ndarray:
    """Take the absolute value of the gradient direction, apply a threshold, and create a binary image result."""
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


def highlight_features(image: np.ndarray) -> (np.ndarray, np.ndarray):
    """Highlight features for line finding independent of color or shadows"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    s_channel = hls[:, :, 2]

    sx_binary = abs_sobel_thresh(gray, 'x', 5, (30, 100))
    sy_binary = abs_sobel_thresh(gray, 'y', 5, (30, 100))
    mag_binary = mag_thresh(gray, 5, (20, 100))
    dir_binary = dir_threshold(gray, 5, (0.7, math.pi / 2))

    s_thresh_min = 90
    s_thresh_max = 255

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    color_binary = np.dstack((np.zeros_like(sx_binary), sx_binary, s_binary)) * 255

    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)((mag_binary==1)&(dir_binary==1))] = 1

    return color_binary, combined_binary
