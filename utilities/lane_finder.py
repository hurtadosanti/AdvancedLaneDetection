import cv2
import numpy as np
from utilities import image_utility
import logging


class Lanes:
    def __init__(self, img):
        self.image = img
        self.out_image = np.dstack((img, img, img)) * 255
        self.histogram = image_utility.calculate_histogram(img)
        self.left_lane_idx = []
        self.right_lane_idx = []
        self.left_fit = None
        self.right_fit = None
        self.plot_y = None

    def _find_lanes(self, number_windows: int = 9, margin: int = 100, min_pixels: int = 50):
        """Return left_x,left_y,right_x,right_y"""
        # Get left and right lanes sections to process
        midpoint = np.int(self.histogram.shape[0] // 2)
        left_base = np.argmax(self.histogram[:midpoint])
        right_base = np.argmax(self.histogram[midpoint:]) + midpoint
        window_height = np.int(self.image.shape[0] // number_windows)

        left_current = left_base
        right_current = right_base

        non_zero = self.image.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        # process each window to find where the line is
        for window in range(number_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.image.shape[0] - (window + 1) * window_height
            win_y_high = self.image.shape[0] - window * window_height
            win_x_left_low = left_current - margin
            win_x_left_high = left_current + margin
            win_x_right_low = right_current - margin
            win_x_right_high = right_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_image, (win_x_left_low, win_y_low),
                          (win_x_left_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.out_image, (win_x_right_low, win_y_low),
                          (win_x_right_high, win_y_high), (0, 255, 0), 2)
            good_left_idx = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & (non_zero_x >= win_x_left_low) &
                             (non_zero_x < win_x_left_high)).nonzero()[0]
            good_right_idx = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & (non_zero_x >= win_x_right_low) &
                              (non_zero_x < win_x_right_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_idx.append(good_left_idx)
            self.right_lane_idx.append(good_right_idx)

            if len(good_left_idx) > min_pixels:
                left_current = np.int(np.mean(non_zero_x[good_left_idx]))
            if len(good_right_idx) > min_pixels:
                right_current = np.int(np.mean(non_zero_x[good_right_idx]))
        try:
            left_lane_idx = np.concatenate(self.left_lane_idx)
            right_lane_idx = np.concatenate(self.right_lane_idx)
        except ValueError as v:
            # Avoids an error if the above is not implemented fully
            logging.error(str(v))
            raise v
        # Extract left and right line pixel positions

        left_x = non_zero_x[left_lane_idx]
        left_y = non_zero_y[left_lane_idx]
        right_x = non_zero_x[right_lane_idx]
        right_y = non_zero_y[right_lane_idx]

        return left_x, left_y, right_x, right_y

    def fit_polynomial(self):
        left_x, left_y, right_x, right_y = self._find_lanes()
        self.left_fit = np.polyfit(left_y, left_x, 2)
        self.right_fit = np.polyfit(right_y, right_x, 2)
        self.plot_y = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
        self.out_image[left_y, left_x] = [255, 0, 0]
        self.out_image[right_y, right_x] = [0, 0, 255]
        return self.plot_y, self.left_fit, self.right_fit

    def generate_plotting_values(self):
        # Generate x and y values for plotting
        self.plot_y = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
        try:
            left_fit_x = self.left_fit[0] * self.plot_y ** 2 + self.left_fit[1] * self.plot_y + self.left_fit[2]
            right_fit_x = self.right_fit[0] * self.plot_y ** 2 + self.right_fit[1] * self.plot_y + self.right_fit[2]
        except TypeError as t:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            logging.error('The function failed to fit a line!', str(t))
            left_fit_x = 1 * self.plot_y ** 2 + 1 * self.plot_y
            right_fit_x = 1 * self.plot_y ** 2 + 1 * self.plot_y
        return self.plot_y, left_fit_x, right_fit_x

    def fit_poly(self, leftx, lefty, rightx, righty):
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        self.plot_y = np.linspace(0, self.image.shape[0] - 1, self.image.shape[0])
        left_fit_x = self.left_fit[0] * self.plot_y ** 2 + self.left_fit[1] * self.plot_y + self.left_fit[2]
        right_fit_x = self.right_fit[0] * self.plot_y ** 2 + self.right_fit[1] * self.plot_y + self.right_fit[2]
        return left_fit_x, right_fit_x, self.plot_y

    def search_around_poly(self, image):
        margin = 100
        # Load a new image
        self.image = image
        # Grab activated pixels
        non_zero = self.image.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        left_lane_inds = ((non_zero_x > (self.left_fit[0] * (non_zero_y ** 2) + self.left_fit[1] * non_zero_y +
                                         self.left_fit[2] - margin)) & (
                                  non_zero_x < (self.left_fit[0] * (non_zero_y ** 2) +
                                                self.left_fit[1] * non_zero_y +
                                                self.left_fit[
                                                    2] + margin)))
        right_lane_inds = ((non_zero_x > (self.right_fit[0] * (non_zero_y ** 2) + self.right_fit[1] * non_zero_y +
                                          self.right_fit[2] - margin)) & (
                                   non_zero_x < (self.right_fit[0] * (non_zero_y ** 2) +
                                                 self.right_fit[1] * non_zero_y + self.right_fit[
                                                     2] + margin)))
        # Again, extract left and right line pixel positions

        # Fit new polynomials
        left_fitx, right_fitx, self.plot_y = self.fit_poly(
            non_zero_x[left_lane_inds], non_zero_y[left_lane_inds],
            non_zero_x[right_lane_inds], non_zero_y[right_lane_inds]
        )
        # Image of the resultant lane
        result_img = np.dstack((self.image, self.image, self.image)) * 255
        window_img = np.zeros_like(result_img)
        result_img[non_zero_y[left_lane_inds], non_zero_x[left_lane_inds]] = [255, 0, 0]
        result_img[non_zero_y[right_lane_inds], non_zero_x[right_lane_inds]] = [0, 0, 255]

        # Calculate the windows to search the points
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, self.plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, self.plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, self.plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, self.plot_y])))])
        # Add to the original image
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        window_lanes = cv2.addWeighted(result_img, 1, window_img, 0.3, 0)

        # Draw the lane section
        pts_left = np.array([np.transpose(np.vstack([left_fitx, self.plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, self.plot_y])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(result_img, np.int_([pts]), (0, 255, 0))
        return window_lanes, result_img, left_line_pts, right_line_pts
