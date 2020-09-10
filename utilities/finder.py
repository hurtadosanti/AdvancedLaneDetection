import logging
import cv2
import numpy as np
from utilities import image_utility


class LaneFinder:
    def __init__(self):
        self.inspection_image = None
        self.result_image = None
        self.left_fit = None
        self.right_fit = None

    def _find_lanes_windows(self, image, number_windows: int = 9, margin: int = 100, min_pixels: int = 50):
        """Return left_x,left_y,right_x,right_y"""
        self.inspection_image = np.dstack((image, image, image)) * 255
        left_lane_idx = []
        right_lane_idx = []
        # Get left and right lanes sections to process
        histogram = image_utility.calculate_histogram(image)
        midpoint = np.int(histogram.shape[0] // 2)
        left_base = np.argmax(histogram[:midpoint])
        right_base = np.argmax(histogram[midpoint:]) + midpoint
        window_height = np.int(image.shape[0] // number_windows)

        left_current = left_base
        right_current = right_base

        non_zero = image.nonzero()
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        # process each window to find where the line is
        for window in range(number_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_x_left_low = left_current - margin
            win_x_left_high = left_current + margin
            win_x_right_low = right_current - margin
            win_x_right_high = right_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.inspection_image, (win_x_left_low, win_y_low),
                          (win_x_left_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.inspection_image, (win_x_right_low, win_y_low),
                          (win_x_right_high, win_y_high), (0, 255, 0), 2)
            good_left_idx = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & (non_zero_x >= win_x_left_low) &
                             (non_zero_x < win_x_left_high)).nonzero()[0]
            good_right_idx = ((non_zero_y >= win_y_low) & (non_zero_y < win_y_high) & (non_zero_x >= win_x_right_low) &
                              (non_zero_x < win_x_right_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_idx.append(good_left_idx)
            right_lane_idx.append(good_right_idx)

            if len(good_left_idx) > min_pixels:
                left_current = np.int(np.mean(non_zero_x[good_left_idx]))
            if len(good_right_idx) > min_pixels:
                right_current = np.int(np.mean(non_zero_x[good_right_idx]))
        try:
            left_lane_idx = np.concatenate(left_lane_idx)
            right_lane_idx = np.concatenate(right_lane_idx)
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

    def plot_polylines(self, image):
        # Find Lane sections
        left_x, left_y, right_x, right_y = self._find_lanes_windows(image)
        # Color each side
        self.inspection_image[left_y, left_x] = [255, 0, 0]
        self.inspection_image[right_y, right_x] = [0, 0, 255]
        # Generate x and y values for plotting
        plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fit_x, right_fit_x = self._fit_poly(plot_y, left_x, left_y, right_x, right_y)
        self.draw_lane_section(left_fit_x, plot_y, self.inspection_image, right_fit_x)
        return plot_y, left_fit_x, right_fit_x

    def search_around_polylines(self, image, margin=100):
        # Grab activated pixels
        non_zero = image.nonzero()
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
        # Fit new polynomials
        plot_y = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fit_x, right_fit_x = self._fit_poly(plot_y,
                                                 non_zero_x[left_lane_inds], non_zero_y[left_lane_inds],
                                                 non_zero_x[right_lane_inds], non_zero_y[right_lane_inds]
                                                 )
        # Image of the resultant lane
        result_img = np.dstack((image, image, image)) * 255
        window_img = np.zeros_like(result_img)
        result_img[non_zero_y[left_lane_inds], non_zero_x[left_lane_inds]] = [255, 0, 0]
        result_img[non_zero_y[right_lane_inds], non_zero_x[right_lane_inds]] = [0, 0, 255]

        # Calculate the windows to search the points
        left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x - margin, plot_y]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x + margin, plot_y])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x - margin, plot_y]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x + margin, plot_y])))])
        # Add to the original image
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        self.inspection_image = cv2.addWeighted(result_img, 1, window_img, 0.3, 0)
        self.draw_lane_section(left_fit_x, plot_y, result_img, right_fit_x)
        return result_img

    def draw_lane_section(self, left_fit_x, plot_y, result_img, right_fit_x):
        # Draw the lane section
        pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(result_img, np.int_([pts]), (0, 255, 0))

    def _fit_poly(self, plot_y, left_x, left_y, right_x, right_y):
        self.left_fit = np.polyfit(left_y, left_x, 2)
        self.right_fit = np.polyfit(right_y, right_x, 2)
        # Generate x and y values for plotting
        left_fit_x = self.left_fit[0] * plot_y ** 2 + self.left_fit[1] * plot_y + self.left_fit[2]
        right_fit_x = self.right_fit[0] * plot_y ** 2 + self.right_fit[1] * plot_y + self.right_fit[2]
        return left_fit_x, right_fit_x
