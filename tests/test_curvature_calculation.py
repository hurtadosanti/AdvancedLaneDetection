import unittest
import numpy as np
from utilities import image_utility


def generate_data(ym_per_pix=None, xm_per_pix=None):
    """
    Generates fake data to use for calculating lane curvature.
    In your own project, you'll ignore this function and instead
    feed in the output of your lane detection algorithm to
    the lane curvature calculation.
    """
    # Set random seed number so results are consistent for grader
    # Comment this out if you'd like to see results on different random data!
    np.random.seed(0)
    # Generate some fake data to represent lane-line pixels
    plot_y = np.linspace(0, 719, num=720)  # to cover same y-range as image
    quadratic_coefficient = 3e-4  # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    left_x = np.array([200 + (y ** 2) * quadratic_coefficient + np.random.randint(-50, high=51)
                       for y in plot_y])
    right_x = np.array([900 + (y ** 2) * quadratic_coefficient + np.random.randint(-50, high=51)
                        for y in plot_y])

    left_x = left_x[::-1]  # Reverse to match top-to-bottom in y
    right_x = right_x[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    # Fit new polynomials to x,y in world space
    left_fit_cr, right_fit_cr = _fit_polynomial(plot_y, left_x, right_x, xm_per_pix, ym_per_pix)

    return plot_y, left_fit_cr, right_fit_cr


def _fit_polynomial(plot_y, left_x, right_x, xm_per_pix=None, ym_per_pix=None):
    """Fit a second order polynomial to pixel positions in each fake lane line"""
    if xm_per_pix is not None and ym_per_pix is not None:
        left_fit = np.polyfit(plot_y * ym_per_pix, left_x * xm_per_pix, 2)
        right_fit = np.polyfit(plot_y * ym_per_pix, right_x * xm_per_pix, 2)
    else:
        left_fit = np.polyfit(plot_y, left_x, 2)
        right_fit = np.polyfit(plot_y, right_x, 2)
    return left_fit, right_fit


class CurvatureCalculationTests(unittest.TestCase):
    def test_real_curvature(self):
        plot_y, left_fit_cr, right_fit_cr = generate_data(image_utility.ym_per_pix,image_utility.xm_per_pix)
        left_curve, right_curve = image_utility.measure_curvature_real(plot_y, left_fit_cr, right_fit_cr)
        self.assertAlmostEqual(left_curve, 533.7525889210938)
        self.assertAlmostEqual(right_curve, 648.157485143441)

    def test_pixel_curvature(self):
        plot_y, left_fit, right_fit = generate_data()
        left_curve, right_curve = image_utility.measure_curvature_pixels(plot_y, left_fit, right_fit)
        self.assertAlmostEqual(left_curve, 1625.0601831657204)
        self.assertAlmostEqual(right_curve, 1976.2967307714334)


if __name__ == '__main__':
    unittest.main()
