import unittest
import cv2
from utilities import calibration
from utilities import image_utility


class CameraCalibrationTests(unittest.TestCase):
    def test_calibration(self):
        c = calibration.CameraCalibration()
        c.calibrate_camera('../camera_cal/')
        original = cv2.imread('../camera_cal/test5.jpg')
        src = cv2.imread('../camera_cal/undistorted_test5.jpg')
        undistorted = c.undistort_image(original)

        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        mse = image_utility.mse(src_gray, dst_gray)
        mse_original = image_utility.mse(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), dst_gray)
        self.assertGreater(mse_original, 1000)
        self.assertLess(mse, 1)


if __name__ == '__main__':
    unittest.main()
