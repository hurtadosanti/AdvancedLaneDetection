#!/usr/bin/env python
import numpy as np
import cv2
import glob
import logging

# Size of the checkers picture

class CameraCalibration:
    def __init__(self,nx=9,ny=6):
        self.nx = nx
        self.ny = ny

    def calibrate_camera(self,path: str, save_output: bool = False) -> ([np.ndarray], [np.ndarray]):
        """Get set of images on a path to get the calibration points, it is possible to store the output images into the
        out_<path> """
        criteria = None
        if save_output:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        nx = self.nx
        ny = self.ny
        # Points found to calibrate the image
        objects_points = np.zeros((nx * ny, 3), np.float32)
        objects_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        # Points to be returned for the set of images
        object_points_found = []  # 3d points in real world space
        image_points_found = []  # 2d points in image plane.

        # Path of the images to calibrate the camera
        images = glob.glob(path)

        # Read
        for image_name in images:
            img = cv2.imread(image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                object_points_found.append(objects_points)
                image_points_found.append(corners)
                if save_output:
                    logging.info('saving image: output_' + image_name)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)
                    if not cv2.imwrite('output_' + image_name, img):
                        message = 'file can not be saved into: out_' + image_name
                        logging.error(message)
                        raise Exception(message)
            else:
                logging.warning('no corners found on file:' + image_name)

        return object_points_found, image_points_found


    def undistort_image(self,path: str, object_points: [np.ndarray], image_points: [np.ndarray]) -> np.ndarray:
        """ Undistort a given image for the object_points and image_points specified"""
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
        if not ret:
            logging.error(ret.msg)
        return cv2.undistort(img, mtx, dist, None, mtx)
    
    def serialize_calibration(self,path:str):
        pass

    def load_calibration(self,path:str):
        pass

if __name__ == '__main__':
    c = CameraCalibration()
    # Example based on the camera_cal images
    op, ip = c.calibrate_camera('camera_cal/calibration*.jpg')
    dst = c.undistort_image('test_images/test5.jpg', op, ip)
    cv2.imshow('undistorted', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
