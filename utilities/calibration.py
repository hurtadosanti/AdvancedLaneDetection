#!/usr/bin/env python
import numpy as np
import cv2
import glob
import os
import logging
import pickle


class CameraCalibration:
    def __init__(self, nx=9, ny=6):
        self.nx = nx
        self.ny = ny
        self.object_points_found = []
        self.image_points_found = []

    def calibrate_camera(self, path: str, save_output: bool = False) -> ([np.ndarray], [np.ndarray]):
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
        self.object_points_found = []  # 3d points in real world space
        self.object_points_found.clear()
        self.image_points_found = []  # 2d points in image plane.
        self.image_points_found.clear()
        # Path of the images to calibrate the camera
        # Read
        for image_name in [i for i in os.listdir(path) if i.endswith('.jpg')]:
            img = cv2.imread(path+image_name)
            logging.info(path+image_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret:
                self.object_points_found.append(objects_points)
                self.image_points_found.append(corners)
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

        return self.object_points_found, self.image_points_found

    def serialize_calibration(self, path: str):
        # TODO: Check that the path is a directory
        pickle.dump(self.object_points_found, open(path + "/calibration_points.b", "wb"))
        pickle.dump(self.image_points_found, open(path + "/calibration_image_points.b", "wb"))
        pass

    def load_calibration(self, path: str):
        self.object_points_found = pickle.load(open(path + "/calibration_points.b", 'rb'))
        self.image_points_found = pickle.load(open(path + "/calibration_image_points.b", 'rb'))
        pass

    def undistort_image(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points_found, self.image_points_found,
                                                           gray.shape[::-1], None, None)
        if not ret:
            logging.error(ret.msg)
        return cv2.undistort(img, mtx, dist, None, mtx), mtx, dist
