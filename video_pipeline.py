import cv2
import numpy as np

from utilities import calibration
from utilities import image_utility
from utilities import finder
from utilities import thresholds


def write_text(frame, text, x, y):
    cv2.putText(frame, text, (x, y), 0, 0.7, (255, 255, 255), 1)


class VideoLaneLineDetection:

    def __init__(self, video_path):
        # Calibrate the camera
        c = calibration.CameraCalibration()
        c.calibrate_camera('./camera_cal/')
        self.calibration = c
        self.cap = cv2.VideoCapture(video_path)

    def process_video(self):
        lane_finder = finder.LaneFinder()
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
        first_frame = True
        while True:
            ret, frame = self.cap.read()
            if not ret:
                cv2.destroyAllWindows()
                return
            undistorted = self.calibration.undistort_image(frame)
            warped, reverse = image_utility.warp_image(undistorted, 140, 470, 100)
            combined_binary = thresholds.highlight_features(warped, 7, (50, 150), (50, 200), (0.5, np.pi / 2),
                                                            (80, 200))
            if first_frame:
                plot_y, left_fit_x, right_fit_x = lane_finder.plot_polylines(combined_binary)
                first_frame = False
            else:
                plot_y, left_fit_x, right_fit_x = lane_finder.search_around_polylines(combined_binary, 25)

            left, right = image_utility.measure_curvature_real(plot_y, lane_finder.left_fit, lane_finder.right_fit)
            pos = image_utility.measure_vehicle_distance(frame.shape[0], frame.shape[1], lane_finder.left_fit,
                                                         lane_finder.right_fit)
            text = f'left curvature:{left:.2f}mts right curvature:{right:.2f}mts center offset:{pos:.2f}mts'
            reversed_frame = image_utility.reverse_warp(frame, lane_finder.result_image, reverse,
                                                        (frame.shape[1], frame.shape[0]),
                                                        0.3)
            write_text(reversed_frame, text, 25, 25)
            cv2.imshow('lanes', reversed_frame)
            out.write(reversed_frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    ll = VideoLaneLineDetection('project_video.mp4')
    ll.process_video()
