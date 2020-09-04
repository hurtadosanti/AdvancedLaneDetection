import numpy as np
import cv2


def mse(a, b):
    """https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/"""
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((a.astype("float") - b.astype("float")) ** 2)
    err /= float(a.shape[0] * b.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def draw_boundaries(img: np.ndarray, top=(400, 350), borders=(100, 100)) -> np.ndarray:
    image_shape = img.shape
    vertices = np.array(
        [[(borders[0], image_shape[0]), top, (image_shape[1] - top[0], top[1]),
          (image_shape[1] - borders[1], image_shape[0])]],
        dtype=np.int32)
    return cv2.polylines(img, vertices, True, (255, 0, 0), 3)
