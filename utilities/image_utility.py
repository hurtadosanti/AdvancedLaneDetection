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


def draw_boundaries(image: np.ndarray, top=(400, 350), borders=(100, 100)) -> np.ndarray:
    image_shape = image.shape
    vertices = np.array(
        [[(borders[0], image_shape[0]), top, (image_shape[1] - top[0], top[1]),
          (image_shape[1] - borders[1], image_shape[0])]],
        dtype=np.int32)
    return cv2.polylines(image, vertices, True, (255, 0, 0), 3)


def warp_image(image, width=130, y=500, border=10):
    half = image.shape[1] // 2
    src = np.float32([
        [half - width, y],
        [half + width, y],
        [image.shape[1] - border, image.shape[0] - border],
        [border, image.shape[0] - border]])
    dst = np.float32([
        [0, 0],
        [image.shape[1], 0],
        [image.shape[1], image.shape[0]],
        [0, image.shape[0]]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


def calculate_histogram(image: np.ndarray) -> np.ndarray:
    histogram_image = image / 255
    bottom_half = histogram_image[histogram_image.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)


if __name__ == '__main__':
    img = cv2.imread('../test_images/test1.jpg')
    cv2.imshow('test1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
