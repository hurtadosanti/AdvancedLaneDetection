import numpy as np
import cv2

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


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
    # half = image.shape[1] // 2
    # src = np.float32([
    #     [half - width, y],
    #     [half + width, y],
    #     [image.shape[1] - border, image.shape[0] - border],
    #     [border, image.shape[0] - border]])
    # dst = np.float32([
    #     [0, 0],
    #     [image.shape[1], 0],
    #     [image.shape[1], image.shape[0]],
    #     [0, image.shape[0]]])
    img_size = (image.shape[1], image.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    m = cv2.getPerspectiveTransform(src, dst)
    # Get the M to be able to return from the bird view back to the original image
    reverse = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]), flags=cv2.INTER_AREA), reverse


def reverse_warp(image, result_img, reverse, shape, beta):
    inv = cv2.warpPerspective(result_img, reverse, shape, flags=cv2.INTER_LINEAR)
    # for alpha we use 1 so the original image is not degraded
    return cv2.addWeighted(image, 1, inv, beta, 0)


def calculate_histogram(image: np.ndarray) -> np.ndarray:
    histogram_image = image / 255
    bottom_half = histogram_image[histogram_image.shape[0] // 2:, :]
    return np.sum(bottom_half, axis=0)


def measure_curvature_pixels(plot_y, left_fit, right_fit):
    """
    Calculates the curvature of polynomial functions in pixels.
    """
    y_eval = np.max(plot_y)
    left_curve = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curve = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curve, right_curve


def measure_curvature_real(plot_y, left_fit, right_fit):
    """
    Calculates the curvature of polynomial functions in meters.
    """
    y_eval = np.max(plot_y)
    left_curve = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curve = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])
    return left_curve, right_curve


def measure_vehicle_distance(point_y, point_x, left_fit, right_fit):
    """
    Calculate the distance at point y
    """
    # Calculate the x points at the y axis defined
    left_x = left_fit[0]*point_y**2 + left_fit[1]*point_y + left_fit[2]
    right_x = right_fit[0]*point_y**2 + right_fit[1]*point_y + right_fit[2]
    # Calculate the center of the two points
    center = (left_x+right_x)//2
    # Calculate the difference to the actual center of the image
    position = (point_x//2)-center
    # convert to meters
    return position*xm_per_pix


if __name__ == '__main__':
    img = cv2.imread('../test_images/test1.jpg')
    cv2.imshow('test1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
