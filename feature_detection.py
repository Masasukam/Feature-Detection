import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations

from transformations import *

def is_within_bounds(array_shape, indices):

    assert len(array_shape) == len(indices)
    for dim_size, index in zip(array_shape, indices):
        if index < 0 or index >= dim_size:
            return False
    return True


# Computes Harris Values for Corners
def computeHv(grayscale_image):

    img_height, img_width = grayscale_image.shape[:2]
    harris_response = np.zeros(grayscale_image.shape[:2])
    gradient_orientation = np.zeros(grayscale_image.shape[:2])

    # Compute image gradients
    grad_x = ndimage.sobel(grayscale_image, axis=1, mode='nearest')
    grad_y = ndimage.sobel(grayscale_image, axis=0, mode='nearest')

    # Compute products of gradients
    grad_x_squared = ndimage.gaussian_filter(grad_x * grad_x, sigma=0.5, mode='nearest')
    grad_y_squared = ndimage.gaussian_filter(grad_y * grad_y, sigma=0.5, mode='nearest')
    grad_xy = ndimage.gaussian_filter(grad_x * grad_y, sigma=0.5, mode='nearest')

    # Compute Harris corner response
    determinant = grad_x_squared * grad_y_squared - grad_xy * grad_xy
    trace = grad_x_squared + grad_y_squared
    harris_response = determinant - 0.1 * (trace * trace)

    # Compute gradient orientation
    gradient_orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi

    return harris_response, gradient_orientation




def computeLM(harris_response):
    
    local_maxima = (harris_response == ndimage.maximum_filter(harris_response, size=7, mode='constant', cval=-1e10))
    return local_maxima


def cornerDetection(harris_response, gradient_orientation):

    img_height, img_width = harris_response.shape[:2]
    detected_features = []

    # Compute local maxima
    local_maxima = computeLM(harris_response)

    for y in range(img_height):
        for x in range(img_width):
            if local_maxima[y, x]:
                feature = (x, y, gradient_orientation[y, x], harris_response[y, x])
                detected_features.append(feature)

    return detected_features




