import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import trans_rot_scale

from trans_rot_scale import *

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


def MOPSDescriptorsComputation(srcImage, features):

    srcImage = srcImage.astype(np.float32)
    srcImage /= 255.

    windowSize = 8

    descriptors = np.zeros((len(features), windowSize * windowSize))
    img_grayed = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    img_grayed = ndimage.gaussian_filter(img_grayed, 0.5)

    count = 0
    for i, f in enumerate(features):
        mxHolder = np.zeros((2, 3))

        x, y, angle, _ = f

        # Lec Slides Descriptors-1 pg.15 - 19

        # T1: Translation to center on feature
        T1 = matrix_transformation(np.array([-x, -y, 0]))
        
        # Rotation to align the patch orientation to the horizontal axis
        angle_rad = -math.radians(angle)  # Negative for clockwise rotation
        R = matrixRotation(0, 0, angle_rad)

        # Scaling down from 40x40 to 8x8
        S = matrix_scale(1/5.0, 1/5.0, 1)

        # Final translation to center within the new 8x8 image
        T2 = matrix_transformation(np.array([windowSize/2.0, windowSize/2.0, 0]))

        # Combine transformations into 2x3, throw out z-values (3rd row and 3rd col)
        result = T2 @ S @ R @ T1 
        mxHolder[:2, :2] = result[:2, :2]
        mxHolder[0][2] = result[0][3]
        mxHolder[1][2] = result[1][3]


        # Call the warp affine function to do the mapping
        # It expects a 2x3 matrix
        outImage = cv2.warpAffine(img_grayed, mxHolder,
            (windowSize, windowSize), flags=cv2.INTER_LINEAR)
        # print(outImage.shape)
        # print(np.var(outImage))
        
        # Normalize the patch
        if np.var(outImage) < 1e-10:
            descriptors[i, :] = np.zeros(windowSize * windowSize)
        else:
            descriptors[i, :] = ((outImage - np.mean(outImage)) / np.std(outImage)).flatten()
            # count += 1

    # print(count)
    return descriptors

