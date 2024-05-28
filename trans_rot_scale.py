import math
import numpy as np


def matrixRotation(theta_x, theta_y, theta_z):

    mx_x_rot = np.array([[1, 0, 0, 0],
                         [0, math.cos(theta_x), -math.sin(theta_x), 0],
                         [0, math.sin(theta_x), math.cos(theta_x), 0],
                         [0, 0, 0, 1]])

    mx_y_rot = np.array([[math.cos(theta_y), 0, math.sin(theta_y), 0],
                         [0, 1, 0, 0],
                         [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                         [0, 0, 0, 1]])

    mx_z_rot = np.array([[math.cos(theta_z), -math.sin(theta_z), 0, 0],
                         [math.sin(theta_z), math.cos(theta_z), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    return np.dot(mx_z_rot, np.dot(mx_y_rot, mx_x_rot))


def matrix_transformation(input_vec):

    assert input_vec.ndim == 1
    assert input_vec.shape[0] == 3

    transformed = np.eye(4)
    transformed[:3, 3] = input_vec

    return transformed


def matrix_scale(drt_x, drt_y, drt_z):

    scaled = np.eye(4)

    for i, s in enumerate([drt_x, drt_y, drt_z]):
        scaled[i, i] = s

    return scaled

