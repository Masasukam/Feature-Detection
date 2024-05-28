import math
import numpy as np


def matrixRotation(theta_x, theta_y, theta_z):
    '''
    Input:
        theta_x -- Rotation around the x axis in radians
        theta_y -- Rotation around the y axis in radians
        theta_z -- Rotation around the z axis in radians
    Output:
        A 4x4 numpy array representing 3D rotations. The order of the rotation
        axes from first to last is x, y, z, if you multiply with the resulting
        rotation matrix from left.
    '''
    # Note: For MOPS, you need to use theta_z only, since we are in 2D

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
    '''
    Input:
        input_vec -- Translation vector represented by an 1D numpy array with 3
        elements
    Output:
        A 4x4 numpy array representing 3D translation.
    '''
    assert input_vec.ndim == 1
    assert input_vec.shape[0] == 3

    transformed = np.eye(4)
    transformed[:3, 3] = input_vec

    return transformed


def matrix_scale(drt_x, drt_y, drt_z):
    '''
    Input:
        drt_x -- Scaling along the x axis
        drt_y -- Scaling along the y axis
        drt_z -- Scaling along the z axis
    Output:
        A 4x4 numpy array representing 3D scaling.
    '''
    # Note: For MOPS, you need to use drt_x and drt_y only, since we are in 2D
    scaled = np.eye(4)

    for i, s in enumerate([drt_x, drt_y, drt_z]):
        scaled[i, i] = s

    return scaled

