
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

from detect_feature import computeHv, cornerDetection, MOPSDescriptorsComputation, match

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_AQUAMARINE = (212, 255, 127)

def filter_keypoints_by_threshold(keypoints, threshold):
    return [keypoint for keypoint in keypoints if keypoint[3] >= threshold]

def concatenate_images(images):
    valid_images = [img for img in images if img is not None]
    if not valid_images:
        return np.zeros((0, 0, 3), np.uint8)
    
    max_height = max([img.shape[0] for img in valid_images])
    total_width = sum([img.shape[1] for img in valid_images])
    
    concatenated_image = np.full((max_height, total_width, 3), 255, np.uint8)
    current_width = 0
    
    for img in valid_images:
        height, width = img.shape[:2]
        concatenated_image[:height, current_width:current_width + width, :] = img
        current_width += width

    return concatenated_image

def draw_feature_matches(image1, keypoints1, image2, keypoints2, matches):

    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]

    combined_image = concatenate_images([image1, image2])

    keypoint_pairs = [[keypoints1[match[0]], keypoints2[match[1]]] for match in matches]
    match_status = np.ones(len(keypoint_pairs), dtype=bool)
    points1 = np.int32([pair[0][:2] for pair in keypoint_pairs])
    points2 = np.int32([pair[1][:2] for pair in keypoint_pairs]) + (width1, 0)

    for (x1, y1), (x2, y2), is_inlier in zip(points1, points2, match_status):
        if is_inlier:
            cv2.circle(combined_image, (x1, y1), 5, COLOR_GREEN, 2)
            cv2.circle(combined_image, (x2, y2), 5, COLOR_GREEN, 2)
        else:
            radius = 5
            line_thickness = 6
            cv2.line(combined_image, (x1 - radius, y1 - radius), (x1 + radius, y1 + radius), COLOR_RED, line_thickness)
            cv2.line(combined_image, (x1 - radius, y1 + radius), (x1 + radius, y1 - radius), COLOR_RED, line_thickness)
            cv2.line(combined_image, (x2 - radius, y2 - radius), (x2 + radius, y2 + radius), COLOR_RED, line_thickness)
            cv2.line(combined_image, (x2 - radius, y2 + radius), (x2 + radius, y2 - radius), COLOR_RED, line_thickness)
    
    for (x1, y1), (x2, y2), is_inlier in zip(points1, points2, match_status):
        if is_inlier:
            cv2.line(combined_image, (x1, y1), (x2, y2), COLOR_AQUAMARINE)

    return combined_image

def draw_matches_with_threshold(percent, image1, image2, threshold_exp, keypoints1, keypoints2):

    thresholded_keypoints1 = filter_keypoints_by_threshold(keypoints1, 10 ** threshold_exp)
    thresholded_keypoints2 = filter_keypoints_by_threshold(keypoints2, 10 ** threshold_exp)
    
    descriptors1 = MOPSDescriptorsComputation(image1, thresholded_keypoints1)
    descriptors2 = MOPSDescriptorsComputation(image2, thresholded_keypoints2)

    all_matches = match(descriptors1, descriptors2)
    all_matches = sorted(all_matches, key=lambda match: match[2])

    if all_matches:
        num_matches_to_draw = int(float(percent) * len(all_matches) / 100)
        selected_matches = all_matches[:num_matches_to_draw]
        
        if selected_matches:
            return draw_feature_matches(image1, thresholded_keypoints1, image2, thresholded_keypoints2, selected_matches)
    
    return concatenate_images([image1, image2])