import cv2
import numpy as np
def detect_sift_features(image_path):
    ########## Detects SIFT keypoints and descriptors from one image ##########
    # reads image from disk
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    # converts image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # sift works on intensity, not color
    # improves contrast slightly before sift
    gray = cv2.equalizeHist(gray) # helps sift find more stable features in low-contrast areas
    # creates sift detector
    sift = cv2.SIFT_create(
        nfeatures=5000,    # limits the maximum number of strongest features
        contrastThreshold=0.02,  # helps sift to detect features even in low-contrast regions
        edgeThreshold=10   # controls how strongly edge-like points are filtered out
    )
    # detects keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    # converts keypoint locations to numopy array
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    return image, points, descriptors, keypoints

def match_descriptors(des1, des2, ratio=0.75):  # matches descriptors from image 1 and image 2
                                                # lowe's ratio was 0.65
    ########## Matches SIFT descriptors using BFMatcher and Lowe's ratio test ##########
    if des1 is None or des2 is None:
        return []
    # sift descriptors are floating-point vectors
    # used L2 distance -> euclidean distance
    bf = cv2.BFMatcher(cv2.NORM_L2)
    # for each descriptor in image 1, find its two nearest descriptors in image 2
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [] # creats empty list to contain good matches
    for match_pair in raw_matches:
        if len(match_pair) < 2: # sometimes opencv may return fewer than 2 matches
            continue # skips that descriptor and moves to the next one
        m, n = match_pair
        # lowe's ratio test:
        if m.distance < ratio * n.distance:
            good_matches.append(m) # keep the match only if the best match is clearly better than the 2nd-best match
    # sort matches from strongest to weakest.
    good_matches = sorted(good_matches, key=lambda x: x.distance) # smaller distance means more similar descriptors

    return good_matches

def get_matched_points(points1, points2, matches):
    ########## Converts OpenCV match objects into two Nx2 point arrays ##########
    matched_1 = [] # matched points in image 1
    matched_2 = [] # matched points in image 2
    for m in matches:
        matched_1.append(points1[m.queryIdx]) # point coordinate in image 1
        matched_2.append(points2[m.trainIdx]) # corresponding point coordinate in image 2 

    return np.array(matched_1, dtype=np.float32), np.array(matched_2, dtype=np.float32) # both point lists as array