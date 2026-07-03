import cv2 # rgb image → grayscale; sift feature detection; bfmatcher descriptor matching
import numpy as np # arrays, 3d point calculations, svd, matrix multiplication, norm/error calculation and random sampling in ransac
from .rgbd_sequence_config import sift_ratio, ransac_iterations, ransac_threshold, max_depth_for_feature_points
########### contains SIFT, 2D-to-3D conversion, Procrustes, RANSAC ###########
def detect_sift_features(image_rgb): # receives one rgb image
    # image → keypoints + descriptors
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) # works on intensity/brightness: rgb → grayscale; in Part 2, images have already been converted to rgb
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.02, edgeThreshold=10) # keeps up to 5000 strongest features, allows lower-contrast features, controls removel of edge-like unstable features
    keypoints, descriptors = sift.detectAndCompute(gray, None) # detects keypoints and computes desctriptors

    return keypoints, descriptors # feature locations and feature appearance vectors

def match_sift_descriptors(des1, des2, ratio=sift_ratio): # matches sift descriptors between two images; des1 -> target descriptors and des2 -> source descriptors; target_point = r * source_point + t
    # descriptors from two images → good feature matches
    if des1 is None or des2 is None:   # if one image has no descriptors
        return []                      # matching is impossible
    bf = cv2.BFMatcher(cv2.NORM_L2)    # creates a brute force matcher; sift descriptors are floating-point vectors, so L2 distance is the one; smaller distance = more similar descriptor
    raw_matches = bf.knnMatch(des1, des2, k=2) # for each descriptor in des1, it finds the best 2 matches in des2; lowe's ratio test compares best match vs 2nd best match
    good_matches = [] # creates an empty list for accepted matches
    for pair in raw_matches: # loops through each match pair: best match and 2nd best match
        if len(pair) < 2: # may return fewer than 2 matches
            continue # skips it, because we need at least 2 candidate matches
        m, n = pair # best and 2nd best matches
        if m.distance < ratio * n.distance: # lowe’s ratio test
            good_matches.append(m) # accept the match only if the best match is clearly better than the 2nd best match
    def get_distance(match): # getting distance for each match
        return match.distance # small distance is much better
    sorted_matches = sorted(good_matches, key=get_distance) # needs to list by each match's respective distance

    return sorted_matches # lists them

def pixel_to_3d(u, v, depth, K, max_depth=max_depth_for_feature_points):
    # one pixel coordinate + depth + K -> one 3D point
    h, w = depth.shape  # gives the size of the depth image: h -> rows/image height while w -> columns/image width
    # converts decimal keypoint to integer pixel
    u_int = int(round(u)) # horizontal pixel coordinate
    v_int = int(round(v)) # vertical pixel coordinate
    if u_int < 0 or u_int >= w or v_int < 0 or v_int >= h: # checks if pixel is inside the image
        return None # this pixel cannot be converted to 3d
    Z = float(depth[v_int, u_int]) # depth value of that pixel
    if Z <= 0 or Z > max_depth: # checks if the depth is usable
        return None # rejected
    fx = float(K[0, 0]) # extracts focal length in x direction parameter from K
    fy = float(K[1, 1]) # extracts focal length in y direction parameter from K
    cx = float(K[0, 2]) # extracts image center x parameter from K
    cy = float(K[1, 2]) # extracts image center y parameter from K
    X = (u_int - cx) * Z / fx # pixel-to-3D conversion
    Y = (v_int - cy) * Z / fy # pixel-to-3D conversion

    return np.array([X, Y, Z], dtype=np.float32) # 3d point as a numpy array

def matched_keypoints_to_3d(keypoints_target, keypoints_source, matches, depth_target, depth_source, K_target, K_source):
    # takes matched 2D keypoints and converts them into matched 3D point pairs
    # target keypoint and source keypoint -> target 3d point and source 3d point
    # same index = same physical feature match
    target_points = [] # stores valid 3d target points
    source_points = [] # stores valid 3d source points
    for m in matches:  # checks every good sift match
        u_target, v_target = keypoints_target[m.queryIdx].pt # target keypoint index; it gets target keypoints with the m.queryIdx index andgets its pixel coordinates
        u_source, v_source = keypoints_source[m.trainIdx].pt # spurce keypoint index; it gets source keypoints with the m.trainIdx index and gets its pixel coordinates
        p_target = pixel_to_3d(u_target, v_target, depth_target, K_target) # converts target pixel into target 3d point
        p_source = pixel_to_3d(u_source, v_source, depth_source, K_source) # converts source pixel into target 3d point
        if p_target is None or p_source is None: # if one of them is missing
            continue # skips this match, jumps back and checks next match pair
        target_points.append(p_target) # stores valid target points
        source_points.append(p_source) # stores valid sournce points

    return (
        np.array(target_points, dtype=np.float32),  # converts into numpy arrays
        np.array(source_points, dtype=np.float32),  # converts into numpy arrays
    )

def estimate_rigid_transform_procrustes(source_points, target_points):
    # source camera/frame point cloud → target/template coordinate system
    # finds R and t that move the source 3D points onto the target 3D points
    # target_point = R @ source_point + t
    # same index means same physical feature
    if source_points.shape[0] < 3: # number of source 3d points
        raise ValueError("At least 3 point pairs are required.") # gives vakueerror
    #removes the translation effect, so we can estimate only the rotation
    source_centroid = np.mean(source_points, axis=0) # average center of source points; calculates down each column
    target_centroid = np.mean(target_points, axis=0) # average center of target points; calculates down each column
    source_centered = source_points - source_centroid # subtracts the centroid from every source point
    target_centered = target_points - target_centroid # subtracts the centroid from every target point
    H = source_centered.T @ target_centered # stores how source inlier shape should rotate to match target inlier shape
    U, singular_values, Vt = np.linalg.svd(H) # singular value decomposition; decomposes H and gives the information needed to compute the best rotation
    # U -> source-side rotation/orientation information
    # singular_values -> how strong each direction is
    # Vt -> target-side rotation/orientation information
    R = Vt.T @ U.T # calculates the best rotation matrix that turns source directions into target directions
    if np.linalg.det(R) < 0: # checking against reflection; if result is not positive, it means the result includes reflection; like a mirror reflectionm
        Vt[-1, :] *= -1 # we need to flip the last row of Vt to fix it
        R = Vt.T @ U.T # and recompute
    t = target_centroid - R @ source_centroid # translkation

    return R, t # rotation matrix and translation vector

def apply_transform(points, R, t):   # source point → transformed source point
    return (R @ points.T).T + t   # transformed_point = R @ point + t

def compute_registration_errors(source_points, target_points, R, t): # how close are the transformed source points to the target points?
    transformed = apply_transform(source_points, R, t)  # source points after moving them into target coordinate system
    return np.linalg.norm(transformed - target_points, axis=1)  # converts each error vector into one distance number: error = sqrt(dx² + dy² + dz²) per point; axis=1 -> calculate one distance per row -> like row 0, row 1, row 2 and etc...

def ransac_procrustes(source_points, target_points, iterations=ransac_iterations, threshold=ransac_threshold):
    # tries many random small groups
    # estimates r and t from each group
    # tests it on all points
    # keeps the r and t that gives the most inliers
    num_points = source_points.shape[0] # gives the number of matched 3d point pairs
    if num_points < 3: # less than 3
        raise ValueError("Not enough 3D point pairs for RANSAC.") # gives valueerror
    best_inlier_mask = None # stores the best ransac result found so far -> [True, True, False, True, False, ...]
    best_inlier_count = 0   # stores the best ransac result found so far -> the best model has 0 inliers because we haven't started yet
    for _ in range(iterations): # repeats the process many times
        sample_indices = np.random.choice(num_points, size=3, replace=False) # randomly chooses 3 different indices
        src_sample = source_points[sample_indices] # selects the same 3 indices from source
        tgt_sample = target_points[sample_indices] # selects the same 3 indices from target
        try:
            R_candidate, t_candidate = estimate_rigid_transform_procrustes(src_sample, tgt_sample) # tries to estimate a temporary possible r and t using only those 3 point pairs
        except Exception: # if procrustes fails
            continue  # goes to the next ransac iteration
        errors = compute_registration_errors(source_points, target_points, R_candidate, t_candidate) # applies the candidate r and t to all source points
        inlier_mask = errors < threshold # creates a true/false array
        inlier_count = int(np.sum(inlier_mask)) # counts how many true values exist
        if inlier_count > best_inlier_count: # if this candidate is better than the previous one
            best_inlier_mask = inlier_mask # saves it as the best one
            best_inlier_count = inlier_count # saves it as the best one 
    if best_inlier_mask is None: # if all random trials failed
        raise RuntimeError("RANSAC failed.") # no valid transformation was found, stops with an error
    R_refined, t_refined = estimate_rigid_transform_procrustes(source_points[best_inlier_mask], target_points[best_inlier_mask]) # gives a more accurate final transformation
    refined_errors = compute_registration_errors(source_points, target_points, R_refined, t_refined) # tests the refined transformation on all matched point pairs again
    refined_inlier_mask = refined_errors < threshold # recomputes the final inlier mask

    return R_refined, t_refined, refined_inlier_mask, refined_errors 

def register_source_to_target(target_rgb, target_depth, target_K, source_rgb, source_depth, source_K, ratio=sift_ratio, ransac_iterations=ransac_iterations, ransac_threshold=ransac_threshold):
    # given one target RGB-D frame and one source RGB-D frame -> estimates R and t that register source to target
    target_keypoints, target_descriptors = detect_sift_features(target_rgb) # detects sift features in the target/reference image
    source_keypoints, source_descriptors = detect_sift_features(source_rgb) # detects sift features in the source image
    matches = match_sift_descriptors(target_descriptors, source_descriptors, ratio=ratio) # matches target descriptors to source descriptors
    target_points, source_points = matched_keypoints_to_3d(target_keypoints, source_keypoints, matches, target_depth, source_depth, target_K, source_K) # converts matched 2d keypoints to 3d points
    if target_points.shape[0] < 10: # checks if enough valid 3d correspondences; 3 is too weak, so we need at least 10 (just a chosen number, there is no any specific reson)
        raise RuntimeError(f"Too few valid 3D correspondences: {target_points.shape[0]}") # gives runtimeerror
    R, t, inlier_mask, errors = ransac_procrustes(source_points, target_points, iterations=ransac_iterations, threshold=ransac_threshold) # estimates the final transformation
    inlier_count = int(np.sum(inlier_mask)) # counts how many matches are inliers
    total_count = int(target_points.shape[0]) # counts all valid 3d correspondences before ransac filtering
    inlier_ratio = inlier_count / total_count # tells what fraction of correspondences are good
    mean_error = float(np.mean(errors[inlier_mask])) # calculates the average error; only for inliers
    median_error = float(np.median(errors[inlier_mask])) # calculates the middle error among inliers
    return {
        "R": R,                               # final rotation matrix
        "t": t,                               # final translation vector
        "target_points": target_points,       # all valid target 3d points from matched features
        "source_points": source_points,       # all valid source 3d points from matched features
        "inlier_mask": inlier_mask,           # true/false values showing which correspondences are reliable
        "errors": errors,                     # 3d error for every correspondence
        "matches": matches,                   # original 2d sift matches
        "total_correspondences": total_count, # number of valid 3d matched pairs before ransac filtering
        "inlier_count": inlier_count,         # how many correspondences survived ransac
        "inlier_ratio": inlier_ratio,         # a quality score of inlier_count/total_correspondences 
        "mean_error": mean_error,             # average 3d alignment error among inliers
        "median_error": median_error,         # middle 3d alignment error among inliers
    }



