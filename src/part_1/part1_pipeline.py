import cv2 
import os  
from src.part_1.features import detect_sift_features, match_descriptors, get_matched_points # come from features.py
from src.part_1.ransac import ransac_homography # estimates a robust homography by rejecting bad matches.
def estimate_pair_homography(template_path, frame_path, ratio=0.75, threshold=5.0): 
    ######### Estimates homography from frame image to template image #########
    template_img, template_pts, template_des, template_kp = detect_sift_features(template_path) # detects sift features in the template image
    # template_img -> original template image; template_pts -> coordinates of detected keypoints; template_des -> sift descriptors; template_kp -> opencv keypoint objects
    frame_img, frame_pts, frame_des, frame_kp = detect_sift_features(frame_path) # detects sift features in the current frame
    # frame_pts and frame_des will later be matched with the template features
    matches = match_descriptors(template_des, frame_des, ratio=ratio) # matches descriptors between the template and the frame, only good and accepted matches
    matched_template, matched_frame = get_matched_points(template_pts, frame_pts, matches) # converts the OpenCV match objects into two NumPy arrays of 2D points
    # matched_template contains points from the template image
    # matched_frame contains corresponding points from the current frame
    H, inliers, K = ransac_homography(src_pts=matched_frame, dst_pts=matched_template, threshold=threshold, P=0.99, w=0.25)
    info = {                           # stores useful numerical information about the homography estimation
        "matches": len(matches),
        "inliers": len(inliers),
        "ransac_iterations": K
    }                                  # later used for quality evaluation and report summary

    return H, info  # homography matrix and information dictionary

def warp_frame_to_template(template_path, frame_path, H, output_path):
    ######### Warpes frame into template image coordinate system #########
    template_img = cv2.imread(template_path) # reads the template image from disk -> its size will define the output image size
    frame_img = cv2.imread(frame_path) # reads the current frame from disk -> the image that will be warped
    h, w = template_img.shape[:2] # gets the height and width of the template image -> the warped frame should have the same size as the template
    warped = cv2.warpPerspective(frame_img, H, (w, h)) # applies the homography to the current frame -> transforms the current frame into the template coordinate system
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # creates the output folder if it does not already exist
    cv2.imwrite(output_path, warped) # saves the warped image to disk

    return warped # later it can be used for overlay visualization

def save_overlay(template_path, warped, output_path):
    ######### Saves 50/50 blend between template and warped frame #########
    template_img = cv2.imread(template_path)   # reads the template image from disk
    overlay = cv2.addWeighted(template_img, 0.5, warped, 0.5, 0)  # blends the template and warped image with equal weight
                                                                  # 0.5 means 50% template and 50% warped image
                                                                  # if alignment is good, objects should overlap clearly
    os.makedirs(os.path.dirname(output_path), exist_ok=True)      # creates the output folder if it does not already exist
    cv2.imwrite(output_path, overlay) # saves the overlay image to disk

    return overlay    # overlay image