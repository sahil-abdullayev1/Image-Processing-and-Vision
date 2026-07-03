import numpy as np
from src.part_1.homography import estimate_homography_dlt, apply_homography  # estimated homography and given projected points
def ransac_iterations(P=0.99, w=0.25, n=4): # assuming 25% of matches are correct; how many random 4-match tests do I need to be 99% sure; that at least one test uses only correct matches?
    # computes required ransac iterations: P = desired success probability; w = expected inlier ratio; n = minimum sample size
    return int(np.ceil(np.log(1 - P) / np.log(1 - w**n))) # np.ceil rounds the number upward

def ransac_homography(src_pts, dst_pts, threshold=5.0, P=0.99, w=0.25): 
    # robustly estimates homography using ransac: src_pts -> Nx2 source points; dst_pts -> Nx2 destination/template points
    n_samples = 4  # to estimate homography, ransac needs at least 4 point pairs
    K = ransac_iterations(P=P, w=w, n=n_samples) # computes how many random tests ransac should do
    N = src_pts.shape[0]  # gets the total number of matched point pairs
    best_inliers = []  # the best inlier matches found so far
    best_H = None  # the best homography found so far
    if N < n_samples:   # if we have fewer than 4 matches
        raise ValueError("not enough matches for homography ransac.") # impossible homography estimation
    for _ in range(K):  # repeats the ransac process K times
        idx = np.random.choice(N, n_samples, replace=False) # randomly choose 4 different matches from all available matches
        try: # estimates a temporary homography using only these 4 random matches
            H = estimate_homography_dlt(src_pts[idx], dst_pts[idx]) 
        except Exception: # if this random group gives an error
            continue # skips it and tries another group
        projected = apply_homography(src_pts, H) # temporary homography to project all source points into destination image
        errors = np.linalg.norm(dst_pts - projected, axis=1) # calculates distance between real destination and projected points
                                                             # small error -> agrees with this homography
                                                             # large error -> probably not
        inliers = np.where(errors < threshold)[0]   # keeps only matches whose reprojection error is smaller than the threshold; threshold -> points within 5 pixels are accepted as inliers
        if len(inliers) > len(best_inliers): # if temporary homography has more inliers than the previous best one
            best_inliers = inliers # saves them as the best ones
            best_H = H   # saves this homography as the best one as well
    if best_H is None or len(best_inliers) < 4: # no valid homography was found
        raise RuntimeError("RANSAC failed to find a valid homography.") # stops it with an error
    final_H = estimate_homography_dlt(src_pts[best_inliers], dst_pts[best_inliers])  # estimates the final homography again by using all best inliers

    return final_H, best_inliers, K # final homography, inlier indices and number of ransac iterations