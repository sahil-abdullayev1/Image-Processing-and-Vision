import numpy as np
def normalize_points(points): # prepares points before estimating the homography
    mean = np.mean(points, axis=0) # calculates the center of all points
    shifted = points - mean # moves the points so that their center becomes close to (0, 0)
    avg_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=1))) # calculates the average distance of the points from the center
    scale = np.sqrt(2) / avg_dist if avg_dist > 1e-12 else 1.0  # chooses a scaling factor
    T = np.array([
        [scale, 0, -scale * mean[0]],     # shifts by mean_x and scale
        [0, scale, -scale * mean[1]],     # shifts by mean_y and scale
        [0, 0, 1]                         # homogeneous coordinate stays unchanged
    ])
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # converts Nx2 points into Nx3 homogeneous points
    norm_h = (T @ points_h.T).T  # applies the normalization matrix to all homogeneous points

    return norm_h[:, :2], T   # normalized 2D points and the normalization matrix

def build_homography_matrix_A(src_pts, dst_pts):  # linear equation matrix used for DLT homography estimation
    A = []  # empty list where all equation rows will be stored
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):  # loops through each matching point pair
        A.append([x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp]) # 1st equation from the point pair
        A.append([0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp]) # 2nd equation from the point pair

    return np.array(A, dtype=np.float64) # converts the list into a numpy matrix

def estimate_homography_dlt(src_pts, dst_pts): # estimates homography using the DLT method
    if src_pts.shape[0] < 4: # should be at least 4 points
        raise ValueError("At least 4 point pairs are required.") # otherwise, gives value error
    src_norm, T_src = normalize_points(src_pts) # normalizes source points and stores source normalization matrix
    dst_norm, T_dst = normalize_points(dst_pts) # normalizes destination points and stores destination normalization matrix
    A = build_homography_matrix_A(src_norm, dst_norm) # builds the DLT equation matrix using normalized points
    _, _, Vt = np.linalg.svd(A) # applies singular value decomposition (SVD) to solve the linear system
    # U-> directions of the equations/rows of A
    # singular_value-> strength of each direction
    # Vt-> directions of possible solutions h
    h = Vt[-1] # takes the last row of Vt, which gives the solution vector for the homography
    H_norm = h.reshape(3, 3) # reshapes the 9-value vector into a 3x3 homography matrix
    # denormalizes
    H = np.linalg.inv(T_dst) @ H_norm @ T_src # homography back to original image coordinate system
    if abs(H[2, 2]) > 1e-12: # checks that the bottom-right value is not too close to zero
        H = H / H[2, 2] # scales the matrix so that H[2, 2] becomes 1

    return H # final 3x3 homography matrix

def apply_homography(points, H): # applies a homography matrix to a set of 2D points
    points_h = np.hstack([points, np.ones((points.shape[0], 1))]) # converts Nx2 points into Nx3 homogeneous points
    projected_h = (H @ points_h.T).T  # multiplies each point by the homography matrix
    z = projected_h[:, 2]  # extracts the third homogeneous coordinate
    valid = np.abs(z) > 1e-12 # checks which projected points have a valid non-zero z value
    projected = np.zeros((points.shape[0], 2)) # empty Nx2 array for final projected 2D points
    projected[valid] = projected_h[valid, :2] / z[valid, None] # converts valid homogeneous points back to normal 2D points

    return projected  # projected 2D points