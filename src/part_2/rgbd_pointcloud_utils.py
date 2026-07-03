import numpy as np # arrays, masks, matrix operations, random sampling, cross product, distances....
from .rgbd_sequence_config import max_depth_for_cloud, point_cloud_stride, plane_ransac_iterations, plane_distance_threshold, plane_center_percentile
# converts RGB-D into 3D point clouds
def depth_rgb_to_colored_cloud(rgb, depth, K, max_depth=max_depth_for_cloud, stride=point_cloud_stride):    
    # takes RGB image + depth map D + camera matrix K aaaand converts pixels into 3D colored points
    h, w = depth.shape # gets the height (rows) and width (columns) of the depth map
    fx = float(K[0, 0]) # extracts focal length in x direction parameter from K
    fy = float(K[1, 1]) # extracts focal length in y direction parameter from K
    cx = float(K[0, 2]) # extracts image center x parameter from K
    cy = float(K[1, 2]) # extracts image center y parameter from K
    u_grid, v_grid = np.meshgrid(   # creates pixel coordinate grids
        np.arange(0, w, stride),    # horizontal pixel coordinate grid: left -> right
        np.arange(0, h, stride)     # vertical pixel coordinate grid: top -> bottom
    )                               # 2 -> 0, 2, 4, 6...
    Z = depth[v_grid, u_grid] # takes depth value for each selected pixel
    valid = (Z > 0) & (Z < max_depth) # creates a mask for valid depth values -> bigger than 0 while smaller than max_depth
                                      # far points are usually noisy or not useful
    u_valid = u_grid[valid].astype(np.float32) # valid pixel x positions
    v_valid = v_grid[valid].astype(np.float32) # valid pixel y positions
    Z_valid = Z[valid].astype(np.float32) # valid depth values
    X_valid = (u_valid - cx) * Z_valid / fx  # pixel-to-3D conversion
    Y_valid = (v_valid - cy) * Z_valid / fy  # pixel-to-3D conversion
    points = np.stack([X_valid, Y_valid, Z_valid], axis=1).astype(np.float32)  # X Y Z coordinates
    colors = rgb[v_grid[valid], u_grid[valid]].astype(np.float32) / 255.0  # R G B colors

    return points, colors

def crop_cloud(points, colors, x_range=None, y_range=None, z_range=None): # keeps just the useful middle part of the 3D scene; removes the rest
    mask = np.ones(points.shape[0], dtype=bool) # creates a boolean mask
    if x_range is not None:
        x_condition = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) # keeps only points whose X coordinate is inside the range
        mask = mask & x_condition

    if y_range is not None:
        y_condition = (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) # keeps only points whose Y coordinate is inside the range
        mask = mask & y_condition

    if z_range is not None:
        z_condition = (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1]) # keeps only points whose Z coordinate is inside the range
        mask = mask & z_condition

    return points[mask], colors[mask]

def random_sample(points, colors=None, max_points=15000): # reduces the number of plotted points
    # only for visualization; does not change registration
    if points.shape[0] <= max_points:  # if cloud already has fewer than max_points, no need to sample
        if colors is None:  # no color
            return points  # only points
        return points, colors  # both points and colors
    idx = np.random.choice(points.shape[0], max_points, replace=False) # randomly chooses max_points indices..; does not choose same point twice..
    if colors is None: # no color
        return points[idx] # only sampled pointst

    return points[idx], colors[idx] # both sampled points and colors

def ransac_plane_mask(points, num_iters=plane_ransac_iterations, threshold=plane_distance_threshold): # finds the dominant plane in a point cloud using ransac
    N = points.shape[0]  # gets number of points
    if N < 3: # less than minimum
        return np.ones(N, dtype=bool)   # all points as valid
    
    best_inliers = np.array([], dtype=int)   # stores the best plane inliers found so far

    for _ in range(num_iters):
        idx = np.random.choice(N, 3, replace=False) # randomly selects 3 points
        p1, p2, p3 = points[idx] # gets those 3 selected points
        n = np.cross(p2 - p1, p3 - p1) # normal vector -> only tells the plane’s direction/orientation -> n = [a, b, c]
        norm = np.linalg.norm(n) # computes the length of the normal vector

        if norm < 1e-8:  # if almost zero -> three points are collinear or almost the same
            continue  # skips this sample

        n = n / norm  # normalizes the normal vector -> normal length becomes 1 -> makes distance computation easier
        d = -np.dot(n, p1)  # n_x*X + n_y*Y + n_z*Z + d = 0 -> n = [a, b, c] -> plane must pass through p1 -> p1 was chosen because its 1st...

        distances = np.abs(points @ n + d)  # computes distance from every point to this plane; n is normalized, this is perpendicular distancve
        inliers = np.where(distances < threshold)[0]  # finds all points close enough to the plane -> points within 2.5 cm of the plane are considered plane points

        if len(inliers) > len(best_inliers): # if it has more inliers than previous one
            best_inliers = inliers  # keeps it as the best one

    mask = np.zeros(N, dtype=bool)  # creates final boolean mask
    mask[best_inliers] = True   # plane inliers

    return mask

def remove_far_plane_outliers(points, center_percentile=plane_center_percentile): # cleans the plane points for visualization
    if points.shape[0] < 50: # point cloud -> fewer than 50 points
        return points   # just keeps them as they are
    xy_points = points[:, :2] # mostly cleaning the top-view of the plane -> takes just X and Y, not Z...
    center_xy = np.median(xy_points, axis=0) # finds the median center of the plane points in XY
    distances = np.linalg.norm(xy_points - center_xy, axis=1) # computes how far each point is from the XY center
    distance_threshold = np.percentile(distances, center_percentile) # keeps the closest 85% of points and removes the farthest 15%
    clean_mask = distances <= distance_threshold # selects only points inside that threshold
    return points[clean_mask]









