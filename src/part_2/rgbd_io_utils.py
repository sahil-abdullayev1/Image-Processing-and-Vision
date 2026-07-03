import os
import re
import numpy as np
import cv2
import scipy.io as sio
# loads rgb images, depth maps and camera intrinsics k 
# finds matching rgb/depth sequence files 
# saves xyzrgb point clouds
def load_rgb_image(path):
    # reads image and converts bgr to rgb
    image_bgr = cv2.imread(path) # reads the image from disk
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # converts the image from bgr to rgb

def load_depth_and_intrinsics(mat_path): # loads depth map and camera intrinsic matrix k
    # reads depth and K from .mat file
    if not os.path.exists(mat_path): # if .mat file does not exist
        raise FileNotFoundError(f"mat file not found: {mat_path}") # gives a clear error
    data = sio.loadmat(mat_path) # reads the matlab file like a python dictionary
    if "depth" not in data or "K" not in data: # checks if data is available
        raise KeyError(
            f"Expected variables: 'depth' and 'K' inside {mat_path}. "
            f"Available variables: {list(data.keys())}"
        )
    depth = data["depth"].astype(np.float32) # depth map
    K = data["K"].astype(np.float32) # camera matrix k

    return depth, K

def list_sequence_files(sequence_dir): # finds all rgb-d frame pairs inside the sequence folder
    # finds matching frame_0000.jpg + frame_0000.mat pairs
    if not os.path.exists(sequence_dir): # checks if folder does exist
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}") # gives a clear errior
    image_pattern = re.compile(r"frame_(\d{4})\.jpg$", re.IGNORECASE) # image name ruele - exactly 4 digits
    depth_pattern = re.compile(r"frame_(\d{4})\.mat$", re.IGNORECASE) # depth map name rule - exactly 4 digits
    image_dict = {} # stores images by frame number
    depth_dict = {} # stores depth maps by frame number
    for filename in os.listdir(sequence_dir):
        image_match = image_pattern.match(filename) # checks if the filename matches the image pattern
        depth_match = depth_pattern.match(filename) # checks if the filename matches the depth pattern
        if image_match: # an image
            idx = image_match.group(1) # extracts frame number
            image_dict[idx] = os.path.join(sequence_dir, filename) # saves it as it is
        if depth_match: # a depth map
            idx = depth_match.group(1) # extracts frame number
            depth_dict[idx] = os.path.join(sequence_dir, filename) # saves it as it is
    common_indices = sorted(set(image_dict.keys()) & set(depth_dict.keys())) # finds frame numbers that exist in both dictionaries
    if len(common_indices) == 0: # no matching rgb-d pairs
        raise RuntimeError(        
            f"No valid RGB-D sequence frames found in {sequence_dir}. "
            "Expected names: frame_0000.jpg and frame_0000.mat."
        ) # stops and tells us the expected naming style
    frame_ids = []
    rgb_files = []
    depth_files = []
    for idx in common_indices:
        frame_ids.append(idx) # frame id
        rgb_files.append(image_dict[idx]) # rgb file path
        depth_files.append(depth_dict[idx]) # depth .mat file path

    return frame_ids, rgb_files, depth_files

def save_xyzrgb(points, colors, output_path): # saves a colored 3d point cloud
    # saves colored point clouds as X Y Z R G B
    if points.shape[0] == 0: # zero points
        np.savetxt(output_path, np.empty((0, 6))) # saves it as an empty file with 6 columns
        return
    xyzrgb = np.hstack([points, colors * 255.0]) # combines 3d points and colors
    np.savetxt(output_path, xyzrgb, fmt="%.6f %.6f %.6f %.0f %.0f %.0f") # saves the file as text contains 6 decimen nums for XYZ and whole nums for RGB


    