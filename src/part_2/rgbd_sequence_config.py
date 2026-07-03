import os
############ Controls paths and parameters ############
base_dataset_dir = r"C:/Users/sahil/Downloads/part_2_datasets" # main folder
dataset_name = "taag3d"  # choosing dataset manually
def configure_paths(): # creates all paths needed by the main script
    dataset_root = os.path.join(base_dataset_dir, dataset_name) # creates the full path to the chosen dataset
    template_dir = os.path.join(dataset_root, "template") # template RGB/depth files
    sequence_dir = os.path.join(dataset_root, "sequence") # sequence frames
    output_dir = os.path.join("outputs", f"part_2", f"part2_{dataset_name}", "sequence_registration_composed") # creates the output folder
    return {
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "template_rgb_path": os.path.join(template_dir, "templatergb.jpg"),
        "template_mat_path": os.path.join(template_dir, "templatedepth.mat"),
        "sequence_dir": sequence_dir,
        "output_dir": output_dir,
        "transform_dir": os.path.join(output_dir, "transforms"),
        "figure_dir": os.path.join(output_dir, "figures"),
        "registered_cloud_dir": os.path.join(output_dir, "registered_clouds"),
    }

############ registration parameters ############
sift_ratio = 0.80 # controls lowe’s ratio test -> higher value means more matches accepted and maybe more wrong matches while lower value means fewer matches accepted, but usually cleaner matches
ransac_iterations = 2000 # 2000 random samples -> more iterations mean safer but slower while less iterations mean faster but less reliable
ransac_threshold = 0.05 # 3D inlier threshold -> if it lands within 5 cm of its target point, it is considered an inlier
max_depth_for_feature_points = 20.0 # feature-based 3D correspondences
max_depth_for_cloud = 20.0 # maximum depth used when creating the dense point cloud -> points farther than 20 m are removed from the dense cloud
#max_depth_for_cloud = 10.0 # 20.0
point_cloud_stride = 2 # controls how many pixels are used for point-cloud generation -> lighter and faster

############ visualization parameters ############
crop_x_range = None # optional crop limit
crop_y_range = None # optional crop limit
crop_z_range = None # optional crop limit
plane_ransac_iterations = 1000 # number of ransac attempts for plane extraction -> used after point clouds are registered, mainly for cleaning/visualization
plane_distance_threshold = 0.025 # distance threshold for plane inliers
plane_center_percentile = 85 # for cleaning plane outliers after finding the plane
