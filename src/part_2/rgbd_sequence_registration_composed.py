import os # file and folder paths
import csv # saves registration quality results into a csv file
import numpy as np # numerical operations
import scipy.io as sio # to read and write .mat files
from .rgbd_sequence_config import configure_paths, sift_ratio, ransac_iterations, ransac_threshold, max_depth_for_cloud, point_cloud_stride, crop_x_range, crop_y_range, crop_z_range, plane_ransac_iterations, plane_distance_threshold, plane_center_percentile
from .rgbd_io_utils import load_rgb_image, load_depth_and_intrinsics, list_sequence_files, save_xyzrgb
from .rgbd_registration_utils import register_source_to_target, apply_transform
from .rgbd_pointcloud_utils import depth_rgb_to_colored_cloud, crop_cloud, ransac_plane_mask, remove_far_plane_outliers
from .rgbd_visualization_utils import save_before_after_inlier_plot, save_error_histogram, save_quality_plot, save_registered_planes_top_view, save_registered_rgb_cloud_3d
# calls all other files and runs the full pipeline
# loads template rgb-d data and sequence rgb-d frames 
# finds best frame directly to the template 
# registers neighboring frames
# composes transformations
# applies transformation to full point clouds
# saves results and plots
############ Registers a whole RGB-D sequence to the template frame using composed transforms ############
def find_best_frame_to_template(template_rgb, template_depth, template_K, frame_ids, frame_data): # tries to register every sequence frame directly to the template
    # finds which sequence frame (frame number with its RGB image, depth map, and K together) registers best to the template
    best_frame_id = None  # stores the id of the best frame
    best_template_result = None  # stores the full registration result for the best frame
    best_score = -1  # starts very low -> any valid score will be better than -1
    for frame_id in frame_ids:  # tries every frame
        print(f"\nTemplate <- Frame {frame_id}")  # prints what is happening -> registers frame_id into template coordinate system -> move frame to template
        try:
            result = register_source_to_target(             # previous registration pipeline: target -> template while source -> current image
                target_rgb=template_rgb,
                target_depth=template_depth,
                target_K=template_K,
                source_rgb=frame_data[frame_id]["rgb"],     # frame_data -> stores all loaded frames
                source_depth=frame_data[frame_id]["depth"], # frame_data -> stores their respecive depth values
                source_K=frame_data[frame_id]["K"],         # frame_data -> and their intrinsic numbers
                ratio=sift_ratio,
                ransac_iterations=ransac_iterations,
                ransac_threshold=ransac_threshold,
            )                                               # it estimates -> template_point = R @ frame_point + t
            print("valid correspondences:", result["total_correspondences"])
            print("inliers:", result["inlier_count"])
            print("inlier ratio:", f"{result['inlier_ratio']:.3f}")
            print("mean error:", f"{result['mean_error']:.4f}")
            score = result["inlier_count"] / max(result["mean_error"], 1e-6)   # creates a score for choosing the best frame -> higher inlier count and lower mean error are better; if mean error is zero or extremely tiny, division can explode
            if score > best_score: 
                best_score = score  # updates best frame
                best_frame_id = frame_id  # updates its frame id
                best_template_result = result  # updates all data about this frame id
        except Exception as e:  # registration fails
            print("Failed:", e)  # prints the error and tries the next frame
    if best_frame_id is None:  # no frame worked
        raise RuntimeError("Could not find any valid frame-to-template registration.")  # gives runtimeerror

    return best_frame_id, best_template_result   # best frame id and registration result of that frame to template

def compose_transforms(frame_ids, frame_data, best_frame_id, best_template_result):
    # calculates global R and t for every frame 
    # so every frame goes into the template coordinate system
    # other frames may be too far from the template -> so we do: next frame → nearby frame → best frame → template
    best_index = frame_ids.index(best_frame_id) # finds where the best frame is in the sequence
    global_R = {} # stores global rotation for each frame to the template
    global_t = {} # stores global rotation for each frame to the template
    qualities = [] # stores quality results for csv and plots
    global_R[best_frame_id] = best_template_result["R"]   # best frame was registered directly to the template by global R
    global_t[best_frame_id] = best_template_result["t"]   # best frame was registered directly to the template by global t
    qualities.append(make_quality_entry(f"{best_frame_id}_to_template", best_template_result))   # saves the quality of best frame to template
    for i in range(best_index - 1, -1, -1):  # goes backward from the best frame -> registers frames before the best frame
        # current frame -> next frame -> template
        current_id = frame_ids[i]  # the frame we are working on now
        next_id = frame_ids[i + 1]  # neighboring frame after current_id, already connected to the template
        print(f"\nPair registration: Frame {next_id} <- Frame {current_id}")
        result = register_pair(frame_data, target_id=next_id, source_id=current_id) # estimates R_pair @ current_point + t_pair ≈ next_point
        R_pair = result["R"] # extracts pair transformation
        t_pair = result["t"] # extracts pair transformation
        global_R[current_id] = global_R[next_id] @ R_pair
        global_t[current_id] = global_R[next_id] @ t_pair + global_t[next_id] # must also be rotated before adding -> t_pair is expressed in the next_id coordinate relation
        qualities.append(make_quality_entry(f"{current_id}_to_{next_id}", result)) 
        print_quality(result)
    for i in range(best_index + 1, len(frame_ids)): # goes forward after the best frame
        # current frame -> previous frame -> template
        current_id = frame_ids[i] # current frame
        previous_id = frame_ids[i - 1]  # previous ( technically best) frame
        print(f"\nPair registration: Frame {previous_id} <- Frame {current_id}")
        result = register_pair(frame_data, target_id=previous_id, source_id=current_id) # estimates R_pair @ current_point + t_pair ≈ previous_point
        R_pair = result["R"] # extracts pair transformation
        t_pair = result["t"] # extracts pair transformation
        global_R[current_id] = global_R[previous_id] @ R_pair
        global_t[current_id] = global_R[previous_id] @ t_pair + global_t[previous_id] # must also be rotated before adding -> t_pair is expressed in the previous_id coordinate relation
        qualities.append(make_quality_entry(f"{current_id}_to_{previous_id}", result))
        print_quality(result)

    return global_R, global_t, qualities  # frame-to-template rotation and translation; registration quality list

def register_pair(frame_data, target_id, source_id):   # registers one sequence frame to another sequence frame using frame ids
    return register_source_to_target(target_rgb=frame_data[target_id]["rgb"], target_depth=frame_data[target_id]["depth"], target_K=frame_data[target_id]["K"], source_rgb=frame_data[source_id]["rgb"], source_depth=frame_data[source_id]["depth"], source_K=frame_data[source_id]["K"], ratio=sift_ratio, ransac_iterations=ransac_iterations, ransac_threshold=ransac_threshold,) # estimates source_id → target_id

def make_quality_entry(name, result): # extracts useful quality values from a registration result
    return {"frame_id": name, "total_correspondences": result["total_correspondences"], "inlier_count": result["inlier_count"], "inlier_ratio": result["inlier_ratio"], "mean_error": result["mean_error"], "median_error": result["median_error"],} # creates a smaller dictionary for csv and plotting

def print_quality(result): # prints the same quality values to the terminal
    print("valid correspondences:", result["total_correspondences"])
    print("inliers:", result["inlier_count"])
    print("inlier ratio:", f"{result['inlier_ratio']:.3f}")
    print("mean error:", f"{result['mean_error']:.4f}")

def save_global_transforms(frame_ids, global_R, global_t, transform_dir):  # saves each frame’s global transform to a mat file
    for frame_id in frame_ids:   # loops through all frames
        if frame_id not in global_R:   # no global transform exists for that frame
            continue   # skips it
        transform_path = os.path.join(transform_dir, f"transform_{frame_id}.mat") # creates output path
        sio.savemat(transform_path,{"R": global_R[frame_id], "T": global_t[frame_id].reshape(3, 1),}) # saves them with their respective frame ids
        print("saved:", transform_path) # prints to the terminakl

def generate_registered_clouds(frame_ids, frame_data, global_R, global_t, registered_cloud_dir): # generates full registered colored point clouds for all frames
    registered_clouds = {}   # stores full registered clouds
    registered_colors = {}   # stores RGB colors for each point
    registered_plane_clouds = {}   # stores extracted clean plane points
    for frame_id in frame_ids:   # loops through all frames
        if frame_id not in global_R:   # no global transform exists for that frame
            continue   # skips it
        rgb = frame_data[frame_id]["rgb"]  # extracts rgb values for respecyive frame id
        depth = frame_data[frame_id]["depth"]  # extracts depth value for respective frame id
        K = frame_data[frame_id]["K"]  # extracts intrinsic number for respective frame id
        cloud, colors = depth_rgb_to_colored_cloud(rgb, depth, K, max_depth=max_depth_for_cloud, stride=point_cloud_stride,) # Nx3 point array and RGB color array
        cloud_registered = apply_transform(cloud, global_R[frame_id], global_t[frame_id],) # moves the whole point cloud into the template coordinate system
        cloud_registered, colors_registered = crop_cloud(cloud_registered, colors, x_range=crop_x_range, y_range=crop_y_range, z_range=crop_z_range,) # removes unwanted points outside selected ranges
        registered_clouds[frame_id] = cloud_registered   # stores registered cloud
        registered_colors[frame_id] = colors_registered  # stores registered colors
        registered_plane_clouds[frame_id] = extract_clean_plane(cloud_registered) # extracts the dominant plane from the registered cloud -> used later for top view plane visualization
        save_xyzrgb(cloud_registered, colors_registered, os.path.join(registered_cloud_dir, f"registered_cloud_{frame_id}.xyzrgb"),) # saves point cloud to file
        print(
            f"Frame {frame_id}: cloud points = {cloud_registered.shape[0]}, "      # number of registered cloud points
            f"clean plane points = {registered_plane_clouds[frame_id].shape[0]}"   # number of clean plane points
        )

    return registered_clouds, registered_colors, registered_plane_clouds       # all generated point clouds for plotting

def extract_clean_plane(cloud_registered):   # extracts clean plane points from one registered point cloud
    if cloud_registered.shape[0] <= 100:  # 100 or less -> ransac plane fitting needs enough points; without this, plane extraction may be unreliable
        return cloud_registered  # does not try -> simply returns the whole cloud
    plane_mask = ransac_plane_mask(cloud_registered, num_iters=plane_ransac_iterations, threshold=plane_distance_threshold,) # finds the dominant plane: true -> point belongs to plane or false -> point doesn't belong to plane
    plane_points = cloud_registered[plane_mask]   # keeps only plane points

    return remove_far_plane_outliers(plane_points, center_percentile=plane_center_percentile,) # removes far-away plane points based on distance from the plane’s center -> even after ransac some far scattered plane points may remain

def save_quality_csv(qualities, csv_path):   # saves registration quality results to a csv file
    with open(csv_path, "w", newline="") as f:   # opens the csv file for writing and prevents extra blank lines on windows
        writer = csv.DictWriter(f, fieldnames=["frame_id", "total_correspondences", "inlier_count", "inlier_ratio", "mean_error", "median_error",],) # reates a csv writer and columns
        writer.writeheader()   # writes the first row
        for q in qualities:   # writes one row for each quality entry
            writer.writerow(q)
    print("Saved CSV:", csv_path)   # prints saved csv path

def main():
    np.random.seed(0)
    paths = configure_paths()
    os.makedirs(paths["output_dir"], exist_ok=True)
    os.makedirs(paths["transform_dir"], exist_ok=True)
    os.makedirs(paths["figure_dir"], exist_ok=True)
    os.makedirs(paths["registered_cloud_dir"], exist_ok=True)
    print("\n----------------- COMPOSED RGB-D SEQUENCE REGISTRATION -----------------")
    print("This script performs RGB-D registration using SIFT, RANSAC, Procrustes and composed transforms!")
    print(f"Selected dataset: {paths['dataset_name']}")
    print(f"Dataset root: {paths['dataset_root']}")
    print("\nLoading template...")
    template_rgb = load_rgb_image(paths["template_rgb_path"])
    template_depth, template_K = load_depth_and_intrinsics(paths["template_mat_path"])
    print("Template RGB shape:", template_rgb.shape)
    print("Template depth shape:", template_depth.shape)
    frame_ids, rgb_files, depth_files = list_sequence_files(paths["sequence_dir"])
    frame_data = {}
    print("\nLoading sequence frames...")
    for frame_id, rgb_path, depth_path in zip(frame_ids, rgb_files, depth_files):
        depth, K = load_depth_and_intrinsics(depth_path)
        frame_data[frame_id] = {"rgb": load_rgb_image(rgb_path), "depth": depth, "K": K,}
    print("Frames:", frame_ids)
    print("\n----------------- STEP 1: FINDING BEST FRAME TO TEMPLATE -----------------")
    best_frame_id, best_template_result = find_best_frame_to_template(template_rgb, template_depth, template_K, frame_ids, frame_data,)
    print("\nBest frame:", best_frame_id)
    print("Best frame index:", frame_ids.index(best_frame_id))
    print("\n----------------- STEP 2: COMPOSING TRANSFORMS -----------------")
    global_R, global_t, qualities = compose_transforms(frame_ids, frame_data, best_frame_id, best_template_result,)
    print("\n----------------- STEP 3: SAVING GLOBAL TRANSFORMS -----------------")
    save_global_transforms(frame_ids, global_R, global_t, paths["transform_dir"],)
    print("\n----------------- STEP 4: GENERATING REGISTERED CLOUDS -----------------")
    registered_clouds, registered_colors, registered_plane_clouds = generate_registered_clouds(frame_ids, frame_data, global_R, global_t, paths["registered_cloud_dir"],)
    print("\n----------------- STEP 5: SAVING SUMMARY AND FIGURES -----------------")
    csv_path = os.path.join(paths["output_dir"], "composed_sequence_registration_quality.csv")
    save_quality_csv(qualities, csv_path)
    before_after_path = os.path.join(paths["figure_dir"], f"best_frame_{best_frame_id}_to_template_before_after.png",)
    error_histogram_path = os.path.join(paths["figure_dir"], f"best_frame_{best_frame_id}_to_template_error_histogram.png",)
    quality_plot_path = os.path.join(paths["figure_dir"], "composed_sequence_registration_quality.png",)
    plane_top_view_path = os.path.join(paths["figure_dir"], "composed_registered_sequence_planes_top_view.png",)
    rgb_cloud_3d_path = os.path.join(paths["figure_dir"], "composed_registered_sequence_rgb_clouds_3d_new.png",)
    #rgb_cloud_3d_path = os.path.join(paths["figure_dir"], "composed_registered_sequence_rgb_clouds_3d.png",)
    save_before_after_inlier_plot(best_template_result, before_after_path)
    save_error_histogram(best_template_result, error_histogram_path)
    save_quality_plot(qualities, quality_plot_path)
    save_registered_planes_top_view(registered_plane_clouds, frame_ids, plane_top_view_path)
    save_registered_rgb_cloud_3d(registered_clouds, registered_colors, frame_ids, rgb_cloud_3d_path)
    print("\nSaved outputs to:", paths["output_dir"])
    print("\nMain report figures:")
    print("1.", before_after_path)
    print("2.", error_histogram_path)
    print("3.", quality_plot_path)
    print("4.", plane_top_view_path)
    print("5.", rgb_cloud_3d_path)
    print("\nExtra saved outputs:")
    print("- Global transforms:", paths["transform_dir"])
    print("- Registered XYZRGB clouds:", paths["registered_cloud_dir"])
    print("- Quality CSV:", csv_path)
    print("\n----------------- FINISHED COMPOSED RGB-D SEQUENCE REGISTRATION -----------------")

if __name__ == "__main__":
    main()