import os
import numpy as np
import cv2
import open3d as o3d
from scipy.io import loadmat
from glob import glob
from .rgbd_sequence_config import configure_paths
def depth_to_pointcloud(depth, rgb, K, max_depth=None):
    h, w = depth.shape
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    Z = depth
    X = (u_grid - cx) * Z / fx
    Y = (v_grid - cy) * Z / fy
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3) / 255.0
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    if max_depth is not None:
        valid = valid & (points[:, 2] < max_depth)
    points = points[valid]
    colors = colors[valid]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    return pc

def create_camera_pyramid(R, T, fx, fy, cx, cy, width, height, scale=1.0, color_counter=0):
    C = np.array(T).reshape(3)
    z = scale
    corners = np.array([
        [(0 - cx) * z / fx,       (0 - cy) * z / fy,        z],
        [(width - cx) * z / fx,   (0 - cy) * z / fy,        z],
        [(width - cx) * z / fx,   (height - cy) * z / fy,   z],
        [(0 - cx) * z / fx,       (height - cy) * z / fy,   z],
    ])
    corners = (R @ corners.T).T + C
    points = np.vstack([C.reshape(1, 3), corners])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if color_counter % 3 == 0:
        color = [1, 0, 0]
    elif color_counter % 3 == 1:
        color = [0, 1, 0]
    else:
        color = [0, 0, 1]
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])

    return line_set

def main():
    paths = configure_paths()
    sequence_dir = paths["sequence_dir"]
    transform_dir = paths["transform_dir"]
    figure_dir = paths["figure_dir"]
    if not os.path.exists(transform_dir):
        raise FileNotFoundError(
            f"Transform folder not found: {transform_dir}\n"
            "Run rgbd_sequence_registration_composed.py first...!"
        )
    os.makedirs(figure_dir, exist_ok=True)
    print("\n--------------- OPEN3D RGB-D REGISTRATION VERIFIER ---------------")
    print("Dataset:", paths["dataset_name"])
    print("Transform folder:", transform_dir)
    rgb_files = sorted(glob(os.path.join(sequence_dir, "*.jpg")))
    pc_list = []
    camera_pyramids = []
    color_counter = 0
    for rgb_path in rgb_files:
        frame_idx = os.path.basename(rgb_path).split("_")[-1].split(".")[0]
        transform_path = os.path.join(transform_dir, f"transform_{frame_idx}.mat")
        if not os.path.exists(transform_path):
            print(f"Skipping frame {frame_idx}: no transform")
            continue
        depth_path = rgb_path.replace(".jpg", ".mat")
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print("Could not read:", rgb_path)
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth_data = loadmat(depth_path)
        depth = depth_data["depth"]
        K = depth_data["K"]
        transform_data = loadmat(transform_path)
        R = transform_data["R"]
        T = transform_data["T"].reshape(3)
        pc = depth_to_pointcloud(depth, rgb, K, max_depth=None)
        points = np.asarray(pc.points)
        points_transformed = (R @ points.T).T + T
        pc.points = o3d.utility.Vector3dVector(points_transformed)
        pc_list.append(pc)
        h, w = depth.shape
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        cam_pyramid = create_camera_pyramid(R, T, fx, fy, cx, cy, w, h, scale=1.0, color_counter=color_counter,)
        camera_pyramids.append(cam_pyramid)
        color_counter += 1
        print(f"Loaded frame {frame_idx}: {np.asarray(pc.points).shape[0]} points")
    if len(pc_list) == 0:
        raise RuntimeError("No point clouds loaded!")
    merged_pc = pc_list[0]
    for pc in pc_list[1:7]:
        merged_pc += pc
    print("Merged cloud points:", np.asarray(merged_pc.points).shape[0])
    render_geometries = [merged_pc, *camera_pyramids]
    print("\nOpening Open3D window...")
    print("Rotate and zoom by mouse!")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D RGB-D registration verifier", width=1400, height=900, visible=True,)
    for geom in render_geometries:
        vis.add_geometry(geom)
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1.0, 1.0, 1.0])
    render_option.point_size = 1.5
    print("\nRotate and zoom the scene!")
    vis.run()
    screenshot_path = os.path.join(figure_dir, "open3d_registered_scene.png")
    vis.capture_screen_image(screenshot_path, do_render=True)
    vis.destroy_window()
    print("Saved Open3D screenshot:", screenshot_path)
    print("\n--------------- FINISHED OPEN3D RGB-D REGISTRATION VERIFIER ---------------")

if __name__ == "__main__":
    main()













