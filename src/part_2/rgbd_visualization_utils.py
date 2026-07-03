import numpy as np
import matplotlib.pyplot as plt
from .rgbd_registration_utils import apply_transform
from .rgbd_pointcloud_utils import random_sample
# saves figures after registration
def save_before_after_inlier_plot(result, output_path):
    target_points = result["target_points"]
    source_points = result["source_points"]
    inlier_mask = result["inlier_mask"]
    R = result["R"]
    t = result["t"]
    target_inliers = target_points[inlier_mask]
    source_inliers = source_points[inlier_mask]
    source_registered = apply_transform(source_inliers, R, t)
    before_errors = np.linalg.norm(source_inliers - target_inliers, axis=1)
    after_errors = np.linalg.norm(source_registered - target_inliers, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].scatter(target_inliers[:, 0], target_inliers[:, 1], s=18, alpha=0.75, label="Target")
    axes[0].scatter(source_inliers[:, 0], source_inliers[:, 1], s=18, alpha=0.75, label="Source before")
    axes[0].set_title("Before Registration")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].grid(True)
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].text(
        0.02, 
        0.96, 
        f"Mean error: {np.mean(before_errors):.4f} m", 
        transform=axes[0].transAxes, 
        verticalalignment="top", 
        bbox=dict(facecolor="white", alpha=0.8)
    )

    axes[1].scatter(target_inliers[:, 0], target_inliers[:, 1], s=18, alpha=0.75, label="Target")
    axes[1].scatter(source_registered[:, 0], source_registered[:, 1], s=18, alpha=0.75, label="Source after")
    axes[1].set_title("After Registration")
    axes[1].set_xlabel("X [m]")
    axes[1].set_ylabel("Y [m]")
    axes[1].grid(True)
    axes[1].axis("equal")
    axes[1].legend()
    axes[1].text(
        0.02,
        0.96,
        f"Mean error: {np.mean(after_errors):.4f} m\n"
        f"Median error: {np.median(after_errors):.4f} m",
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )

    fig.suptitle("Pairwise RGB-D Registration using RANSAC + Procrustes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_error_histogram(result, output_path):
    errors = result["errors"]
    inlier_mask = result["inlier_mask"]
    inlier_errors = errors[inlier_mask]
    plt.figure(figsize=(9, 6))
    plt.hist(inlier_errors, bins=25, alpha=0.85)
    plt.title("Registration Error Distribution for RANSAC Inliers")
    plt.xlabel("3D alignment error [m]")
    plt.ylabel("Number of correspondences")
    plt.grid(True)
    text = (
        f"Mean: {np.mean(inlier_errors):.4f} m\n"
        f"Median: {np.median(inlier_errors):.4f} m\n"
        f"Max: {np.max(inlier_errors):.4f} m"
    )
    plt.text(
        0.97, 
        0.95, 
        text, 
        transform=plt.gca().transAxes, 
        horizontalalignment="right", 
        verticalalignment="top", 
        bbox=dict(facecolor="white", alpha=0.8)
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_quality_plot(qualities, output_path):
    frame_ids = [q["frame_id"] for q in qualities]
    inlier_ratios = [q["inlier_ratio"] for q in qualities]
    mean_errors = [q["mean_error"] for q in qualities]
    x = np.arange(len(frame_ids))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(x, inlier_ratios, marker="o", label="Inlier ratio")
    ax1.set_xlabel("Pair/frame")
    ax1.set_ylabel("RANSAC inlier ratio")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(x, mean_errors, marker="s", linestyle="--", label="Mean inlier error")
    ax2.set_ylabel("Mean inlier error [m]")

    ax1.set_xticks(x)
    ax1.set_xticklabels(frame_ids, rotation=45, ha="right")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.title("Composed RGB-D Sequence Registration Quality")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_registered_planes_top_view(registered_plane_clouds, frame_ids, output_path):
    plt.figure(figsize=(10, 9))
    for frame_id in frame_ids:
        points = registered_plane_clouds.get(frame_id, np.empty((0, 3)))
        if points.shape[0] == 0:
            continue
        points_sampled = random_sample(points, max_points=7000)
        plt.scatter(
            points_sampled[:, 0],
            points_sampled[:, 1],
            s=1.2,
            alpha=0.45,
            label=f"Frame {frame_id}"
        )

    plt.title("Composed RGB-D Sequence Registration: Dominant Plane Top View")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.grid(True)
    plt.axis("equal")
    plt.legend(markerscale=8, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_registered_rgb_cloud_3d(registered_clouds, registered_colors, frame_ids, output_path):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    for frame_id in frame_ids:
        points = registered_clouds.get(frame_id, np.empty((0, 3)))
        colors = registered_colors.get(frame_id, np.empty((0, 3)))
        if points.shape[0] == 0:
            continue
        points_sampled, colors_sampled = random_sample(
            points,
            colors,
            max_points=7000
        )

        ax.scatter(
            points_sampled[:, 0],
            points_sampled[:, 1],
            points_sampled[:, 2],
            c=colors_sampled,
            s=0.6,
            alpha=0.55
        )

    ax.set_title("Composed RGB-D Sequence - 3D Colored Point Clouds")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    #ax.view_init(elev=-150, azim=-70)
    ax.view_init(elev=25, azim=-65)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)