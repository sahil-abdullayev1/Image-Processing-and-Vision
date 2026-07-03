# Image Registration and 3D Vision

This repository contains an image processing and computer vision project focused on 2D image registration, homography estimation, RGB-D processing and 3D point cloud registration.

The project has two main parts:

- **Part 1:** 2D image registration using SIFT, feature matching, RANSAC and homography estimation.
- **Part 2:** RGB-D 3D registration using SIFT feature matching, depth-based 3D point conversion, RANSAC-Procrustes registration and point cloud alignment.

A more detailed project description is available in `docs/project_overview.md`.

## Part 1: 2D Image Registration

**Part 1** estimates homographies between image frames.

The main steps are:

1. Detect SIFT keypoints and descriptors in the template image and input frame.
2. Match descriptors using BFMatcher and Lowe's ratio test.
3. Use RANSAC to reject incorrect matches.
4. Estimate a homography matrix using the Direct Linear Transform method.
5. Warp the input frame into the template coordinate system.
6. Save homography matrices, warped images, overlay images and quality summaries.

Two methods are implemented:

- **Direct-to-template method:** each frame is matched directly to the template image.
- **Sequence-composed method:** consecutive frames are matched first and the estimated homographies are composed back to the original template.

## Part 2: RGB-D 3D Registration

**Part 2** performs 3D registration using RGB-D data.

The main steps are:

1. Detect and match SIFT features between RGB images.
2. Convert matched 2D pixels into 3D points using depth maps and the camera intrinsic matrix.
3. Estimate a rigid transformation using Procrustes alignment.
4. Use RANSAC to reject incorrect 3D correspondences.
5. Compose pairwise transformations across the sequence.
6. Generate and visualize registered point clouds.

The final result is a set of point clouds aligned in a common template coordinate frame.

## Repository Structure

```text
Image-Registration-and-3D-Vision/
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   └── project_overview.md
│
├── src/
│   ├── part_1/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── homography.py
│   │   ├── main_part1_universal.py
│   │   ├── part1_pipeline.py
│   │   ├── quality.py
│   │   └── ransac.py
│   │
│   └── part_2/
│       ├── __init__.py
│       ├── rgbd_io_utils.py
│       ├── rgbd_open3d_verifier.py
│       ├── rgbd_pointcloud_utils.py
│       ├── rgbd_registration_utils.py
│       ├── rgbd_sequence_config.py
│       ├── rgbd_sequence_registration_composed.py
│       └── rgbd_visualization_utils.py
│
└── results/
    ├── part_1/
    │   ├── ireland/
    │   │   ├── homographies/
    │   │   ├── overlays/
    │   │   ├── warped/
    │   │   ├── ireland_direct_registration_quality.png
    │   │   └── ireland_summary.csv
    │   │
    │   ├── lisbon/
    │   │   ├── homographies/
    │   │   ├── overlays/
    │   │   ├── warped/
    │   │   ├── lisbon_direct_registration_quality.png
    │   │   └── lisbon_summary.csv
    │   │
    │   └── taagpiv/
    │       ├── homographies/
    │       ├── overlays/
    │       ├── warped/
    │       ├── taagpiv_sequence_registration_quality.png
    │       └── taagpiv_summary.csv
    │
    └── part_2/
        ├── lion dataset/
        │   ├── plane_segmentation/
        │   ├── depth_map.png
        │   ├── valid_depth_mask.png
        │   ├── lion_cloud_0.xyz
        │   ├── lion_cloud_clean.xyz
        │   ├── lion_colored_cloud_0.xyzrgb
        │   └── visualization images
        │
        ├── plondres dataset/
        │   └── sequence_registration_composed/
        │
        ├── plondres2 dataset/
        │   └── sequence_registration_composed/
        │
        └── taag3d/
            ├── registration/
            ├── sequence_registration/
            └── sequence_registration_composed/

```

## Results

Reduced result outputs are included in `results/`.

For Part 1, a selected subset of warped and overlay images is included instead of the complete output folders:

- Ireland: 23 warped images and 23 overlay images out of 215 frames
- Lisbon: 46 warped images and 46 overlay images out of 900 frames
- Homography files and summary CSV files are included for the selected runs

For Part 2, the result outputs are included in the repository, including point cloud registration results, transformation files, plots and visualization examples.

The original full datasets are not included due to file size limitations.

## Datasets

The original full datasets are not included in this repository due to file size limitations. The repository contains the implementation and selected/generated result outputs needed to demonstrate the project.

## Main Methods Used

- SIFT feature detection
- BFMatcher descriptor matching
- Lowe's ratio test
- Homography estimation using DLT
- RANSAC
- RGB-D pixel-to-3D conversion
- Procrustes rigid registration
- RANSAC-Procrustes registration
- Transform composition
- Point cloud generation
- Open3D visualization

## Notes

The repository focuses on the final implementation and reduced result outputs. The original full datasets and cache files are excluded to keep the repository size manageable.
