# Project Overview

This repository contains the implementation of an Image Processing and Vision project focused on 2D image registration and RGB-D 3D registration.

The project is divided into two main parts.

## Part 1: 2D Image Registration Using Homography

The first part focuses on estimating the geometric transformation between image frames. SIFT features are detected in a template image and in the input frames. The corresponding descriptors are matched using a brute-force matcher and Lowe's ratio test.

Since feature matching can include incorrect correspondences, RANSAC is used to robustly estimate a homography matrix. The estimated homography is then used to warp the input frame into the coordinate system of the template image.

Two strategies are implemented:

- **Direct-to-template registration:** each frame is matched directly to the template image.
- **Sequence-composed registration:** consecutive frames are matched to each other and the resulting homographies are composed to transform each frame back to the original template.

The outputs of this part include homography matrices, warped images, overlay images and registration quality summaries.

## Part 2: RGB-D 3D Registration

The second part extends the registration problem to 3D using RGB-D data. SIFT features are detected and matched between RGB images. The matched 2D feature points are then converted into 3D points using the corresponding depth maps and the camera intrinsic matrix.

A rigid transformation between frames is estimated using Procrustes alignment inside a RANSAC loop. RANSAC is used to reject incorrect 3D correspondences and estimate a robust rotation and translation between frames.

For sequence registration, pairwise transformations are composed so that multiple frames can be registered into a common template coordinate frame. The registered RGB-D frames are then used to generate and visualize aligned point clouds.

The outputs of this part include rigid transformations, registration error statistics, registered point clouds and visualization results.

## Main Methods

The main techniques used in this project are:

- SIFT feature detection
- Descriptor matching with BFMatcher
- Lowe's ratio test
- Homography estimation using DLT
- RANSAC for robust model estimation
- RGB-D pixel-to-3D conversion
- Procrustes rigid registration
- Transform composition
- Point cloud generation and visualization

## Data and Results

The full datasets and complete generated output folders are not included directly in this repository due to file size limitations. Reduced/generated result outputs are provided in the repository, while the original full datasets are excluded due to file size limitations.