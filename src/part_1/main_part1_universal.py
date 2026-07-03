import os # path operations
import glob # finds image files inside a folder
import csv # creates the summary.csv file
import numpy as np # matrix operations
from src.part_1.part1_pipeline import estimate_pair_homography, warp_frame_to_template, save_overlay # pipeline outputs
from src.part_1.quality import homography_quality  # quality checker
dataset_dir = r"C:/Users/sahil/Downloads/part_1_datasets/imageslisbon/imageslisbon" # folder contains the input images
method = "direct" # "direct" -> each frame is matched directly to the template; ireland and lisbon
# "sequence" -> each frame is matched to the previous frame, then homographies are composed; taagpiv
template_name = None # first image in the folder is used as the template
# template_name = os.path.join(dataset_dir, "20251028_165814.jpg") # taagpiv
output_dir = "outputs/part_1/universal/part1_universal_lisbon_direct" # all results will be saved in this folder
ratio = 0.75 # lowe's ratio test value
threshold = 5.0 # ransac reprojection threshold, basically max error in pixels
stop_on_bad = True # stops sequence when bad homography appears -> mainly used for sequence-composed method

########### MAIN HELPERS ###########
def get_images(dataset_dir):
    image_files = []    # contains received input images
    for ext in ("*.jpg", "*.jpeg", "*.png"):  # checks three possible image extensions
        image_files.extend(glob.glob(os.path.join(dataset_dir, ext)))  # finds all images with the current extension
    return sorted(image_files)  # sorting is important -> frame order matters

def create_output_folders(output_dir):   
    homography_dir = os.path.join(output_dir, "homographies")  # homography matrices
    warped_dir = os.path.join(output_dir, "warped") # warped images
    overlay_dir = os.path.join(output_dir, "overlays") # overlay images
    os.makedirs(homography_dir, exist_ok=True) # creates folder
    os.makedirs(warped_dir, exist_ok=True) # creates folder
    os.makedirs(overlay_dir, exist_ok=True) # creates folder
    return homography_dir, warped_dir, overlay_dir

def save_summary(output_dir, rows):
    summary_path = os.path.join(output_dir, "summary.csv") # creates the full path for the summary file
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["frame_name", "target_frame", "matches", "inliers", "ratio", "quality", "status",],)
        writer.writeheader() # column names
        writer.writerows(rows) # all stored result rows into the csv file
    print("\nSummary saved to:", summary_path)

def save_alignment_result(template_path, frame_path, H, homography_dir, warped_dir, overlay_dir):  # saves all outputs for one successful frame
    frame_name = os.path.splitext(os.path.basename(frame_path))[0]   # extracts the image name without extension
    np.save(os.path.join(homography_dir, f"H_{frame_name}.npy"), H)   # saves the homography matrix as a .npy file
    warped_path = os.path.join(warped_dir, f"warped_{frame_name}.jpg")   # saves the warped image as a .jpg file
    overlay_path = os.path.join(overlay_dir, f"overlay_{frame_name}.jpg")  # saves the overlay image as a .jpg file
    warped = warp_frame_to_template(template_path, frame_path, H, warped_path) # calls the pipeline function that warps the current frame into the template coordinate system and saves the warped image
    save_overlay(template_path, warped, overlay_path) # saves overlay results

def make_summary_row(frame_path, target_path, quality_info, status):  # creates one row for summary.csv
    return {
        "frame_name": os.path.basename(frame_path),
        "target_frame": os.path.basename(target_path) if target_path else "TEMPLATE",
        "matches": quality_info.get("matches", 0),
        "inliers": quality_info.get("inliers", 0),
        "ratio": quality_info.get("ratio", 0.0),
        "quality": quality_info.get("quality", "failed"),
        "status": status,
    }

########### DIRECT-TO-TEMPLATE METHOD ###########
def run_direct(frames, template_index):
    template_path = frames[template_index]
    homography_dir, warped_dir, overlay_dir = create_output_folders(output_dir)
    rows = [make_summary_row(template_path, None, {"matches": 0, "inliers": 0, "ratio": 0.0, "quality": "template"}, "template image",)]
    print("\nRunning DIRECT-TO-TEMPLATE method")
    print("Template:", os.path.basename(template_path))
    for i, frame_path in enumerate(frames):
        if i == template_index:
            continue
        print(f"\nEstimating {os.path.basename(frame_path)} -> {os.path.basename(template_path)}")
        try:
            H, info = estimate_pair_homography(template_path=template_path, frame_path=frame_path, ratio=ratio, threshold=threshold,)
            quality_info = homography_quality(info["matches"], info["inliers"])
            print(
                "Matches:", quality_info["matches"],
                "Inliers:", quality_info["inliers"], "/", quality_info["matches"],
                "Ratio:", f'{quality_info["ratio"]:.3f}',
                "Quality:", quality_info["quality"],
            )
            if quality_info["quality"] == "bad":
                print("Skipping bad homography.")
                rows.append(make_summary_row(frame_path, template_path, quality_info, "skipped bad homography"))
                continue
            save_alignment_result(template_path, frame_path, H, homography_dir, warped_dir, overlay_dir)
            rows.append(make_summary_row(frame_path, template_path, quality_info, "saved"))
            print("Saved.")
        except Exception as error:
            print("Failed:", error)
            rows.append(make_summary_row(frame_path, template_path, {}, str(error)))

    save_summary(output_dir, rows)

########### SEQUENCE-COMPOSED METHOD ###########
def run_sequence(frames, template_index):
    template_path = frames[template_index]
    homography_dir, warped_dir, overlay_dir = create_output_folders(output_dir)
    H_to_template = {template_index: np.eye(3)}
    rows = [make_summary_row(template_path, None, {"matches": 0, "inliers": 0, "ratio": 0.0, "quality": "template"}, "template image",)]
    print("\nRunning SEQUENCE-COMPOSED method")
    print("Template:", os.path.basename(template_path))
    for i in range(template_index + 1, len(frames)):
        current_frame = frames[i]
        previous_frame = frames[i - 1]
        print(f"\nEstimating {os.path.basename(current_frame)} -> {os.path.basename(previous_frame)}")
        try:
            H_pair, info = estimate_pair_homography(template_path=previous_frame, frame_path=current_frame, ratio=ratio, threshold=threshold,)
            quality_info = homography_quality(info["matches"], info["inliers"])
            print(
                "Matches:", quality_info["matches"],
                "Inliers:", quality_info["inliers"], "/", quality_info["matches"],
                "Ratio:", f'{quality_info["ratio"]:.3f}',
                "Quality:", quality_info["quality"],
            )
            if quality_info["quality"] == "bad":
                print("Bad homography.")
                rows.append(make_summary_row(current_frame, previous_frame, quality_info, "bad homography"))
                if stop_on_bad:
                    print("stopping sequence to avoid error propagation.")
                    break
                continue
            H_to_template[i] = H_to_template[i - 1] @ H_pair
            save_alignment_result(template_path, current_frame, H_to_template[i], homography_dir, warped_dir, overlay_dir,)
            rows.append(make_summary_row(current_frame, previous_frame, quality_info, "saved"))
            print("Saved.")
        except Exception as error:
            print("failed:", error)
            rows.append(make_summary_row(current_frame, previous_frame, {}, str(error)))
            if stop_on_bad:
                break

    save_summary(output_dir, rows)

########### START PROGRAM ###########
def main():
    frames = get_images(dataset_dir)
    if len(frames) < 2:
        raise RuntimeError("not enough images found.")
    
    if template_name is None:
        template_index = 0
    else:
        template_path = os.path.normpath(os.path.join(dataset_dir, template_name))
        frames_norm = [os.path.normpath(path) for path in frames]

        if template_path not in frames_norm:
            raise RuntimeError(f"template image not found: {template_name}")

        template_index = frames_norm.index(template_path)
    print("Number of images:", len(frames))
    print("Method:", method)
    print("Output folder:", output_dir)

    if method == "direct":
        run_direct(frames, template_index)
    elif method == "sequence":
        run_sequence(frames, template_index)
    else:
        raise ValueError('method must be either "direct" or "sequence".')

if __name__ == "__main__":
    main()