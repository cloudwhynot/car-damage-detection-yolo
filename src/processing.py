import json
import os
import shutil
from pathlib import Path

CLASSES = {
    "damage": 0,
    "headlamp": 1,
    "rear_bumper": 2,
    "door": 3,
    "hood": 4,
    "front_bumper": 5,
}


def convert_to_yolo_bbox(img_w, img_h, bbox):
    """
    Converts COCO format [x, y, width, height]
    to YOLO format [x_center, y_center, width, height] (normalized 0-1).
    """
    x, y, w, h = bbox

    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    return x_center, y_center, w_norm, h_norm


def process_dataset_split(split_name, raw_dir, processed_dir):
    """Processes a single dataset split (e.g., train or val)."""
    print(f"Processing split: {split_name}...")

    # Define file paths
    img_dir = raw_dir / split_name
    damage_json_path = raw_dir / f"{split_name}/COCO_{split_name}_annos.json"
    parts_json_path = raw_dir / f"{split_name}/COCO_mul_{split_name}_annos.json"

    # Create YOLO directory structure (images and labels)
    out_img_dir = processed_dir / "images" / split_name
    out_lbl_dir = processed_dir / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Load JSON annotation files
    with open(damage_json_path, "r") as f:
        damage_data = json.load(f)
    with open(parts_json_path, "r") as f:
        parts_data = json.load(f)

    # Load JSON annotation files
    with open(damage_json_path, "r") as f:
        damage_data = json.load(f)
    with open(parts_json_path, "r") as f:
        parts_data = json.load(f)

    # Create an image dictionary mapping image_id to image info
    images_info = {img["id"]: img for img in damage_data["images"]}

    # Aggregate all annotations by image_id
    annotations_by_image = {img_id: [] for img_id in images_info.keys()}

    # Add damage annotations (class 0)
    for ann in damage_data["annotations"]:
        img_id = ann["image_id"]

        x_c, y_c, w_n, h_n = convert_to_yolo_bbox(
            images_info[img_id]["width"], images_info[img_id]["height"], ann["bbox"]
        )

        annotations_by_image[img_id].append(
            f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
        )

    # Add part annotations (classes 1-5)
    for ann in parts_data["annotations"]:
        img_id = ann["image_id"]
        yolo_class_id = ann["category_id"]

        x_c, y_c, w_n, h_n = convert_to_yolo_bbox(
            images_info[img_id]["width"], images_info[img_id]["height"], ann["bbox"]
        )

        annotations_by_image[img_id].append(
            f"{yolo_class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
        )

    # Save files to processed directory
    for img_id, ann_list in annotations_by_image.items():
        img_info = images_info[img_id]
        img_filename = img_info["file_name"]

        src_img_path = img_dir / img_filename
        if src_img_path.exists():
            shutil.copy(src_img_path, out_img_dir / img_filename)

            txt_filename = img_filename.replace(".jpg", ".txt").replace(".png", ".txt")
            with open(out_lbl_dir / txt_filename, "w") as f:
                f.write("\n".join(ann_list))

    print(f"Done! Saved {len(annotations_by_image)} files for {split_name}.")


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DIR = BASE_DIR / "data" / "processed"

    if (RAW_DIR / "train").exists():
        process_dataset_split("train", RAW_DIR, PROCESSED_DIR)
    if (RAW_DIR / "val").exists():
        process_dataset_split("val", RAW_DIR, PROCESSED_DIR)

    test_raw = RAW_DIR / "test"
    test_proc = PROCESSED_DIR / "images" / "test"
    if test_raw.exists():
        print("Copying test images (no labels)...")
        shutil.copytree(test_raw, test_proc, dirs_exist_ok=True)
        print("Done! Copied test images.")


if __name__ == "__main__":
    main()
