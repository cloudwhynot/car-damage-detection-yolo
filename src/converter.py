import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_yolo_yaml(output_dir: Path):
    """
    Generates the data.yaml configuration file required by YOLO.

    Args:
        output_dir (Path): The root directory where the data.yaml file will be saved
                           (e.g., 'data/processed/cardamages-seg').
    """
    yaml_content = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: dent
  1: scratch
  2: crack
  3: glass shatter
  4: lamp broken
  5: tire flat
"""
    yaml_path = output_dir / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Created YOLO YAML config: {yaml_path}")


def convert_coco_to_yolo_seg(
    coco_json_path: Path, img_out_dir: Path, lbl_out_dir: Path, source_img_dir: Path
):
    """
    Converts COCO segmentation annotations to YOLO format and copies corresponding images.

    Args:
        coco_json_path (Path): Path to the original COCO annotations JSON file.
        img_out_dir (Path): Destination directory where the images will be copied.
        lbl_out_dir (Path): Destination directory where the YOLO label .txt files will be saved.
        source_img_dir (Path): Source directory containing the original raw images.
    """
    print(f"Processing annotations: {coco_json_path.name}...")

    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    image_dict = {img["id"]: img for img in coco_data.get("images", [])}

    annotations_dict = {}
    for ann in coco_data.get("annotations", []):
        img_id = ann["image_id"]

        if img_id not in annotations_dict:
            annotations_dict[img_id] = []

        annotations_dict[img_id].append(ann)

    processed_count = 0
    missing_images = 0

    for img_id, img_info in image_dict.items():
        width = img_info["width"]
        height = img_info["height"]
        file_name = img_info["file_name"]

        src_img_path = source_img_dir / file_name
        dst_img_path = img_out_dir / file_name

        if src_img_path.exists():
            if not dst_img_path.exists():
                shutil.copy2(src_img_path, dst_img_path)
        else:
            print(f"  -> Warning: Image {file_name} not found in {source_img_dir}")
            missing_images += 1
            continue

        txt_filename = Path(file_name).stem + ".txt"
        txt_path = lbl_out_dir / txt_filename

        annotations = annotations_dict.get(img_id, [])

        with open(txt_path, "w", encoding="utf-8") as txt_file:
            for ann in annotations:
                yolo_class_id = ann["category_id"] - 1
                segmentations = ann.get("segmentation", [])

                if isinstance(segmentations, list):
                    for polygon in segmentations:
                        yolo_polygon = []
                        for i in range(0, len(polygon), 2):
                            x = polygon[i]
                            y = polygon[i + 1]

                            x_norm = max(0.0, min(1.0, x / width))
                            y_norm = max(0.0, min(1.0, y / height))

                            yolo_polygon.append(f"{x_norm:.6f} {y_norm:.6f}")

                        if yolo_polygon:
                            line = f"{yolo_class_id} " + " ".join(yolo_polygon) + "\n"
                            txt_file.write(line)

        processed_count += 1

    print(
        f"Successfully processed {processed_count} files for {img_out_dir.parent.name}/{img_out_dir.name}"
    )
    if missing_images > 0:
        print(f"Failed to find {missing_images} images.")
    print("-" * 40)


if __name__ == "__main__":
    raw_cardd_dir = PROJECT_ROOT / "data" / "raw" / "CarDD_release" / "CarDD_COCO"
    raw_ann_dir = raw_cardd_dir / "annotations"

    out_base_dir = PROJECT_ROOT / "data" / "processed" / "cardamages-seg"

    datasets = [
        {
            "split_name": "train",
            "json_path": raw_ann_dir / "instances_train2017.json",
            "src_img_dir": raw_cardd_dir / "train2017",
        },
        {
            "split_name": "val",
            "json_path": raw_ann_dir / "instances_val2017.json",
            "src_img_dir": raw_cardd_dir / "val2017",
        },
        {
            "split_name": "test",
            "json_path": raw_ann_dir / "instances_test2017.json",
            "src_img_dir": raw_cardd_dir / "test2017",
        },
    ]

    print(f"Starting conversion to YOLO format at: {out_base_dir}")

    for ds in datasets:
        split = ds["split_name"]

        img_out = out_base_dir / "images" / split
        lbl_out = out_base_dir / "labels" / split

        if ds["json_path"].exists():
            if ds["src_img_dir"].exists():
                convert_coco_to_yolo_seg(
                    coco_json_path=ds["json_path"],
                    img_out_dir=img_out,
                    lbl_out_dir=lbl_out,
                    source_img_dir=ds["src_img_dir"],
                )
            else:
                print(f"Error: Original image directory not found: {ds['src_img_dir']}")
        else:
            print(f"Error: Annotation file not found: {ds['json_path']}")

    create_yolo_yaml(out_base_dir)
    print("Conversion completed successfully.")
