import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OVERSAMPLE_MAP = {
    2: 3.0,  # Crack
    5: 2.0,  # Tire flat
    0: 1.5,  # Dent
    1: 1.5,  # Scratch
}

NUM_BG_IMAGES = 500
CLASS_NAMES = ["dent", "scratch", "crack", "glass_shatter", "lamp_broken", "tire_flat"]


def get_image_score(image_path: Path) -> float:
    """
    Analyzes an image for complexity (reflections, shadows, textures).
    Uses Laplacian variance (edge density) and standard deviation (contrast).

    Args:
        image_path (Path): Path to the image file.

    Returns:
        float: A calculated complexity score. Higher means more complex.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    contrast = np.std(img)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

    return float(contrast + (laplacian_var / 100))


def get_classes_from_label(label_path: Path) -> list[int]:
    """
    Reads a YOLO format label file and extracts all present class IDs.

    Args:
        label_path (Path): Path to the YOLO .txt label file.

    Returns:
        list[int]: A list of integer class IDs found in the image.
    """
    if not label_path.exists():
        return []
    with open(label_path, "r", encoding="utf-8") as f:
        return [int(line.split()[0]) for line in f.readlines() if line.strip()]


def select_background_images(
    source_dir: Path, num_images: int, sample_pool_size: int = 2000
) -> list[Path]:
    """
    Selects the most complex background images from a dataset based on edge density and contrast.

    Args:
        source_dir (Path): Directory containing the source background images.
        num_images (int): Target number of images to select.
        sample_pool_size (int): Size of the random pool to analyze before picking the top N.

    Returns:
        list[Path]: List of Paths to the selected background images.
    """
    extensions = {".jpg", ".jpeg", ".png"}
    all_images = [p for p in source_dir.rglob("*") if p.suffix.lower() in extensions]

    if not all_images:
        print(f"Warning: No background images found in {source_dir}")
        return []

    print("Analyzing background images for visual complexity...")
    sample_size = min(sample_pool_size, len(all_images))
    sample_pool = random.sample(all_images, sample_size)

    scored_images = []
    for img_path in sample_pool:
        score = get_image_score(img_path)
        scored_images.append((img_path, score))

    scored_images.sort(key=lambda x: x[1], reverse=True)
    selected_images = [img for img, score in scored_images[:num_images]]

    return selected_images


def balance_dataset(source_dataset: Path, bg_source_dir: Path, output_dataset: Path):
    """
    Creates a new balanced dataset by applying probabilistic oversampling to minority classes
    and injecting complex background images to improve precision.

    Args:
        source_dataset (Path): Path to the original processed YOLO dataset.
        bg_source_dir (Path): Path to the directory containing raw background images.
        output_dataset (Path): Path where the new balanced dataset will be created.
    """
    if output_dataset.exists():
        print("Cleaning up existing output directory...")
        shutil.rmtree(output_dataset)

    for split in ["train", "val", "test"]:
        (output_dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Generating balanced training set via probabilistic oversampling...")
    train_labels = list((source_dataset / "labels" / "train").glob("*.txt"))
    stats = Counter()
    total_train_images = 0

    for lbl_path in train_labels:
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            potential_path = (
                source_dataset / "images" / "train" / f"{lbl_path.stem}{ext}"
            )
            if potential_path.exists():
                img_path = potential_path
                break

        if not img_path:
            continue

        classes = get_classes_from_label(lbl_path)

        multipliers = [OVERSAMPLE_MAP.get(c, 1.0) for c in classes]
        raw_multiplier = max(multipliers) if multipliers else 1.0

        base_copies = int(raw_multiplier)
        decimal_part = raw_multiplier - base_copies
        final_multiplier = (
            base_copies + 1 if random.random() < decimal_part else base_copies
        )

        for i in range(final_multiplier):
            suffix = f"_rec{i}" if i > 0 else ""
            new_name = f"{lbl_path.stem}{suffix}"

            shutil.copy2(
                img_path,
                output_dataset / "images" / "train" / f"{new_name}{img_path.suffix}",
            )
            shutil.copy2(
                lbl_path, output_dataset / "labels" / "train" / f"{new_name}.txt"
            )

            stats.update(classes)
            total_train_images += 1

    print(
        f"Integrating {NUM_BG_IMAGES} complex background images into the training set..."
    )
    selected_bgs = select_background_images(bg_source_dir, NUM_BG_IMAGES)

    for bg_img in selected_bgs:
        new_name = f"bg_{bg_img.name}"
        shutil.copy2(bg_img, output_dataset / "images" / "train" / new_name)

        label_name = f"{Path(new_name).stem}.txt"
        with open(
            output_dataset / "labels" / "train" / label_name, "w", encoding="utf-8"
        ) as f:
            pass
        total_train_images += 1

    print("Copying unmodified validation and test splits...")
    for split in ["val", "test"]:
        for folder in ["images", "labels"]:
            src_dir = source_dataset / folder / split
            if src_dir.exists():
                for item in src_dir.glob("*"):
                    shutil.copy2(item, output_dataset / folder / split / item.name)

    yaml_src = source_dataset / "data.yaml"
    if yaml_src.exists():
        shutil.copy2(yaml_src, output_dataset / "data.yaml")

    print("-" * 50)
    print("Recall-optimized class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"Class ID {i} ({name.ljust(13)}): {stats[i]} instances")

    print(f"Final training pool size: {total_train_images} images")
    print(f"Dataset successfully balanced and saved to: {output_dataset.name}")


if __name__ == "__main__":
    src_data = PROJECT_ROOT / "data" / "processed" / "cardamages-seg"
    bg_data = PROJECT_ROOT / "data" / "raw" / "stanford_cars"
    out_data = PROJECT_ROOT / "data" / "processed" / "cardamages-seg-balanced"

    if src_data.exists():
        balance_dataset(
            source_dataset=src_data, bg_source_dir=bg_data, output_dataset=out_data
        )
    else:
        print(f"Error: Source dataset not found at {src_data}")
