import json
import cv2
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DISTINCT_COLORS = [
    (255, 255, 0),  # Neon Cyan
    (255, 0, 255),  # Hot Magenta
    (255, 100, 255),  # Bright Rose
    (255, 0, 127),  # Electric Violet
    (255, 255, 255),  # Phosphor White
    (255, 150, 50),  # Neon Sky Blue
    (255, 50, 200),  # Acid Pink
    (255, 180, 255),  # Soft Neon Lavender
]

REPLACE_CLASSES = ["lamp broken", "glass shatter"]
DAMAGE_WEIGHTS = {"scratch": 1.0, "dent": 2.0, "crack": 3.0}

SEVERITY_COLORS = {
    "Light": (0, 255, 0),  # Green
    "Medium": (0, 255, 255),  # Yellow
    "Heavy": (0, 165, 255),  # Orange
    "Critical / Replace": (0, 0, 255),  # Red
}


def get_text_color(background_bgr: tuple) -> tuple:
    """
    Determines the text color (black or white) for the best contrast against a given background.

    Args:
        background_bgr (tuple): The BGR color tuple of the background.

    Returns:
        tuple: (0, 0, 0) for black text or (255, 255, 255) for white text.
    """
    b, g, r = background_bgr
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)


def get_polygon_area_mask(polygon: list, img_shape: tuple) -> np.ndarray:
    """
    Creates a binary mask from polygon coordinates to calculate the precise pixel area.

    Args:
        polygon (list): List of polygon coordinates (x, y).
        img_shape (tuple): Shape of the original image (height, width, channels).

    Returns:
        np.ndarray: A binary mask array representing the polygon area.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if len(polygon) > 0:
        cv2.fillPoly(mask, [np.array(polygon, np.int32)], 1)
    return mask


def process_image(
    img_path: Path,
    car_model: YOLO,
    parts_model: YOLO,
    damage_model: YOLO,
    output_dir: Path = None,
) -> tuple:
    """
    Processes a single image: filters the background, identifies car parts, damages, and evaluates their severity.

    Args:
        img_path (Path): Path to the input image file.
        car_model (YOLO): Model for detecting the global car silhouette.
        parts_model (YOLO): Model for segmenting individual car parts.
        damage_model (YOLO): Model for detecting specific surface damages.
        output_dir (Path, optional): Directory to save the annotated image. Defaults to None.

    Returns:
        tuple: A tuple containing the annotated image (np.ndarray) and the report dictionary (dict).
               Returns (None, None) if the image fails to load.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Failed to load {img_path.name}")
        return None, None

    img_shape = img.shape
    car_res = car_model(img, conf=0.25, retina_masks=True, verbose=False)[0]
    parts_res = parts_model(img, conf=0.3, retina_masks=True, verbose=False)[0]
    damage_res = damage_model(img, conf=0.25, retina_masks=True, verbose=False)[0]

    part_names, damage_names = parts_res.names, damage_res.names
    report, vis_img = {}, img.copy()

    global_car_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if car_res.masks:
        for poly in car_res.masks.xy:
            if len(poly) > 0:
                cv2.fillPoly(global_car_mask, [np.array(poly, np.int32)], 1)

    total_car_area = max(np.sum(global_car_mask), 1)

    part_data = []
    if parts_res.masks:
        for poly, cls_id in zip(parts_res.masks.xy, parts_res.boxes.cls.cpu().numpy()):
            mask = get_polygon_area_mask(poly, img_shape)
            part_data.append(
                {
                    "name": part_names[int(cls_id)],
                    "mask": mask,
                    "area": max(np.sum(mask), 1),
                }
            )

    if damage_res.masks:
        font_scale = max(0.7, img_shape[1] / 1500)
        thickness = max(2, int(img_shape[1] / 750))

        for idx, (poly, cls_id, box) in enumerate(
            zip(
                damage_res.masks.xy,
                damage_res.boxes.cls.cpu().numpy(),
                damage_res.boxes.xyxy.cpu().numpy(),
            )
        ):
            damage_name = damage_names[int(cls_id)]
            raw_d_mask = get_polygon_area_mask(poly, img_shape)

            d_mask_filtered = np.logical_and(raw_d_mask, global_car_mask)
            damage_pixel_area = np.sum(d_mask_filtered)

            if damage_pixel_area == 0:
                continue

            weighted_pct = (
                damage_pixel_area
                * DAMAGE_WEIGHTS.get(damage_name, 1.0)
                / total_car_area
            ) * 100
            severity = (
                "Critical / Replace"
                if damage_name in REPLACE_CLASSES
                else (
                    "Light"
                    if weighted_pct < 0.5
                    else "Medium" if weighted_pct < 2.0 else "Heavy"
                )
            )

            c_distinct, c_severity = (
                DISTINCT_COLORS[idx % len(DISTINCT_COLORS)],
                SEVERITY_COLORS[severity],
            )

            overlay = vis_img.copy()
            cv2.fillPoly(overlay, [np.array(poly, np.int32)], c_distinct)
            cv2.addWeighted(overlay, 0.5, vis_img, 0.5, 0, vis_img)

            x1, y1, x2, y2 = map(int, box)
            font = cv2.FONT_HERSHEY_SIMPLEX
            t1, t2 = f"{damage_name}", f"[{severity}]"

            (w1, h1), _ = cv2.getTextSize(t1, font, font_scale, thickness)
            (w2, h2), _ = cv2.getTextSize(t2, font, font_scale, thickness)

            max_w = max(w1, w2) + 12
            h_sec1, h_sec2 = h1 + int(12 * font_scale), h2 + int(12 * font_scale)
            total_h = h_sec1 + h_sec2
            ly = y1 if y1 - total_h > 0 else y1 + total_h + 10

            cv2.rectangle(
                vis_img,
                (x1 - 2, ly - total_h - 2),
                (x1 + max_w + 2, ly + 2),
                (0, 0, 0),
                thickness + 1,
            )
            cv2.rectangle(
                vis_img, (x1, ly - total_h), (x1 + max_w, ly - h_sec2), c_distinct, -1
            )
            cv2.rectangle(vis_img, (x1, ly - h_sec2), (x1 + max_w, ly), c_severity, -1)

            cv2.putText(
                vis_img,
                t1,
                (x1 + 6, ly - h_sec2 - int(6 * font_scale)),
                font,
                font_scale,
                get_text_color(c_distinct),
                thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis_img,
                t2,
                (x1 + 6, ly - int(6 * font_scale)),
                font,
                font_scale,
                get_text_color(c_severity),
                thickness,
                cv2.LINE_AA,
            )

            best_part, max_ovl, part_pct = "unknown_part", 0, "N/A"
            for part in part_data:
                ovl = np.sum(np.logical_and(d_mask_filtered, part["mask"]))
                if ovl > max_ovl:
                    max_ovl, best_part, part_pct = (
                        ovl,
                        part["name"],
                        round((ovl / part["area"]) * 100, 2),
                    )

            if best_part not in report:
                report[best_part] = []

            report[best_part].append(
                {
                    "type": damage_name,
                    "severity": severity,
                    "pixel_area": int(damage_pixel_area),
                    "car_area_percentage": round(
                        (damage_pixel_area / total_car_area) * 100, 2
                    ),
                    "part_area_percentage": part_pct,
                }
            )

    if output_dir:
        cv2.imwrite(str(output_dir / f"analyzed_{img_path.name}"), vis_img)

    return vis_img, report


class DamageDetectionEngine:
    """
    Expert System Engine that encapsulates the 3 YOLO models.
    Loads model weights once during initialization for efficient inference.
    """

    def __init__(self):
        print(f"{'-'*50}\nInitializing Expert AI Engine\n{'-'*50}")
        self.car_model = YOLO(str(PROJECT_ROOT / "models" / "yolo26m-seg.pt"))
        self.parts_model = YOLO(
            str(PROJECT_ROOT / "models" / "parts_model" / "weights" / "best.pt")
        )
        self.damage_model = YOLO(
            str(PROJECT_ROOT / "models" / "damage_model" / "weights" / "best.pt")
        )
        print("All models loaded successfully. Engine is ready.")

    def predict(self, image_path: str, output_dir: Path = None) -> tuple:
        """
        Executes the analysis on a single image.

        Args:
            image_path (str): Path to the input image file.
            output_dir (Path, optional): Directory to save the output visual artifacts. Defaults to None.

        Returns:
            tuple: Annotated image as np.ndarray and the JSON-compatible report dictionary.
        """
        return process_image(
            Path(image_path),
            self.car_model,
            self.parts_model,
            self.damage_model,
            output_dir,
        )


def run_inference(source_path: str):
    """
    Orchestrator for processing a folder or a single file natively (CLI/Demo version).

    Args:
        source_path (str): Path to the source image file or directory containing images.
    """
    engine = DamageDetectionEngine()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / f"prediction_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = Path(source_path)
    images = [source] if source.is_file() else list(source.glob("*"))
    images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    print(f"\nProcessing {len(images)} image(s). Saving results to: {output_dir}\n")

    final_report = {}
    for img_p in images:
        print(f"Analysing: {img_p.name}")
        shutil.copy2(img_p, output_dir / img_p.name)
        _, rep = engine.predict(str(img_p), output_dir)
        if rep:
            final_report.update({img_p.name: rep})

    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    print(f"\n{'-'*50}\nInference completed successfully.\n{'-'*50}")


if __name__ == "__main__":
    demo_path = str(PROJECT_ROOT / "assets" / "demo-images" / "kia-picanto")
    run_inference(demo_path)
