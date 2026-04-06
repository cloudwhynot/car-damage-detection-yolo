import os
import torch
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_optimal_device() -> str:
    """
    Determines the best available hardware accelerator for PyTorch.
    Prioritizes CUDA (supporting multi-GPU), then Apple MPS, and falls back to CPU.

    Returns:
        str: The device string identifier compatible with Ultralytics YOLO
             (e.g., '0,1' for dual GPU, 'mps' for Mac, or 'cpu').
    """
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            devices = ",".join(str(i) for i in range(gpu_count))
            print(
                f"CUDA detected with {gpu_count} GPUs. Using DDP mode on devices: {devices}"
            )
            return devices
        else:
            print("CUDA detected with 1 GPU. Using device: 0")
            return "0"
    elif torch.backends.mps.is_available():
        print("Apple Silicon MPS detected. Using device: mps")
        return "mps"
    else:
        print("No hardware acceleration detected. Falling back to device: cpu")
        return "cpu"


def enforce_absolute_yaml_path(yaml_path: Path):
    """
    Dynamically rewrites the 'path:' variable in the data.yaml file
    to use an absolute system path. This prevents YOLO's relative pathing errors.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(yaml_path, "w", encoding="utf-8") as f:
        for line in lines:
            if line.startswith("path:"):
                f.write(f"path: {yaml_path.parent.absolute()}\n")
            else:
                f.write(line)


def train_parts_model():
    """
    Initializes and trains the YOLO segmentation model for car parts detection.
    Uses dynamic paths, optimal hardware detection, and specific augmentations.
    """
    print("Initializing Car Parts Model Training...")

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    data_yaml_path = (
        PROJECT_ROOT / "data" / "processed" / "carparts-seg" / "carparts-seg.yaml"
    )
    models_dir = PROJECT_ROOT / "models"

    if not data_yaml_path.exists():
        print(f"Error: Dataset configuration not found at {data_yaml_path}")
        print("Please ensure the dataset is downloaded and structured correctly.")
        return

    enforce_absolute_yaml_path(data_yaml_path)

    models_dir.mkdir(parents=True, exist_ok=True)
    base_model_path = models_dir / "yolo26m-seg.pt"

    model = YOLO(str(base_model_path))
    target_device = get_optimal_device()

    print("Starting training process...")
    model.train(
        data=str(data_yaml_path),
        epochs=100,
        imgsz=896,
        batch=16,
        device=target_device,
        amp=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=15.0,
        project=str(models_dir),
        name="parts_model",
        exist_ok=True,
    )

    print(
        f"Training completed. Artifacts and weights saved to: {models_dir / 'parts_model'}"
    )


if __name__ == "__main__":
    train_parts_model()
