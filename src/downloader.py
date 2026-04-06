import zipfile
import shutil
import gdown
import kagglehub
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_from_gdrive(file_id: str, output_path: Path, extract_to: Path = None):
    """
    Downloads a file from Google Drive by its ID and extracts it if it is a zip archive.

    Args:
        file_id (str): Google Drive file ID.
        output_path (Path): Pathlib Path object to save the downloaded file.
        extract_to (Path, optional): Pathlib Path to extract the contents.
    """
    print(f"Starting download for file ID: {file_id}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(url, str(output_path), quiet=False)

    if output_path.suffix == ".zip":
        if extract_to is None:
            extract_to = output_path.parent

        print(f"Extracting archive to {extract_to}...")
        with zipfile.ZipFile(output_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        print("Removing the .zip archive to save space...")
        output_path.unlink()

    print("Download and extraction completed successfully.")


def download_from_kaggle(dataset_name: str, output_path: Path):
    """
    Downloads a dataset using kagglehub and moves it to the target directory.
    No API key required for public datasets.

    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., 'eduardo4jesus/stanford-cars-dataset').
        output_path (Path): Directory path to save the dataset in the project.
    """
    print(f"Starting Kaggle download via kagglehub for: {dataset_name}...")

    cached_path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded to cache: {cached_path}")

    print(f"Moving dataset to project directory: {output_path}...")

    if output_path.exists():
        shutil.rmtree(output_path)

    shutil.copytree(cached_path, output_path)

    print("Kaggle dataset prepared successfully.")


if __name__ == "__main__":
    CARDD_GDRIVE_ID = "1bbyqVCKZX5Ur5Zg-uKj0jD0maWAVeOLx"
    CARDD_ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "cardd.zip"

    KAGGLE_DATASET = "eduardo4jesus/stanford-cars-dataset"
    STANFORD_CARS_DIR = PROJECT_ROOT / "data" / "raw" / "stanford_cars"

    download_from_gdrive(file_id=CARDD_GDRIVE_ID, output_path=CARDD_ZIP_PATH)
    download_from_kaggle(dataset_name=KAGGLE_DATASET, output_path=STANFORD_CARS_DIR)
