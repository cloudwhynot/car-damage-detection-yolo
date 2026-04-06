import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_script(script_name: str):
    """
    Executes a target Python script as a subprocess.

    Args:
        script_name (str): The filename of the script to run, located in the 'src' directory.

    Raises:
        SystemExit: If the subprocess returns a non-zero exit code.
    """
    script_path = PROJECT_ROOT / "src" / script_name

    print(f"\n{'-'*50}")
    print(f"Executing pipeline module: {script_name}")
    print(f"{'-'*50}\n")

    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(
            f"\nError: Module '{script_name}' failed with exit status {e.returncode}."
        )
        print("Pipeline execution aborted.")
        sys.exit(1)

    print(f"\nModule '{script_name}' completed successfully.")


if __name__ == "__main__":
    print("Initializing the Data Preparation Pipeline...")

    pipeline_steps = ["downloader.py", "converter.py", "balancer.py"]

    for step in pipeline_steps:
        run_script(step)

    print(f"\n{'='*50}")
    print("Data Preparation Pipeline completed successfully.")
    print(f"{'='*50}")
