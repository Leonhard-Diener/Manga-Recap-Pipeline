"""Download the RT-DETRv4-X Manga109-s ONNX model."""

from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download

REPO_ID = "tori29umai/rtdetrv4-x-manga109s"
FILENAME = "model.onnx"
TARGET = Path(__file__).resolve().parent / "models" / "model.onnx"


def main() -> None:
    """Download the model into the project's models directory."""
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=TARGET.parent,
        )
    )
    if downloaded.resolve() != TARGET.resolve():
        shutil.copy2(downloaded, TARGET)
    print(f"Model saved to: {TARGET}")


if __name__ == "__main__":
    main()
