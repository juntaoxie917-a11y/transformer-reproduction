"""Shared runtime and persistent artifact locations."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TOKENIZER_WORK_DIR = PROJECT_ROOT / "data" / "tokenizer"
DRIVE_ROOT = Path("/content/drive/MyDrive")
ARTIFACT_ROOT = Path(os.environ.get(
    "TRANSFORMER_ARTIFACT_ROOT", str(DRIVE_ROOT / "transformer-reproduction")
)).expanduser()
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"
RESULT_DIR = ARTIFACT_ROOT / "results"
TOKENIZER_PATH = ARTIFACT_ROOT / "tokenizer" / "bpe.model"


def ensure_artifact_dirs():
    """Fail before writing if the default Drive location is unavailable."""
    if ARTIFACT_ROOT.is_relative_to(DRIVE_ROOT) and not DRIVE_ROOT.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted. Run drive.mount('/content/drive') "
            "in Colab first, or set TRANSFORMER_ARTIFACT_ROOT for local use."
        )
    for directory in (CHECKPOINT_DIR, RESULT_DIR, TOKENIZER_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)
