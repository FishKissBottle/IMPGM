from pathlib import Path

from IMPGM_Config import DATASET_NAME, PROJECT_ROOT, PROMPT_DICT


ARTIFACT_ROOT = Path(PROJECT_ROOT) / "Quality_Evaluation" / "Evaluation_UNet"
MODEL_DIR = ARTIFACT_ROOT / "Models" / DATASET_NAME
LOG_DIR = ARTIFACT_ROOT / "Logs" / DATASET_NAME
RGB_DIR = ARTIFACT_ROOT / "RGBs" / DATASET_NAME
MODEL_PATH = MODEL_DIR / f"Evaluation_UNet_{DATASET_NAME}.pth"
LAST_MODEL_PATH = MODEL_DIR / f"Evaluation_UNet_{DATASET_NAME}_last.pth"
TRAINING_INFO_PATH = MODEL_DIR / f"Evaluation_UNet_{DATASET_NAME}_TrainingInfo.txt"
TEST_INFO_PATH = MODEL_DIR / f"Evaluation_UNet_{DATASET_NAME}_test_Metrics.json"

BASE_CHANNELS = 64
CONDITION_CHANNELS = 16
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
THRESHOLD = 0.5


def number_of_classes():
    if not PROMPT_DICT:
        raise RuntimeError("Evaluation_UNet requires a non-empty dataset.prompt_map.")
    values = sorted(int(value) for value in PROMPT_DICT.values())
    expected = list(range(len(values)))
    if values != expected:
        raise ValueError(
            f"dataset.prompt_map IDs must be contiguous {expected}, got {values}."
        )
    return len(values)


def ensure_artifact_directories():
    for path in (MODEL_DIR, LOG_DIR, RGB_DIR):
        path.mkdir(parents=True, exist_ok=True)
