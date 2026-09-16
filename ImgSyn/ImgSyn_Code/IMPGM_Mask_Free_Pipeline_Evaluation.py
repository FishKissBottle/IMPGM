import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from IMPGM_Config import DATASET_NAME, VAE_TRAINING_REGIME
from Evaluation.Evaluation_Code.IMPGM_Generation_Evaluation import MANIFEST_FILENAME
from Evaluation.Evaluation_Code.IMPGM_Mask_Free_Evaluation import (
    run_mask_free_generation_evaluation_cli,
)


def main():
    default_root = PROJECT_ROOT / "Evaluation" / "Full_IMPGM_Mask_Free" / DATASET_NAME
    if VAE_TRAINING_REGIME != "from_scratch":
        default_root /= VAE_TRAINING_REGIME
    run_mask_free_generation_evaluation_cli(
        default_root / "test" / MANIFEST_FILENAME,
        description="Evaluate existing IMPGM mask-free generation outputs.",
    )


if __name__ == "__main__":
    main()
