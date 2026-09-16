import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

from IMPGM_Config import DATASET_NAME
from Evaluation.Evaluation_Code.IMPGM_Generation_Evaluation import (
    MANIFEST_FILENAME,
    run_generation_evaluation_cli,
)


def main():
    default_manifest = (
        PROJECT_ROOT
        / "Evaluation"
        / "Full_IMPGM"
        / DATASET_NAME
        / "test"
        / MANIFEST_FILENAME
    )
    run_generation_evaluation_cli(
        default_manifest,
        description="Evaluate existing Full IMPGM benchmark outputs.",
    )


if __name__ == "__main__":
    main()
