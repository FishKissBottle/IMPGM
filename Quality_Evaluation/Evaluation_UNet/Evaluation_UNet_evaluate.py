import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from torch import nn

from IMPGM_Config import DEVICE, FGSEG_TEST_MICROBATCH_SIZE, TEST_BATCH_SIZE
from IMPGM_Utils import load_model_for_eval
from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_Config import (
    MODEL_PATH,
    TEST_INFO_PATH,
    ensure_artifact_directories,
)
from Quality_Evaluation.Evaluation_UNet.Evaluation_UNet_train import (
    build_evaluation_unet,
    build_loader,
    evaluate_loader,
)


def main(split_name="test"):
    ensure_artifact_directories()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Evaluation_UNet checkpoint does not exist: {MODEL_PATH}")
    model = build_evaluation_unet().to(DEVICE)
    model, extra = load_model_for_eval(MODEL_PATH, model, map_location=DEVICE)
    loader = build_loader(split_name, TEST_BATCH_SIZE, seed=1001)
    metrics = evaluate_loader(loader, model, nn.BCEWithLogitsLoss(), FGSEG_TEST_MICROBATCH_SIZE)
    result = {
        "split": split_name,
        "checkpoint": str(MODEL_PATH.resolve()),
        "checkpoint_metadata": {
            "dataset_name": extra.get("dataset_name"),
            "num_classes": extra.get("num_classes"),
            "input_domain": extra.get("input_domain"),
            "conditioning": extra.get("conditioning"),
        },
        "metrics": metrics,
    }
    output_path = TEST_INFO_PATH.with_name(
        f"Evaluation_UNet_{extra.get('dataset_name', 'dataset')}_{split_name}_Metrics.json"
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the IMPGM full-image evaluator UNet.")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    args = parser.parse_args()
    main(args.split)
