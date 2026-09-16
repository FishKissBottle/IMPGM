import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ImgSyn.ImgSyn_Code.ImgSyn_Evaluation import evaluate_imgsyn_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate ImgSyn ControlNet.")
    parser.add_argument(
        "--split",
        choices=("valid", "test", "draw"),
        default="test",
    )
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override both the data-loader and inference microbatch sizes.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-save-outputs", action="store_true")
    args = parser.parse_args()
    evaluate_imgsyn_model(
        task_name="imgsyn_controlnet",
        split_name=args.split,
        sampler_mode=args.sampler,
        save_outputs=not args.no_save_outputs,
        max_batches=args.max_batches,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
