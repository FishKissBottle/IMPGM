import argparse
import csv
import json
from pathlib import Path


def _nested(payload, *keys):
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def summarize_generation_metrics(metric_paths, output_root, method_labels=None):
    """Create a compact cross-model table without discarding full JSON results."""
    metric_paths = [Path(path).resolve() for path in metric_paths]
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    datasets = {payload.get("dataset") for payload in payloads}
    splits = {payload.get("split") for payload in payloads}
    if None in datasets or len(datasets) != 1:
        raise ValueError(
            "All generation metric files must declare the same dataset; "
            f"found {sorted(str(value) for value in datasets)}."
        )
    if None in splits or len(splits) != 1:
        raise ValueError(
            "All generation metric files must declare the same split; "
            f"found {sorted(str(value) for value in splits)}."
        )
    source_methods = [payload.get("method") for payload in payloads]
    if any(method is None for method in source_methods):
        raise ValueError("Every generation metric file must declare a method.")
    if method_labels is not None:
        methods = [str(label).strip() for label in method_labels]
        if len(methods) != len(payloads):
            raise ValueError(
                "--method-labels must provide exactly one label per metric file; "
                f"got {len(methods)} labels for {len(payloads)} files."
            )
        if any(not method for method in methods):
            raise ValueError("Summary method labels must not be empty.")
    else:
        methods = source_methods
    duplicate_methods = sorted({method for method in methods if methods.count(method) > 1})
    if duplicate_methods:
        raise ValueError(
            "Generation metric summary contains duplicate methods: "
            f"{duplicate_methods}. Regenerate manifests with unique method names, "
            "or provide explicit --method-labels for legacy metric files."
        )
    rows = []
    for metric_path, payload, method in zip(metric_paths, payloads, methods):
        rows.append({
            "method": method,
            "source_method": payload.get("method"),
            "dataset": payload.get("dataset"),
            "split": payload.get("split"),
            "metrics_path": str(metric_path),
            "n": payload.get("num_primary_samples"),
            "fid_rgb": _nested(payload, "distribution", "global", "fid_rgb"),
            "kid_rgb": _nested(payload, "distribution", "global", "kid_rgb"),
            "fid_nir": _nested(
                payload, "distribution", "global", "fid_nir"
            ),
            "kid_nir": _nested(
                payload, "distribution", "global", "kid_nir"
            ),
            "swd_all_bands": _nested(payload, "distribution", "global", "swd_all_bands"),
            "spectral_sam_macro_deg": _nested(
                payload, "spectral_distribution", "macro", "mean_spectrum_sam_deg"
            ),
            "spectral_w1_macro": _nested(
                payload, "spectral_distribution", "macro", "mean_w1"
            ),
            "rgb_lpips_diversity": _nested(payload, "diversity", "mean_rgb_lpips"),
            "nir_lpips_diversity": _nested(payload, "diversity", "mean_nir_lpips"),
            "all_band_pairwise_pixel_rmse": _nested(
                payload, "diversity", "mean_all_band_pixel_rmse"
            ),
            "condition_segmentation_iou": _nested(
                payload, "condition_segmentation_diagnostics", "generated_consistency", "iou"
            ),
            "condition_segmentation_dice": _nested(
                payload, "condition_segmentation_diagnostics", "generated_consistency", "dice"
            ),
            "condition_segmentation_real_ceiling_iou": _nested(
                payload,
                "condition_segmentation_diagnostics",
                "real_image_evaluator_ceiling",
                "iou",
            ),
        })

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "generation_metrics_summary.json"
    csv_path = output_root / "generation_metrics_summary.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize IMPGM generation metric JSON files.")
    parser.add_argument("metrics", nargs="+")
    parser.add_argument(
        "--method-labels",
        nargs="+",
        help="Optional unique labels, one per metric file, for legacy manifests.",
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    rows = summarize_generation_metrics(
        args.metrics,
        args.output_root,
        method_labels=args.method_labels,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
