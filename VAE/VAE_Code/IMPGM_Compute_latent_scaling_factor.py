import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from IMPGM_Config import (
    DATASET_DICT,
    DATASET_NAME,
    DATASET_YAML_PATH,
    DEVICE,
    INPUT_CHANNELS,
    LATENT_SCALING_FACTOR_FIELD,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    TEST_BATCH_SIZE,
    VAE_MODEL_SAVEPATH,
)
from IMPGM_Dataset import IMPGM_Dataset
from IMPGM_Utils import (
    build_dataloader_generator,
    build_standard_vae,
    load_model_for_eval,
    seed_dataloader_worker,
    set_random_seed,
)


class RunningScalarStats:
    """Track streaming scalar mean and variance statistics."""
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, tensor):
        tensor32 = tensor.detach().to(dtype=torch.float32)
        self.count += int(tensor32.numel())
        self.sum += float(tensor32.sum().item())
        self.sum_sq += float(tensor32.square().sum().item())

    def finalize(self):
        if self.count <= 0:
            raise RuntimeError("No latent values were accumulated.")
        mean = self.sum / float(self.count)
        var = max(self.sum_sq / float(self.count) - mean * mean, 1e-12)
        std = var ** 0.5
        scaling_factor = 1.0 / std
        return {
            "numel": int(self.count),
            "mean": float(mean),
            "std": float(std),
            "scaling_factor": float(scaling_factor),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Compute the latent scaling factor for the active dataset YAML.")
    parser.add_argument("--batch-size", type=int, default=TEST_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def write_scaling_factor_to_dataset_yaml(scaling_factor):
    yaml_path = Path(DATASET_YAML_PATH)
    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config.setdefault("latent", {})[LATENT_SCALING_FACTOR_FIELD] = round(
        float(scaling_factor), 8
    )

    temp_path = yaml_path.with_suffix(yaml_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    temp_path.replace(yaml_path)
    return yaml_path


@torch.no_grad()
def compute_scaling_factor(batch_size, num_workers, max_batches, seed):
    set_random_seed(seed, deterministic=False)
    data_generator = build_dataloader_generator(seed)

    dataset = IMPGM_Dataset(
        img_rootdir_list=DATASET_DICT["img_rootdir_list_forTrain"],
        msk_rootdir_list=DATASET_DICT["msk_rootdir_list_forTrain"],
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_dataloader_worker,
        generator=data_generator,
    )

    save_path = VAE_MODEL_SAVEPATH
    vae_model = build_standard_vae(img_channel=INPUT_CHANNELS, device=DEVICE)
    vae_model, _ = load_model_for_eval(save_path, vae_model, map_location=DEVICE)
    vae_model.eval()
    for parameter in vae_model.parameters():
        parameter.requires_grad = False
    if vae_model.training:
        raise RuntimeError("VAE must be in eval mode when estimating the latent scaling factor.")
    stats = RunningScalarStats()
    processed_batches = 0

    total = len(loader) if max_batches is None else min(len(loader), max_batches)
    with tqdm(loader, total=total, ncols=120, desc=f"Compute scaling factor ({DATASET_NAME})") as pbar:
        for batch_idx, (_, fg_imgs, _, syn_imgs, _, _) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            fg_imgs = fg_imgs.to(next(vae_model.parameters()).device)
            syn_imgs = syn_imgs.to(next(vae_model.parameters()).device)

            for imgs in (fg_imgs, syn_imgs):
                latent_params = vae_model.encode(imgs)
                # This post-quant latent is the exact representation consumed by diffusion models.
                latent, _, _ = vae_model.reparameterize(latent_params)
                if not bool(torch.isfinite(latent).all().item()):
                    raise RuntimeError("VAE produced non-finite post-quant latent values.")
                stats.update(latent)

            processed_batches += 1
            pbar.set_postfix({"batches": str(processed_batches)})

    result = stats.finalize()
    written_yaml = write_scaling_factor_to_dataset_yaml(result["scaling_factor"])

    print(f"\nDataset: {DATASET_NAME}")
    print(f"VAE checkpoint: {Path(save_path).resolve()}")
    print(f"Updated YAML: {written_yaml.resolve()}")
    print(f"Updated field: latent.{LATENT_SCALING_FACTOR_FIELD}")
    print(f"numel={result['numel']}, mean={result['mean']:.8f}, std={result['std']:.8f}")
    print(f"scaling_factor={result['scaling_factor']:.8f}")
    print("\nRecommended usage:")
    print("  latent_scaled = latent * scaling_factor")
    print("  latent = latent_scaled / scaling_factor")
    return result


def main():
    args = parse_args()
    compute_scaling_factor(
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_batches=args.max_batches,
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
