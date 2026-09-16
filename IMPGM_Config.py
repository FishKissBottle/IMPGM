import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from IMPGM_ConfigLoader import (
    load_yaml_config,
    merge_configs,
    validate_config,
    _namespace_to_dict,
    _to_namespace,
    _flatten_dict,
)

# YAML config paths
#   IMPGM_DATASET_YAML  dataset YAML (default: configs/datasets/main.yaml)
#   IMPGM_<TASK>_ARTIFACT_TAG  optional task-specific artifact filename tag
#   IMPGM_<TASK>_FREQUENCY_WEIGHT / IMPGM_<TASK>_EDGE_WEIGHT
#       optional task-specific loss-weight overrides
# Each model family has a fixed task YAML path. Runtime, VAE, and dataset
# settings remain shared across all four task configurations.
PROJECT_ROOT = Path(__file__).resolve().parent

_DATASET_YAML_PATH = os.environ.get(
    "IMPGM_DATASET_YAML",
    str(PROJECT_ROOT / "configs" / "datasets" / "main.yaml"),
)
_RUNTIME_YAML_PATH = str(PROJECT_ROOT / "configs" / "runtime" / "default_train.yaml")
_VAE_YAML_PATH = str(PROJECT_ROOT / "configs" / "tasks" / "vae.yaml")
_FGGEN_DIFFUSION_YAML_PATH = str(
    PROJECT_ROOT / "configs" / "tasks" / "fggen_diffusion.yaml"
)
_FGGEN_CONTROLNET_YAML_PATH = str(
    PROJECT_ROOT / "configs" / "tasks" / "fggen_controlnet.yaml"
)
_IMGSYN_DIFFUSION_YAML_PATH = str(
    PROJECT_ROOT / "configs" / "tasks" / "imgsyn_diffusion.yaml"
)
_IMGSYN_CONTROLNET_YAML_PATH = str(
    PROJECT_ROOT / "configs" / "tasks" / "imgsyn_controlnet.yaml"
)

# Load and merge configs: runtime -> shared VAE -> task -> dataset

def _load_and_merge_configs(task_yaml_path=None):
    """Load and merge runtime, shared VAE, task, and dataset YAMLs.

    Merge order: runtime -> shared VAE -> task -> dataset.
    Later layers override earlier ones.
    """
    runtime_cfg = load_yaml_config(_RUNTIME_YAML_PATH)
    vae_cfg = load_yaml_config(_VAE_YAML_PATH)
    task_cfg = (
        load_yaml_config(task_yaml_path)
        if task_yaml_path is not None
        else SimpleNamespace()
    )
    dataset_cfg = load_yaml_config(_DATASET_YAML_PATH)

    dataset_dict = _namespace_to_dict(dataset_cfg)
    return merge_configs(
        runtime_cfg,
        vae_cfg,
        task_cfg,
        _to_namespace(_flatten_dict(dataset_dict)),
    )


_cfg = _load_and_merge_configs()
_FGGEN_DIFFUSION_CFG = _load_and_merge_configs(_FGGEN_DIFFUSION_YAML_PATH)
_FGGEN_CONTROLNET_CFG = _load_and_merge_configs(_FGGEN_CONTROLNET_YAML_PATH)
_IMGSYN_DIFFUSION_CFG = _load_and_merge_configs(_IMGSYN_DIFFUSION_YAML_PATH)
_IMGSYN_CONTROLNET_CFG = _load_and_merge_configs(_IMGSYN_CONTROLNET_YAML_PATH)
DATASET_YAML_PATH = _DATASET_YAML_PATH

# Validation
for _config_value in (
    _cfg,
    _FGGEN_DIFFUSION_CFG,
    _FGGEN_CONTROLNET_CFG,
    _IMGSYN_DIFFUSION_CFG,
    _IMGSYN_CONTROLNET_CFG,
):
    validate_config(_config_value, strict=False)

# Nested config access

def _get_from_config(cfg, attr_path: str, default=None):
    node = cfg
    for part in attr_path.split("."):
        if not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node


def _get(attr_path: str, default=None):
    return _get_from_config(_cfg, attr_path, default)


# Legacy bridge helpers

def _build_legacy_dataset_dict(cfg):
    d = {}
    mapping = {
        "train": "forTrain",
        "valid": "forValid",
        "test": "forTest",
        "draw": "forDraw",
    }
    for split, old_suffix in mapping.items():
        img = _get_from_config(cfg, f"dataset.splits.{split}.image_roots")
        msk = _get_from_config(cfg, f"dataset.splits.{split}.mask_roots")
        if img is not None:
            d[f"img_rootdir_list_{old_suffix}"] = list(img)
        if msk is not None:
            d[f"msk_rootdir_list_{old_suffix}"] = list(msk)
    _require_legacy_valid_split(d)
    return d


def _is_no_image_root(rootdir: str) -> bool:
    root_name = Path(rootdir).name
    label_name = root_name.split("_")[-1]
    return label_name.startswith("No")


def _require_legacy_valid_split(dataset_dict):
    required_keys = (
        "img_rootdir_list_forValid",
        "msk_rootdir_list_forValid",
    )
    missing_keys = [key for key in required_keys if key not in dataset_dict]
    if not missing_keys:
        return

    raise ValueError(
        "A dedicated valid split is required for training and may not fall back to test. "
        f"Missing keys in dataset YAML ({_DATASET_YAML_PATH}): {missing_keys}. "
        "Please define dataset.splits.valid.image_roots and dataset.splits.valid.mask_roots explicitly."
    )


def assert_no_image_roots_at_end(image_roots, split_name: str):
    if image_roots is None:
        return

    first_no_root = None
    for idx, rootdir in enumerate(image_roots):
        if _is_no_image_root(rootdir):
            if first_no_root is None:
                first_no_root = rootdir
            continue

        if first_no_root is not None:
            ordered_root_names = [Path(path).name for path in image_roots]
            raise ValueError(
                f"dataset.splits.{split_name}.image_roots must place every '_No*' directory at the end. "
                f"Found non-_No* directory '{Path(rootdir).name}' at index {idx} after "
                f"'{Path(first_no_root).name}'. Current order: {ordered_root_names}"
            )


def _build_legacy_prompt_dict(cfg):
    pm = _get_from_config(cfg, "dataset.prompt_map")
    if pm is None:
        return None
    if isinstance(pm, SimpleNamespace):
        return {k: getattr(pm, k) for k in dir(pm) if not k.startswith("_")}
    return dict(pm)


# Dataset and experiment identity
DATASET_NAME = str(_get("dataset.name", ""))

DATASET_DICT = _build_legacy_dataset_dict(_cfg)
assert_no_image_roots_at_end(DATASET_DICT.get("img_rootdir_list_forTrain"), "train")
assert_no_image_roots_at_end(DATASET_DICT.get("img_rootdir_list_forValid"), "valid")
assert_no_image_roots_at_end(DATASET_DICT.get("img_rootdir_list_forTest"), "test")
assert_no_image_roots_at_end(DATASET_DICT.get("img_rootdir_list_forDraw"), "draw")

PROMPT_DICT = _build_legacy_prompt_dict(_cfg)

# Geometry
IMG_SIZE = int(_get("geometry.image_size", 256))
INPUT_CHANNELS = int(_get("geometry.input_channels", 4))
MASK_CHANNELS = int(_get("geometry.mask_channels", 1))
LATENT_RESOLUTION = int(_get("geometry.latent_resolution", 64))
SIDELENGTH_SCALE_FACTOR = IMG_SIZE // LATENT_RESOLUTION

VAE_IMG_CHANNEL = INPUT_CHANNELS  # backward-compatible alias

# Visualization and normalization
VIS_BAND_ORDER = list(_get("visualization.vis_band_order", [0, 1, 2]))
RGB_VIS_STRETCH = str(_get("visualization.rgb_stretch", "sqrt")).strip().lower()

# Validate VIS_BAND_ORDER
if len(VIS_BAND_ORDER) != 3:
    raise ValueError(
        f"VIS_BAND_ORDER must have exactly 3 elements, got {VIS_BAND_ORDER}"
    )
if not all(isinstance(v, int) for v in VIS_BAND_ORDER):
    raise ValueError(
        f"VIS_BAND_ORDER must contain integers, got {VIS_BAND_ORDER}"
    )
if not all(0 <= v < INPUT_CHANNELS for v in VIS_BAND_ORDER):
    raise ValueError(
        f"VIS_BAND_ORDER indices must be in [0, {INPUT_CHANNELS - 1}], got {VIS_BAND_ORDER}"
    )
if len(set(VIS_BAND_ORDER)) != 3:
    raise ValueError(
        f"VIS_BAND_ORDER must not contain duplicates, got {VIS_BAND_ORDER}"
    )
if RGB_VIS_STRETCH in ("identity", "linear", "none", "off"):
    RGB_VIS_STRETCH = "none"
elif RGB_VIS_STRETCH != "sqrt":
    raise ValueError(
        f"visualization.rgb_stretch must be one of ['none', 'sqrt'], got {RGB_VIS_STRETCH!r}"
    )

_IMAGE_MEAN_RAW = _get("normalization.image_mean")
if _IMAGE_MEAN_RAW is None:
    raise ValueError(
        "Config error: normalization.image_mean is missing. "
        "Run IMPGM_Compute_Mean_Std.py --dataset-yaml <cfg> --update-yaml to populate it."
    )
IMAGE_MEAN = list(_IMAGE_MEAN_RAW)

_IMAGE_STD_RAW = _get("normalization.image_std")
if _IMAGE_STD_RAW is None:
    raise ValueError(
        "Config error: normalization.image_std is missing. "
        "Run IMPGM_Compute_Mean_Std.py --dataset-yaml <cfg> --update-yaml to populate it."
    )
IMAGE_STD = list(_IMAGE_STD_RAW)

# Placeholder georeference for synthetic TIF exports
PLACEHOLDER_TOP_LEFT_X = 0.0
PLACEHOLDER_TOP_LEFT_Y = 0.0
PLACEHOLDER_PIXEL_WIDTH = 1.0
PLACEHOLDER_PIXEL_HEIGHT = -1.0
PLACEHOLDER_EPSG = 3857  # Web Mercator; explicitly indicates placeholder coords

# Latent space
_VAE_PRELOAD_ENABLED_FOR_LATENT = bool(_get("vae.preload.enabled", False))
LATENT_SCALING_FACTOR_FIELD = (
    "scaling_factor_transfer"
    if _VAE_PRELOAD_ENABLED_FOR_LATENT
    else "scaling_factor"
)
_LATENT_SCALING_FACTOR_RAW = _get(
    f"latent.{LATENT_SCALING_FACTOR_FIELD}",
    None if _VAE_PRELOAD_ENABLED_FOR_LATENT else 1.0,
)
LATENT_SCALING_FACTOR = (
    None
    if _LATENT_SCALING_FACTOR_RAW is None
    else float(_LATENT_SCALING_FACTOR_RAW)
)
if LATENT_SCALING_FACTOR is not None and LATENT_SCALING_FACTOR <= 0.0:
    raise ValueError(
        f"Invalid latent.{LATENT_SCALING_FACTOR_FIELD} for dataset {DATASET_NAME!r}: "
        f"{LATENT_SCALING_FACTOR}. It must be > 0 in {DATASET_YAML_PATH}."
    )
LATENT_HIDDENCHANNEL = int(_get("vae.model.z_channel", 8))

# Runtime and training budget
TRAIN_BATCH_SIZE = int(_get("batch_sizes.train", 8))
_VALID_BATCH_SIZE_RAW = _get("batch_sizes.valid")
if _VALID_BATCH_SIZE_RAW is None:
    raise ValueError("Define batch_sizes.valid in the runtime configuration.")
VALID_BATCH_SIZE = int(_VALID_BATCH_SIZE_RAW)
TEST_BATCH_SIZE = int(_get("batch_sizes.test", VALID_BATCH_SIZE))
DRAW_BATCH_SIZE = int(_get("batch_sizes.draw", 6))

VAE_TRAIN_MICROBATCH_SIZE = int(_get("microbatch_sizes.vae.train", TRAIN_BATCH_SIZE))
VAE_VALID_MICROBATCH_SIZE = int(_get("microbatch_sizes.vae.valid", VALID_BATCH_SIZE))
VAE_DRAW_MICROBATCH_SIZE = int(_get("microbatch_sizes.vae.draw", DRAW_BATCH_SIZE))

DIFFUSION_TRAIN_MICROBATCH_SIZE = int(_get("microbatch_sizes.diffusion.train", TRAIN_BATCH_SIZE))
DIFFUSION_VALID_MICROBATCH_SIZE = int(_get("microbatch_sizes.diffusion.valid", VALID_BATCH_SIZE))
DIFFUSION_TEST_MICROBATCH_SIZE = int(_get("microbatch_sizes.diffusion.test", TEST_BATCH_SIZE))
DIFFUSION_DRAW_MICROBATCH_SIZE = int(_get("microbatch_sizes.diffusion.draw", DRAW_BATCH_SIZE))

FGSEG_TRAIN_MICROBATCH_SIZE = int(_get("microbatch_sizes.fgseg.train", TRAIN_BATCH_SIZE))
FGSEG_VALID_MICROBATCH_SIZE = int(_get("microbatch_sizes.fgseg.valid", VALID_BATCH_SIZE))
FGSEG_TEST_MICROBATCH_SIZE = int(_get("microbatch_sizes.fgseg.test", TEST_BATCH_SIZE))
FGSEG_DRAW_MICROBATCH_SIZE = int(_get("microbatch_sizes.fgseg.draw", DRAW_BATCH_SIZE))

TRAIN_MAX_EPOCHS = int(_get("training_budget.max_epochs", 300))
TRAIN_MAX_ITERATIONS = int(_get("training_budget.max_iterations", 75000))
EXTRA_MODEL_SAVE_EPOCH_NUM = list(_get("training_budget.extra_model_save_epoch_num", []))
PRETRAIN_EXTRA_MODEL_SAVE_EPOCH_NUM = EXTRA_MODEL_SAVE_EPOCH_NUM
NUM_WORKERS = int(_get("hardware.num_workers", 4))

LEARNING_RATE = float(_get("optimization.learning_rate", 1.0e-4))
MIN_LEARNING_RATE = float(_get("optimization.min_learning_rate", 1.0e-6))
ADAM_BETAS = tuple(float(value) for value in _get("optimization.adam_betas", [0.9, 0.999]))
if len(ADAM_BETAS) != 2 or not all(0.0 <= value < 1.0 for value in ADAM_BETAS):
    raise ValueError(
        f"optimization.adam_betas must contain two values in [0, 1), got {ADAM_BETAS}"
    )
EMA_DECAY = float(_get("optimization.ema_decay", 0.995))
PATIENCE_THRESHOLD_NUM = int(_get("optimization.scheduler_patience", 3))
PRETRAINED_PATIENCE_THRESHOLD_NUM = int(_get("optimization.pretrained_scheduler_patience", 3))

# Diffusion
def _resolve_task_scheduler(get, task_name):
    scheduler_type = str(get("diffusion.scheduler.type", "cosine")).strip().lower()
    supported_types = {"cosine", "linear", "power", "sigmoid"}
    if scheduler_type not in supported_types:
        raise ValueError(
            f"{task_name}: unsupported diffusion.scheduler.type {scheduler_type!r}; "
            f"choose from {sorted(supported_types)}."
        )

    power_val = 3
    min_beta = 1.0e-4
    max_beta = 0.999
    cosine_s = 0.008
    sigmoid_start = -12.0
    sigmoid_end = -2.0

    if scheduler_type == "cosine":
        prefix = "diffusion.scheduler.cosine"
        min_beta = float(get(f"{prefix}.min_beta", min_beta))
        max_beta = float(get(f"{prefix}.max_beta", max_beta))
        cosine_s = float(get(f"{prefix}.offset", cosine_s))
        if cosine_s < 0.0:
            raise ValueError(f"{task_name}: cosine.offset must be non-negative.")
    elif scheduler_type == "linear":
        prefix = "diffusion.scheduler.linear"
        min_beta = float(get(f"{prefix}.min_beta", min_beta))
        max_beta = float(get(f"{prefix}.max_beta", 0.01))
    elif scheduler_type == "power":
        prefix = "diffusion.scheduler.power"
        min_beta = float(get(f"{prefix}.min_beta", min_beta))
        max_beta = float(get(f"{prefix}.max_beta", max_beta))
        power_val = int(get(f"{prefix}.exponent", power_val))
        if power_val <= 0:
            raise ValueError(f"{task_name}: power.exponent must be positive.")
    else:
        prefix = "diffusion.scheduler.sigmoid"
        min_beta = float(get(f"{prefix}.min_beta", min_beta))
        max_beta = float(get(f"{prefix}.max_beta", max_beta))
        sigmoid_start = float(get(f"{prefix}.start", sigmoid_start))
        sigmoid_end = float(get(f"{prefix}.end", sigmoid_end))
        if sigmoid_start >= sigmoid_end:
            raise ValueError(
                f"{task_name}: sigmoid.start must be smaller than sigmoid.end."
            )

    if not 0.0 < min_beta < max_beta <= 1.0:
        raise ValueError(
            f"{task_name}: scheduler beta bounds must satisfy "
            f"0 < min_beta < max_beta <= 1, got {min_beta} and {max_beta}."
        )

    return SimpleNamespace(
        type=scheduler_type,
        power_val=power_val,
        min_beta=min_beta,
        max_beta=max_beta,
        cosine_s=cosine_s,
        sigmoid_start=sigmoid_start,
        sigmoid_end=sigmoid_end,
    )


def _build_task_config(task_name, yaml_path, cfg):
    """Build one task-specific legacy-compatible configuration view."""
    get = lambda path, default=None: _get_from_config(cfg, path, default)
    task_cfg = load_yaml_config(yaml_path)
    task_get = lambda path, default=None: _get_from_config(task_cfg, path, default)
    scheduler = _resolve_task_scheduler(task_get, task_name)
    scheduler_type = scheduler.type
    scheduler_power = scheduler.power_val
    scheduler_tag = (
        f"power_p{scheduler_power}"
        if scheduler_type == "power"
        else scheduler_type
    )
    display_names = {
        "fggen_diffusion": "Fggen_Diffusion",
        "fggen_controlnet": "Fggen_ControlNet",
        "imgsyn_diffusion": "ImgSyn_Diffusion",
        "imgsyn_controlnet": "ImgSyn_ControlNet",
    }
    display_name = display_names[task_name]
    component = "FgGen" if task_name.startswith("fggen") else "ImgSyn"
    component_root = PROJECT_ROOT / component
    dataset_name = DATASET_NAME
    env_prefix = f"IMPGM_{task_name.upper()}"
    artifact_tag = os.environ.get(f"{env_prefix}_ARTIFACT_TAG", "").strip().lower()
    if artifact_tag and any(
        not (char.isalnum() or char in "-_") for char in artifact_tag
    ):
        raise ValueError(
            f"{task_name}: {env_prefix}_ARTIFACT_TAG may contain only letters, "
            f"numbers, '-' and '_', got {artifact_tag!r}."
        )
    artifact_suffix = f"_{artifact_tag}" if artifact_tag else ""
    experiment_name = (
        f"{display_name}_{DATASET_NAME}_{scheduler_tag}{artifact_suffix}"
    )
    preload_root = (
        "controlnet.preload"
        if task_name.endswith("controlnet")
        else "diffusion.preload"
    )
    preload_enabled = bool(get(f"{preload_root}.enabled", False))
    preload_source_path = str(get(f"{preload_root}.source_path", "")).strip()
    preload_skip_prefix = str(get(f"{preload_root}.skip_prefix", "")).strip()
    preload_source_dataset = str(
        get(f"{preload_root}.source_dataset", "")
    ).strip().lower()
    if preload_enabled and not preload_source_dataset:
        raise ValueError(
            f"{task_name}: set {preload_root}.source_dataset when preload is enabled."
        )
    if preload_source_dataset and any(
        not (char.isalnum() or char in "-_") for char in preload_source_dataset
    ):
        raise ValueError(
            f"{task_name}: invalid preload source_dataset {preload_source_dataset!r}."
        )
    training_regime = (
        f"transfer_from_{preload_source_dataset}"
        if preload_enabled
        else "from_scratch"
    )
    artifact_parent = Path(dataset_name)
    if preload_enabled:
        artifact_parent /= training_regime

    if task_name.endswith("controlnet"):
        log_dir = (
            component_root
            / f"{display_name}_Logs"
            / artifact_parent
            / f"{display_name}_Logs_{DATASET_NAME}_{scheduler_tag}{artifact_suffix}"
        )
        rgb_prefix = f"{component}_ControlNet"
        rgb_dir = (
            component_root
            / f"{rgb_prefix}_RGBs"
            / artifact_parent
            / f"{rgb_prefix}_RGBs_{DATASET_NAME}_{scheduler_tag}{artifact_suffix}"
        )
        tif_dir = (
            component_root
            / f"{rgb_prefix}_TIFs"
            / artifact_parent
            / f"{rgb_prefix}_TIFs_{DATASET_NAME}_{scheduler_tag}{artifact_suffix}"
        )
    else:
        log_dir = (
            component_root
            / f"{display_name}_Logs"
            / artifact_parent
            / f"{display_name}_Logs_{DATASET_NAME}_{scheduler_tag}{artifact_suffix}"
        )
        rgb_prefix = f"{component}_Diffusion"
        rgb_dir = (
            component_root
            / f"{rgb_prefix}_RGBs"
            / artifact_parent
            / f"{rgb_prefix}_RGBs_{DATASET_NAME}_{scheduler_tag}{artifact_suffix}"
        )
        tif_dir = (
            component_root
            / f"{rgb_prefix}_TIFs"
            / artifact_parent
            / f"{rgb_prefix}_TIFs_{DATASET_NAME}_{scheduler_tag}{artifact_suffix}"
        )

    model_savepath = (
        component_root
        / f"{display_name}_Models_SaveFolder"
        / artifact_parent
        / f"{experiment_name}.pth"
    )
    frequency_loss_weight = float(
        os.environ.get(
            f"{env_prefix}_FREQUENCY_WEIGHT",
            get("diffusion.losses.frequency_weight", 0.0),
        )
    )
    edge_loss_weight = float(
        os.environ.get(
            f"{env_prefix}_EDGE_WEIGHT",
            get("diffusion.losses.edge_weight", 0.0),
        )
    )
    if not np.isfinite(frequency_loss_weight) or frequency_loss_weight < 0.0:
        raise ValueError(
            f"{task_name}: frequency loss weight must be non-negative, "
            f"got {frequency_loss_weight}."
        )
    if not np.isfinite(edge_loss_weight) or edge_loss_weight < 0.0:
        raise ValueError(
            f"{task_name}: edge loss weight must be non-negative, "
            f"got {edge_loss_weight}."
        )
    conditional_ch = int(get("controlnet.conditional_channels", 1))
    controlnet_weight = float(get("controlnet.weight", 1.0))
    background_highpass_filter_scale = float(
        get("controlnet.background_highpass_filter_scale", 0.20)
    )
    if task_name.endswith("controlnet") and conditional_ch <= 0:
        raise ValueError(
            f"{task_name}: controlnet.conditional_channels must be positive, "
            f"got {conditional_ch}"
        )
    if (
        task_name == "imgsyn_controlnet"
        and not 0.0 < background_highpass_filter_scale <= 1.0
    ):
        raise ValueError(
            f"{task_name}: controlnet.background_highpass_filter_scale "
            "must be in (0, 1], got "
            f"{background_highpass_filter_scale}"
        )

    return SimpleNamespace(
        CFG=cfg,
        TASK_NAME=task_name,
        TASK_YAML_PATH=str(yaml_path),
        EXP_NAME=experiment_name,
        ARTIFACT_TAG=artifact_tag,
        STEPS=int(get("diffusion.sampling.steps", 1000)),
        SCHEDULER_TYPE=scheduler_type,
        SCHEDULER_POWER_VAL=scheduler_power,
        SCHEDULER_MIN_BETA=scheduler.min_beta,
        SCHEDULER_MAX_BETA=scheduler.max_beta,
        SCHEDULER_COSINE_S=scheduler.cosine_s,
        SCHEDULER_SIGMOID_START=scheduler.sigmoid_start,
        SCHEDULER_SIGMOID_END=scheduler.sigmoid_end,
        MODEL_CH=int(get("diffusion.unet.ch", 96)),
        MODEL_CH_MULT=tuple(get("diffusion.unet.ch_mult", [1, 2, 3, 4])),
        MODEL_ATTN_RESOLUTIONS=list(
            get("diffusion.unet.attn_resolutions", [8, 16, 32, 64])
        ),
        MODEL_DROPOUT=float(get("diffusion.unet.dropout", 0.0)),
        MODEL_RESAMP_WITH_CONV=bool(get("diffusion.unet.resamp_with_conv", True)),
        MODEL_RESOLUTION=int(get("diffusion.unet.resolution", 64)),
        MODEL_IN_CHANNELS=int(get("diffusion.unet.in_channels", 8)),
        MODEL_OUT_CH=int(get("diffusion.unet.out_channels", 8)),
        MODEL_CONDITIONAL_CH=conditional_ch,
        MODEL_CONTROLNET_WEIGHT=controlnet_weight,
        CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE=(
            background_highpass_filter_scale
        ),
        LOSS_FREQ_POWER=float(get("diffusion.losses.freq_power", 1.0)),
        FREQUENCY_LOSS_WEIGHT=frequency_loss_weight,
        EDGE_LOSS_WEIGHT=edge_loss_weight,
        TRAINING_REGIME=training_regime,
        PRELOAD_ENABLED=preload_enabled,
        PRELOAD_SOURCE_DATASET=preload_source_dataset,
        PRELOAD_SOURCE_PATH=preload_source_path,
        PRELOAD_SKIP_PREFIX=preload_skip_prefix,
        PRELOAD_METADATA={
            "training_regime": training_regime,
            "preload_enabled": preload_enabled,
            "source_dataset": preload_source_dataset if preload_enabled else None,
            "source_checkpoint": preload_source_path if preload_enabled else None,
            "skip_prefix": preload_skip_prefix if preload_enabled else None,
        },
        LOG_DIR=str(log_dir),
        TRAIN_INFO_PATH=str(
            model_savepath.with_name(f"{experiment_name}_TrainingInfo.txt")
        ),
        RGB_DIR=str(rgb_dir),
        TIF_DIR=str(tif_dir),
        MODEL_SAVEPATH=str(model_savepath),
    )


FGGEN_DIFFUSION_CONFIG = _build_task_config(
    "fggen_diffusion", _FGGEN_DIFFUSION_YAML_PATH, _FGGEN_DIFFUSION_CFG
)
FGGEN_CONTROLNET_CONFIG = _build_task_config(
    "fggen_controlnet", _FGGEN_CONTROLNET_YAML_PATH, _FGGEN_CONTROLNET_CFG
)
IMGSYN_DIFFUSION_CONFIG = _build_task_config(
    "imgsyn_diffusion", _IMGSYN_DIFFUSION_YAML_PATH, _IMGSYN_DIFFUSION_CFG
)
IMGSYN_CONTROLNET_CONFIG = _build_task_config(
    "imgsyn_controlnet", _IMGSYN_CONTROLNET_YAML_PATH, _IMGSYN_CONTROLNET_CFG
)

_TASK_CONFIG_EXPORT_NAMES = (
    "TASK_NAME",
    "TASK_YAML_PATH",
    "EXP_NAME",
    "ARTIFACT_TAG",
    "STEPS",
    "SCHEDULER_TYPE",
    "SCHEDULER_POWER_VAL",
    "SCHEDULER_MIN_BETA",
    "SCHEDULER_MAX_BETA",
    "SCHEDULER_COSINE_S",
    "SCHEDULER_SIGMOID_START",
    "SCHEDULER_SIGMOID_END",
    "MODEL_CH",
    "MODEL_CH_MULT",
    "MODEL_ATTN_RESOLUTIONS",
    "MODEL_DROPOUT",
    "MODEL_RESAMP_WITH_CONV",
    "MODEL_RESOLUTION",
    "MODEL_IN_CHANNELS",
    "MODEL_OUT_CH",
    "MODEL_CONDITIONAL_CH",
    "MODEL_CONTROLNET_WEIGHT",
    "CONTROLNET_BACKGROUND_HIGHPASS_FILTER_SCALE",
    "LOSS_FREQ_POWER",
    "FREQUENCY_LOSS_WEIGHT",
    "EDGE_LOSS_WEIGHT",
    "TRAINING_REGIME",
    "PRELOAD_ENABLED",
    "PRELOAD_SOURCE_DATASET",
    "PRELOAD_SOURCE_PATH",
    "PRELOAD_SKIP_PREFIX",
    "PRELOAD_METADATA",
    "LOG_DIR",
    "TRAIN_INFO_PATH",
    "RGB_DIR",
    "TIF_DIR",
    "MODEL_SAVEPATH",
)


def apply_task_config(namespace, task_config):
    for name in _TASK_CONFIG_EXPORT_NAMES:
        namespace[name] = getattr(task_config, name)


# Backward-compatible defaults for shared utilities that do not select a task.
apply_task_config(globals(), FGGEN_DIFFUSION_CONFIG)
FGGEN_CONTROLNET_CONDITIONAL_CH = FGGEN_CONTROLNET_CONFIG.MODEL_CONDITIONAL_CH
FGGEN_CONTROLNET_WEIGHT = FGGEN_CONTROLNET_CONFIG.MODEL_CONTROLNET_WEIGHT
IMGSYN_CONTROLNET_CONDITIONAL_CH = IMGSYN_CONTROLNET_CONFIG.MODEL_CONDITIONAL_CH
IMGSYN_CONTROLNET_WEIGHT = IMGSYN_CONTROLNET_CONFIG.MODEL_CONTROLNET_WEIGHT

# VAE
VAE_DOWN_CHANNELS = list(_get("vae.model.down_channels", [32, 64, 128]))
VAE_MID_CHANNELS = list(_get("vae.model.mid_inout_channels", [128, 128]))
VAE_NUM_DOWN_LAYERS = int(_get("vae.model.num_down_layers", 1))
VAE_NUM_MID_LAYERS = int(_get("vae.model.num_mid_layers", 1))
VAE_NUM_UP_LAYERS = int(_get("vae.model.num_up_layers", 1))
VAE_Z_CHANNEL = int(_get("vae.model.z_channel", 8))
VAE_NORM_CHANNELS = int(_get("vae.model.norm_channels", 8))
VAE_PRELOAD_ENABLED = _VAE_PRELOAD_ENABLED_FOR_LATENT
VAE_PRELOAD_SOURCE_PATH = str(_get("vae.preload.source_path", "")).strip()
VAE_PRELOAD_SOURCE_DATASET = str(
    _get("vae.preload.source_dataset", "")
).strip().lower()
if VAE_PRELOAD_ENABLED and not VAE_PRELOAD_SOURCE_DATASET:
    raise ValueError("Set vae.preload.source_dataset when preload is enabled.")
if VAE_PRELOAD_SOURCE_DATASET and any(
    not (char.isalnum() or char in "-_") for char in VAE_PRELOAD_SOURCE_DATASET
):
    raise ValueError(
        f"Invalid vae.preload.source_dataset {VAE_PRELOAD_SOURCE_DATASET!r}."
    )
VAE_TRAINING_REGIME = (
    f"transfer_from_{VAE_PRELOAD_SOURCE_DATASET}"
    if VAE_PRELOAD_ENABLED
    else "from_scratch"
)
VAE_PRELOAD_METADATA = {
    "training_regime": VAE_TRAINING_REGIME,
    "preload_enabled": VAE_PRELOAD_ENABLED,
    "source_dataset": VAE_PRELOAD_SOURCE_DATASET if VAE_PRELOAD_ENABLED else None,
    "source_checkpoint": VAE_PRELOAD_SOURCE_PATH if VAE_PRELOAD_ENABLED else None,
    "skip_prefix": None,
}
KL_LOSS_WEIGHT = float(_get("vae.losses.kl_weight", 0.01))

# Logging
SAVE_TIF_IMAGES = bool(_get("logging.save_tif_images", True))
DRAW_SAMPLER_MODE = str(_get("draw.sampler", "ddpm")).strip().lower()
if DRAW_SAMPLER_MODE not in {"ddpm", "ddim"}:
    raise ValueError(
        f"Unsupported draw.sampler {DRAW_SAMPLER_MODE!r}; use 'ddpm' or 'ddim'."
    )
DRAW_RANDOM_SEED = int(_get("draw.seed", 999))
DRAW_INTERVAL_EPOCHS = int(_get("draw.interval_epochs", 5))
TIF_INTERVAL_EPOCHS = int(_get("draw.tif_interval_epochs", 25))
if DRAW_INTERVAL_EPOCHS <= 0 or TIF_INTERVAL_EPOCHS <= 0:
    raise ValueError("draw interval values must be positive integers")

# Experiment paths
_VAE_ARTIFACT_PARENT = Path(DATASET_NAME)
if VAE_PRELOAD_ENABLED:
    _VAE_ARTIFACT_PARENT /= VAE_TRAINING_REGIME
VAE_LOG_DIR = str(PROJECT_ROOT / "VAE" / "VAE_Logs" / _VAE_ARTIFACT_PARENT)
VAE_MODEL_SAVEPATH = str(
    PROJECT_ROOT
    / "VAE"
    / "VAE_Models_SaveFolder"
    / _VAE_ARTIFACT_PARENT
    / f"VAE_{DATASET_NAME}.pth"
)
VAE_TRAIN_INFO_PATH = str(
    Path(VAE_MODEL_SAVEPATH).with_name(f"VAE_{DATASET_NAME}_TrainingInfo.txt")
)
VAE_RGB_DIR = str(PROJECT_ROOT / "VAE" / "VAE_RGBs" / _VAE_ARTIFACT_PARENT)
VAE_TIF_DIR = str(PROJECT_ROOT / "VAE" / "VAE_TIFs" / _VAE_ARTIFACT_PARENT)
FGSEG_LOG_DIR = str(PROJECT_ROOT / "FgSeg_UNet" / "FgSeg_Logs" / DATASET_NAME)
FGSEG_UNET_MODEL_SAVEPATH = str(
    PROJECT_ROOT
    / "FgSeg_UNet"
    / "FgSeg_Models_SaveFolder"
    / DATASET_NAME
    / f"FgSeg_UNet_{DATASET_NAME}.pth"
)
FGSEG_TRAIN_INFO_PATH = str(
    Path(FGSEG_UNET_MODEL_SAVEPATH).with_name(
        f"FgSeg_UNet_{DATASET_NAME}_TrainingInfo.txt"
    )
)
FGSEG_TEST_INFO_PATH = str(
    Path(FGSEG_LOG_DIR) / f"FgSeg_UNet_{DATASET_NAME}_TestInfo.txt"
)
FGSEG_RGB_DIR = str(PROJECT_ROOT / "FgSeg_UNet" / "FgSeg_RGBs" / DATASET_NAME)
FGSEG_TIF_DIR = str(PROJECT_ROOT / "FgSeg_UNet" / "FgSeg_TIFs" / DATASET_NAME)
FGGEN_DIFFUSION_MODEL_SAVEPATH = FGGEN_DIFFUSION_CONFIG.MODEL_SAVEPATH
FGGEN_CONTROLNET_MODEL_SAVEPATH = FGGEN_CONTROLNET_CONFIG.MODEL_SAVEPATH
IMGSYN_DIFFUSION_MODEL_SAVEPATH = IMGSYN_DIFFUSION_CONFIG.MODEL_SAVEPATH
IMGSYN_CONTROLNET_MODEL_SAVEPATH = IMGSYN_CONTROLNET_CONFIG.MODEL_SAVEPATH
FGGEN_BASE_DIFFUSION_MODEL_SAVEPATH = FGGEN_DIFFUSION_MODEL_SAVEPATH
IMGSYN_BASE_DIFFUSION_MODEL_SAVEPATH = IMGSYN_DIFFUSION_MODEL_SAVEPATH

# Legacy constants kept for old scripts
UNET_OUTPUT_CHANNELS = MASK_CHANNELS
RANDOM_SEED = 999
DETERMINISTIC_EVAL = True

def _resolve_device():
    if torch.cuda.is_available():
        return "cuda"
    xpu_backend = getattr(torch, "xpu", None)
    if xpu_backend is not None:
        is_available = getattr(xpu_backend, "is_available", None)
        if callable(is_available) and is_available():
            return "xpu"
    return "cpu"

DEVICE = _resolve_device()
DEVICE_TYPE = torch.device(DEVICE).type

# Only autocast uses bf16; model parameters and explicit calculations stay fp32.
AMP_ENABLED = True
AMP_DTYPE = torch.bfloat16
PIN_MEMORY = DEVICE_TYPE in ("cuda", "xpu")
DATA_TRANSFER_NON_BLOCKING = PIN_MEMORY
METRIC_DEVICE = DEVICE

# Transforms
class CustomNormalize:
    """Apply per-channel image normalization from dataset statistics."""
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32).reshape(-1)
        self.std = np.array(std, dtype=np.float32).reshape(-1)
        if np.any(self.std <= 0):
            raise ValueError(
                f"CustomNormalize: all elements of std must be strictly positive, got {std}"
            )
        self.available_keys = ["mean", "std"]

    def __call__(self, image, **kwargs):
        # image: (H, W, C) float32 ndarray from albumentations
        mean = self.mean.reshape(1, 1, -1)
        std = self.std.reshape(1, 1, -1)
        normalized_image = (image - mean) / std
        return normalized_image

custom_normalize = CustomNormalize(mean=IMAGE_MEAN, std=IMAGE_STD)


all_train_transforms = A.Compose(
    [
        A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ],
    additional_targets={
        "image0": "image",
    },
)

all_test_transforms = A.Compose(
    [
        A.CenterCrop(width=IMG_SIZE, height=IMG_SIZE, p=1.0),
    ],
    additional_targets={
        "image0": "image",
    },
)

transform_only_tif = A.Compose(
    [
        A.Lambda(image=custom_normalize),
        ToTensorV2(),
    ]
)

transform_only_msk = A.Compose(
    [
        ToTensorV2(),
    ]
)

transform_only_wgt = A.Compose(
    [
        ToTensorV2(),
    ]
)
