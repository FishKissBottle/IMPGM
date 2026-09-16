"""Configuration for the downstream synthetic-data replacement U-Net."""

from IMPGM_Config import PROMPT_DICT


ARCHITECTURE_ID = "impgm_class_conditioned_downstream_unet_v1"
UNET_CHANNELS = (16, 32, 64, 128, 256)
CONDITION_CHANNELS = 16


def number_of_classes():
    if not PROMPT_DICT:
        raise RuntimeError("Downstream_UNet requires a non-empty dataset.prompt_map.")
    values = sorted(int(value) for value in PROMPT_DICT.values())
    expected = list(range(len(values)))
    if values != expected:
        raise ValueError(
            f"dataset.prompt_map IDs must be contiguous {expected}, got {values}."
        )
    return len(values)
