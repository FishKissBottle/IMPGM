import torch
from torch.utils.data import Dataset

from IMPGM_Config import PROMPT_DICT
from IMPGM_Dataset import IMPGM_Dataset


class EvaluationUNetDataset(Dataset):
    """Expose normalized full images, class IDs, and aligned binary masks."""

    def __init__(self, image_roots, mask_roots, is_train):
        self.base = IMPGM_Dataset(
            img_rootdir_list=list(image_roots),
            msk_rootdir_list=list(mask_roots),
            is_train=bool(is_train),
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        label, _, mask, full_image, _, _ = self.base[index]
        if label not in PROMPT_DICT:
            raise KeyError(f"Unknown evaluation label {label!r}.")
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Evaluation_UNet requires tensor masks.")
        return {
            "image": full_image,
            "mask": (mask[:1] >= 0.5).to(torch.float32),
            "label_id": int(PROMPT_DICT[label]),
            "label_text": label,
        }
