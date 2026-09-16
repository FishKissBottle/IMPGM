import numpy as np
from pathlib import Path

from IMPGM_Config import *
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from IMPGM_TifReader import Tif_Read_and_Write
from osgeo import gdal
from IMPGM_Utils import (
    prepare_rgb_vis_tensor,
    prepare_mask_vis_tensor,
    seed_dataloader_worker,
    build_dataloader_generator,
)

gdal.UseExceptions()


def build_unique_resolved_entries(dataset):
    """Map indices to unique nonzero source paths actually returned by a dataset."""
    entries = []
    seen_paths = set()
    for catalog_index in range(len(dataset)):
        source_path = str(Path(dataset.resolve_image_path(catalog_index)).resolve())
        source_key = source_path.casefold()
        if source_key in seen_paths:
            continue
        seen_paths.add(source_key)
        entries.append((catalog_index, source_path))
    if not entries:
        raise RuntimeError("Dataset contains no nonzero image samples.")
    return entries


def _list_sorted_files(rootdir):
    raster_suffixes = {".tif", ".tiff"}
    return sorted(
        str(path)
        for path in Path(rootdir).iterdir()
        if path.is_file() and path.suffix.lower() in raster_suffixes
    )


def _extract_label(img_path):
    return Path(img_path).stem.split("_")[-2]


def _build_mask_name_to_path(msk_rootdir_list):
    mask_name_to_path = {}
    mask_path_catalog = []

    for msk_rootdir in msk_rootdir_list:
        for msk_path in _list_sorted_files(msk_rootdir):
            msk_name = Path(msk_path).name
            existing_path = mask_name_to_path.get(msk_name)
            if existing_path is not None and existing_path != msk_path:
                raise ValueError(
                    "Duplicate mask filename detected across mask roots: "
                    f"'{msk_name}' exists in both '{existing_path}' and '{msk_path}'."
                )
            mask_name_to_path[msk_name] = msk_path
            mask_path_catalog.append(msk_path)

    return mask_name_to_path, mask_path_catalog


def _resolve_mask_path(img_path, mask_name_to_path):
    img_name = Path(img_path).name
    candidate_names = []

    if "_data_" in img_name:
        candidate_names.append(img_name.replace("_data_", "_mask_", 1))
    candidate_names.append(img_name)

    seen = set()
    deduped_candidates = []
    for candidate_name in candidate_names:
        if candidate_name not in seen:
            seen.add(candidate_name)
            deduped_candidates.append(candidate_name)

    for candidate_name in deduped_candidates:
        msk_path = mask_name_to_path.get(candidate_name)
        if msk_path is not None:
            return msk_path

    raise FileNotFoundError(
        f"Mask not found for image '{img_path}'. Tried mask names: {deduped_candidates}"
    )


class IMPGM_Dataset(Dataset):
    """Load aligned IMPGM image-mask samples with shared preprocessing."""
    def __init__(self, img_rootdir_list, msk_rootdir_list=None, is_train=True):
        self.img_path_catalog = []
        self.msk_path_catalog = None
        self._mask_name_to_path = None

        for img_rootdir in img_rootdir_list:
            self.img_path_catalog.extend(_list_sorted_files(img_rootdir))

        if msk_rootdir_list is not None:
            self._mask_name_to_path, self.msk_path_catalog = _build_mask_name_to_path(msk_rootdir_list)

        self.is_train = is_train

    def __len__(self):
        return len(self.img_path_catalog)

    def _read_nonzero_image(self, index):
        """Resolve and read the actual nonzero catalog sample used for an index."""
        sample_num = len(self.img_path_catalog)
        for offset in range(sample_num):
            img_path = self.img_path_catalog[(index + offset) % sample_num]
            img, proj, geo = Tif_Read_and_Write().Tif_Read(img_path)
            if np.max(img) != 0.00:
                return img_path, img, proj, geo
        raise RuntimeError("All images in IMPGM_Dataset have max pixel value equal to 0.")

    def resolve_image_path(self, index):
        """Return the source path actually selected after zero-image skipping."""
        img_path, _, _, _ = self._read_nonzero_image(index)
        return img_path

    def __getitem__(self, index):
        """Load one sample and return it as a 6-tuple (label, fg_img, msk, syn_img, proj, geo).

        label : str
            Class label extracted from the image file name (second-to-last
            underscore-separated token of the stem, e.g. "Water", "NoObj").
        fg_img : torch.Tensor, shape (INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)
            Foreground image: the normalized syn_img multiplied by the mask
            (fg_img = syn_img_normalized * msk), so foreground pixels hold
            normalized values and background pixels are 0.
        msk : torch.Tensor, shape (1, IMG_SIZE, IMG_SIZE)
            Mask tensor whose source values are preserved without z-score
            normalization. Source datasets are expected to provide binary masks.
            Samples whose label starts with "No" (background-only class)
            automatically get an all-zero mask.
        syn_img : torch.Tensor, shape (INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)
            Full image after per-channel z-score normalization
            (CustomNormalize with IMAGE_MEAN / IMAGE_STD).
        proj : str
            WKT projection string of the source TIF.
        geo : tuple
            6-element GDAL affine geotransform of the source TIF.

        Images whose maximum pixel value is 0 are skipped (the following
        catalog entry is tried instead). If the dataset was built without
        mask roots, msk and fg_img are the sentinel value -999, not tensors.
        """
        img_path, img, proj, geo = self._read_nonzero_image(index)

        img = np.transpose(img, (1, 2, 0))
        label = _extract_label(img_path)

        if self.msk_path_catalog is not None:
            if not label.startswith("No"):
                msk_path = _resolve_mask_path(img_path, self._mask_name_to_path)
                msk, _, _ = Tif_Read_and_Write().Tif_Read(msk_path)
            else:
                msk = np.zeros(img.shape[:2], dtype=np.float32)

        if self.is_train and self.msk_path_catalog is not None:
            augmentataions_train = all_train_transforms(image=img, image0=msk)
            img, msk = augmentataions_train["image"], augmentataions_train["image0"]

            syn_img = transform_only_tif(image=img)["image"]
            msk = transform_only_msk(image=msk)["image"]
            fg_img = transform_only_tif(image=img)["image"]
            fg_img = fg_img * msk

        elif self.is_train and self.msk_path_catalog is None:
            augmentataions_train = all_train_transforms(image=img)
            img = augmentataions_train["image"]

            syn_img = transform_only_tif(image=img)["image"]
            msk = -999
            fg_img = -999

        elif not self.is_train and self.msk_path_catalog is not None:
            augmentataions_test = all_test_transforms(image=img, image0=msk)
            img, msk = augmentataions_test["image"], augmentataions_test["image0"]

            syn_img = transform_only_tif(image=img)["image"]
            msk = transform_only_msk(image=msk)["image"]
            fg_img = transform_only_tif(image=img)["image"]
            fg_img = fg_img * msk

        else:
            augmentataions_test = all_test_transforms(image=img)
            img = augmentataions_test["image"]

            syn_img = transform_only_tif(image=img)["image"]
            msk = -999
            fg_img = -999

        return label, fg_img, msk, syn_img, proj, geo


if __name__ == "__main__":
    img_rootdir_list = [
        r"./data/sfq2019/train/images/Paddy",
    ]
    img_rootdir_list.sort(key=lambda x: "No" in x)

    msk_rootdir_list = [
        r"./data/sfq2019/train/masks/Paddy",
    ]

    dataset = IMPGM_Dataset(
        img_rootdir_list=img_rootdir_list,
        msk_rootdir_list=msk_rootdir_list,
        is_train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_dataloader_worker,
        generator=build_dataloader_generator(None),
    )

    for lab, fg_img, msk, syn_img, proj, geo in loader:
        print("label: ", lab)
        print(proj)
        print(geo)
        print("min: ", torch.min(syn_img))
        print("max: ", torch.max(syn_img))
        plt.figure(figsize=(30, 30))
        if torch.all(fg_img != -999):
            plt.subplot(1, 3, 1)
            plt.imshow(prepare_rgb_vis_tensor(fg_img[0]).cpu().permute(1, 2, 0).numpy())
        if torch.all(msk != -999):
            plt.subplot(1, 3, 2)
            plt.imshow(prepare_mask_vis_tensor(msk[0]).cpu().numpy())
        plt.subplot(1, 3, 3)
        plt.imshow(prepare_rgb_vis_tensor(syn_img[0]).cpu().permute(1, 2, 0).numpy())

        plt.show()
