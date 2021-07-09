from pathlib import Path
from typing import List, Dict, Any
from typing import Tuple

import albumentations as albu
import numpy as np
import torch
from iglovikov_helper_functions.utils.image_utils import load_rgb, load_grayscale
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, Path]],
        transform: albu.Compose,
        deg_transform: albu.Compose = None,
        length: int = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.deg_transform = deg_transform

        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = idx % len(self.samples)

        image_path, mask_path = self.samples[idx]

        image = load_rgb(image_path, lib="cv2")
        mask = load_grayscale(mask_path)

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]
        if self.deg_transform is not None:
            degraded_sample = self.deg_transform(image=image, mask=mask)
            degraded_image, degraded_mask = degraded_sample["image"], degraded_sample["mask"]
        else:
            degraded_image = image
            degraded_mask = mask

        degraded_mask = torch.from_numpy(degraded_mask)
        mask = torch.from_numpy(mask)
        return {
            "image_id": image_path.stem,
            "features": tensor_from_rgb_image(degraded_image),
            "masks": torch.unsqueeze(degraded_mask, 0).float(),
            "features_HQ": tensor_from_rgb_image(image) / 255.,
            "mask_HQ": torch.unsqueeze(mask, 0).float(),
        }

