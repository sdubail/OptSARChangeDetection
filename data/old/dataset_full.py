from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class MultimodalDamageDataset(Dataset):
    """Dataset for multimodal damage assessment with optical pre-event and SAR post-event images."""

    def __init__(self, root_dir, split="train", transform=None, crop_size=256):
        """
        Args:
            root_dir: Root directory containing the dataset
            split: 'train' or 'val'
            transform: Optional transforms to apply
            crop_size: Size of patches for training
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.crop_size = crop_size

        post_event_dir = self.root_dir / split / "post-event"
        self.image_ids = [
            f.name.replace("_post_disaster.tif", "")
            for f in post_event_dir.glob("*_post_disaster.tif")
        ]

        print(f"Found {len(self.image_ids)} images for {split} split")

    def __len__(self):
        return len(self.image_ids)

    def load_tiff_data(self, path):
        """Load and preprocess TIFF images."""
        data_reader = rasterio.open(path)
        try:
            return np.stack(
                [data_reader.read(1), data_reader.read(2), data_reader.read(3)], axis=2
            )
        except:
            return data_reader.read(1)[..., np.newaxis]

    def __getitem__(self, idx):
        """Get a pre-event optical, post-event SAR, and damage label triplet."""
        image_id = self.image_ids[idx]

        pre_path = (
            self.root_dir / self.split / "pre-event" / f"{image_id}_pre_disaster.tif"
        )
        pre_img = self.load_tiff_data(str(pre_path))
        if pre_img.shape[-1] > 3:  # Ensure 3 channels
            pre_img = pre_img[:, :, :3]

        post_path = (
            self.root_dir / self.split / "post-event" / f"{image_id}_post_disaster.tif"
        )
        post_img = self.load_tiff_data(str(post_path))

        if post_img.shape[-1] == 1:
            post_img = np.repeat(post_img, 3, axis=2)

        label_path = (
            self.root_dir / self.split / "target" / f"{image_id}_building_damage.tif"
        )
        label = self.load_tiff_data(str(label_path))

        loc_label = label.copy()
        loc_label[loc_label > 0] = 1  # Any damage becomes 1

        if self.transform:
            sample = self.transform(pre_img, post_img, label)
            pre_img, post_img, label = (
                sample["pre_image"],
                sample["post_image"],
                sample["label"],
            )

        pre_img = self._normalize_and_to_tensor(pre_img)
        post_img = self._normalize_and_to_tensor(post_img)

        label = torch.from_numpy(label.squeeze().copy()).long()
        loc_label = torch.from_numpy(loc_label.squeeze().copy()).long()

        return {
            "pre_image": pre_img,
            "post_image": post_img,
            "label": label,
            "loc_label": loc_label,
            "image_id": image_id,
        }

    def _normalize_and_to_tensor(self, img):
        """Normalize and convert image to tensor."""
        img = img.astype(np.float32)

        img = (img - img.mean()) / (img.std() + 1e-8)

        img = torch.from_numpy(img.transpose(2, 0, 1))

        return img
