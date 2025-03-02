"""
Dataset loader for pre-processed patch datasets.
This loads patches from the HDF5 files created by create_patch_dataset.py.
"""

import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PreprocessedPatchDataset(Dataset):
    """Dataset for loading pre-processed patches for contrastive learning."""

    def __init__(self, patch_dir, split="train", transform=None, cache_size=100):
        """
        Args:
            patch_dir: Directory containing pre-processed patches
            split: 'train' or 'val'
            transform: Optional transforms to apply to patches
            cache_size: Number of patches to cache in memory
        """
        self.patch_dir = Path(patch_dir)
        self.split = split
        self.transform = transform
        self.cache_size = cache_size

        # Load metadata
        metadata_path = self.patch_dir / f"{split}_metadata.json"
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Open HDF5 file
        self.h5_path = self.patch_dir / f"{split}_patches.h5"
        self.h5_file = None  # Will be opened on first access

        # Cache for frequently accessed patches
        self.cache = {}

        # Load summary for stats
        summary_path = self.patch_dir / f"{split}_summary.json"
        if summary_path.exists():
            with open(summary_path, "r") as f:
                self.summary = json.load(f)
                logger.info(
                    f"Loaded {split} dataset with {self.summary['total_patches']} patches "
                    f"({self.summary['positive_patches']} positive, "
                    f"{self.summary['negative_patches']} negative)"
                )
        else:
            self.summary = None
            logger.info(f"Loaded {split} dataset with {len(self.metadata)} patches")

    def __len__(self):
        return len(self.metadata)

    def _open_h5(self):
        """Open the HDF5 file if not already open."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

    def _close_h5(self):
        """Close the HDF5 file if open."""
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def __getitem__(self, idx):
        """Get a patch pair."""
        self._open_h5()

        # Get metadata for this index
        meta = self.metadata[idx]
        h5_idx = meta["index"]

        # Try to get from cache first
        if h5_idx in self.cache:
            pre_patch = self.cache[h5_idx]["pre_patch"]
            post_patch = self.cache[h5_idx]["post_patch"]
            label = self.cache[h5_idx]["label"]
        else:
            # Load from HDF5
            pre_patch = self.h5_file["pre_patches"][h5_idx]
            post_patch = self.h5_file["post_patches"][h5_idx]
            label = self.h5_file["labels"][h5_idx]

            # Squeeze singleton dimensions from label
            if label.ndim > 3:
                label = np.squeeze(label, axis=-1)

            # Add to cache
            if len(self.cache) < self.cache_size:
                self.cache[h5_idx] = {
                    "pre_patch": pre_patch,
                    "post_patch": post_patch,
                    "label": label,
                }

        # Get metadata
        is_positive = meta["is_positive"]

        # Apply transforms
        if self.transform:
            transformed = self.transform(pre_patch, post_patch, label)
            pre_patch = transformed["pre_image"]
            post_patch = transformed["post_image"]
            label = transformed["label"]

        # Convert to tensor if not already
        if not isinstance(pre_patch, torch.Tensor):
            pre_patch = self._to_tensor(pre_patch)
            post_patch = self._to_tensor(post_patch)
            label = torch.from_numpy(label.copy()).long()

        return {
            "pre_patch": pre_patch,
            "post_patch": post_patch,
            "label": label,
            "is_positive": torch.tensor([float(is_positive)], dtype=torch.float32),
            "idx": idx,
            "image_id": meta["image_id"],
            "position": meta["position"],
            "damage_ratio": meta["damage_ratio"],
        }

    def _to_tensor(self, img):
        """Convert image to tensor and normalize."""
        # Convert to float32
        img = img.astype(np.float32)

        # Normalize
        img = (img - img.mean()) / (img.std() + 1e-8)

        # Convert to tensor with channel-first format (HWC -> CHW)
        img = torch.from_numpy(img.transpose(2, 0, 1))

        return img

    def get_pos_neg_ratio(self):
        """Get ratio of positive to negative samples."""
        if self.summary:
            return self.summary["positive_ratio"]

        # Count from metadata
        pos_count = sum(1 for meta in self.metadata if meta["is_positive"])
        total = len(self.metadata)
        return pos_count / total if total > 0 else 0

    def __del__(self):
        """Close the HDF5 file on deletion."""
        self._close_h5()
