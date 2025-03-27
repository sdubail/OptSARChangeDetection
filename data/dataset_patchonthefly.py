# data/dataset_onthefly_numpy.py
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

SPLIT_DATA = {"train": "train", "val": "train", "test": "val"}


class OnTheFlyPatchDataset(Dataset):
    """Dataset that extracts patches on-the-fly from pre-computed metadata in NumPy format."""

    def __init__(
        self,
        root_dir,
        metadata_dir,
        split="train",
        transform=None,
        cache_size=10,
        subset_fraction=1.0,
        target_neg_ratio=None,
        seed=42,
    ):
        """
        Args:
            root_dir: Root directory containing original images
            metadata_dir: Directory with patch metadata
            split: 'train' or 'val' or 'test'
            transform: Optional transforms for data augmentation
            cache_size: Number of full images to cache
            subset_fraction: Fraction of dataset to use
            target_neg_ratio: Target ratio of negative samples (0-1)
            seed: Random seed
        """
        self.root_dir = Path(root_dir)
        self.metadata_dir = Path(metadata_dir)
        self.split = split
        self.transform = transform
        self.cache_size = cache_size

        # Load metadata
        self.image_ids = np.load(
            self.metadata_dir / f"{split}_image_ids.npy", allow_pickle=True
        )
        self.roi_positions = np.load(self.metadata_dir / f"{split}_roi_positions.npy")
        self.context_positions = np.load(
            self.metadata_dir / f"{split}_context_positions.npy"
        )
        self.is_positive = np.load(self.metadata_dir / f"{split}_is_positive.npy")
        self.damage_ratios = np.load(self.metadata_dir / f"{split}_damage_ratios.npy")

        self.has_building = np.load(self.metadata_dir / f"{split}_has_building.npy")
        # Load positive and negative indices separately
        positive_indices = np.load(self.metadata_dir / f"{split}_positive_indices.npy")
        negative_indices = np.load(self.metadata_dir / f"{split}_negative_indices.npy")

        # Load summary for stats
        with open(self.metadata_dir / f"{split}_summary.json", "r") as f:
            self.summary = json.load(f)

        # Set random seed
        np.random.seed(seed)

        # Compute the actual dataset size based on subset_fraction
        total_size = int(len(self.image_ids) * subset_fraction)

        if target_neg_ratio is not None:
            # Compute target negative and positive counts
            target_neg_count = int(total_size * target_neg_ratio)
            target_pos_count = total_size - target_neg_count

            # Adjust counts if we don't have enough samples
            if target_neg_count > len(negative_indices):
                target_neg_count = len(negative_indices)
                target_pos_count = total_size - target_neg_count

            if target_pos_count > len(positive_indices):
                target_pos_count = len(positive_indices)
                target_neg_count = total_size - target_pos_count
        else:
            target_neg_count = int(total_size * (1 - self.summary["positive_ratio"]))
            target_pos_count = total_size - target_neg_count

        # Select samples
        self.selected_pos_indices = np.random.choice(
            positive_indices, size=target_pos_count, replace=False
        )
        self.selected_neg_indices = np.random.choice(
            negative_indices, size=target_neg_count, replace=False
        )

        # Combine and shuffle
        self.indices = np.concatenate(
            [self.selected_pos_indices, self.selected_neg_indices]
        )
        np.random.shuffle(self.indices)

        # Image cache
        self.image_cache = {}

        # Log dataset info
        logger.info(
            f"Loaded {split} dataset with {len(self.indices)} patches "
            f"({target_pos_count} positive, {target_neg_count} negative)"
        )
        logger.info(f"Actual negative ratio: {target_neg_count/len(self.indices):.4f}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.indices)

    def _load_image(self, image_id):
        """Load an image into cache if not already there."""
        if image_id in self.image_cache:
            return self.image_cache[image_id]

        # Load pre-event and post-event images (no label needed for training)
        pre_path = (
            self.root_dir
            / SPLIT_DATA[self.split]
            / "pre-event"
            / f"{image_id}_pre_disaster.tif"
        )
        post_path = (
            self.root_dir
            / SPLIT_DATA[self.split]
            / "post-event"
            / f"{image_id}_post_disaster.tif"
        )

        # Load with rasterio
        with rasterio.open(str(pre_path)) as src:
            pre_img = np.stack(
                [src.read(i + 1) for i in range(min(3, src.count))], axis=2
            )

        with rasterio.open(str(post_path)) as src:
            post_img = np.stack(
                [src.read(i + 1) for i in range(min(3, src.count))], axis=2
            )

        # Store in cache
        self.image_cache[image_id] = (pre_img, post_img)

        # Limit cache size
        if len(self.image_cache) > self.cache_size:
            # Remove least recently used item
            lru_key = next(iter(self.image_cache))
            del self.image_cache[lru_key]

        return pre_img, post_img

    def __getitem__(self, idx):
        """Get a patch by extracting it from the original image."""
        metadata_idx = idx  # no need for the translation like ```self.indices[idx]````
        # - because the sampler samples directly from the metadata indices

        # Get metadata for this patch
        image_id = self.image_ids[metadata_idx]
        roi_y, roi_x = self.roi_positions[metadata_idx]
        ctx_y_start, ctx_x_start, ctx_y_end, ctx_x_end = self.context_positions[
            metadata_idx
        ]
        is_positive = self.is_positive[metadata_idx]
        damage_ratio = self.damage_ratios[metadata_idx]

        # Load images (from cache if available)
        pre_img, post_img = self._load_image(image_id)

        # Extract patches from context coordinates
        pre_patch = pre_img[ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end].copy()
        post_patch = post_img[ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end].copy()

        # Apply transforms if any
        if self.transform:
            transformed = self.transform(pre_patch, post_patch)
            pre_patch = transformed["pre_image"]
            post_patch = transformed["post_image"]

        # Convert to tensor
        pre_patch = self._to_tensor_optical(pre_patch)
        post_patch = self._to_tensor_sar(post_patch)

        has_building = self.has_building[metadata_idx]

        return {
            "pre_patch": pre_patch,
            "post_patch": post_patch,
            "is_positive": torch.tensor([float(is_positive)], dtype=torch.float32),
            "idx": idx,
            "image_id": image_id,
            "position": (roi_y, roi_x),
            "damage_ratio": damage_ratio,
            "has_building": has_building,
        }

    def _to_tensor_optical(self, img):
        """Convert optical image to tensor with normalization."""
        # Convert to float32
        img = img.astype(np.float32)

        # Per-channel normalization
        means = img.mean(axis=(0, 1), keepdims=True)
        stds = img.std(axis=(0, 1), keepdims=True) + 1e-8
        img = (img - means) / stds

        # Convert to tensor with channel-first format
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img

    def _to_tensor_sar(self, img):
        """Convert SAR image to tensor with appropriate preprocessing."""
        # Convert to float32
        img = img.astype(np.float32)

        # Check if it's three identical channels (likely repeated grayscale)
        if (
            img.shape[2] == 3
            and np.allclose(img[:, :, 0], img[:, :, 1])
            and np.allclose(img[:, :, 0], img[:, :, 2])
        ):
            # Extract just one channel
            img = img[:, :, 0:1]

        # Apply log transformation
        img = np.log1p(img)  # natural log of (1 + x)

        # Normalize each channel
        means = img.mean(axis=(0, 1), keepdims=True)
        stds = img.std(axis=(0, 1), keepdims=True) + 1e-8
        img = (img - means) / stds

        # Convert to tensor
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img

    def get_pos_neg_ratio(self):
        """Get ratio of positive to negative samples in the dataset."""
        # Count actual positives in the sampled dataset
        pos_count = np.sum(self.is_positive[self.indices])
        total = len(self.indices)
        return float(pos_count) / total if total > 0 else 0
