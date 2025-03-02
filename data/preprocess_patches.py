"""
Pre-process satellite image dataset by extracting patches for contrastive learning.
This script extracts patch pairs from the training dataset, classifies them as positive
(no change) or negative (with change), and splits them into train and validation sets.

Stored in two separate h5 files (train, val). Very temporary until we agree on a
definitive data management solution.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SimpleDatasetLoader:
    """Simple loader for datasets when the main dataset class isn't available."""

    def __init__(self, root_dir, split="train", limit=None):
        """
        Args:
            root_dir: Root directory containing the dataset
            split: 'train' or 'val'
            limit: limit number of images used - default to None
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.limit = limit
        # Get list of image names
        post_event_dir = self.root_dir / split / "post-event"
        self.image_ids = [
            f.name.replace("_post_disaster.tif", "")
            for f in post_event_dir.glob("*_post_disaster.tif")
        ][:limit]

        logger.info(f"Found {len(self.image_ids)} images for {split} split")

    def __len__(self):
        if self.limit is None:
            return len(self.image_ids)
        elif self.limit is not None and self.limit <= len(self.image_ids):
            return self.limit

    def load_tiff_data(self, path):
        """Load and preprocess TIFF images."""
        with rasterio.open(path) as data_reader:
            if data_reader.count > 1:
                # Multi-band image
                return np.stack(
                    [data_reader.read(i + 1) for i in range(min(3, data_reader.count))],
                    axis=2,
                )
            else:
                # Single-band image
                return data_reader.read(1)[..., np.newaxis]

    def __getitem__(self, idx):
        """Get a pre-event optical, post-event SAR, and damage label triplet."""
        image_id = self.image_ids[idx]

        # Load pre-event optical image (RGB)
        pre_path = (
            self.root_dir / self.split / "pre-event" / f"{image_id}_pre_disaster.tif"
        )
        pre_img = self.load_tiff_data(str(pre_path))
        if pre_img.shape[-1] > 3:  # Ensure 3 channels
            pre_img = pre_img[:, :, :3]

        # Load post-event SAR image
        post_path = (
            self.root_dir / self.split / "post-event" / f"{image_id}_post_disaster.tif"
        )
        post_img = self.load_tiff_data(str(post_path))

        # Convert single-channel to 3-channel for consistent processing
        if post_img.shape[-1] == 1:
            post_img = np.repeat(post_img, 3, axis=2)

        # Load damage label
        label_path = (
            self.root_dir / self.split / "target" / f"{image_id}_building_damage.tif"
        )
        label = self.load_tiff_data(str(label_path))

        return {
            "pre_image": pre_img,
            "post_image": post_img,
            "label": label,
            "image_id": image_id,
        }


class PatchExtractor:
    """Extract positive and negative patch pairs from satellite images."""

    def __init__(
        self,
        patch_size=64,
        stride=32,
        pos_threshold=0.05,
        min_valid_pixels=0.8,
        balance_ratio=0.5,
        max_pairs_per_image=500,
        random_seed=42,
    ):
        """
        Args:
            patch_size: Size of patches to extract (square)
            stride: Stride for patch extraction
            pos_threshold: Maximum damage ratio for positive pairs
            min_valid_pixels: Minimum ratio of valid pixels needed
            balance_ratio: Target ratio of positive samples
            max_pairs_per_image: Maximum number of pairs to extract per image
            random_seed: Random seed for reproducibility
        """
        self.patch_size = patch_size
        self.stride = stride
        self.pos_threshold = pos_threshold
        self.min_valid_pixels = min_valid_pixels
        self.balance_ratio = balance_ratio
        self.max_pairs_per_image = max_pairs_per_image
        self.random_seed = random_seed

        random.seed(random_seed)
        np.random.seed(random_seed)

    def extract_patches(self, pre_img, post_img, label):
        """Extract patches from image triplet."""
        h, w = pre_img.shape[:2]

        patches = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                # Extract patches
                pre_patch = pre_img[
                    y : y + self.patch_size, x : x + self.patch_size
                ].copy()
                post_patch = post_img[
                    y : y + self.patch_size, x : x + self.patch_size
                ].copy()
                label_patch = label[
                    y : y + self.patch_size, x : x + self.patch_size
                ].copy()

                # Check if patch is valid (contains enough valid pixels)
                valid_mask = ~np.isnan(pre_patch).any(axis=2) & ~np.isnan(
                    post_patch
                ).any(axis=2)
                valid_ratio = np.sum(valid_mask) / valid_mask.size

                if valid_ratio < self.min_valid_pixels:
                    continue

                # Calculate damage ratio
                if label_patch.ndim > 2:
                    label_patch_flat = label_patch.squeeze()
                else:
                    label_patch_flat = label_patch

                # Skip patches with only NaN or invalid values
                if np.isnan(label_patch_flat).all() or (label_patch_flat == 0).all():
                    continue

                # Calculate damage ratio
                damage_pixels = np.sum(label_patch_flat > 0)
                total_valid_pixels = np.sum(~np.isnan(label_patch_flat))

                if total_valid_pixels == 0:
                    continue

                damage_ratio = damage_pixels / total_valid_pixels

                # Determine if positive or negative sample
                is_positive = damage_ratio <= self.pos_threshold

                patches.append(
                    {
                        "pre_patch": pre_patch,
                        "post_patch": post_patch,
                        "label": label_patch,
                        "is_positive": is_positive,
                        "position": (y, x),
                        "damage_ratio": damage_ratio,
                    }
                )

        return patches

    def create_balanced_dataset(self, patches):
        """Create a balanced dataset with desired ratio of positive samples."""
        # Separate positive and negative samples
        pos_patches = [p for p in patches if p["is_positive"]]
        neg_patches = [p for p in patches if not p["is_positive"]]

        logger.info(
            f"  - Found {len(pos_patches)} positive and {len(neg_patches)} negative patches"
        )

        # Determine samples to keep for balance
        total_samples = min(
            len(pos_patches) + len(neg_patches), self.max_pairs_per_image
        )

        # Calculate target counts
        target_pos = int(total_samples * self.balance_ratio)
        target_neg = total_samples - target_pos

        # Adjust if needed
        if target_pos > len(pos_patches):
            target_pos = len(pos_patches)
            target_neg = min(total_samples - target_pos, len(neg_patches))

        if target_neg > len(neg_patches):
            target_neg = len(neg_patches)
            target_pos = min(total_samples - target_neg, len(pos_patches))

        # Randomly sample
        random.shuffle(pos_patches)
        random.shuffle(neg_patches)

        selected_patches = pos_patches[:target_pos] + neg_patches[:target_neg]
        random.shuffle(selected_patches)

        return selected_patches


def split_train_val(all_patches, train_ratio=0.8, balance_ratio=0.5, random_seed=42):
    """
    Split patches into training and validation sets while maintaining class balance.

    Args:
        all_patches: List of all patch dictionaries
        train_ratio: Ratio of training to total data
        balance_ratio: Desired ratio of positive samples in each split
        random_seed: Random seed for reproducibility

    Returns:
        train_patches, val_patches: Split patch lists
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Separate positive and negative samples
    pos_patches = [p for p in all_patches if p["is_positive"]]
    neg_patches = [p for p in all_patches if not p["is_positive"]]

    # Shuffle both lists
    random.shuffle(pos_patches)
    random.shuffle(neg_patches)

    # Calculate split indices
    pos_train_count = int(len(pos_patches) * train_ratio)
    neg_train_count = int(len(neg_patches) * train_ratio)

    # Split positive and negative patches
    pos_train = pos_patches[:pos_train_count]
    pos_val = pos_patches[pos_train_count:]
    neg_train = neg_patches[:neg_train_count]
    neg_val = neg_patches[neg_train_count:]

    # Combine and shuffle
    train_patches = pos_train + neg_train
    val_patches = pos_val + neg_val
    random.shuffle(train_patches)
    random.shuffle(val_patches)

    # Log statistics
    logger.info(
        f"Split dataset into {len(train_patches)} training and {len(val_patches)} validation patches"
    )
    logger.info(f"  - Training: {len(pos_train)} positive, {len(neg_train)} negative")
    logger.info(f"  - Validation: {len(pos_val)} positive, {len(neg_val)} negative")

    return train_patches, val_patches


def save_patches_hdf5(output_dir, split, patches, config):
    """Save patches to HDF5 files with metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create HDF5 file for this split
    h5_path = output_dir / f"{split}_patches.h5"

    patch_size = config.get("patch_size", 64)
    total_patches = len(patches)

    logger.info(f"Saving {total_patches} patches to {h5_path}")

    # Create a list to store metadata
    metadata = []

    with h5py.File(h5_path, "w") as hf:
        # Pre-create datasets
        hf.create_dataset(
            "pre_patches",
            shape=(total_patches, patch_size, patch_size, 3),
            dtype="float32",
            chunks=(1, patch_size, patch_size, 3),
            compression="gzip",
        )

        hf.create_dataset(
            "post_patches",
            shape=(total_patches, patch_size, patch_size, 3),
            dtype="float32",
            chunks=(1, patch_size, patch_size, 3),
            compression="gzip",
        )

        hf.create_dataset(
            "labels",
            shape=(total_patches, patch_size, patch_size, 1),
            dtype="int8",
            chunks=(1, patch_size, patch_size, 1),
            compression="gzip",
        )

        # Create datasets for metadata
        hf.create_dataset("is_positive", shape=(total_patches,), dtype="bool")

        # Fill datasets
        for idx, patch in enumerate(tqdm(patches, desc=f"Saving {split} patches")):
            # Store main data
            hf["pre_patches"][idx] = patch["pre_patch"]
            hf["post_patches"][idx] = patch["post_patch"]

            # Ensure label has right shape
            label = patch["label"]
            if label.ndim == 2:
                label = label[..., np.newaxis]
            hf["labels"][idx] = label

            # Store metadata
            hf["is_positive"][idx] = patch["is_positive"]

            # Add to metadata list
            metadata.append(
                {
                    "index": idx,
                    "image_id": patch.get("image_id", "unknown"),
                    "position": patch["position"],
                    "is_positive": bool(patch["is_positive"]),
                    "damage_ratio": float(patch["damage_ratio"]),
                }
            )

    # Save metadata JSON
    with open(output_dir / f"{split}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save summary statistics
    pos_count = sum(1 for p in metadata if p["is_positive"])
    neg_count = sum(1 for p in metadata if not p["is_positive"])

    summary = {
        "total_patches": total_patches,
        "positive_patches": pos_count,
        "negative_patches": neg_count,
        "positive_ratio": pos_count / total_patches if total_patches > 0 else 0,
        "patch_size": patch_size,
        "config": config,
    }

    with open(output_dir / f"{split}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Saved {total_patches} patches ({pos_count} positive, {neg_count} negative)"
    )
    return h5_path


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save patches for contrastive learning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed_patches",
        help="Directory to save processed patches",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images used to extract patches",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of patches to use for training vs validation",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract patch config parameters
    patch_config = {
        "patch_size": config.get("patch_size", 64),
        "stride": config.get("patch_stride", 32),
        "pos_threshold": config.get("pos_threshold", 0.05),
        "min_valid_pixels": config.get("min_valid_pixels", 0.8),
        "balance_ratio": config.get("positive_ratio", 0.5),
        "max_pairs_per_image": config.get("max_pairs_per_image", 500),
    }

    # Create patch extractor
    extractor = PatchExtractor(**patch_config, random_seed=args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "extraction_config.yaml", "w") as f:
        yaml.dump(patch_config, f)

    # Load training data only
    split = "train"  # We always use the training data and split it ourselves
    logger.info(f"Processing {split} split...")

    dataset = SimpleDatasetLoader(
        root_dir=config["data"]["root_dir"], split=split, limit=args.limit
    )

    # Extract patches from each image
    all_patches = []

    for i in tqdm(range(len(dataset)), desc=f"Extracting patches from {split}"):
        sample = dataset[i]
        image_id = sample["image_id"]

        # Convert tensors to numpy if needed
        pre_img = sample["pre_image"]
        post_img = sample["post_image"]
        label = sample["label"]

        if isinstance(pre_img, torch.Tensor):
            pre_img = pre_img.numpy().transpose(1, 2, 0)
            post_img = post_img.numpy().transpose(1, 2, 0)
            label = label.numpy()

        # Extract patches
        image_patches = extractor.extract_patches(pre_img, post_img, label)

        # Balance dataset within each image
        balanced_patches = extractor.create_balanced_dataset(image_patches)

        # Add image ID to each patch
        for patch in balanced_patches:
            patch["image_id"] = image_id

        # Collect all patches
        all_patches.extend(balanced_patches)

    # Split patches into train and validation sets
    train_patches, val_patches = split_train_val(
        all_patches,
        train_ratio=args.train_ratio,
        balance_ratio=patch_config["balance_ratio"],
        random_seed=args.seed,
    )

    # Save train patches to disk
    save_patches_hdf5(output_dir, "train", train_patches, patch_config)

    # Save validation patches to disk
    save_patches_hdf5(output_dir, "val", val_patches, patch_config)

    logger.info(f"Processing complete. Patches saved to {output_dir}")


if __name__ == "__main__":
    main()
