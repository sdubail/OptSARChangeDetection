import argparse
import json
import logging
import random
from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
import yaml
from tqdm import tqdm

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


def extract_and_save_patches(
    config,
    image_id,
    pre_img,
    post_img,
    label,
    train_h5,
    val_h5,
    train_meta,
    val_meta,
    train_ratio=0.8,
):
    """
    Extract patches from an image and save them directly to HDF5 files.

    Args:
        config: Configuration dictionary
        image_id: ID of the current image
        pre_img: Pre-event image
        post_img: Post-event image
        label: Label image
        train_h5: HDF5 file handle for training data
        val_h5: HDF5 file handle for validation data
        train_meta: List to store training metadata
        val_meta: List to store validation metadata
        train_ratio: Ratio of patches to use for training

    Returns:
        tuple: (num_train_pos, num_train_neg, num_val_pos, num_val_neg)
    """
    # Extract patch parameters from config
    roi_patch_size = config.get("roi_patch_size", 64)
    context_patch_size = config.get("context_patch_size", 256)
    stride = config.get("patch_stride", 32)
    pos_threshold = config.get("pos_threshold", 0.05)
    min_valid_pixels = config.get("min_valid_pixels", 0.8)
    balance_ratio = config.get("positive_ratio", 0.5)
    max_pairs_per_image = config.get("max_pairs_per_image", 500)

    # Calculate padding size
    pad_size = (context_patch_size - roi_patch_size) // 2
    h, w = pre_img.shape[:2]

    # Calculate valid extraction region
    start_y = pad_size
    start_x = pad_size
    end_y = h - pad_size - roi_patch_size + 1
    end_x = w - pad_size - roi_patch_size + 1

    # Check if we have valid region
    if start_y >= end_y or start_x >= end_x:
        logger.warning(f"Image {image_id} too small for extraction: {h}x{w}")
        return 0, 0, 0, 0

    # Collect positive and negative patches separately
    positive_patches = []
    negative_patches = []

    # Extract patches
    for y in range(start_y, end_y, stride):
        for x in range(start_x, end_x, stride):
            # Extract context patches
            ctx_y_start = y - pad_size
            ctx_x_start = x - pad_size
            ctx_y_end = y + roi_patch_size + pad_size
            ctx_x_end = x + roi_patch_size + pad_size

            pre_context = pre_img[ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end].copy()
            post_context = post_img[ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end].copy()

            # Extract ROI for labeling
            label_roi = label[y : y + roi_patch_size, x : x + roi_patch_size].copy()

            # Verify dimensions
            if (
                pre_context.shape[0] != context_patch_size
                or pre_context.shape[1] != context_patch_size
            ):
                continue

            # Check for valid pixels
            valid_mask = ~np.isnan(
                pre_img[y : y + roi_patch_size, x : x + roi_patch_size]
            ).any(axis=2) & ~np.isnan(
                post_img[y : y + roi_patch_size, x : x + roi_patch_size]
            ).any(axis=2)
            valid_ratio = np.sum(valid_mask) / valid_mask.size

            if valid_ratio < min_valid_pixels:
                continue

            # Process the label ROI
            if label_roi.ndim > 2:
                label_roi_flat = label_roi.squeeze()
            else:
                label_roi_flat = label_roi

            # Skip invalid labels
            if np.isnan(label_roi_flat).all() or (label_roi_flat == 0).all():
                continue

            # Calculate damage ratio
            label_patch_flat_binary = np.where(label_roi_flat > 1, 1, 0)
            damage_pixels = np.sum(label_patch_flat_binary > 0)
            total_valid_pixels = np.sum(~np.isnan(label_roi_flat))

            if total_valid_pixels == 0:
                continue

            damage_ratio = damage_pixels / total_valid_pixels
            is_positive = damage_ratio <= pos_threshold

            # Create patch dict
            patch = {
                "pre_patch": pre_context,
                "post_patch": post_context,
                "label": label_roi,
                "is_positive": is_positive,
                "position": (y, x),
                "damage_ratio": damage_ratio,
            }

            # Add to appropriate list
            if is_positive:
                positive_patches.append(patch)
            else:
                negative_patches.append(patch)

    # Apply balancing if requested
    num_pos = len(positive_patches)
    num_neg = len(negative_patches)

    logger.info(
        f"  Image {image_id}: Found {num_pos} positive and {num_neg} negative patches"
    )

    total_samples = min(num_pos + num_neg, max_pairs_per_image)
    target_pos = int(total_samples * balance_ratio)
    target_neg = total_samples - target_pos

    if target_pos > num_pos:
        target_pos = num_pos
        target_neg = min(total_samples - target_pos, num_neg)

    if target_neg > num_neg:
        target_neg = num_neg
        target_pos = min(total_samples - target_neg, num_pos)

    random.shuffle(positive_patches)
    random.shuffle(negative_patches)

    pos_selected = positive_patches[:target_pos]
    neg_selected = negative_patches[:target_neg]

    pos_train_count = int(len(pos_selected) * train_ratio)
    neg_train_count = int(len(neg_selected) * train_ratio)

    pos_train = pos_selected[:pos_train_count]
    pos_val = pos_selected[pos_train_count:]
    neg_train = neg_selected[:neg_train_count]
    neg_val = neg_selected[neg_train_count:]

    train_size = train_h5["pre_patches"].shape[0]
    val_size = val_h5["pre_patches"].shape[0]

    num_train_pos = len(pos_train)
    num_train_neg = len(neg_train)
    train_patches = pos_train + neg_train

    if train_patches:
        # Resize datasets to accommodate new patches
        new_train_size = train_size + len(train_patches)
        resize_h5_datasets(train_h5, new_train_size)

        # Add patches to HDF5
        for i, patch in enumerate(train_patches):
            idx = train_size + i
            train_h5["pre_patches"][idx] = patch["pre_patch"]
            train_h5["post_patches"][idx] = patch["post_patch"]

            # Ensure label has right shape
            label = patch["label"]
            if label.ndim == 2:
                label = label[..., np.newaxis]
            train_h5["labels"][idx] = label

            train_h5["is_positive"][idx] = patch["is_positive"]

            # Add to metadata
            train_meta.append(
                {
                    "index": idx,
                    "image_id": image_id,
                    "position": patch["position"],
                    "is_positive": bool(patch["is_positive"]),
                    "damage_ratio": float(patch["damage_ratio"]),
                }
            )

    # Add to validation set
    num_val_pos = len(pos_val)
    num_val_neg = len(neg_val)
    val_patches = pos_val + neg_val

    if val_patches:
        # Resize datasets to accommodate new patches
        new_val_size = val_size + len(val_patches)
        resize_h5_datasets(val_h5, new_val_size)

        # Add patches to HDF5
        for i, patch in enumerate(val_patches):
            idx = val_size + i
            val_h5["pre_patches"][idx] = patch["pre_patch"]
            val_h5["post_patches"][idx] = patch["post_patch"]

            # Ensure label has right shape
            label = patch["label"]
            if label.ndim == 2:
                label = label[..., np.newaxis]
            val_h5["labels"][idx] = label

            val_h5["is_positive"][idx] = patch["is_positive"]

            # Add to metadata
            val_meta.append(
                {
                    "index": idx,
                    "image_id": image_id,
                    "position": patch["position"],
                    "is_positive": bool(patch["is_positive"]),
                    "damage_ratio": float(patch["damage_ratio"]),
                }
            )

    # Flush to disk to save memory
    train_h5.flush()
    val_h5.flush()

    return num_train_pos, num_train_neg, num_val_pos, num_val_neg


def resize_h5_datasets(h5_file, new_size):
    """Resize HDF5 datasets to accommodate more data."""
    for key in h5_file.keys():
        if isinstance(h5_file[key], h5py.Dataset):
            current_shape = h5_file[key].shape
            new_shape = list(current_shape)
            new_shape[0] = new_size
            h5_file[key].resize(new_shape)


def preprocess_patches(config, output_dir, train_ratio=0.8, limit=None, seed=42):
    """
    Extract and save patches for contrastive learning with memory efficiency.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save processed patches
        train_ratio: Ratio of patches for training vs validation
        limit: Maximum number of images to process (None for all)
        seed: Random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    roi_patch_size = config.get("roi_patch_size", 64)
    context_patch_size = config.get("context_patch_size", 256)

    dataset = SimpleDatasetLoader(
        root_dir=config["data"]["root_dir"], split="train", limit=limit
    )

    train_h5_path = output_dir / "train_patches.h5"
    val_h5_path = output_dir / "val_patches.h5"

    with h5py.File(train_h5_path, "w") as train_h5, h5py.File(
        val_h5_path, "w"
    ) as val_h5:
        train_h5.create_dataset(
            "pre_patches",
            shape=(0, context_patch_size, context_patch_size, 3),
            maxshape=(None, context_patch_size, context_patch_size, 3),
            dtype="float32",
            chunks=(1, context_patch_size, context_patch_size, 3),
            compression="gzip",
        )

        train_h5.create_dataset(
            "post_patches",
            shape=(0, context_patch_size, context_patch_size, 3),
            maxshape=(None, context_patch_size, context_patch_size, 3),
            dtype="float32",
            chunks=(1, context_patch_size, context_patch_size, 3),
            compression="gzip",
        )

        train_h5.create_dataset(
            "labels",
            shape=(0, roi_patch_size, roi_patch_size, 1),
            maxshape=(None, roi_patch_size, roi_patch_size, 1),
            dtype="int8",
            chunks=(1, roi_patch_size, roi_patch_size, 1),
            compression="gzip",
        )

        train_h5.create_dataset(
            "is_positive",
            shape=(0,),
            maxshape=(None,),
            dtype="bool",
        )

        val_h5.create_dataset(
            "pre_patches",
            shape=(0, context_patch_size, context_patch_size, 3),
            maxshape=(None, context_patch_size, context_patch_size, 3),
            dtype="float32",
            chunks=(1, context_patch_size, context_patch_size, 3),
            compression="gzip",
        )

        val_h5.create_dataset(
            "post_patches",
            shape=(0, context_patch_size, context_patch_size, 3),
            maxshape=(None, context_patch_size, context_patch_size, 3),
            dtype="float32",
            chunks=(1, context_patch_size, context_patch_size, 3),
            compression="gzip",
        )

        val_h5.create_dataset(
            "labels",
            shape=(0, roi_patch_size, roi_patch_size, 1),
            maxshape=(None, roi_patch_size, roi_patch_size, 1),
            dtype="int8",
            chunks=(1, roi_patch_size, roi_patch_size, 1),
            compression="gzip",
        )

        val_h5.create_dataset(
            "is_positive",
            shape=(0,),
            maxshape=(None,),
            dtype="bool",
        )

        train_metadata = []
        val_metadata = []

        total_train_pos = 0
        total_train_neg = 0
        total_val_pos = 0
        total_val_neg = 0

        for i in tqdm(range(len(dataset)), desc="Processing images"):
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

            # Extract patches and save directly to HDF5
            train_pos, train_neg, val_pos, val_neg = extract_and_save_patches(
                config,
                image_id,
                pre_img,
                post_img,
                label,
                train_h5,
                val_h5,
                train_metadata,  # is modified in place by the function
                val_metadata,  # is modified in place by the function
                train_ratio,
            )

            # Update statistics
            total_train_pos += train_pos
            total_train_neg += train_neg
            total_val_pos += val_pos
            total_val_neg += val_neg

            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i+1}/{len(dataset)} images")
                logger.info(
                    f"  Training: {total_train_pos} positive, {total_train_neg} negative"
                )
                logger.info(
                    f"  Validation: {total_val_pos} positive, {total_val_neg} negative"
                )

        # Save metadata to files
        with open(output_dir / "train_metadata.json", "w") as f:
            json.dump(train_metadata, f)

        with open(output_dir / "val_metadata.json", "w") as f:
            json.dump(val_metadata, f)

        # Save summary statistics
        train_summary = {
            "total_patches": total_train_pos + total_train_neg,
            "positive_patches": total_train_pos,
            "negative_patches": total_train_neg,
            "positive_ratio": total_train_pos / (total_train_pos + total_train_neg)
            if (total_train_pos + total_train_neg) > 0
            else 0,
            "roi_patch_size": roi_patch_size,
            "context_patch_size": context_patch_size,
            "config": config,
        }

        val_summary = {
            "total_patches": total_val_pos + total_val_neg,
            "positive_patches": total_val_pos,
            "negative_patches": total_val_neg,
            "positive_ratio": total_val_pos / (total_val_pos + total_val_neg)
            if (total_val_pos + total_val_neg) > 0
            else 0,
            "roi_patch_size": roi_patch_size,
            "context_patch_size": context_patch_size,
            "config": config,
        }

        with open(output_dir / "train_summary.json", "w") as f:
            json.dump(train_summary, f)

        with open(output_dir / "val_summary.json", "w") as f:
            json.dump(val_summary, f)

    logger.info("Patch extraction complete!")
    logger.info(f"Training: {total_train_pos} positive, {total_train_neg} negative")
    logger.info(f"Validation: {total_val_pos} positive, {total_val_neg} negative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and save patches for contrastive learning with context"
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

    # Process patches
    preprocess_patches(
        config=config,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        limit=args.limit,
        seed=args.seed,
    )
