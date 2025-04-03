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
import random
from pathlib import Path

import h5py
import numpy as np
import rasterio
import torch
import yaml
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
        roi_patch_size=64,  # Size of the actual region of interest
        context_patch_size=256,  # Size of the context window
        stride=32,  # Stride for extraction
        pos_threshold=0.05,  # Maximum damage ratio for positive pairs
        min_valid_pixels=0.8,  # Minimum ratio of valid pixels needed
        balance_ratio=0.5,  # Target ratio of positive samples
        max_pairs_per_image=500,  # Maximum number of pairs to extract per image
        random_seed=42,  # Random seed for reproducibility
    ):
        self.roi_patch_size = roi_patch_size
        self.context_patch_size = context_patch_size
        self.stride = stride
        self.pos_threshold = pos_threshold
        self.min_valid_pixels = min_valid_pixels
        self.balance_ratio = balance_ratio
        self.max_pairs_per_image = max_pairs_per_image
        self.random_seed = random_seed

        # Calculate the padding needed to center the ROI in the context window
        self.pad_size = (context_patch_size - roi_patch_size) // 2

        random.seed(random_seed)
        np.random.seed(random_seed)

    def extract_patches(self, pre_img, post_img, label):
        """Extract patches from image triplet with context, skipping boundaries."""
        h, w = pre_img.shape[:2]

        # Calculate valid extraction region considering context padding
        start_y = self.pad_size
        start_x = self.pad_size
        end_y = h - self.pad_size - self.roi_patch_size + 1
        end_x = w - self.pad_size - self.roi_patch_size + 1

        # Check if we have valid region to extract from
        if start_y >= end_y or start_x >= end_x:
            logger.warning(
                f"Image too small for extraction with current parameters: {h}x{w}"
            )
            return []

        patches = []
        for y in range(start_y, end_y, self.stride):
            for x in range(start_x, end_x, self.stride):
                # Extract full-sized context patch
                ctx_y_start = y - self.pad_size
                ctx_x_start = x - self.pad_size
                ctx_y_end = y + self.roi_patch_size + self.pad_size
                ctx_x_end = x + self.roi_patch_size + self.pad_size

                # Extract patches
                pre_context = pre_img[
                    ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end
                ].copy()
                post_context = post_img[
                    ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end
                ].copy()

                # Extract ROI for labeling
                label_roi = np.zeros(
                    (self.context_patch_size, self.context_patch_size, 1),
                    dtype=label.dtype,
                )
                roi_label_data = label[
                    y : y + self.roi_patch_size, x : x + self.roi_patch_size
                ].copy()

                # Place the ROI label in the center of the zero-padded context-sized label
                label_roi[
                    self.pad_size : self.pad_size + self.roi_patch_size,
                    self.pad_size : self.pad_size + self.roi_patch_size,
                ] = roi_label_data

                # Verify the dimensions are as expected
                assert (
                    pre_context.shape[0] == self.context_patch_size
                ), f"Context height mismatch: {pre_context.shape[0]} != {self.context_patch_size}"
                assert (
                    pre_context.shape[1] == self.context_patch_size
                ), f"Context width mismatch: {pre_context.shape[1]} != {self.context_patch_size}"

                # Check if patch is valid (contains enough valid pixels in ROI)
                valid_mask = ~np.isnan(
                    pre_img[y : y + self.roi_patch_size, x : x + self.roi_patch_size]
                ).any(axis=2) & ~np.isnan(
                    post_img[y : y + self.roi_patch_size, x : x + self.roi_patch_size]
                ).any(axis=2)
                valid_ratio = np.sum(valid_mask) / valid_mask.size

                if valid_ratio < self.min_valid_pixels:
                    continue

                # Calculate damage ratio based only on the ROI
                if label_roi.ndim > 2:
                    label_roi_flat = label_roi.squeeze()
                else:
                    label_roi_flat = label_roi

                # Skip patches with only NaN or invalid values
                if np.isnan(label_roi_flat).all() or (label_roi_flat == 0).all():
                    continue

                # Calculate damage ratio
                label_patch_flat_binary = np.where(
                    label_roi_flat > 1, 1, 0
                )  # Binary damage: >1 is damaged
                damage_pixels = np.sum(label_patch_flat_binary > 0)
                total_valid_pixels = np.sum(~np.isnan(label_roi_flat))

                if total_valid_pixels == 0:
                    continue

                damage_ratio = damage_pixels / total_valid_pixels

                # Determine if positive or negative sample
                is_positive = damage_ratio <= self.pos_threshold

                patches.append(
                    {
                        "pre_patch": pre_context,
                        "post_patch": post_context,
                        "label": label_roi,
                        "is_positive": is_positive,
                        "position": (y, x),  # Original position in the full image
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


def save_batch_to_h5(batch_file, patches, batch_metadata):
    """
    Save a batch of patches to a temporary HDF5 file.

    Args:
        batch_file: Path to save the HDF5 file
        patches: List of patch dictionaries containing image data
        batch_metadata: List of metadata dictionaries for this batch
    """
    if not patches:
        return

    patch_size = patches[0]["pre_patch"].shape[0]
    num_patches = len(patches)

    with h5py.File(batch_file, "w") as hf:
        # Create datasets for image data
        hf.create_dataset(
            "pre_patches",
            shape=(num_patches, patch_size, patch_size, 3),
            dtype="float32",
            chunks=(1, patch_size, patch_size, 3),
            compression="gzip",
        )

        hf.create_dataset(
            "post_patches",
            shape=(num_patches, patch_size, patch_size, 3),
            dtype="float32",
            chunks=(1, patch_size, patch_size, 3),
            compression="gzip",
        )

        hf.create_dataset(
            "labels",
            shape=(num_patches, patch_size, patch_size, 1),
            dtype="int8",
            chunks=(1, patch_size, patch_size, 1),
            compression="gzip",
        )

        # Fill the datasets
        for i, patch in enumerate(patches):
            hf["pre_patches"][i] = patch["pre_patch"]
            hf["post_patches"][i] = patch["post_patch"]

            # Ensure label has right shape
            label = patch["label"]
            if label.ndim == 2:
                label = label[..., np.newaxis]
            hf["labels"][i] = label

        # Store metadata as attribute
        hf.attrs["metadata"] = json.dumps(batch_metadata)

        logger.info(
            f"Saved batch with {num_patches} patches ({sum(1 for m in batch_metadata if m['is_positive'])} positive)"
        )


def split_metadata(all_metadata, train_ratio=0.8, balance_ratio=0.5, random_seed=42):
    """
    Split metadata into training and validation sets while maintaining class balance.

    Args:
        all_metadata: List of all metadata dictionaries
        train_ratio: Ratio of training to total data
        balance_ratio: Desired ratio of positive samples in each split
        random_seed: Random seed for reproducibility

    Returns:
        train_metadata, val_metadata: Split metadata lists
    """
    # Set random seed for reproducibility
    random.seed(random_seed)

    # Separate positive and negative samples
    pos_metadata = [m for m in all_metadata if m["is_positive"]]
    neg_metadata = [m for m in all_metadata if not m["is_positive"]]

    # Shuffle both lists
    random.shuffle(pos_metadata)
    random.shuffle(neg_metadata)

    # Calculate split indices
    pos_train_count = int(len(pos_metadata) * train_ratio)
    neg_train_count = int(len(neg_metadata) * train_ratio)

    # Split positive and negative metadata
    pos_train = pos_metadata[:pos_train_count]
    pos_val = pos_metadata[pos_train_count:]
    neg_train = neg_metadata[:neg_train_count]
    neg_val = neg_metadata[neg_train_count:]

    # Combine and shuffle
    train_metadata = pos_train + neg_train
    val_metadata = pos_val + neg_val
    random.shuffle(train_metadata)
    random.shuffle(val_metadata)

    logger.info(
        f"Split metadata into {len(train_metadata)} training and {len(val_metadata)} validation entries"
    )
    logger.info(f"  - Training: {len(pos_train)} positive, {len(neg_train)} negative")
    logger.info(f"  - Validation: {len(pos_val)} positive, {len(neg_val)} negative")

    return train_metadata, val_metadata


def create_final_datasets(
    temp_dir, output_dir, train_metadata, val_metadata, patch_config=None
):
    """
    Create final train/val HDF5 files using the split metadata.

    Args:
        temp_dir: Directory containing temporary batch HDF5 files
        output_dir: Directory to save final datasets
        train_metadata: List of metadata for training set
        val_metadata: List of metadata for validation set
        patch_config: Patch extraction configuration (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process training set
    _create_final_dataset(temp_dir, output_dir, train_metadata, "train", patch_config)

    # Process validation set
    _create_final_dataset(temp_dir, output_dir, val_metadata, "val", patch_config)


def _create_final_dataset(temp_dir, output_dir, metadata, split, patch_config=None):
    """
    Helper function to create a final dataset (train or val).

    Args:
        temp_dir: Directory containing temporary batch HDF5 files
        output_dir: Directory to save final datasets
        metadata: List of metadata for this split
        split: 'train' or 'val'
        patch_config: Patch extraction configuration (optional)
    """
    if not metadata:
        logger.warning(f"No metadata for {split} split!")
        return

    # Get sample patch to determine dimensions
    sample_batch_file = next(Path(temp_dir).glob("batch_*.h5"))
    with h5py.File(sample_batch_file, "r") as sample_f:
        patch_shape = sample_f["pre_patches"].shape
        patch_size = patch_shape[1]

    # Create final HDF5 file
    h5_path = output_dir / f"{split}_patches.h5"
    logger.info(f"Creating final {split} dataset with {len(metadata)} patches")

    with h5py.File(h5_path, "w") as hf:
        # Create datasets
        hf.create_dataset(
            "pre_patches",
            shape=(len(metadata), patch_size, patch_size, 3),
            dtype="float32",
            chunks=(1, patch_size, patch_size, 3),
            compression="gzip",
        )

        hf.create_dataset(
            "post_patches",
            shape=(len(metadata), patch_size, patch_size, 3),
            dtype="float32",
            chunks=(1, patch_size, patch_size, 3),
            compression="gzip",
        )

        hf.create_dataset(
            "labels",
            shape=(len(metadata), patch_size, patch_size, 1),
            dtype="int8",
            chunks=(1, patch_size, patch_size, 1),
            compression="gzip",
        )

        hf.create_dataset("is_positive", shape=(len(metadata),), dtype="bool")

        # Fill datasets
        batch_files = {}  # Cache for opened batch files

        for idx, meta in enumerate(tqdm(metadata, desc=f"Creating {split} dataset")):
            batch_num = meta["batch"]
            idx_in_batch = meta["index_in_batch"]

            # Open batch file if not already open
            if batch_num not in batch_files:
                batch_file = temp_dir / f"batch_{batch_num}.h5"
                batch_files[batch_num] = h5py.File(batch_file, "r")

            # Copy data from batch file to final file
            hf["pre_patches"][idx] = batch_files[batch_num]["pre_patches"][idx_in_batch]
            hf["post_patches"][idx] = batch_files[batch_num]["post_patches"][
                idx_in_batch
            ]
            hf["labels"][idx] = batch_files[batch_num]["labels"][idx_in_batch]
            hf["is_positive"][idx] = meta["is_positive"]

        # Close batch files
        for f in batch_files.values():
            f.close()

    # Save metadata JSON
    cleaned_metadata = []
    for idx, meta in enumerate(metadata):
        cleaned_metadata.append(
            {
                "index": idx,
                "image_id": meta["image_id"],
                "position": meta["position"],
                "is_positive": meta["is_positive"],
                "damage_ratio": meta["damage_ratio"],
            }
        )

    with open(output_dir / f"{split}_metadata.json", "w") as f:
        json.dump(cleaned_metadata, f, indent=2)

    # Save summary statistics
    pos_count = sum(1 for m in metadata if m["is_positive"])
    neg_count = len(metadata) - pos_count

    summary = {
        "total_patches": len(metadata),
        "positive_patches": pos_count,
        "negative_patches": neg_count,
        "positive_ratio": pos_count / len(metadata) if len(metadata) > 0 else 0,
        "patch_size": patch_size,
        "config": patch_config,
    }

    with open(output_dir / f"{split}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Created {split} dataset with {len(metadata)} patches ({pos_count} positive, {neg_count} negative)"
    )


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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of images to process before saving a batch",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract patch config parameters
    patch_config = {
        "roi_patch_size": config.get("patch_size", 64),
        "context_patch_size": config.get("context_patch_size", 256),
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

    # Create temporary directory for batch processing
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "extraction_config.yaml", "w") as f:
        yaml.dump(patch_config, f)

    # Load training data only
    split = "train"  # We always use the training data and split it ourselves
    logger.info(f"Processing {split} split...")

    dataset = SimpleDatasetLoader(
        root_dir=config["data"]["root_dir"], split=split, limit=args.limit
    )

    # Initialize batch tracking variables
    batch_patches = []
    batch_metadata = []
    batch_counter = 0
    patch_counter = 0
    total_patches_count = 0

    # Process each image
    for i in tqdm(range(len(dataset)), desc="Extracting patches"):
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

        # Extract patches from this image
        image_patches = extractor.extract_patches(pre_img, post_img, label)

        # Balance dataset within each image
        balanced_patches = extractor.create_balanced_dataset(image_patches)

        # Add image ID to each patch
        for patch in balanced_patches:
            patch["image_id"] = image_id

        # Add patches to current batch
        batch_patches.extend(balanced_patches)

        # Add metadata
        for p in balanced_patches:
            batch_metadata.append(
                {
                    "image_id": p["image_id"],
                    "position": p["position"],
                    "is_positive": bool(p["is_positive"]),
                    "damage_ratio": float(p["damage_ratio"]),
                    "batch": batch_counter,
                    "index_in_batch": patch_counter,
                }
            )
            patch_counter += 1

        total_patches_count += len(balanced_patches)

        # If we've accumulated enough patches or this is the last image, save the batch
        if len(batch_patches) >= args.batch_size * 100 or i == len(dataset) - 1:
            if batch_patches:  # Check if we have any patches
                batch_file = temp_dir / f"batch_{batch_counter}.h5"
                logger.info(
                    f"Saving batch {batch_counter} with {len(batch_patches)} patches"
                )
                save_batch_to_h5(batch_file, batch_patches, batch_metadata)

                # Reset for next batch
                batch_counter += 1
                batch_patches = []
                batch_metadata = []
                patch_counter = 0

    logger.info(
        f"Extracted {total_patches_count} patches across {batch_counter} batches"
    )

    # Read and combine all metadata from batch files
    logger.info("Combining metadata from all batches...")
    all_metadata = []

    for batch_idx in range(batch_counter):
        batch_file = temp_dir / f"batch_{batch_idx}.h5"
        with h5py.File(batch_file, "r") as f:
            batch_meta = json.loads(f.attrs["metadata"])
            all_metadata.extend(batch_meta)

    # Split metadata into train/val
    logger.info("Splitting metadata into train and validation sets...")
    train_metadata, val_metadata = split_metadata(
        all_metadata,
        train_ratio=args.train_ratio,
        balance_ratio=patch_config["balance_ratio"],
        random_seed=args.seed,
    )

    # Create final datasets
    logger.info("Creating final train and validation datasets...")
    create_final_datasets(
        temp_dir, output_dir, train_metadata, val_metadata, patch_config
    )

    # Clean up temporary files
    logger.info("Cleaning up temporary files...")
    import shutil

    shutil.rmtree(temp_dir)

    logger.info(f"Processing complete. Datasets saved to {output_dir}")


if __name__ == "__main__":
    main()
