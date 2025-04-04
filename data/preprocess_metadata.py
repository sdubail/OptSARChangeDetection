"""
A code to process metadata for a dataset containing pre-event optical images, post-event SAR images, and damage labels.
The script extracts patches from the images, calculates damage ratios, and saves the metadata for training and validation splits.
It also includes options for limiting the number of images processed, selecting specific disasters, and excluding blacklisted images.

The metadata includes information about the image ID, ROI positions, context positions, positive/negative flags, damage ratios, and building presence.
Metadata acts as an index for the dataset, allowing for efficient loading of patches during training.
"""

# data/preprocess_metadata_numpy.py
import argparse
import json
import logging
from pathlib import Path

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

    def __init__(
        self,
        root_dir,
        blacklist_path,
        split="train",
        limit=None,
        exclude_blacklist=True,
        select_disaster="",
    ):
        """
        Args:
            root_dir: Root directory containing the dataset
            split: 'train' or 'val'
            limit: limit number of images used - default to None
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.limit = limit
        blacklist = np.loadtxt(blacklist_path, dtype=str, delimiter=",")[:, 0]
        # list of image names
        post_event_dir = self.root_dir / split / "post-event"

        self.image_ids = [
            f.name.replace("_post_disaster.tif", "")
            for f in post_event_dir.glob("*_post_disaster.tif")
            if (
                not exclude_blacklist
                or f.name.replace("_post_disaster.tif", "") not in blacklist
            )
            and select_disaster in f.name
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

    def get_by_image_id(self, image_id):
        """Get a sample by image ID."""
        try:
            # find the index of the image with the given ID
            idx = self.image_ids.index(image_id)
            return self[idx]
        except ValueError:
            raise ValueError(f"Image ID '{image_id}' not found in the dataset")

    def __getitem__(self, idx):
        """Get a pre-event optical, post-event SAR, and damage label triplet."""
        image_id = self.image_ids[idx]

        # load pre-event optical image (RGB)
        pre_path = (
            self.root_dir / self.split / "pre-event" / f"{image_id}_pre_disaster.tif"
        )
        pre_img = self.load_tiff_data(str(pre_path))
        if pre_img.shape[-1] > 3:  # Ensure 3 channels
            pre_img = pre_img[:, :, :3]

        # load post-event SAR image
        post_path = (
            self.root_dir / self.split / "post-event" / f"{image_id}_post_disaster.tif"
        )
        post_img = self.load_tiff_data(str(post_path))

        if post_img.shape[-1] == 1:
            post_img = np.repeat(post_img, 3, axis=2)

        # damage label
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


def extract_patch_metadata(
    pre_img,
    post_img,
    label,
    image_id,
    roi_patch_size=16,
    context_patch_size=256,
    stride=8,
    pos_threshold=0.05,
    min_valid_pixels=0.8,
    building_threshold=0.2,
):
    """
    Extract metadata for valid patches directly into arrays.

    Args:
        pre_img: Pre-event image
        post_img: Post-event image
        label: Label image
        image_id: Image identifier
        roi_patch_size: Size of the region of interest
        context_patch_size: Size of the context window
        stride: Stride for patch extraction
        pos_threshold: Maximum damage ratio for positive pairs
        min_valid_pixels: Minimum ratio of valid pixels needed
        building_threshold: Minimum ratio of building pixels needed for building label

    Returns:
        Arrays with extracted metadata
    """
    h, w = pre_img.shape[:2]
    pad_size = (context_patch_size - roi_patch_size) // 2

    # get the valid extraction region
    start_y = pad_size
    start_x = pad_size
    end_y = h - pad_size - roi_patch_size + 1
    end_x = w - pad_size - roi_patch_size + 1

    if start_y >= end_y or start_x >= end_x:
        logger.warning(f"Image {image_id} too small for extraction: {h}x{w}")
        return [], [], [], [], [], []

    image_ids = []
    roi_positions = []
    context_positions = []
    is_positive_flags = []
    damage_ratios = []
    has_building_flags = []

    for y in range(start_y, end_y, stride):
        for x in range(start_x, end_x, stride):
            # check for valid pixels in ROI
            roi = pre_img[y : y + roi_patch_size, x : x + roi_patch_size]

            # skip if ROI contains NaN values
            if hasattr(roi, "ndim") and roi.ndim > 2:
                valid_mask = ~np.isnan(roi).any(axis=2)
                valid_ratio = np.sum(valid_mask) / valid_mask.size

                if valid_ratio < min_valid_pixels:
                    continue

            # process the label ROI
            label_roi = label[y : y + roi_patch_size, x : x + roi_patch_size]

            if label_roi.ndim > 2:
                label_roi_flat = label_roi.squeeze()
            else:
                label_roi_flat = label_roi

            if np.isnan(label_roi_flat).all() or (label_roi_flat == 0).all():
                continue

            # damage ratio
            label_binary = np.where(label_roi_flat > 1, 1, 0)
            damage_pixels = np.sum(label_binary > 0)
            total_valid_pixels = np.sum(~np.isnan(label_roi_flat))

            if total_valid_pixels == 0:
                continue

            damage_ratio = damage_pixels / total_valid_pixels
            is_positive = damage_ratio <= pos_threshold

            building_pixels = np.sum(label_roi_flat > 0)
            building_ratio = building_pixels / total_valid_pixels
            has_building = building_ratio >= building_threshold

            # context coords
            ctx_y_start = y - pad_size
            ctx_x_start = x - pad_size
            ctx_y_end = y + roi_patch_size + pad_size
            ctx_x_end = x + roi_patch_size + pad_size

            image_ids.append(image_id)
            roi_positions.append((y, x))
            context_positions.append((ctx_y_start, ctx_x_start, ctx_y_end, ctx_x_end))
            is_positive_flags.append(is_positive)
            damage_ratios.append(float(damage_ratio))
            has_building_flags.append(has_building)

    return (
        image_ids,
        roi_positions,
        context_positions,
        is_positive_flags,
        damage_ratios,
        has_building_flags,
    )


def process_and_save_metadata(
    all_image_ids,
    all_roi_positions,
    all_context_positions,
    all_is_positive,
    all_damage_ratios,
    all_has_building,
    output_dir,
    train_ratio=0.8,
    seed=42,
):
    """
    Process all metadata and split into train/val sets.

    Args:
        all_image_ids: List of all image IDs
        all_roi_positions: List of all ROI positions
        all_context_positions: List of all context positions
        all_is_positive: List of all positive flags
        all_damage_ratios: List of all damage ratios
        all_has_building: List of all building flags
        output_dir: Directory to save processed metadata
        train_ratio: Ratio of data to use for training
        seed: Random seed for splitting

    Returns:
        Dictionary with statistics
    """
    np.random.seed(seed)

    all_image_ids = np.array(all_image_ids, dtype=object)
    all_roi_positions = np.array(all_roi_positions, dtype=np.int32)
    all_context_positions = np.array(all_context_positions, dtype=np.int32)
    all_is_positive = np.array(all_is_positive, dtype=np.bool_)
    all_damage_ratios = np.array(all_damage_ratios, dtype=np.float32)
    all_has_building = np.array(all_has_building, dtype=np.bool_)

    # random permutation for splitting
    n_samples = len(all_image_ids)
    indices = np.random.permutation(n_samples)

    train_split_idx = int(n_samples * train_ratio)
    train_indices = indices[:train_split_idx]
    val_indices = indices[train_split_idx:]

    train_stats = save_split_metadata(
        all_image_ids[train_indices],
        all_roi_positions[train_indices],
        all_context_positions[train_indices],
        all_is_positive[train_indices],
        all_damage_ratios[train_indices],
        all_has_building[train_indices],
        output_dir,
        "train",
    )

    val_stats = save_split_metadata(
        all_image_ids[val_indices],
        all_roi_positions[val_indices],
        all_context_positions[val_indices],
        all_is_positive[val_indices],
        all_damage_ratios[val_indices],
        all_has_building[val_indices],
        output_dir,
        "val",
    )

    return {"train": train_stats, "val": val_stats, "total": n_samples}


def save_split_metadata(
    image_ids,
    roi_positions,
    context_positions,
    is_positive,
    damage_ratios,
    has_building,
    output_dir,
    split,
):
    """
    Save metadata for a specific split (train/val).

    Args:
        image_ids: Array of image IDs
        roi_positions: Array of ROI positions
        context_positions: Array of context positions
        is_positive: Array of positive flags
        damage_ratios: Array of damage ratios
        has_building: Array of building flags
        output_dir: Directory to save metadata
        split: 'train' or 'val'

    Returns:
        Dictionary with statistics
    """
    n_items = len(image_ids)

    if n_items == 0:
        logger.warning(f"No items in {split} split!")
        return {
            "total_patches": 0,
            "positive_patches": 0,
            "negative_patches": 0,
            "positive_ratio": 0,
            "building_patches": 0,
            "non_building_patches": 0,
            "building_ratio": 0,
        }

    # indices of positive and negative samples
    positive_indices = np.where(is_positive)[0]
    negative_indices = np.where(~is_positive)[0]

    # indices of patches with and without buildings
    building_indices = np.where(has_building)[0]
    non_building_indices = np.where(~has_building)[0]

    np.save(output_dir / f"{split}_image_ids.npy", image_ids)
    np.save(output_dir / f"{split}_roi_positions.npy", roi_positions)
    np.save(output_dir / f"{split}_context_positions.npy", context_positions)
    np.save(output_dir / f"{split}_is_positive.npy", is_positive)
    np.save(output_dir / f"{split}_damage_ratios.npy", damage_ratios)
    np.save(output_dir / f"{split}_has_building.npy", has_building)

    np.save(output_dir / f"{split}_positive_indices.npy", positive_indices)
    np.save(output_dir / f"{split}_negative_indices.npy", negative_indices)

    np.save(output_dir / f"{split}_building_indices.npy", building_indices)
    np.save(output_dir / f"{split}_non_building_indices.npy", non_building_indices)

    # statistics
    n_positive = len(positive_indices)
    n_negative = len(negative_indices)
    pos_ratio = n_positive / n_items if n_items > 0 else 0

    n_building = len(building_indices)
    building_ratio = n_building / n_items if n_items > 0 else 0

    stats = {
        "total_patches": n_items,
        "positive_patches": n_positive,
        "negative_patches": n_negative,
        "positive_ratio": pos_ratio,
        "building_patches": n_building,
        "non_building_patches": len(non_building_indices),
        "building_ratio": building_ratio,
    }

    with open(output_dir / f"{split}_summary.json", "w") as f:
        json.dump(stats, f)

    logger.info(
        f"{split.capitalize()} set: {n_items} patches "
        f"({n_positive} positive, {n_negative} negative) "
        f"({n_building} with buildings, {len(non_building_indices)} without buildings)"
    )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract patch metadata without storing patches"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/metadata", help="Output directory"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of images"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train/val split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--include_blacklist",
        action="store_true",
        help="Include the images present in the blacklist",
    )
    parser.add_argument(
        "--select_disaster",
        type=str,
        default="",
        help="Select a scene / disaster",
    )
    parser.add_argument(
        "--building_threshold",
        type=float,
        default=0.2,
        help="Minimum ratio of building pixels needed for building label",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    roi_patch_size = config.get("roi_patch_size", 16)
    context_patch_size = config.get("context_patch_size", 256)
    stride = config.get("patch_stride", 8)
    pos_threshold = config.get("pos_threshold", 0.05)
    min_valid_pixels = config.get("min_valid_pixels", 0.8)
    building_threshold = args.building_threshold

    dataset = SimpleDatasetLoader(
        root_dir=config["data"]["root_dir"],
        blacklist_path=config["data"]["blacklist_path"],
        split="train",
        limit=args.limit,
        exclude_blacklist=not args.include_blacklist,
        select_disaster=args.select_disaster,
    )

    all_image_ids = []
    all_roi_positions = []
    all_context_positions = []
    all_is_positive = []
    all_damage_ratios = []
    all_has_building = []

    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        sample = dataset[idx]
        image_id = sample["image_id"]

        pre_img = sample["pre_image"]
        post_img = sample["post_image"]
        label = sample["label"]

        if isinstance(pre_img, torch.Tensor):
            pre_img = pre_img.numpy().transpose(1, 2, 0)
            post_img = post_img.numpy().transpose(1, 2, 0)
            label = label.numpy()

        # extract metadata for this image directly into arrays
        (img_ids, roi_pos, ctx_pos, is_pos, dmg_ratios, has_building) = (
            extract_patch_metadata(
                pre_img,
                post_img,
                label,
                image_id,
                roi_patch_size=roi_patch_size,
                context_patch_size=context_patch_size,
                stride=stride,
                pos_threshold=pos_threshold,
                min_valid_pixels=min_valid_pixels,
                building_threshold=building_threshold,
            )
        )

        all_image_ids.extend(img_ids)
        all_roi_positions.extend(roi_pos)
        all_context_positions.extend(ctx_pos)
        all_is_positive.extend(is_pos)
        all_damage_ratios.extend(dmg_ratios)
        all_has_building.extend(has_building)

        logger.info(f"Image {image_id}: Found {len(img_ids)} valid patches")

    stats = process_and_save_metadata(
        all_image_ids,
        all_roi_positions,
        all_context_positions,
        all_is_positive,
        all_damage_ratios,
        all_has_building,
        output_dir,
        args.train_ratio,
        args.seed,
    )

    config_summary = {
        "roi_patch_size": roi_patch_size,
        "context_patch_size": context_patch_size,
        "stride": stride,
        "pos_threshold": pos_threshold,
        "min_valid_pixels": min_valid_pixels,
        "building_threshold": building_threshold,
        "train_ratio": args.train_ratio,
        "random_seed": args.seed,
        "stats": stats,
    }

    with open(output_dir / "config_summary.json", "w") as f:
        json.dump(config_summary, f)

    logger.info(f"Metadata processing complete! Files saved to {output_dir}")
    logger.info(f"Total patches: {stats['total']}")
    logger.info(
        f"Training: {stats['train']['total_patches']} patches "
        f"({stats['train']['positive_patches']} positive, {stats['train']['negative_patches']} negative) "
        f"({stats['train']['building_patches']} with buildings, {stats['train']['non_building_patches']} without buildings)"
    )
    logger.info(
        f"Validation: {stats['val']['total_patches']} patches "
        f"({stats['val']['positive_patches']} positive, {stats['val']['negative_patches']} negative) "
        f"({stats['val']['building_patches']} with buildings, {stats['val']['non_building_patches']} without buildings)"
    )


if __name__ == "__main__":
    main()
