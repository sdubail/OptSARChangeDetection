import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PatchDataset(Dataset):
    """Dataset for batched patch extraction from a single image pair."""

    def __init__(
        self,
        pre_img: np.ndarray,
        post_img: np.ndarray,
        roi_patch_size: int,
        context_patch_size: int,
        stride: int,
        transform=None,
    ):
        """
        Args:
            pre_img: Pre-event optical image
            post_img: Post-event SAR image
            roi_patch_size: Size of the region of interest
            context_patch_size: Size of the context window
            stride: Stride for patch extraction
            transform: Optional transforms to apply
        """
        self.pre_img = pre_img
        self.post_img = post_img
        self.roi_patch_size = roi_patch_size
        self.context_patch_size = context_patch_size
        self.stride = stride
        self.transform = transform
        self.pad_size = (context_patch_size - roi_patch_size) // 2

        self.height, self.width = pre_img.shape[:2]

        # extract valid patch positions
        self.patch_positions = self._get_patch_positions()

    def _get_patch_positions(self) -> List[Tuple[int, int]]:
        """Get all valid patch positions for the image pair."""
        # extraction boundaries
        start_y = self.pad_size
        start_x = self.pad_size
        end_y = self.height - self.pad_size - self.roi_patch_size + 1
        end_x = self.width - self.pad_size - self.roi_patch_size + 1

        # Check if extraction is possible
        if start_y >= end_y or start_x >= end_x:
            logger.warning(
                f"Image too small for extraction: {self.height}x{self.width}"
            )
            return []

        # Collect all valid patch positions
        positions = []

        for y in range(start_y, end_y, self.stride):
            for x in range(start_x, end_x, self.stride):
                positions.append((y, x))
        return positions

    def __len__(self) -> int:
        """Return the number of patches."""
        return len(self.patch_positions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a patch pair at the specified position."""
        y, x = self.patch_positions[idx]

        # Extract context patch coordinates
        ctx_y_start = y - self.pad_size
        ctx_x_start = x - self.pad_size
        ctx_y_end = y + self.roi_patch_size + self.pad_size
        ctx_x_end = x + self.roi_patch_size + self.pad_size

        # Extract patches
        pre_patch = self.pre_img[ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end].copy()
        post_patch = self.post_img[ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end].copy()

        if self.transform:
            transformed = self.transform(pre_patch, post_patch)
            pre_patch = transformed["pre_image"]
            post_patch = transformed["post_image"]

        # Convert to tensor with normalization
        pre_patch = self._to_tensor_optical(pre_patch)
        post_patch = self._to_tensor_sar(post_patch)

        return {
            "pre_patch": pre_patch,
            "post_patch": post_patch,
            "position": (y, x),
        }

    def _to_tensor_optical(self, img: np.ndarray) -> torch.Tensor:
        """Convert optical image to tensor with normalization."""
        img = img.astype(np.float32)

        # Per-channel normalization
        means = img.mean(axis=(0, 1), keepdims=True)
        stds = img.std(axis=(0, 1), keepdims=True) + 1e-8
        img = (img - means) / stds

        # Convert to tensor with channel-first format
        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img

    def _to_tensor_sar(self, img: np.ndarray) -> torch.Tensor:
        """Convert SAR image to tensor with appropriate preprocessing."""
        img = img.astype(np.float32)

        if (
            img.shape[2] == 3
            and np.allclose(img[:, :, 0], img[:, :, 1])
            and np.allclose(img[:, :, 0], img[:, :, 2])
        ):
            img = img[:, :, 0:1]

        # log transformation
        img = np.log1p(img)

        # Normalize
        means = img.mean(axis=(0, 1), keepdims=True)
        stds = img.std(axis=(0, 1), keepdims=True) + 1e-8
        img = (img - means) / stds

        img = torch.from_numpy(img.transpose(2, 0, 1))
        return img


def set_border_to_zero(array, border_width):
    """
    Set the border of an array to zero without changing its dimensions.

    Args:
        array: Input numpy array
        border_width: Width of the border to set to zero

    Returns:
        Modified array with borders set to zero
    """
    result = array.copy()

    if array.ndim == 2:
        h, w = array.shape

        result[:border_width, :] = 0
        result[h - border_width :, :] = 0

        result[:, :border_width] = 0
        result[:, w - border_width :] = 0

    elif array.ndim == 3:
        h, w, c = array.shape

        result[:border_width, :, :] = 0
        result[h - border_width :, :, :] = 0

        result[:, :border_width, :] = 0
        result[:, w - border_width :, :] = 0

    else:
        raise ValueError(f"Unsupported array dimensionality: {array.ndim}")

    return result


def visualize_damage(
    damage_img,
    is_patch=False,
    roi_patch_size=16,
    context_patch_size=256,
    stride=8,
    pos_threshold=0.05,
):
    """
    Create a color visualization of damage levels or patch-level contrastive labels.

    Args:
        damage_img: Input damage image or binary label
        is_patch: If True, visualize patch-level contrastive training labels
        roi_patch_size: Size of the region of interest for patches
        pad_size: Padding size around the ROI
        stride: Stride for patch extraction
        pos_threshold: Threshold for positive/negative classification

    Returns:
        RGB visualization image
    """
    if not is_patch:
        # Standard damage level visualization (original function)
        # Define colors for damage levels (0 to 4)
        colors = np.array(
            [
                [0, 0, 0],  # 0: No damage (black)
                [0, 255, 0],  # 1: Minor damage (green)
                [255, 255, 0],  # 2: Moderate damage (yellow)
                [255, 127, 0],  # 3: Major damage (orange)
                [255, 0, 0],  # 4: Destroyed (red)
            ]
        )

        # Create RGB image
        h, w = damage_img.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Fill with colors based on damage values
        for i in range(5):  # 5 damage levels
            mask = damage_img == i
            if np.any(mask):
                rgb[mask] = colors[i]
    else:
        # Patch-level contrastive label visualization
        h, w = damage_img.shape
        pad_size = (context_patch_size - roi_patch_size) // 2
        # Create accumulation map for patch labels and count map for averaging
        label_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.int32)

        # Create binary version of the damage map (any damage > 0)
        label_binary = np.where(damage_img > 1, 1, 0)

        # Calculate valid extraction region
        start_y = pad_size
        start_x = pad_size
        end_y = h - pad_size - roi_patch_size + 1
        end_x = w - pad_size - roi_patch_size + 1

        if start_y >= end_y or start_x >= end_x:
            # Image too small for extraction
            # Return a red "error" image
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            rgb[:, :, 0] = 255
            return rgb

        # Process each patch
        for y in range(start_y, end_y, stride):
            for x in range(start_x, end_x, stride):
                # Extract ROI
                roi = label_binary[y : y + roi_patch_size, x : x + roi_patch_size]

                # Calculate damage ratio in this ROI
                damage_pixels = np.sum(roi > 0)
                total_valid_pixels = roi.size - np.sum(np.isnan(roi))

                if total_valid_pixels == 0:
                    continue

                damage_ratio = damage_pixels / total_valid_pixels
                is_positive = int(damage_ratio <= pos_threshold)

                # Assign the patch label (0 or 1) to all pixels in the ROI region
                # This is where patches overlap and will be averaged
                label_map[y : y + roi_patch_size, x : x + roi_patch_size] += (
                    1 - is_positive
                )
                count_map[y : y + roi_patch_size, x : x + roi_patch_size] += 1

        # Average overlapping regions
        label_map = np.divide(
            label_map, count_map, out=np.zeros_like(label_map), where=count_map > 0
        )

        # Create RGB visualization (grayscale)
        # 0 (negative/damaged) = black, 1 (positive/intact) = white
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        grayscale = (label_map * 255).astype(np.uint8)

        rgb[:, :, 0] = grayscale
        rgb[:, :, 1] = grayscale
        rgb[:, :, 2] = grayscale

    return rgb


def dome_window(size, power=2):
    """
    Generates a 2D dome window.
    Maximum at center, and exactly zero weight at corners.

    Args:
        size (tuple): (height, width) of the window.
        power (float): Controls the curvature of the dome
                        (2=parabolic, larger=sharper).

    Returns:
        window (ndarray): 2D window with values in [0,1], zero at corners.
    """
    height, width = size

    # Normalized grid: x and y in [-1, 1]
    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)

    # Compute distance to corners (max distance = sqrt(2))
    radius = np.sqrt(xx**2 + yy**2) / np.sqrt(
        2
    )  # Normalize so that corners have radius = 1

    # Dome function: maximum at center, zero at corners
    window = 1 - (radius**power)

    # Ensure no negative values
    window = np.clip(window, 0, None)

    return window


def create_change_map(
    model: torch.nn.Module,
    pre_img: np.ndarray,
    post_img: np.ndarray,
    config: dict,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 4,
    window_method: str = "dome",  # classic or dome
    window_power: float = 1.5,
) -> np.ndarray:
    """
    Create a change map from pre-event optical and post-event SAR images.

    Args:
        model: Trained model for change detection
        pre_img: Pre-event optical image
        post_img: Post-event SAR image
        config: Configuration dictionary with patch parameters
        device: Device to run inference on
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading

    Returns:
        A change map with the same spatial dimensions as the input images
    """
    roi_patch_size = config.get("roi_patch_size", 16)
    context_patch_size = config.get("context_patch_size", 256)
    stride = config.get("patch_stride", 8)

    model.eval()

    # Create dataset and dataloader for patches
    patch_dataset = PatchDataset(
        pre_img=pre_img,
        post_img=post_img,
        roi_patch_size=roi_patch_size,
        context_patch_size=context_patch_size,
        stride=stride,
    )

    if len(patch_dataset) == 0:
        logger.warning("No valid patches found. Image may be too small.")
        return np.zeros((pre_img.shape[0], pre_img.shape[1]), dtype=np.float32)

    patch_loader = DataLoader(
        patch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Create an accumulation map for change scores and a count map for averaging
    height, width = pre_img.shape[:2]
    change_map = np.zeros((height, width), dtype=np.float32)
    count_map = np.zeros(
        (height, width), dtype=np.int32 if window_method == "classic" else np.float32
    )

    # Process patches in batches
    with torch.no_grad():
        for batch in tqdm(patch_loader, desc="Generating change map"):
            pre_patches = batch["pre_patch"].to(device)
            post_patches = batch["post_patch"].to(device)
            positions = batch["position"]

            # Forward pass through model
            outputs = model(optical=pre_patches, sar=post_patches)

            # Get change scores
            if "change_score" in outputs:
                change_scores = outputs["change_score"].cpu().numpy()

            # Accumulate change scores to the change map

            for i, (y, x) in enumerate(zip(positions[0], positions[1])):
                score = change_scores[i]

                # Update the region of interest in the change map
                # For each patch's ROI, add the change score
                roi_y_end = y + roi_patch_size
                roi_x_end = x + roi_patch_size

                # define window importance
                window = (
                    1
                    if window_method == "classic"
                    else dome_window(
                        (roi_patch_size, roi_patch_size), power=window_power
                    )
                )

                change_map[y:roi_y_end, x:roi_x_end] += score * window
                count_map[y:roi_y_end, x:roi_x_end] += window

    # Average overlapping regions
    change_map = np.divide(
        change_map, count_map, out=np.zeros_like(change_map), where=count_map > 0
    )

    return change_map


def compute_iou(
    pred_map: np.ndarray, target_map: np.ndarray, threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute Intersection over Union between predicted and target change maps.

    Args:
        pred_map: Predicted change map
        target_map: Target change map (ground truth)
        threshold: Threshold for binarizing change maps

    Returns:
        IoU score, precision, and recall
    """
    # Ensure binary maps
    pred_binary = pred_map > threshold

    # Target is already binary (1 for damage, >0 for any level of damage)
    target_binary = target_map > 0

    # Calculate intersection and union
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()

    # Calculate IoU, precision, and recall
    iou = intersection / union if union > 0 else 0

    # Calculate precision and recall
    true_positives = intersection
    false_positives = pred_binary.sum() - true_positives
    false_negatives = target_binary.sum() - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    return iou, precision, recall
