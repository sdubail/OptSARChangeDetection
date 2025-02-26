# data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from . import transforms as T

class MultimodalDamageDataset(Dataset):
    """Dataset for multimodal damage assessment with optical pre-event and SAR post-event images."""
    
    def __init__(self, root_dir, split='train', transform=None, crop_size=256):
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
        
        # Get list of image names
        post_event_dir = self.root_dir / split / 'post-event'
        self.image_ids = [f.name.replace('_post_disaster.tif', '') 
                          for f in post_event_dir.glob('*_post_disaster.tif')]
        
        print(f"Found {len(self.image_ids)} images for {split} split")
    
    def __len__(self):
        return len(self.image_ids)
    
    def load_tiff_data(self, path):
        """Load and preprocess TIFF images."""
        data_reader = rasterio.open(path)
        try:
            return np.stack([data_reader.read(1), data_reader.read(2), data_reader.read(3)], axis=2)
        except:
            return data_reader.read(1)[..., np.newaxis]
    
    def __getitem__(self, idx):
        """Get a pre-event optical, post-event SAR, and damage label triplet."""
        image_id = self.image_ids[idx]
        
        # Load pre-event optical image (RGB)
        pre_path = self.root_dir / self.split / 'pre-event' / f"{image_id}_pre_disaster.tif"
        pre_img = self.load_tiff_data(str(pre_path))
        if pre_img.shape[-1] > 3:  # Ensure 3 channels
            pre_img = pre_img[:, :, :3]
        
        # Load post-event SAR image
        post_path = self.root_dir / self.split / 'post-event' / f"{image_id}_post_disaster.tif"
        post_img = self.load_tiff_data(str(post_path))
        
        # Convert single-channel to 3-channel for consistent processing
        if post_img.shape[-1] == 1:
            post_img = np.repeat(post_img, 3, axis=2)
        
        # Load damage label
        label_path = self.root_dir / self.split / 'target' / f"{image_id}_building_damage.tif"
        label = self.load_tiff_data(str(label_path))
        
        # Create binary label for location (0: no building, 1: building)
        loc_label = label.copy()
        loc_label[loc_label > 0] = 1  # Any building becomes 1
        
        # Create binary damage label (0: no/minor damage, 1: major damage)
        damage_label = np.zeros_like(label)
        damage_label[label == 3] = 1  # Major damage is labeled as 1
        
        # Apply transforms for training
        if self.transform:
            sample = self.transform(pre_img, post_img, label)
            pre_img, post_img, label = sample['pre_image'], sample['post_image'], sample['label']
        
        # Convert to tensors and normalize
        pre_img = self._normalize_and_to_tensor(pre_img)
        post_img = self._normalize_and_to_tensor(post_img)
        
        label = torch.from_numpy(label.squeeze()).long()
        loc_label = torch.from_numpy(loc_label.squeeze()).long()
        damage_label = torch.from_numpy(damage_label.squeeze()).long()
        
        return {
            'pre_image': pre_img,
            'post_image': post_img,
            'label': label,
            'loc_label': loc_label,
            'damage_label': damage_label,
            'image_id': image_id
        }
    
    def _normalize_and_to_tensor(self, img):
        """Normalize and convert image to tensor."""
        img = img.astype(np.float32)
        
        # Standard normalization
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        # Convert to tensor with channel-first format
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        return img
    
    def extract_patches(self, image, label, size, stride):
        """Extract patches from image and corresponding labels."""
        patches, labels = [], []
        h, w = image.shape[:2]
        
        for y in range(0, h - size + 1, stride):
            for x in range(0, w - size + 1, stride):
                patch = image[y:y+size, x:x+size]
                patch_label = label[y:y+size, x:x+size]
                patches.append(patch)
                labels.append(patch_label)
                
        return patches, labels
    
    def create_contrastive_pairs(self, pre_img, post_img, damage_label):
        """
        Create positive and negative pairs based on damage labels.
        
        Positive pairs: Areas with no major damage (0s in damage_label)
        Negative pairs: Areas with major damage (1s in damage_label)
        """
        # Extract patches from images and labels
        pre_patches, labels = self.extract_patches(pre_img, damage_label, self.crop_size, self.crop_size//2)
        post_patches, _ = self.extract_patches(post_img, damage_label, self.crop_size, self.crop_size//2)
        
        positive_pairs, negative_pairs = [], []
        
        # Create pairs based on damage label
        for pre_patch, post_patch, patch_label in zip(pre_patches, post_patches, labels):
            # Calculate percentage of damaged pixels
            damage_percentage = np.mean(patch_label)
            
            if damage_percentage < 0.1:  # Less than 10% damaged pixels -> positive pair
                positive_pairs.append((pre_patch, post_patch))
            elif damage_percentage > 0.3:  # More than 30% damaged pixels -> negative pair
                negative_pairs.append((pre_patch, post_patch))
        
        return positive_pairs, negative_pairs