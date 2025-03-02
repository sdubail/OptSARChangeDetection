# 🛰️ OptSARChangeDetection 🛰️

Multimodal change detection framework based on supervised contrastive learning for Optical and SAR remote sensing images.

## 📋 Overview

This project implements a supervised contrastive learning approach for multimodal change detection between optical and SAR remote sensing images. The method builds a common representation space for both modalities to accurately detect changes, with a particular focus on building damage assessment following disasters (earthquakes, floods, etc).

Unlike traditional unsupervised methods, this approach leverages change annotations to guide the contrastive learning process by selecting positive pairs (no change) and negative pairs (with change) for more effective feature learning.

## 🚀 Getting Started

### Prerequisites

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. **IMPORTANT**: Place the dataset in the following path:
   ```
   data/dfc25_track2_trainval/
   ```

### 🔄 Pipeline

The project workflow consists of three main steps:

1. **Data preprocessing** - Extracts and prepares training patches
2. **Model training** - Trains the contrastive learning model
3. **Inference** - Applies the model to detect changes (TODO)

## 📁 Project Structure and Usage

### 1️⃣ Data Preprocessing

First, you need to preprocess the raw satellite images into balanced patch pairs:

```bash
python -m data.preprocess_patches --config configs/default.yaml --output_dir data/processed_patches
```

> 💡 **Tip for testing**: When developing or testing your code, use the `--limit` parameter to process fewer images:
> ```bash
> python -m data.preprocess_patches --config configs/default.yaml --output_dir data/processed_patches --limit 5
> ```

Key files:
- `data/preprocess_patches.py`: Extracts patch pairs from images and classifies them as positive/negative pairs
- `configs/default.yaml`: Configuration parameters for preprocessing and training

### 2️⃣ Model Training

Once preprocessing is complete, train the contrastive learning model:

```bash
python __main__.py --config configs/default.yaml --patch_dir data/processed_patches
```

Key files:
- `__main__.py`: Entry point for training
- `models/pseudo_siamese.py`: Contains the multimodal network architecture with optical and SAR encoders
- `losses/contrastive_loss.py`: Implements the supervised contrastive loss function
- `trainer/trainer.py`: Manages the training loop, validation, and checkpoints

### 3️⃣ Key Components

#### Dataset Handling
- `data/dataset_patches.py`: Loads preprocessed patch pairs from HDF5 files
- `data/dataset_full.py`: Alternative loader for full images (without patch extraction)
- `data/transforms.py`: Data augmentation transforms for training

#### Model Architecture
- `models/pseudo_siamese.py`: 
  - `OpticalEncoder`: Modified ResNet18 for optical images
  - `SAREncoder`: Modified ResNet18 for SAR images 
  - `MultimodalDamageNet`: Main model combining both encoders with projection heads

#### Training
- `losses/contrastive_loss.py`: Implements supervised contrastive loss
- `trainer/trainer.py`: Manages training, validation, checkpoints and visualization

## 📊 Configuration

The `configs/default.yaml` file contains important parameters:

- **Data parameters**: Paths and loading settings
- **Patch extraction**: Sizes, thresholds, and balancing ratios
- **Model architecture**: Channel configurations and projection dimensions
- **Training parameters**: Batch size, learning rate, etc.

## 🔍 Project Goals

This implementation differs from traditional methods (like Chen and Bruzzone [2]) by using a supervised approach with change detection annotations to guide the contrastive learning process. The goal is to:

1. Build a common representation space for SAR and optical images
2. Allow accurate detection of semantic changes between modalities
3. Enable expansion to other modality combinations (SAR-SAR, optical-optical)

Happy coding! 🚀