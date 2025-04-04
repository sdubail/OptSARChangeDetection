I'll create a comprehensive README that reflects the current state of your project based on the repository contents. Here's a new README that accurately describes your OptSARChangeDetection project:

# 🛰️ OptSARChangeDetection

Multimodal change detection framework using supervised contrastive learning for optical and SAR remote sensing images.

## 📋 Overview

This project implements a supervised contrastive learning approach to detect changes between pre-event optical imagery and post-event SAR (Synthetic Aperture Radar) imagery. The framework builds a common representation space for both modalities, enabling accurate detection of semantic changes like building damage following disasters (earthquakes, floods, volcanic eruptions, etc.).

Unlike traditional unsupervised methods that struggle with multimodal data, this approach leverages change annotations to guide the contrastive learning process. It selects positive pairs (no change/undamaged buildings) and negative pairs (with change/damaged buildings) to learn more effective feature representations that can distinguish semantic changes across modalities.

The project is based on research in self-supervised learning for remote sensing change detection, extending it with supervision signals to improve multimodal feature alignment.

## 🧠 Key concepts

- **Multimodal Change Detection**: Combining optical and SAR imagery to detect changes more effectively
- **Supervised Contrastive Learning**: Using damage annotations to guide the learning of a shared representation space
- **Pseudo-Siamese Architecture**: Two specialized branches (optical and SAR) with similar structure but different weights
- **InfoNCE Loss**: Modified contrastive loss function that leverages negative examples within batches

## 🚀 Getting started

### Prerequisites

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Important**: Place the dataset in the following path:
   ```
   data/dfc25_track2_trainval/
   ```

## 🔄 Pipeline

The project workflow consists of three main steps:

### 1️⃣ Metadata processing

First, extract metadata from raw images without storing the patches themselves:

```bash
python -m data.preprocess_metadata --config configs/default.yaml --output_dir data/metadata
```

Options:
- `--limit N`: Process only N images (useful for testing)
- `--building_threshold 0.2`: Set minimum ratio of building pixels for building presence label
- `--train_ratio 0.8`: Set split ratio between train and validation sets
- `--include_blacklist`: Include images from blacklist
- `--select_disaster NAME`: Select a specific disaster scene (e.g., "turkey-earthquake")

### 2️⃣ Model training

Train the contrastive learning model using the extracted metadata:

```bash
python __main__.py train start --config configs/default.yaml --metadata_dir data/metadata
```

Options:
- `--subset_fraction 0.1`: Use only a fraction of the dataset (0.0-1.0)
- `--target_neg_ratio 0.8`: Target ratio of negative samples (0.0-1.0)
- `--use_transforms`: Apply data augmentation transforms
- `--monitor_gradients`: Enable gradient flow monitoring

### 3️⃣ Inference and Evaluation

Generate change maps and evaluate the model:

```bash
# Generate change map for a specific image
python __main__.py infer predict --model output/patch_contrastive/best_model.pth --image-id "morocco-earthquake_00000257"

# Find optimal threshold for binary classification
python __main__.py infer threshold --model output/patch_contrastive/best_model.pth
```

## 🧪 Feature exploration

Visualize the learned feature space:

```bash
python -m models.features_explorer --model_path output/patch_contrastive/best_model.pth --use_tsne --use_umap
```

## 📊 Visualize dataset

Explore the dataset with these visualization tools:

```bash
# Explore patch pairs
python -m data.vizu.patch_explorer --patch_dir data/processed_patches

# Explore full images
python -m data.vizu.image_explorer --data_dir data/dfc25_track2_trainval --split train
```

## 🔍 Project structure

### Data handling
- `data/preprocess_metadata.py`: Creates metadata index for on-the-fly patch extraction
- `data/dataset_patchonthefly.py`: Efficient dataset that extracts patches during training
- `data/transforms.py`: Data augmentation for training

### Model architecture
- `models/pseudo_siamese.py`:
  - `OpticalEncoder`: Modified ResNet for optical images
  - `SAREncoder`: Modified ResNet for SAR images
  - `MultimodalDamageNet`: Main model combining both encoders with projection heads

### Training
- `losses/contrastive_loss.py`: Implements InfoNCE contrastive loss with positive/negative pair weighting
- `trainer/trainer.py`: Training loop with validation and model checkpointing
- `sampler/sampler.py`: Balanced batch sampler for controlling positive/negative ratio

### Inference
- `inference/change_map_generator.py`: Creates change maps from model outputs

### Visualization
- `models/features_explorer.py`: Visualizes learned feature spaces with dimensionality reduction
- `data/vizu/patch_explorer.py`: Interactive tool for exploring image patches
- `data/vizu/image_explorer.py`: Interactive tool for exploring full images

## ⚙️ Configuration

The `configs/default.yaml` file contains important parameters:

```yaml
# Data parameters
data:
  root_dir: "data/dfc25_track2_trainval"
  blacklist_path: "data/image_labels.txt"
  num_workers: 4

# Patch extraction parameters
roi_patch_size: 16              # Actual region of interest size
context_patch_size: 256         # Context window size
patch_stride: 8                 # Stride for patch extraction
pos_threshold: 0.05             # Max damage ratio for positive pairs

# Model parameters
model:
  optical_channels: 3           # RGB for optical images
  sar_channels: 1               # Single channel for SAR
  projection_dim: 128           # Dimension of feature space
  resnet_version: 34            # ResNet backbone version

# Training parameters
training:
  batch_size: 64
  learning_rate: 0.0001
  weight_decay: 0.0001
  num_epochs: 10
  temperature: 0.07             # Temperature for contrastive loss
```

## 🔬 Research context

This implementation extends work on self-supervised change detection in remote sensing images by adding supervision through damage annotations. By learning a common representation space, the model can detect semantic changes between different modalities more effectively than traditional methods.

The approach is particularly valuable for disaster monitoring where optical images might be available before an event, but only SAR images (which can see through clouds) are available immediately after a disaster.

## 📝 Citation

This implementation is based on research from both remote sensing and deep learning fields:

- Chen, Y., & Bruzzone, L. (2021). Self-supervised Change Detection in Multi-view Remote Sensing Images. *arXiv preprint arXiv:2103.05969*