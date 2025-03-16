# main_contrastive.py
import argparse
from pprint import pprint

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset_patches import PreprocessedPatchDataset
from data.dataset_patchonthefly import OnTheFlyPatchDataset
from data.transforms import get_transform
from losses.contrastive_loss import InfoNCEContrastiveLoss, SupervisedContrastiveLoss
from models.pseudo_siamese import (
    MultimodalDamageNet,  # Using your original model with minimal changes
)
from sampler.sampler import RatioSampler  # , WarmupSampler
from trainer.trainer import ContrastiveTrainer


def main(args):
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pprint(f"Using config: {config}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets using preprocessed patches
    train_transform = get_transform("train") if args.use_transforms else None
    val_transform = get_transform("val") if args.use_transforms else None

    # print("Loading preprocessed patch datasets...")
    # train_dataset = PreprocessedPatchDataset(
    #     patch_dir=args.patch_dir,
    #     split="train",
    #     transform=train_transform,
    #     cache_size=args.cache_size,
    #     subset_fraction=args.subset_fraction,
    #     seed=args.subset_seed,
    # )

    # val_dataset = PreprocessedPatchDataset(
    #     patch_dir=args.patch_dir,
    #     split="val",
    #     transform=val_transform,
    #     cache_size=args.cache_size,
    #     subset_fraction=args.subset_fraction,
    #     seed=args.subset_seed,
    # )

    train_dataset = OnTheFlyPatchDataset(
        root_dir=config["data"]["root_dir"],
        metadata_dir=args.metadata_dir,
        split="train",
        transform=train_transform,
        cache_size=args.image_cache_size,
        subset_fraction=args.subset_fraction,
        target_neg_ratio=args.target_neg_ratio,
        seed=args.subset_seed,
    )

    val_dataset = OnTheFlyPatchDataset(
        root_dir=config["data"]["root_dir"],
        metadata_dir=args.metadata_dir,
        split="val",
        transform=val_transform,
        cache_size=args.image_cache_size,
        subset_fraction=args.subset_fraction,
        target_neg_ratio=args.target_neg_ratio,
        seed=args.subset_seed,
    )

    # Print dataset statistics
    print(f"Train dataset: {len(train_dataset)} patch pairs")
    print(f"  - Positive ratio: {train_dataset.get_pos_neg_ratio():.2f}")
    print(f"Val dataset: {len(val_dataset)} patch pairs")
    print(f"  - Positive ratio: {val_dataset.get_pos_neg_ratio():.2f}")

    # Create data loaders

    # Taking into account class imbalance - this is experimental ...
    # pos_weight = 1 / train_dataset.get_pos_neg_ratio()
    # neg_weight = 1 / (1 - train_dataset.get_pos_neg_ratio())

    # # Assign weights to all samples
    # weights = [
    #     pos_weight if meta["is_positive"] else neg_weight
    #     for meta in train_dataset.metadata
    # ]

    # # Create a sampler
    # sampler = WeightedRandomSampler(weights, len(weights))

    # Additionally : For distributed GPUS :
    # from torch.utils.data.distributed import DistributedSampler

    # # Create sampler for distributing data across GPUs
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=world_size,  # Number of GPUs
    #     rank=local_rank,  # Current GPU ID
    #     shuffle=True
    # )

    # sampler = WarmupSampler(
    #     train_dataset,
    #     batch_size=config["training"]["batch_size"],
    #     warmup_epochs=config["warmup"]["warmup_epochs"],  # Adjust as needed
    # )

    train_sampler = RatioSampler(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        neg_ratio=args.target_neg_ratio,
    )

    val_sampler = RatioSampler(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        neg_ratio=args.target_neg_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=train_sampler,
        shuffle=False,  # Important: set to False when using a custom sampler
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=val_sampler,  # testing with a sampler...
        shuffle=False,  # Important: set to False when using a custom sampler
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # Create model with your original architecture (minus classification head)
    model = MultimodalDamageNet(
        resnet_version=config["model"]["resnet_version"],
        freeze_resnet=config["model"]["freeze_resnet"],
        optical_channels=config["model"]["optical_channels"],
        sar_channels=config["model"]["sar_channels"],
        projection_dim=config["model"]["projection_dim"],
    )

    # recap of the model architecture
    print("Model architecture:")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # Create simplified contrastive loss function
    # criterion = SupervisedContrastiveLoss(
    #     temperature=config["training"].get("temperature", 0.07),
    # )
    # Create infoNCE contrastive loss function
    criterion = InfoNCEContrastiveLoss(
        temperature=config["training"].get("temperature", 0.07),
    )

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config["training"]["num_epochs"],
        output_dir=config["training"]["output_dir"],
        save_best=True,
        log_interval=args.log_interval,
        loading_checkpoint=config["training"]["loading_checkpoint"],
        monitor_gradients=args.monitor_gradients,
    )

    # Train model
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train contrastive learning model with preprocessed patches"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        default="data/metadata_noblacklist",
        help="Directory containing patch metadata",
    )
    parser.add_argument(
        "--image_cache_size",
        type=int,
        default=50,
        help="Number of full images to cache in memory",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        default="data/processed_patches",
        help="Directory containing preprocessed patches",
    )
    parser.add_argument(
        "--use_transforms",
        action="store_true",
        help="Whether to apply data augmentation transforms",
    )
    parser.add_argument(
        "--cache_size",
        type=int,
        default=1000,
        help="Number of patches to cache in memory",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="How often to log training metrics (in batches)",
    )
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use (0.0-1.0)",
    )
    parser.add_argument(
        "--target_neg_ratio",
        type=float,
        default=0.8,
        help="Target negative pair ratios to have in training dataset (0.0-1.0)",
    )
    parser.add_argument(
        "--subset_seed", type=int, default=42, help="Random seed for subset selection"
    )
    parser.add_argument(
        "--monitor_gradients",
        action="store_true",
        help="Enable monitoring and visualization of gradients and activations",
    )
    args = parser.parse_args()

    main(args)
