import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import sklearn.metrics
import torch
import torch.optim as optim
import typer
import yaml
from matplotlib.colors import LinearSegmentedColormap
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text
from rich.traceback import install
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

from data.dataset_patches import PreprocessedPatchDataset
from data.dataset_patchonthefly import OnTheFlyPatchDataset
from data.preprocess_metadata import SimpleDatasetLoader
from data.transforms import get_transform
from inference.change_map_generator import (
    compute_iou,
    create_change_map,
    set_border_to_zero,
    visualize_change_map,
    visualize_damage,
)
from losses.contrastive_loss import InfoNCEContrastiveLoss, SupervisedContrastiveLoss
from models.pseudo_siamese import MultimodalDamageNet
from sampler.sampler import RatioSampler
from trainer.trainer import ContrastiveTrainer

install(show_locals=False, width=120, word_wrap=True)

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="OptSARChangeDetection - Multimodal Change Detection Framework",
)
console = Console()

train_app = typer.Typer(pretty_exceptions_show_locals=False, help="Training commands")
infer_app = typer.Typer(pretty_exceptions_show_locals=False, help="Inference commands")

app.add_typer(train_app, name="train")
app.add_typer(infer_app, name="infer")


@train_app.command("start")
def train_start(
    config: Path = typer.Option(
        "configs/default.yaml", help="Path to configuration file", exists=True
    ),
    metadata_dir: Path = typer.Option(
        "data/metadata_noblacklist", help="Directory containing patch metadata"
    ),
    image_cache_size: int = typer.Option(
        50, help="Number of full images to cache in memory"
    ),
    patch_dir: Path = typer.Option(
        "data/processed_patches", help="Directory containing preprocessed patches"
    ),
    use_transforms: bool = typer.Option(
        False, help="Whether to apply data augmentation transforms"
    ),
    cache_size: int = typer.Option(1000, help="Number of patches to cache in memory"),
    log_interval: int = typer.Option(
        10, help="How often to log training metrics (in batches)"
    ),
    subset_fraction: float = typer.Option(
        1.0, min=0.0, max=1.0, help="Fraction of dataset to use"
    ),
    target_neg_ratio: float = typer.Option(
        0.8, min=0.0, max=1.0, help="Target negative pair ratios in training dataset"
    ),
    subset_seed: int = typer.Option(42, help="Random seed for subset selection"),
    monitor_gradients: bool = typer.Option(
        False, help="Enable monitoring of gradients and activations"
    ),
):
    """
    Train the multimodal change detection model using contrastive learning.
    """
    # Create a rich console progress display
    with console.status("[bold green]Loading configuration...") as status:
        # Load configuration
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        # Display configuration in a nice table
        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")

        # Add main config categories
        for category, params in config_data.items():
            if isinstance(params, dict):
                for param, value in params.items():
                    config_table.add_row(f"{category}.{param}", str(value))
            else:
                config_table.add_row(category, str(params))

        console.print(config_table)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"Using device: [bold]{device}[/bold]")

        # Update status
        status.update("[bold green]Loading datasets...")

        # Create datasets
        train_transform = get_transform("train") if use_transforms else None
        val_transform = get_transform("val") if use_transforms else None

        train_dataset = OnTheFlyPatchDataset(
            root_dir=config_data["data"]["root_dir"],
            metadata_dir=metadata_dir,
            split="train",
            transform=train_transform,
            cache_size=image_cache_size,
            subset_fraction=subset_fraction,
            target_neg_ratio=target_neg_ratio,
            seed=subset_seed,
        )

        val_dataset = OnTheFlyPatchDataset(
            root_dir=config_data["data"]["root_dir"],
            metadata_dir=metadata_dir,
            split="val",
            transform=val_transform,
            cache_size=image_cache_size,
            subset_fraction=subset_fraction,
            target_neg_ratio=target_neg_ratio,
            seed=subset_seed,
        )

        # Display dataset info
        dataset_table = Table(title="Dataset Information")
        dataset_table.add_column("Dataset", style="cyan")
        dataset_table.add_column("Size", style="green")
        dataset_table.add_column("Positive Ratio", style="yellow")

        dataset_table.add_row(
            "Training",
            str(len(train_dataset)),
            f"{train_dataset.get_pos_neg_ratio():.2f}",
        )
        dataset_table.add_row(
            "Validation",
            str(len(val_dataset)),
            f"{val_dataset.get_pos_neg_ratio():.2f}",
        )

        console.print(dataset_table)

        # Update status
        status.update("[bold green]Creating data loaders...")

        # Create samplers and data loaders
        train_sampler = RatioSampler(
            train_dataset,
            batch_size=config_data["training"]["batch_size"],
            neg_ratio=target_neg_ratio,
        )

        val_sampler = RatioSampler(
            val_dataset,
            batch_size=config_data["training"]["batch_size"],
            neg_ratio=target_neg_ratio,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config_data["training"]["batch_size"],
            sampler=train_sampler,
            shuffle=False,
            num_workers=config_data["data"]["num_workers"],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config_data["training"]["batch_size"],
            sampler=val_sampler,
            shuffle=False,
            num_workers=config_data["data"]["num_workers"],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        # Update status
        status.update("[bold green]Building model...")

        # Create model
        model = MultimodalDamageNet(
            resnet_version=config_data["model"]["resnet_version"],
            freeze_resnet=config_data["model"]["freeze_resnet"],
            optical_channels=config_data["model"]["optical_channels"],
            sar_channels=config_data["model"]["sar_channels"],
            projection_dim=config_data["model"]["projection_dim"],
        )

        # Display model architecture summary
        model_table = Table(title="Model Architecture")
        model_table.add_column("Layer", style="cyan")
        model_table.add_column("Trainable", style="green")

        trainable_params = 0
        non_trainable_params = 0

        for name, param in model.named_parameters():
            model_table.add_row(name, "✓" if param.requires_grad else "✗")
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                non_trainable_params += param.numel()

        total_params = trainable_params + non_trainable_params

        console.print(model_table)
        console.print(
            f"Trainable parameters: [bold green]{trainable_params:,}[/bold green] ({trainable_params/total_params:.1%})"
        )
        console.print(
            f"Non-trainable parameters: [bold yellow]{non_trainable_params:,}[/bold yellow] ({non_trainable_params/total_params:.1%})"
        )
        console.print(f"Total parameters: [bold]{total_params:,}[/bold]")

        # Update status
        status.update("[bold green]Setting up training...")

        # Create loss function
        criterion = InfoNCEContrastiveLoss(
            temperature=config_data["training"].get("temperature", 0.07),
        )

        # Create optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=config_data["training"]["learning_rate"],
            weight_decay=config_data["training"]["weight_decay"],
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
            num_epochs=config_data["training"]["num_epochs"],
            output_dir=config_data["training"]["output_dir"],
            save_best=True,
            log_interval=log_interval,
            loading_checkpoint=config_data["training"]["loading_checkpoint"],
            monitor_gradients=monitor_gradients,
        )

        # Display training settings
        training_table = Table(title="Training Settings")
        training_table.add_column("Setting", style="cyan")
        training_table.add_column("Value", style="green")

        training_table.add_row("Batch Size", str(config_data["training"]["batch_size"]))
        training_table.add_row(
            "Learning Rate", str(config_data["training"]["learning_rate"])
        )
        training_table.add_row(
            "Weight Decay", str(config_data["training"]["weight_decay"])
        )
        training_table.add_row(
            "Number of Epochs", str(config_data["training"]["num_epochs"])
        )
        training_table.add_row(
            "Temperature", str(config_data["training"].get("temperature", 0.07))
        )
        training_table.add_row(
            "Output Directory", str(config_data["training"]["output_dir"])
        )
        training_table.add_row(
            "Load Checkpoint", str(config_data["training"]["loading_checkpoint"])
        )
        training_table.add_row("Monitor Gradients", str(monitor_gradients))

        console.print(training_table)

        # Start training
        console.print(Panel("[bold green]Starting training...[/bold green]"))

    # Train model - outside of status context to show progress
    trainer.train()

    console.print(Panel("[bold green]Training complete![/bold green]"))


@infer_app.command("predict")
def infer_predict(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to the trained model checkpoint", exists=True
    ),
    config: Path = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
    ),
    image_id: str = typer.Option(
        ..., "--image-id", "-i", help="ID of the image to predict"
    ),
    data_dir: Path = typer.Option(
        "data/dfc25_track2_trainval",
        "--data",
        "-d",
        help="Directory containing the dataset",
    ),
    split: str = typer.Option(
        "val", "--split", "-s", help="Dataset split containing the image"
    ),
    output_dir: Path = typer.Option(
        "output/predictions",
        "--output",
        "-o",
        help="Directory to save prediction results",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help="Device to run inference on",
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", "-b", help="Batch size for inference"
    ),
    context_patch_size: int = typer.Option(
        256, "--context_patch-size", "-b", help="context_patch size for inference"
    ),
    roi_patch_size: int = typer.Option(
        16, "--roi_patch-size", "-b", help="roi_patch size for inference"
    ),
    num_workers: int = typer.Option(
        4, "--workers", "-w", help="Number of workers for data loading"
    ),
):
    """
    Predict change map for a specific image and visualize with multiple thresholds.
    """
    # Print header
    console.print(
        Panel(
            f"[bold green]OptSARChangeDetection - Predicting for image: {image_id}[/bold green]"
        )
    )

    # Load configuration
    with console.status("[bold green]Loading configuration...") as status:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        # Override some config values with command line arguments
        config_data["batch_size"] = batch_size
        config_data["num_workers"] = num_workers

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        status.update("[bold green]Loading model...")

        # Create model with configuration
        model = MultimodalDamageNet(
            resnet_version=config_data["model"]["resnet_version"],
            freeze_resnet=config_data["model"]["freeze_resnet"],
            optical_channels=config_data["model"]["optical_channels"],
            sar_channels=config_data["model"]["sar_channels"],
            projection_dim=config_data["model"]["projection_dim"],
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        # Load dataset
        status.update(f"[bold green]Looking for image ID: {image_id}...")
        dataset = SimpleDatasetLoader(
            root_dir=data_dir,
            blacklist_path="data/image_labels.txt",  # Default blacklist
            split=split,
            exclude_blacklist=False,  # Include all images when looking for a specific ID
        )

        try:
            # Get the sample with the specified image ID
            sample = dataset.get_by_image_id(image_id)
            console.print(
                f"[bold green]Image found in the {split} dataset[/bold green]"
            )
        except ValueError:
            console.print(
                f"[bold red]Error: Image ID '{image_id}' not found in the {split} dataset[/bold red]"
            )
            raise typer.Exit(code=1)

    # Process the image
    console.print("[bold]Processing image...[/bold]")

    # Get image data
    pre_img = sample["pre_image"]
    post_img = sample["post_image"]
    label = sample["label"]

    # Convert tensors to numpy if needed
    if isinstance(pre_img, torch.Tensor):
        pre_img = pre_img.numpy().transpose(1, 2, 0)
        post_img = post_img.numpy().transpose(1, 2, 0)
        label = label.numpy()

    # Ensure label is 2D
    if label.ndim > 2:
        label = label.squeeze()

    # Create change map
    console.print("[bold]Generating change map...[/bold]")
    change_map = create_change_map(
        model=model,
        pre_img=pre_img,
        post_img=post_img,
        config=config_data,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Define 5 thresholds to evaluate
    thresholds = [0.5, 0.8, 1.0, 1.2, 1.5]

    # Compute metrics for each threshold
    results = []
    label_binary = np.where(label > 1, 1, 0)
    pad_size = (context_patch_size - roi_patch_size) // 2
    change_map_padded = set_border_to_zero(change_map, pad_size)
    label_binary_padded = set_border_to_zero(label_binary, pad_size)

    for thresh in thresholds:
        iou, precision, recall = compute_iou(
            change_map_padded, label_binary_padded, threshold=thresh
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        results.append(
            {
                "threshold": thresh,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    # Create visualization with multiple thresholds
    console.print("[bold]Creating visualizations...[/bold]")

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))

    # Create a custom colormap (blue to red)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list("change_cmap", colors, N=256)

    # Continuous change map (top-left)
    im = axes[0, 0].imshow(change_map, cmap=cmap, vmin=0, vmax=2)
    axes[0, 0].set_title("Change Intensity")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    axes[0, 0].axis("off")

    # Ground truth - our labels
    axes[0, 1].imshow(visualize_damage(label, is_patch=True))
    axes[0, 1].set_title("Ground Truth - patch based")
    axes[0, 1].axis("off")

    # Ground truth dataset (top-middle)
    label_padded = set_border_to_zero(label, pad_size)
    axes[0, 2].imshow(visualize_damage(label_padded))
    axes[0, 2].set_title("Ground Truth - original")
    axes[0, 2].axis("off")

    # Empty plot for balance (top-right)
    axes[0, 3].axis("off")

    # Create a table with metrics
    metrics_table = axes[0, 3].table(
        cellText=[
            [
                f"{r['threshold']:.2f}",
                f"{r['iou']:.3f}",
                f"{r['precision']:.3f}",
                f"{r['recall']:.3f}",
                f"{r['f1']:.3f}",
            ]
            for r in results
        ],
        colLabels=["Threshold", "IoU", "Precision", "Recall", "F1"],
        loc="center",
        cellLoc="center",
    )

    axes[0, 2].set_title("Metrics by Threshold")

    # Binary change maps for different thresholds (bottom row)
    for i, thresh in enumerate(thresholds[:4]):
        binary_map = (change_map > thresh).astype(np.uint8)
        result = results[i]
        axes[1, i].imshow(binary_map, cmap="gray")
        axes[1, i].set_title(
            f'Threshold: {thresh:.2f}\nIoU: {result["iou"]:.3f}, F1: {result["f1"]:.3f}'
        )
        axes[1, i].axis("off")

    # Add overall title
    plt.suptitle(f"Change Detection Results for Image: {image_id}", fontsize=16)
    plt.tight_layout()

    # Save the figure
    viz_path = output_dir / f"{image_id}_change_prediction.png"
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")

    # Save individual files
    # Save continuous change map as TIFF
    with rasterio.open(
        output_dir / f"{image_id}_continuous_change.tif",
        "w",
        driver="GTiff",
        height=change_map.shape[0],
        width=change_map.shape[1],
        count=1,
        dtype=change_map.dtype,
    ) as dst:
        dst.write(change_map, 1)

    # Find best threshold based on F1 score
    best_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
    best_threshold = results[best_idx]["threshold"]

    # Save binary change map with best threshold
    binary_map = (change_map > best_threshold).astype(np.uint8)
    with rasterio.open(
        output_dir / f"{image_id}_binary_change.tif",
        "w",
        driver="GTiff",
        height=binary_map.shape[0],
        width=binary_map.shape[1],
        count=1,
        dtype=binary_map.dtype,
    ) as dst:
        dst.write(binary_map, 1)

    # Print summary
    console.print("[bold green]Prediction complete![/bold green]")
    console.print(f"Visualization saved to: {viz_path}")
    console.print(
        f"Best threshold: {best_threshold:.4f} (F1: {results[best_idx]['f1']:.4f})"
    )

    # Create results table
    results_table = Table(title=f"Metrics for {image_id}")
    results_table.add_column("Threshold", style="cyan")
    results_table.add_column("IoU", style="green")
    results_table.add_column("Precision", style="yellow")
    results_table.add_column("Recall", style="magenta")
    results_table.add_column("F1", style="blue")

    for r in results:
        results_table.add_row(
            f"{r['threshold']:.2f}",
            f"{r['iou']:.4f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
        )

    console.print(results_table)


@infer_app.command("threshold")
def infer_threshold(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to the trained model checkpoint", exists=True
    ),
    config: Path = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
    ),
    metadata_dir: Path = typer.Option(
        "data/metadata_noblacklist", help="Directory containing patch metadata"
    ),
    output_dir: Path = typer.Option(
        "output/threshold_analysis",
        "--output",
        "-o",
        help="Directory to save analysis results",
    ),
    image_cache_size: int = typer.Option(
        50, help="Number of full images to cache in memory"
    ),
    target_neg_ratio: float = typer.Option(
        0.8, min=0.0, max=1.0, help="Target negative pair ratio in validation dataset"
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", "-b", help="Batch size for inference"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help="Device to run inference on",
    ),
    subset_fraction: float = typer.Option(
        1.0, min=0.0, max=1.0, help="Fraction of validation dataset to use"
    ),
    subset_seed: int = typer.Option(42, help="Random seed for subset selection"),
):
    """
    Analyze model performance across different thresholds on the validation dataset.

    This command:
    1. Loads a trained model and validation dataset
    2. Computes optical and SAR features for the entire validation set
    3. Calculates change scores for all sample pairs
    4. Evaluates performance metrics (accuracy, precision, recall, F1) at different thresholds
    5. Generates ROC curves and finds optimal thresholds
    6. Saves all results to the specified output directory
    """
    # Print header
    console.print(
        Panel("[bold green]OptSARChangeDetection - Threshold Analysis[/bold green]")
    )

    # Load configuration
    with console.status("[bold green]Loading configuration...") as status:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration for reference
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

        # Set device
        console.print(f"Using device: [bold]{device}[/bold]")

        # Load validation dataset
        status.update("[bold green]Loading validation dataset...")

        val_dataset = OnTheFlyPatchDataset(
            root_dir=config_data["data"]["root_dir"],
            metadata_dir=metadata_dir,
            split="val",
            transform=None,  # No transforms for validation
            cache_size=image_cache_size,
            subset_fraction=subset_fraction,
            target_neg_ratio=target_neg_ratio,
            seed=subset_seed,
        )

        # Display dataset info
        console.print(
            f"Loaded validation dataset with [bold]{len(val_dataset)}[/bold] samples"
        )
        console.print(
            f"Positive ratio: [bold]{val_dataset.get_pos_neg_ratio():.4f}[/bold]"
        )

        # Create validation sampler and dataloader
        val_sampler = RatioSampler(
            val_dataset,
            batch_size=batch_size,
            neg_ratio=target_neg_ratio,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=config_data["data"]["num_workers"],
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

        # Load model
        status.update("[bold green]Loading model...")

        # Create model with configuration
        model = MultimodalDamageNet(
            resnet_version=config_data["model"]["resnet_version"],
            freeze_resnet=config_data["model"]["freeze_resnet"],
            optical_channels=config_data["model"]["optical_channels"],
            sar_channels=config_data["model"]["sar_channels"],
            projection_dim=config_data["model"]["projection_dim"],
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

    # Compute features and change scores
    console.print("[bold]Computing features and change scores...[/bold]")

    # Initialize storage for features, scores, and labels
    optical_features = []
    sar_features = []
    change_scores = []
    is_positive_labels = []

    # Process validation set
    with torch.no_grad():
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Processing validation samples", total=len(val_loader)
            )

            for batch in val_loader:
                # Move data to device
                pre_patches = batch["pre_patch"].to(device)
                post_patches = batch["post_patch"].to(device)
                is_positive = batch["is_positive"].to(device)

                # Forward pass
                outputs = model(optical=pre_patches, sar=post_patches)

                # Store features and scores
                optical_features.append(outputs["optical_features"].cpu().numpy())
                sar_features.append(outputs["sar_features"].cpu().numpy())
                optical_projected = outputs["optical_projected"].cpu().numpy()
                sar_projected = outputs["sar_projected"].cpu().numpy()

                # Calculate change scores
                if "change_score" in outputs:
                    # Use model's change score if available
                    batch_scores = outputs["change_score"].cpu().numpy()
                else:
                    # Calculate manually from projections
                    optical_norm = optical_projected / np.linalg.norm(
                        optical_projected, axis=1, keepdims=True
                    )
                    sar_norm = sar_projected / np.linalg.norm(
                        sar_projected, axis=1, keepdims=True
                    )
                    similarity = np.sum(optical_norm * sar_norm, axis=1)
                    batch_scores = 1.0 - similarity

                change_scores.append(batch_scores)
                is_positive_labels.append(is_positive.cpu().numpy())

                progress.update(task, advance=1)

    # Concatenate results
    optical_features = np.vstack(optical_features)
    sar_features = np.vstack(sar_features)
    change_scores = np.concatenate(change_scores)
    is_positive_labels = np.concatenate(is_positive_labels).flatten()

    # Save features and scores
    np.save(output_dir / "optical_features.npy", optical_features)
    np.save(output_dir / "sar_features.npy", sar_features)
    np.save(output_dir / "change_scores.npy", change_scores)
    np.save(output_dir / "is_positive_labels.npy", is_positive_labels)

    console.print(
        f"[green]Saved features and scores for [bold]{len(change_scores)}[/bold] samples[/green]"
    )

    # Analyze threshold values
    console.print("[bold]Analyzing threshold values...[/bold]")

    # Define threshold range
    thresholds = np.arange(0, 2.1, 0.1)

    # Initialize metrics storage
    results = {
        "threshold": thresholds.tolist(),
        "accuracy": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "specificity": [],  # True negative rate for ROC curve
        "tpr": [],  # True positive rate for ROC curve
        "fpr": [],  # False positive rate for ROC curve
    }

    # Calculate metrics for each threshold
    for threshold in thresholds:
        # Create binary predictions
        predictions = (change_scores >= threshold).astype(int)

        # Calculate metrics
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            1 - is_positive_labels, predictions, labels=[0, 1]
        ).ravel()

        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        balanced_accuracy = (
            ((tp / (tp + fn)) + (tn / (tn + fp))) / 2
            if (tp + fn) > 0 and (tn + fp) > 0
            else 0
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # ROC curve metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        tpr = recall  # True positive rate = recall
        fpr = 1 - specificity  # False positive rate = 1 - specificity

        # Store results
        results["accuracy"].append(float(accuracy))
        results["balanced_accuracy"].append(float(balanced_accuracy))
        results["precision"].append(float(precision))
        results["recall"].append(float(recall))
        results["f1_score"].append(float(f1))
        results["specificity"].append(float(specificity))
        results["tpr"].append(float(tpr))
        results["fpr"].append(float(fpr))

    # Save results
    with open(output_dir / "threshold_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Find optimal thresholds
    best_accuracy_idx = np.argmax(results["accuracy"])
    best_balanced_idx = np.argmax(results["balanced_accuracy"])
    best_f1_idx = np.argmax(results["f1_score"])

    # Compute AUC for ROC
    auc = sklearn.metrics.auc(results["fpr"], results["tpr"])

    # Print optimal thresholds
    console.print("\n[bold cyan]Optimal thresholds:[/bold cyan]")
    console.print(
        f"  Best accuracy: {thresholds[best_accuracy_idx]:.3f} (Accuracy: {results['accuracy'][best_accuracy_idx]:.4f})"
    )
    console.print(
        f"  Best balanced accuracy: {thresholds[best_balanced_idx]:.3f} (Balanced Accuracy: {results['balanced_accuracy'][best_balanced_idx]:.4f})"
    )
    console.print(
        f"  Best F1 score: {thresholds[best_f1_idx]:.3f} (F1: {results['f1_score'][best_f1_idx]:.4f})"
    )
    console.print(f"  ROC AUC: {auc:.4f}")

    # Create summary
    summary = {
        "best_accuracy_threshold": float(thresholds[best_accuracy_idx]),
        "best_balanced_accuracy_threshold": float(thresholds[best_balanced_idx]),
        "best_f1_threshold": float(thresholds[best_f1_idx]),
        "best_accuracy": float(results["accuracy"][best_accuracy_idx]),
        "best_balanced_accuracy": float(
            results["balanced_accuracy"][best_balanced_idx]
        ),
        "best_f1": float(results["f1_score"][best_f1_idx]),
        "auc": float(auc),
        "num_samples": int(len(change_scores)),
        "positive_ratio": float(np.mean(is_positive_labels)),
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Generate plots
    console.print("[bold]Generating plots...[/bold]")

    # Create a styled figure for the metrics
    plt.figure(figsize=(12, 10))

    # Plot accuracy metrics
    plt.subplot(2, 2, 1)
    plt.plot(
        thresholds, results["accuracy"], label="Accuracy", marker="o", markersize=4
    )
    plt.plot(
        thresholds,
        results["balanced_accuracy"],
        label="Balanced Accuracy",
        marker="s",
        markersize=4,
    )
    plt.axvline(
        x=thresholds[best_balanced_idx], color="green", linestyle="--", alpha=0.7
    )
    plt.grid(True, alpha=0.3)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Metrics")
    plt.legend()

    # Plot precision and recall
    plt.subplot(2, 2, 2)
    plt.plot(
        thresholds, results["precision"], label="Precision", marker="o", markersize=4
    )
    plt.plot(thresholds, results["recall"], label="Recall", marker="s", markersize=4)
    plt.plot(
        thresholds, results["f1_score"], label="F1 Score", marker="^", markersize=4
    )
    plt.axvline(x=thresholds[best_f1_idx], color="red", linestyle="--", alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall, and F1 Score")
    plt.legend()

    # Plot ROC curve
    plt.subplot(2, 2, 3)
    plt.plot(
        results["fpr"],
        results["tpr"],
        label=f"ROC Curve (AUC = {auc:.4f})",
        marker="o",
        markersize=4,
    )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)  # Diagonal line
    plt.grid(True, alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # Plot histogram of change scores
    plt.subplot(2, 2, 4)
    plt.hist(
        [
            change_scores[is_positive_labels == 1],  # Positive samples
            change_scores[is_positive_labels == 0],  # Negative samples
        ],
        bins=30,
        alpha=0.7,
        label=["Positive (no damage)", "Negative (damage)"],
        color=["green", "red"],
    )
    plt.axvline(
        x=thresholds[best_f1_idx],
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"Best F1 threshold: {thresholds[best_f1_idx]:.3f}",
    )
    plt.grid(True, alpha=0.3)
    plt.xlabel("Change Score")
    plt.ylabel("Count")
    plt.title("Distribution of Change Scores")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "threshold_analysis.png", dpi=300)
    plt.close()

    # Also create separate precision-recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(results["recall"], results["precision"], marker="o", markersize=4)
    plt.grid(True, alpha=0.3)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    # Mark best F1 point
    best_recall = results["recall"][best_f1_idx]
    best_precision = results["precision"][best_f1_idx]
    plt.scatter(
        [best_recall],
        [best_precision],
        color="red",
        s=100,
        zorder=5,
        label=f"Best F1: {results['f1_score'][best_f1_idx]:.4f}",
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300)
    plt.close()

    # Create table with threshold values
    threshold_table = Table(title="Threshold Analysis Results")
    threshold_table.add_column("Threshold", style="cyan", justify="center")
    threshold_table.add_column("Accuracy", style="green", justify="center")
    threshold_table.add_column("Balanced Acc", style="green", justify="center")
    threshold_table.add_column("Precision", style="yellow", justify="center")
    threshold_table.add_column("Recall", style="magenta", justify="center")
    threshold_table.add_column("F1 Score", style="blue", justify="center")

    for i, t in enumerate(thresholds):
        threshold_table.add_row(
            f"{t:.2f}",
            f"{results['accuracy'][i]:.4f}",
            f"{results['balanced_accuracy'][i]:.4f}",
            f"{results['precision'][i]:.4f}",
            f"{results['recall'][i]:.4f}",
            f"{results['f1_score'][i]:.4f}",
        )

    console.print(threshold_table)

    # Create results panel with optimal thresholds
    result_panel = Panel(
        f"""[bold green]Threshold Analysis Complete![/bold green]
        
[bold cyan]Optimal Thresholds:[/bold cyan]
  Best accuracy: {thresholds[best_accuracy_idx]:.3f} (Accuracy: {results['accuracy'][best_accuracy_idx]:.4f})
  Best balanced accuracy: {thresholds[best_balanced_idx]:.3f} (Balanced Accuracy: {results['balanced_accuracy'][best_balanced_idx]:.4f})
  Best F1 score: {thresholds[best_f1_idx]:.3f} (F1: {results['f1_score'][best_f1_idx]:.4f})
  ROC AUC: {auc:.4f}
        
[bold]Dataset Information:[/bold]
  Total samples: {len(change_scores)}
  Positive ratio: {np.mean(is_positive_labels):.4f}
        
[bold]Output Location:[/bold]
  {output_dir}
        """,
        title="Threshold Analysis Results",
        expand=False,
    )

    console.print(result_panel)


@infer_app.command("evaluate")
def infer_evaluate(
    model_path: Path = typer.Option(
        ..., "--model", "-m", help="Path to the trained model checkpoint", exists=True
    ),
    config: Path = typer.Option(
        "configs/default.yaml",
        "--config",
        "-c",
        help="Path to configuration file",
        exists=True,
    ),
    data_dir: Path = typer.Option(
        "data/dfc25_track2_trainval",
        "--data",
        "-d",
        help="Directory containing the dataset",
    ),
    split: str = typer.Option(
        "val", "--split", "-s", help="Dataset split to evaluate on"
    ),
    output_dir: Path = typer.Option(
        "output/inference", "--output", "-o", help="Directory to save inference results"
    ),
    blacklist_path: Path = typer.Option(
        "data/image_labels.txt", "--blacklist", help="Path to blacklist file"
    ),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        "-t",
        help="Threshold for change detection (None = auto optimize)",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-l", help="Limit number of images to evaluate"
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help="Device to run inference on",
    ),
    save_vis: bool = typer.Option(
        True,
        "--save-visualizations/--no-save-visualizations",
        help="Save visualization images",
    ),
    batch_size: int = typer.Option(
        8, "--batch-size", "-b", help="Batch size for inference"
    ),
    num_workers: int = typer.Option(
        4, "--workers", "-w", help="Number of workers for data loading"
    ),
):
    """
    Evaluate a trained multimodal change detection model.
    """
    # Print header
    console.print(
        Panel("[bold green]OptSARChangeDetection - Model Evaluation[/bold green]")
    )

    # Load configuration
    with console.status("[bold green]Loading configuration...") as status:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        # Override some config values with command line arguments
        config_data["batch_size"] = batch_size
        config_data["num_workers"] = num_workers

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        with open(output_dir / "inference_config.yaml", "w") as f:
            yaml.dump(config_data, f)

        # Display configuration
        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Model Path", str(model_path))
        config_table.add_row("Data Directory", str(data_dir))
        config_table.add_row("Split", split)
        config_table.add_row("Output Directory", str(output_dir))
        config_table.add_row("Device", device)
        config_table.add_row("Batch Size", str(batch_size))

        if threshold is not None:
            config_table.add_row("Threshold", f"{threshold:.4f}")
        else:
            config_table.add_row("Threshold", "Auto optimize")

        console.print(config_table)

        # Load model
        status.update("[bold green]Loading model...")

        # Create model with configuration
        model = MultimodalDamageNet(
            resnet_version=config_data["model"]["resnet_version"],
            freeze_resnet=config_data["model"]["freeze_resnet"],
            optical_channels=config_data["model"]["optical_channels"],
            sar_channels=config_data["model"]["sar_channels"],
            projection_dim=config_data["model"]["projection_dim"],
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(device)
        model.eval()

        # Load dataset
        status.update("[bold green]Loading dataset...")
        dataset = SimpleDatasetLoader(
            root_dir=data_dir,
            blacklist_path=blacklist_path,
            split=split,
            limit=limit,
            exclude_blacklist=True,  # Exclude blacklisted images
        )

        console.print(f"Loaded dataset with [bold]{len(dataset)}[/bold] images")

    # Find optimal threshold if not provided
    if threshold is None:
        console.print("[bold]Finding optimal threshold...[/bold]")

        # Candidate thresholds to test
        thresholds = np.linspace(0.3, 0.7, 9).tolist()  # [0.3, 0.35, 0.4, ..., 0.7]

        # Containers for threshold evaluation
        results = {
            "threshold": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "f1": [],
        }

        # Limit the number of images for threshold optimization
        num_images = min(len(dataset), 10)  # Use max 10 images

        # Get a subset of images
        image_data = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Loading images for threshold tuning", total=num_images
            )

            for idx in range(num_images):
                sample = dataset[idx]

                pre_img = sample["pre_image"]
                post_img = sample["post_image"]
                label = sample["label"]

                # Convert tensors to numpy if needed
                if isinstance(pre_img, torch.Tensor):
                    pre_img = pre_img.numpy().transpose(1, 2, 0)
                    post_img = post_img.numpy().transpose(1, 2, 0)
                    label = label.numpy()

                # Ensure label is 2D
                if label.ndim > 2:
                    label = label.squeeze()

                image_data.append((pre_img, post_img, label))
                progress.update(task, advance=1)

        # Generate change maps for all images
        change_maps = []

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating change maps", total=len(image_data))

            for pre_img, post_img, _ in image_data:
                change_map = create_change_map(
                    model=model,
                    pre_img=pre_img,
                    post_img=post_img,
                    config=config_data,
                    device=device,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
                change_maps.append(change_map)
                progress.update(task, advance=1)

        # Evaluate each threshold
        threshold_table = Table(title="Threshold Evaluation")
        threshold_table.add_column("Threshold", style="cyan", justify="center")
        threshold_table.add_column("IoU", style="green", justify="center")
        threshold_table.add_column("Precision", style="yellow", justify="center")
        threshold_table.add_column("Recall", style="magenta", justify="center")
        threshold_table.add_column("F1 Score", style="blue", justify="center")

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating thresholds", total=len(thresholds))

            for thresh in thresholds:
                # Initialize metrics for this threshold
                ious, precisions, recalls = [], [], []

                # Evaluate each image
                for change_map, (_, _, label) in zip(change_maps, image_data):
                    label_binary = np.where(label > 1, 1, 0)
                    iou, precision, recall = compute_iou(
                        change_map, label_binary, threshold=thresh
                    )
                    ious.append(iou)
                    precisions.append(precision)
                    recalls.append(recall)

                # Calculate averages
                avg_iou = np.mean(ious)
                avg_precision = np.mean(precisions)
                avg_recall = np.mean(recalls)
                f1 = (
                    2 * avg_precision * avg_recall / (avg_precision + avg_recall)
                    if (avg_precision + avg_recall) > 0
                    else 0
                )

                # Store results
                results["threshold"].append(float(thresh))
                results["iou"].append(float(avg_iou))
                results["precision"].append(float(avg_precision))
                results["recall"].append(float(avg_recall))
                results["f1"].append(float(f1))

                # Add to table
                threshold_table.add_row(
                    f"{thresh:.3f}",
                    f"{avg_iou:.3f}",
                    f"{avg_precision:.3f}",
                    f"{avg_recall:.3f}",
                    f"{f1:.3f}",
                )

                progress.update(task, advance=1)

        # Print threshold evaluation table
        console.print(threshold_table)

        # Find optimal threshold based on F1 score
        best_idx = np.argmax(results["f1"])
        threshold = results["threshold"][best_idx]

        console.print(
            f"[bold green]Optimal threshold: {threshold:.4f} with F1: {results['f1'][best_idx]:.4f}[/bold green]"
        )

    # Evaluate model with the (potentially optimized) threshold
    console.print(f"[bold]Evaluating model with threshold {threshold:.4f}[/bold]")

    # Container for results
    results = {
        "image_id": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }

    # Limit the number of images to process if specified
    num_images = min(len(dataset), limit) if limit else len(dataset)

    # Create progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating images", total=num_images)

        for idx in range(num_images):
            # Load image data
            sample = dataset[idx]
            image_id = sample["image_id"]

            pre_img = sample["pre_image"]
            post_img = sample["post_image"]
            label = sample["label"]

            # Convert tensors to numpy if needed
            if isinstance(pre_img, torch.Tensor):
                pre_img = pre_img.numpy().transpose(1, 2, 0)
                post_img = post_img.numpy().transpose(1, 2, 0)
                label = label.numpy()

            # Ensure label is 2D
            if label.ndim > 2:
                label = label.squeeze()

            # Create change map
            change_map = create_change_map(
                model=model,
                pre_img=pre_img,
                post_img=post_img,
                config=config_data,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # Compute metrics
            label_binary = np.where(label > 1, 1, 0)
            iou, precision, recall = compute_iou(
                change_map, label_binary, threshold=threshold
            )

            # Store results
            results["image_id"].append(image_id)
            results["iou"].append(float(iou))
            results["precision"].append(float(precision))
            results["recall"].append(float(recall))

            # Save visualization if requested
            if save_vis:
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(parents=True, exist_ok=True)

                # Visualize change map
                visualize_change_map(
                    change_map,
                    threshold=threshold,
                    output_path=vis_dir / f"{image_id}_change_map.png",
                )

                # Save change map as binary TIFF
                binary_map = (change_map > threshold).astype(np.uint8)
                with rasterio.open(
                    vis_dir / f"{image_id}_binary_change.tif",
                    "w",
                    driver="GTiff",
                    height=binary_map.shape[0],
                    width=binary_map.shape[1],
                    count=1,
                    dtype=binary_map.dtype,
                ) as dst:
                    dst.write(binary_map, 1)

                # Save continuous change map as TIFF
                with rasterio.open(
                    vis_dir / f"{image_id}_continuous_change.tif",
                    "w",
                    driver="GTiff",
                    height=change_map.shape[0],
                    width=change_map.shape[1],
                    count=1,
                    dtype=change_map.dtype,
                ) as dst:
                    dst.write(change_map, 1)

            progress.update(task, advance=1)

    # Calculate average metrics
    avg_iou = np.mean(results["iou"])
    avg_precision = np.mean(results["precision"])
    avg_recall = np.mean(results["recall"])
    f1_score = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )

    # Create summary
    summary = {
        "avg_iou": float(avg_iou),
        "avg_precision": float(avg_precision),
        "avg_recall": float(avg_recall),
        "f1_score": float(f1_score),
        "threshold": float(threshold),
        "num_images": num_images,
    }

    # Save detailed results
    with open(output_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create results table
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("IoU", f"{avg_iou:.4f}")
    results_table.add_row("Precision", f"{avg_precision:.4f}")
    results_table.add_row("Recall", f"{avg_recall:.4f}")
    results_table.add_row("F1 Score", f"{f1_score:.4f}")
    results_table.add_row("Threshold", f"{threshold:.4f}")
    results_table.add_row("Number of Images", str(num_images))

    console.print(results_table)

    # Final message
    console.print(
        Panel(
            f"[bold green]Evaluation complete! Results saved to {output_dir}[/bold green]"
        )
    )


if __name__ == "__main__":
    app()
