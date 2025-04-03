import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import sklearn.metrics
import torch
import torch.optim as optim
import typer
import yaml
from matplotlib.colors import LinearSegmentedColormap
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.traceback import install
from torch.utils.data import DataLoader

from data.dataset_patchonthefly import OnTheFlyPatchDataset
from data.preprocess_metadata import SimpleDatasetLoader
from data.transforms import get_transform
from inference.change_map_generator import (
    compute_iou,
    create_change_map,
    set_border_to_zero,
    visualize_damage,
)
from losses.contrastive_loss import InfoNCEContrastiveLoss
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
    with console.status("[bold green]Loading configuration...") as status:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        config_table = Table(title="Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="green")

        for category, params in config_data.items():
            if isinstance(params, dict):
                for param, value in params.items():
                    config_table.add_row(f"{category}.{param}", str(value))
            else:
                config_table.add_row(category, str(params))

        console.print(config_table)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        console.print(f"Using device: [bold]{device}[/bold]")

        status.update("[bold green]Loading datasets...")

        train_transform = get_transform("train") if use_transforms else None
        val_transform = get_transform("val") if use_transforms else None

        # create datasets
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

        status.update("[bold green]Creating data loaders...")

        # create samplers
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

        # create dataloaders
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

        status.update("[bold green]Building model...")

        # create model
        model = MultimodalDamageNet(
            resnet_version=config_data["model"]["resnet_version"],
            freeze_resnet=config_data["model"]["freeze_resnet"],
            optical_channels=config_data["model"]["optical_channels"],
            sar_channels=config_data["model"]["sar_channels"],
            projection_dim=config_data["model"]["projection_dim"],
        )

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

        console.print(Panel("[bold green]Starting training...[/bold green]"))

    # train model
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
    window_method: str = typer.Option(
        "classic", "--window_method", "-w_met", help="Window method for inference"
    ),
    window_power: float = typer.Option(
        2.0, "--window_power", "-w_pow", help="Window power for inference"
    ),
):
    """
    Predict change map for a specific image and visualize with multiple thresholds.
    """
    console.print(
        Panel(
            f"[bold green]OptSARChangeDetection - Predicting for image: {image_id}[/bold green]"
        )
    )

    # Load configuration
    with console.status("[bold green]Loading configuration...") as status:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        config_data["batch_size"] = batch_size
        config_data["num_workers"] = num_workers

        # output directory
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
            blacklist_path="data/image_labels.txt",
            split=split,
            exclude_blacklist=False,
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

    console.print("[bold]Processing image...[/bold]")

    pre_img = sample["pre_image"]
    post_img = sample["post_image"]
    label = sample["label"]

    if isinstance(pre_img, torch.Tensor):
        pre_img = pre_img.numpy().transpose(1, 2, 0)
        post_img = post_img.numpy().transpose(1, 2, 0)
        label = label.numpy()

    if label.ndim > 2:
        label = label.squeeze()

    # change map
    console.print("[bold]Generating change map...[/bold]")
    change_map = create_change_map(
        model=model,
        pre_img=pre_img,
        post_img=post_img,
        config=config_data,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        window_method=window_method,
        window_power=window_power,
    )

    # 5 thresholds to evaluate
    thresholds = [0.8, 1.0, 1.25, 1.5]

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

    console.print("[bold]Creating visualizations...[/bold]")

    fig, axes = plt.subplots(2, 4, figsize=(18, 12))

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

    plt.suptitle(f"Change Detection Results for Image: {image_id}", fontsize=16)
    plt.tight_layout()

    viz_path = output_dir / f"{image_id}_change_prediction.png"
    plt.savefig(viz_path, dpi=300, bbox_inches="tight")

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

    # best threshold based on F1 score
    best_idx = max(range(len(results)), key=lambda i: results[i]["f1"])
    best_threshold = results[best_idx]["threshold"]

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

    console.print("[bold green]Prediction complete![/bold green]")
    console.print(f"Visualization saved to: {viz_path}")
    console.print(
        f"Best threshold: {best_threshold:.4f} (F1: {results[best_idx]['f1']:.4f})"
    )

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
    console.print(
        Panel("[bold green]OptSARChangeDetection - Threshold Analysis[/bold green]")
    )

    with console.status("[bold green]Loading configuration...") as status:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)

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

    console.print("[bold]Computing features and change scores...[/bold]")

    optical_features = []
    sar_features = []
    change_scores = []
    is_positive_labels = []

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
                    batch_scores = outputs["change_score"].cpu().numpy()
                else:
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

    optical_features = np.vstack(optical_features)
    sar_features = np.vstack(sar_features)
    change_scores = np.concatenate(change_scores)
    is_positive_labels = np.concatenate(is_positive_labels).flatten()

    np.save(output_dir / "optical_features.npy", optical_features)
    np.save(output_dir / "sar_features.npy", sar_features)
    np.save(output_dir / "change_scores.npy", change_scores)
    np.save(output_dir / "is_positive_labels.npy", is_positive_labels)

    console.print(
        f"[green]Saved features and scores for [bold]{len(change_scores)}[/bold] samples[/green]"
    )

    console.print("[bold]Analyzing threshold values...[/bold]")

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
        predictions = (change_scores >= threshold).astype(int)

        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(
            1 - is_positive_labels, predictions, labels=[0, 1]
        ).ravel()

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

        results["accuracy"].append(float(accuracy))
        results["balanced_accuracy"].append(float(balanced_accuracy))
        results["precision"].append(float(precision))
        results["recall"].append(float(recall))
        results["f1_score"].append(float(f1))
        results["specificity"].append(float(specificity))
        results["tpr"].append(float(tpr))
        results["fpr"].append(float(fpr))

    with open(output_dir / "threshold_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Find optimal thresholds
    best_accuracy_idx = np.argmax(results["accuracy"])
    best_balanced_idx = np.argmax(results["balanced_accuracy"])
    best_f1_idx = np.argmax(results["f1_score"])

    # Compute AUC for ROC
    auc = sklearn.metrics.auc(results["fpr"], results["tpr"])

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

    console.print("[bold]Generating plots...[/bold]")

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


if __name__ == "__main__":
    app()
