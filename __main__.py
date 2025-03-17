from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import typer
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from torch.utils.data import DataLoader

from data.dataset_patches import PreprocessedPatchDataset
from data.dataset_patchonthefly import OnTheFlyPatchDataset
from data.transforms import get_transform
from losses.contrastive_loss import InfoNCEContrastiveLoss, SupervisedContrastiveLoss
from models.pseudo_siamese import MultimodalDamageNet
from sampler.sampler import RatioSampler
from trainer.trainer import ContrastiveTrainer

app = typer.Typer(help="OptSARChangeDetection - Multimodal Change Detection Framework")
console = Console()

train_app = typer.Typer(help="Training commands")
infer_app = typer.Typer(help="Inference commands")

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
        ..., help="Path to the trained model checkpoint", exists=True
    ),
    config: Path = typer.Option(
        "configs/default.yaml", help="Path to configuration file", exists=True
    ),
    input_dir: Path = typer.Option(
        ..., help="Directory containing input images (pre and post event)"
    ),
    output_dir: Path = typer.Option(
        "output/predictions", help="Directory to save prediction results"
    ),
    threshold: float = typer.Option(
        0.5, min=0.0, max=1.0, help="Threshold for change detection"
    ),
):
    """
    Run inference on new image pairs using a trained model.
    """
    console.print(
        Panel(
            "[bold yellow]Inference functionality not implemented yet. Coming soon![/bold yellow]"
        )
    )

    # This will be implemented later - placeholder for now
    console.print(f"Model path: {model_path}")
    console.print(f"Config path: {config}")
    console.print(f"Input directory: {input_dir}")
    console.print(f"Output directory: {output_dir}")
    console.print(f"Threshold: {threshold}")


if __name__ == "__main__":
    app()
