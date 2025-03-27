import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F

from data.transforms import get_transform
from data.dataset_patchonthefly import OnTheFlyPatchDataset
from rich import print as rprintpip
from rich.console import Console


import argparse
import os
from pathlib import Path
import numpy as np
import torch
from tkinter import (
    BOTH, BOTTOM, HORIZONTAL, LEFT, RIGHT, TOP, Button, Entry, Frame,
    Label, OptionMenu, Scale, StringVar, Tk, X, Y
)
from models.pseudo_siamese import (
    MultimodalDamageNet,  
)


from data.dataset_patches import PreprocessedPatchDataset
from data.transforms import get_transform
import yaml
import h5py
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm

console = Console()

class ModelVisualizer:
    def __init__(self, model, dataloader, device="cuda", output_dir="visualizations", normalize=False):
        """
        Initialize the model visualizer.
        
        Args:
            model: The MultimodalDamageNet model
            dataloader: DataLoader containing the dataset
            device: Device to run the model on
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create empty lists to store features and labels
        self.sar_features = []
        self.optical_features = []
        self.sar_projected = []
        self.optical_projected = []
        self.is_positive_labels = []
        self.has_building_labels = []
        self.normalize = normalize
        self.chosen_methods = ['pca', 'tsne']

    def collect_features(self):
        """
        Collect features from the model for visualization.

        """
        self.model.eval()
        
        print("\nCollecting features...")
        # Iterate through the dataloader and collect features
        with torch.no_grad():
            with tqdm(
                self.dataloader
            ) as pbar:
                for _, batch in enumerate(pbar):
                    pre_patches = batch["pre_patch"].to(self.device)
                    post_patches = batch["post_patch"].to(self.device)
                    is_positive = batch["is_positive"].to(self.device)
                    has_building = batch["has_building"].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(optical=pre_patches, sar=post_patches)
                    
                    # Store features and labels
                    self.sar_features.append(outputs["sar_features"].cpu().numpy())
                    self.optical_features.append(outputs["optical_features"].cpu().numpy())
                    self.is_positive_labels.append(is_positive.cpu().numpy())
                    self.has_building_labels.append(has_building.cpu().numpy())
                    if self.normalize:
                        self.sar_projected.append(F.normalize(outputs["sar_projected"], dim=1).cpu().numpy())
                        self.optical_projected.append(F.normalize(outputs["optical_projected"], dim=1).cpu().numpy())
                    else:
                        self.sar_projected.append(outputs["sar_projected"].cpu().numpy())
                        self.optical_projected.append(outputs["optical_projected"].cpu().numpy())
                    
        
        # Concatenate all batches
        self.sar_features = np.concatenate(self.sar_features)
        self.optical_features = np.concatenate(self.optical_features)
        self.sar_projected = np.concatenate(self.sar_projected)
        self.optical_projected = np.concatenate(self.optical_projected)
        self.is_positive_labels = np.concatenate(self.is_positive_labels)
        self.has_building_labels = np.concatenate(self.has_building_labels)
        
    def _apply_dimension_reduction(self, data, method='pca', n_components=2, perplexity=30, n_neighbors=15):
        """
        Apply dimension reduction technique to the data.
        
        Args:
            data: Input data for dimension reduction
            method: 'pca', 'tsne'
            n_components: Number of components for dimension reduction
            perplexity: Perplexity parameter for t-SNE
            
        Returns:
            Reduced data
        """
        # Adjust perplexity if needed for small samples
        n_samples = data.shape[0]
        adjusted_perplexity = min(perplexity, n_samples // 3)
        adjusted_perplexity = max(adjusted_perplexity, 5)  # Minimum perplexity
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            return reducer.fit_transform(data)
        
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=adjusted_perplexity, random_state=42)
            return reducer.fit_transform(data)
        
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    def create_visualizations(self, save_to_file=True, show_labels=False, not_showing_building_labels=False, use_tsne=False):
        """
        Create all visualizations for the collected features.
        
        Args:
            save_to_file: Whether to save the visualizations to files
            show_labels: Whether to show point labels (indices)
        """
        # Generate all visualizations
        self._visualize_encoders(save_to_file, show_labels, not_showing_building_labels, use_tsne)
        self._visualize_projection_space(save_to_file, show_labels, not_showing_building_labels, use_tsne)
    
    def _visualize_encoders(self, save_to_file=True, show_labels=False, not_showing_building_labels=False, use_tsne=False):
        """
        Visualize the SAR encoder output space with different dimensionality reduction techniques.
        
        Args:
            save_to_file: Whether to save the visualizations to files
            show_labels: Whether to show point labels (indices)
        """
        methods = ['pca', 'tsne'] if use_tsne else ['pca']

        n_samples = self.sar_features.shape[0]
        
        for method in methods:
            print(f"\nCreating {method.upper()} visualization for SAR encoder outputs...")
            try:
                print(f"Reducing dimensions using {method.upper()}...")
                reduced_data = self._apply_dimension_reduction(self.sar_features, method=method)
                # save reduced data for later use
                # first create the directory if it doesn't exist
                (self.output_dir / 'sar_encoder').mkdir(exist_ok=True, parents=True)
                # then save the reduced data
                np.save(self.output_dir / f'sar_encoder/reduced_data_{method}.npy', reduced_data)
                
                print(f"Creating {method.upper()} visualization...\n")
                plt.figure(figsize=(10, 8))
                
                # Ensure is_positive_labels is flattened to 1D
                is_positive_flat = self.is_positive_labels.flatten()
                has_building_flat = self.has_building_labels.flatten()
                
                # Create scatter plots by damage label (1=intact, 0=damaged)
                intact_mask = is_positive_flat == 1
                damaged_mask = is_positive_flat == 0
                
                # Create scatter plot by has_building label (1=has building, 0=no building)
                sar_building_intact_mask = (has_building_flat == 1) & (is_positive_flat == 1)
                sar_building_damaged_mask = (has_building_flat == 1) & (is_positive_flat == 0)
                sar_no_building_mask = has_building_flat == 0
                # save labels for later use
                # first create the directory if it doesn't exist
                (self.output_dir / 'sar_encoder').mkdir(exist_ok=True, parents=True)
                # then save the labels
                np.save(self.output_dir / 'sar_encoder/labels.npy', {
                    'intact_mask': intact_mask,
                    'damaged_mask': damaged_mask,
                    'sar_building_intact_mask': sar_building_intact_mask,
                    'sar_building_damaged_mask': sar_building_damaged_mask,
                    'sar_no_building_mask': sar_no_building_mask
                })
                
                if not_showing_building_labels:
                    plt.scatter(
                        reduced_data[intact_mask, 0], 
                        reduced_data[intact_mask, 1], 
                        c='blue',
                        marker='o', 
                        alpha=0.7,
                        label='No Change - SAR'
                    )
                    
                    plt.scatter(
                        reduced_data[damaged_mask, 0], 
                        reduced_data[damaged_mask, 1], 
                        c='red',
                        marker='o', 
                        alpha=0.7,
                        label='Change - SAR'
                    )
                else:
                    plt.scatter(
                        reduced_data[sar_building_intact_mask, 0], 
                        reduced_data[sar_building_intact_mask, 1], 
                        c='blue',
                        marker='o', 
                        alpha=0.7,
                        label='No Change - SAR Building '
                    )
                    plt.scatter(
                        reduced_data[sar_building_damaged_mask, 0],
                        reduced_data[sar_building_damaged_mask, 1],
                        c='red',
                        marker='o',
                        alpha=0.7,
                        label='Change - SAR Building '
                    )
                    plt.scatter(
                        reduced_data[sar_no_building_mask, 0],
                        reduced_data[sar_no_building_mask, 1],
                        c='green',
                        marker='o',
                        alpha=0.7,
                        label='SAR No Building'
                    )
                
                plt.legend(loc='best')
                plt.title(f'SAR Encoder Features - {method.upper()} (n={n_samples})')
                plt.xlabel(f'{method.upper()} Component 1')
                plt.ylabel(f'{method.upper()} Component 2')
                
                # Add sample indices as annotations, only if requested
                if show_labels:
                    for i, (x, y) in enumerate(reduced_data):
                        plt.annotate(
                            str(i),
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 5),
                            ha='center',
                            fontsize=8
                        )
                
                if save_to_file:
                    plt.savefig(self.output_dir / f'sar_encoder_{method}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
                
            except Exception as e:
                print(f"Error creating {method.upper()} visualization: {e}")
    
    def _visualize_projection_space(self, save_to_file=True, show_labels=False, not_showing_building_labels=False, use_tsne=False):
        """
        Visualize the SAR and optical projection space with different dimensionality reduction techniques.
        
        Args:
            save_to_file: Whether to save the visualizations to files
            show_labels: Whether to show point labels (indices)
        """
        methods = ['pca', 'tsne'] if use_tsne else ['pca']
        
        # Combine optical and SAR projections
        combined_projections = np.concatenate([self.optical_projected, self.sar_projected])
        
        # Create modality labels (0=optical, 1=SAR)
        n_optical = len(self.optical_projected)
        n_sar = len(self.sar_projected)
        optical_labels = np.zeros(n_optical)
        sar_labels = np.ones(n_sar)
        modality_labels = np.concatenate([optical_labels, sar_labels])
        
        # Ensure is_positive_labels is flattened to 1D
        is_positive_flat = self.is_positive_labels.flatten()
        has_building_flat = self.has_building_labels.flatten()
                
        # Create scatter plot by has_building label (1=has building, 0=no building)
        sar_building_intact_mask = (has_building_flat == 1) & (is_positive_flat == 1)
        sar_building_damaged_mask = (has_building_flat == 1) & (is_positive_flat == 0)
        sar_no_building_mask = has_building_flat == 0
        
        # Add damage information
        # Note: optical images (pre-event) are always intact, while SAR may be intact or damaged
        optical_damage_labels = np.ones(n_optical)  # All optical are intact
        damage_labels = np.concatenate([optical_damage_labels, is_positive_flat])
        
        optical_labels = np.zeros(n_optical)  # All optical are intact
        sar_no_building_labels = np.concatenate([optical_labels, sar_no_building_mask])
        sar_building_change_labels = np.concatenate([optical_labels, sar_building_damaged_mask])
        sar_building_no_change_labels = np.concatenate([optical_labels, sar_building_intact_mask])
        # save labels for later use
        # first create the directory if it doesn't exist
        (self.output_dir / 'projection_space').mkdir(exist_ok=True, parents=True)
        # then save the labels
        np.save(self.output_dir / 'projection_space/labels.npy', {
            'optical_labels': optical_labels,
            'sar_no_building_labels': sar_no_building_labels,
            'sar_building_change_labels': sar_building_change_labels,
            'sar_building_no_change_labels': sar_building_no_change_labels
        })
        n_samples = len(self.sar_features)
        
        if not_showing_building_labels:
            for method in methods:
                if method in self.chosen_methods:
                    print(f"Creating {method.upper()} visualization for projection spaces...")
                    try:
                        print(f"Reducing dimensions using {method.upper()}...")
                        reduced_data = self._apply_dimension_reduction(combined_projections, method=method)
                        # save reduced data for later use
                        # first create the directory if it doesn't exist
                        (self.output_dir / 'projection_space').mkdir(exist_ok=True, parents=True)
                        np.save(self.output_dir / f'projection_space/reduced_data_{method}.npy', reduced_data)

                        print(f"Creating {method.upper()} visualization...")
                        # 1. Plot by modality and damage together
                        plt.figure(figsize=(12, 9))
                        
                        # Separate data by modality and damage status
                        optical_indices = np.where(modality_labels == 0)[0]
                        sar_intact_indices = np.where((modality_labels == 1) & (damage_labels == 1))[0]
                        sar_damaged_indices = np.where((modality_labels == 1) & (damage_labels == 0))[0]
                        
                        # Plot each group with different markers and colors
                        plt.scatter(
                            reduced_data[optical_indices, 0],
                            reduced_data[optical_indices, 1],
                            marker='^',  # Triangle for optical
                            color='blue',
                            alpha=0.7,
                            label='Optical'
                        )
                        
                        plt.scatter(
                            reduced_data[sar_intact_indices, 0],
                            reduced_data[sar_intact_indices, 1],
                            marker='o',  # Circle for SAR
                            color='green',
                            alpha=0.7,
                            label='No Change - SAR'
                        )
                        
                        plt.scatter(
                            reduced_data[sar_damaged_indices, 0],
                            reduced_data[sar_damaged_indices, 1],
                            marker='o',  # Circle for SAR
                            color='red',
                            alpha=0.7,
                            label='Change - SAR'
                        )
                        
                        plt.legend(loc='best')
                        plt.title(f'Projection Space - {method.upper()} (n={n_samples*2})')
                        plt.xlabel(f'{method.upper()} Component 1')
                        plt.ylabel(f'{method.upper()} Component 2')
                        
                        # Add point labels only if requested
                        if show_labels:
                            for i, (x, y) in enumerate(reduced_data):
                                # First half are optical, second half are SAR
                                if i < n_optical:
                                    label = f"{i}O"  # Optical
                                else:
                                    label = f"{i-n_optical}S"  # SAR
                                    
                                plt.annotate(
                                    label,
                                    (x, y),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha='center',
                                    fontsize=8
                                )
                        
                        if save_to_file:
                            plt.savefig(self.output_dir / f'projection_space_combined_{method}.png', dpi=300, bbox_inches='tight')
                            plt.close()
                        else:
                            plt.show()
                            
                        print(f"Created {method.upper()} visualization for projection spaces")
                        
    
                    except Exception as e:
                        print(f"Error creating {method.upper()} visualization: {e}")
        else:
            for method in methods:
                if method in self.chosen_methods:
                    print(f"Creating {method.upper()} visualization for projection spaces...")
                    try:
                        print(f"Reducing dimensions using {method.upper()}...")
                        reduced_data = self._apply_dimension_reduction(combined_projections, method=method)
                        # save reduced data for later use
                        # first create the directory if it doesn't exist
                        (self.output_dir / 'projection_space').mkdir(exist_ok=True, parents=True)
                        np.save(self.output_dir / f'projection_space/reduced_data_{method}.npy', reduced_data)
                        
                        print(f"\nCreating {method.upper()} visualization...")
                        # 1. Plot by modality and damage together
                        plt.figure(figsize=(12, 9))
                        
                        # Separate data by modality and damage status
                        optical_indices = np.where(modality_labels == 0)[0]

                        sar_no_building_indices = np.where((modality_labels == 1) & (sar_no_building_labels == 1))[0]
                        change_sar_building_indices = np.where((modality_labels == 1) & (sar_building_change_labels == 1))[0]
                        no_change_sar_building_indices = np.where((modality_labels == 1) & (sar_building_no_change_labels == 1))[0]

                        # Plot each group with different markers and colors
                        plt.scatter(
                            reduced_data[optical_indices, 0],
                            reduced_data[optical_indices, 1],
                            marker='^',  # Triangle for optical
                            color='blue',
                            alpha=0.7,
                            label='Optical'
                        )
                        
                        plt.scatter(
                            reduced_data[no_change_sar_building_indices, 0],
                            reduced_data[no_change_sar_building_indices, 1],
                            marker='o',  # Circle for SAR
                            color='green',
                            alpha=0.7,
                            label='No Change - SAR Building'
                        )
                        
                        plt.scatter(
                            reduced_data[change_sar_building_indices, 0],
                            reduced_data[change_sar_building_indices, 1],
                            marker='o',  # Circle for SAR
                            color='red',
                            alpha=0.7,
                            label='Change - SAR Building'
                        )
                        
                        plt.scatter(
                            reduced_data[sar_no_building_indices, 0],
                            reduced_data[sar_no_building_indices, 1],
                            marker='o',  # Circle for SAR
                            color='black',
                            alpha=0.7,
                            label='SAR No Building'
                        )
                        
                        plt.legend(loc='best')
                        plt.title(f'Projection Space - {method.upper()} (n={n_samples*2})')
                        plt.xlabel(f'{method.upper()} Component 1')
                        plt.ylabel(f'{method.upper()} Component 2')
                        
                        # Add point labels only if requested
                        if show_labels:
                            for i, (x, y) in enumerate(reduced_data):
                                # First half are optical, second half are SAR
                                if i < n_optical:
                                    label = f"{i}O"  # Optical
                                else:
                                    label = f"{i-n_optical}S"  # SAR
                                    
                                plt.annotate(
                                    label,
                                    (x, y),
                                    textcoords="offset points",
                                    xytext=(0, 5),
                                    ha='center',
                                    fontsize=8
                                )
                        
                        if save_to_file:
                            plt.savefig(self.output_dir / f'projection_space_combined_{method}.png', dpi=300, bbox_inches='tight')
                            plt.close()
                        else:
                            plt.show()
                            
                        print(f"Created {method.upper()} visualization for projection spaces")
                        
    
                    except Exception as e:
                        print(f"Error creating {method.upper()} visualization: {e}")
                                        


def main():
    parser = argparse.ArgumentParser(description="Visualize model outputs")
    parser.add_argument("--config", type=Path, default="configs/default.yaml", 
                      help="Path to configuration file")
    parser.add_argument("--model_path", type=str, default="output/patch_contrastive/best_model.pth", 
                      help="Path to the saved model")
    parser.add_argument("--config_path", type=str, default="configs/default.yaml", 
                      help="Path to the model config file")
    parser.add_argument("--output_dir", type=str, default="output/features_explorer", 
                      help="Directory to save output")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cuda/cpu)")
    parser.add_argument("--show_labels", action="store_true", help="Show point labels in static visualizations")
    parser.add_argument("--not_showing_building_labels", action="store_true", help="Show building labels in static visualizations")
    # using tsne or not
    parser.add_argument("--use_tsne", action="store_true", help="Use t-SNE for visualization")

    # Dataset parameters
    parser.add_argument("--patch_dir", type=str, default="data/processed_patches",
                    help="Directory containing processed patches")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata-noblacklist-turkey",
                    help="Directory containing metadata")
    parser.add_argument("--split", type=str, default="val",
                    help="Dataset split to use (train/val/test)")
    parser.add_argument("--cache_size", type=int, default=50,
                    help="Number of samples to cache in memory")
    parser.add_argument("--subset_fraction", type=float, default=0.1,
                    help="Fraction of the dataset to use (0.0-1.0)")
    parser.add_argument("--target_neg_ratio", type=float, default=None,
                    help="Negative sample ratio for sampling")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for dataset shuffling and sampling")
    parser.add_argument("--use_transform", type=bool, default=False,
                    help="Whether to apply data augmentation transforms")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize feature vectors before visualization",
    )

    
    args = parser.parse_args()

    # Load configuration
    with console.status("[bold green]Loading configuration...") as status:
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)
    
    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = "cpu"
        
    # Create model with default parameters
    with open("configs/default.yaml", "r") as f:
                config = yaml.safe_load(f)
                
    model = MultimodalDamageNet(
                resnet_version=config["model"]["resnet_version"],
                freeze_resnet=config["model"]["freeze_resnet"],
                optical_channels=config["model"]["optical_channels"],
                sar_channels=config["model"]["sar_channels"],
                projection_dim=config["model"]["projection_dim"],
                )
    # Try loading with explicit weights_only=False
    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(args.device)
    print(f"Model loaded from {args.model_path}")
             
    # Create dataset and dataloader
    val_transform = get_transform("val") if args.use_transform else None
    val_dataset = OnTheFlyPatchDataset(
        root_dir=config_data["data"]["root_dir"],
        metadata_dir=args.metadata_dir,
        split="val",
        transform=val_transform,
        cache_size=args.cache_size,
        subset_fraction=args.subset_fraction,
        target_neg_ratio=args.target_neg_ratio,
        seed=args.seed,
    )


    dataloader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )    
    
    print(f"Validation dataset loaded with {len(val_dataset)} samples")
    
    # Generate static visualizations
    visualizer = ModelVisualizer(
        model, 
        dataloader, 
        device=args.device, 
        output_dir=args.output_dir,
        normalize=args.normalize,
    )
    visualizer.collect_features()
    
    visualizer.create_visualizations(save_to_file=True, show_labels=args.show_labels, not_showing_building_labels=args.not_showing_building_labels, use_tsne=args.use_tsne)
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()