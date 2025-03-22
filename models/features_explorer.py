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
    MultimodalDamageNet,  # Using your original model with minimal changes
)

# import sys
# sys.path.append('../')  # Add parent directory to path


from data.dataset_patches import PreprocessedPatchDataset
from data.transforms import get_transform
import yaml
import h5py
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
# matplotlib.use("TkAgg")  # Use TkAgg backend

console = Console()

# Optional: Import UMAP if available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with 'pip install umap-learn' for UMAP visualizations.")

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
        self.normalize = normalize
        
    def collect_features(self, max_batches=None):
        """
        Collect features from the model for visualization.
        
        Args:
            max_batches: Maximum number of batches to process (None for all)
        """
        self.model.eval()
        batch_count = 0
        
        with torch.no_grad():
            for batch in self.dataloader:
                pre_patches = batch["pre_patch"].to(self.device)
                post_patches = batch["post_patch"].to(self.device)
                is_positive = batch["is_positive"].to(self.device)
                
                # Forward pass
                outputs = self.model(optical=pre_patches, sar=post_patches)
                
                # Store features and labels
                self.sar_features.append(outputs["sar_features"].cpu().numpy())
                self.optical_features.append(outputs["optical_features"].cpu().numpy())
                if self.normalize:
                    self.sar_projected.append(F.normalize(outputs["sar_projected"], dim=1).cpu().numpy())
                    self.optical_projected.append(F.normalize(outputs["optical_projected"], dim=1).cpu().numpy())
                else:
                    self.sar_projected.append(outputs["sar_projected"].cpu().numpy())
                    self.optical_projected.append(outputs["optical_projected"].cpu().numpy())
                self.is_positive_labels.append(is_positive.cpu().numpy())
                
                batch_count += 1
                if max_batches is not None and batch_count >= max_batches:
                    break
        
        # Concatenate all batches
        self.sar_features = np.concatenate(self.sar_features)
        self.optical_features = np.concatenate(self.optical_features)
        self.sar_projected = np.concatenate(self.sar_projected)
        self.optical_projected = np.concatenate(self.optical_projected)
        self.is_positive_labels = np.concatenate(self.is_positive_labels)
        
        print(f"Collected features from {len(self.sar_features)} samples")
    
    def _apply_dimension_reduction(self, data, method='pca', n_components=2, perplexity=30, n_neighbors=15):
        """
        Apply dimension reduction technique to the data.
        
        Args:
            data: Input data for dimension reduction
            method: 'pca', 'tsne', or 'umap'
            n_components: Number of components for dimension reduction
            perplexity: Perplexity parameter for t-SNE
            n_neighbors: Number of neighbors for UMAP
            
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
        
        elif method.lower() == 'umap' and UMAP_AVAILABLE:
            reducer = UMAP(n_components=n_components, n_neighbors=min(n_neighbors, n_samples // 2), random_state=42)
            return reducer.fit_transform(data)
        
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    def create_visualizations(self, save_to_file=True, show_labels=False):
        """
        Create all visualizations for the collected features.
        
        Args:
            save_to_file: Whether to save the visualizations to files
            show_labels: Whether to show point labels (indices)
        """
        # Generate all visualizations
        self._visualize_encoders(save_to_file, show_labels)
        self._visualize_projection_space(save_to_file, show_labels)
    
    def _visualize_encoders(self, save_to_file=True, show_labels=False):
        """
        Visualize the SAR encoder output space with different dimensionality reduction techniques.
        
        Args:
            save_to_file: Whether to save the visualizations to files
            show_labels: Whether to show point labels (indices)
        """
        methods = ['pca', 'tsne']
        if UMAP_AVAILABLE:
            methods.append('umap')
        
        n_samples = self.sar_features.shape[0]
        
        for method in methods:
            print(f"Creating {method.upper()} visualization for SAR encoder outputs...")
            try:
                reduced_data = self._apply_dimension_reduction(self.sar_features, method=method)
                
                plt.figure(figsize=(10, 8))
                
                # Ensure is_positive_labels is flattened to 1D
                is_positive_flat = self.is_positive_labels.flatten()
                
                # Create scatter plots by damage label (0=intact, 1=damaged)
                intact_mask = is_positive_flat == 0
                damaged_mask = is_positive_flat == 1
                
                plt.scatter(
                    reduced_data[intact_mask, 0], 
                    reduced_data[intact_mask, 1], 
                    c='blue',
                    marker='o', 
                    alpha=0.7,
                    label='Intact (0)'
                )
                
                plt.scatter(
                    reduced_data[damaged_mask, 0], 
                    reduced_data[damaged_mask, 1], 
                    c='red',
                    marker='o', 
                    alpha=0.7,
                    label='Damaged (1)'
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
    
    def _visualize_projection_space(self, save_to_file=True, show_labels=False):
        """
        Visualize the SAR and optical projection space with different dimensionality reduction techniques.
        
        Args:
            save_to_file: Whether to save the visualizations to files
            show_labels: Whether to show point labels (indices)
        """
        methods = ['pca', 'tsne']
        if UMAP_AVAILABLE:
            methods.append('umap')
        
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
        
        # Add damage information
        # Note: optical images (pre-event) are always intact, while SAR may be intact or damaged
        optical_damage_labels = np.zeros(n_optical)  # All optical are intact
        damage_labels = np.concatenate([optical_damage_labels, is_positive_flat])
        
        n_samples = len(self.sar_features)
        
        for method in methods:
            print(f"Creating {method.upper()} visualization for projection spaces...")
            try:
                reduced_data = self._apply_dimension_reduction(combined_projections, method=method)
                
                # 1. Plot by modality and damage together
                plt.figure(figsize=(12, 9))
                
                # Separate data by modality and damage status
                optical_indices = np.where(modality_labels == 0)[0]
                sar_intact_indices = np.where((modality_labels == 1) & (damage_labels == 0))[0]
                sar_damaged_indices = np.where((modality_labels == 1) & (damage_labels == 1))[0]
                
                # Plot each group with different markers and colors
                plt.scatter(
                    reduced_data[optical_indices, 0],
                    reduced_data[optical_indices, 1],
                    marker='^',  # Triangle for optical
                    color='blue',
                    alpha=0.7,
                    label='Optical (Intact)'
                )
                
                plt.scatter(
                    reduced_data[sar_intact_indices, 0],
                    reduced_data[sar_intact_indices, 1],
                    marker='o',  # Circle for SAR
                    color='green',
                    alpha=0.7,
                    label='SAR (Intact)'
                )
                
                plt.scatter(
                    reduced_data[sar_damaged_indices, 0],
                    reduced_data[sar_damaged_indices, 1],
                    marker='o',  # Circle for SAR
                    color='red',
                    alpha=0.7,
                    label='SAR (Damaged)'
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
                
                # 2. Still create the original modality-only plot
                plt.figure(figsize=(10, 8))
                
                # Plot by modality (optical vs SAR)
                plt.scatter(
                    reduced_data[optical_indices, 0],
                    reduced_data[optical_indices, 1],
                    marker='^',
                    color='blue',
                    alpha=0.7,
                    label='Optical'
                )
                
                plt.scatter(
                    reduced_data[np.where(modality_labels == 1)[0], 0],
                    reduced_data[np.where(modality_labels == 1)[0], 1],
                    marker='o',
                    color='red',
                    alpha=0.7,
                    label='SAR'
                )
                
                plt.legend(loc='best')
                plt.title(f'Projection Space by Modality - {method.upper()}')
                plt.xlabel(f'{method.upper()} Component 1')
                plt.ylabel(f'{method.upper()} Component 2')
                
                # Add point labels only if requested
                if show_labels:
                    for i, (x, y) in enumerate(reduced_data):
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
                    plt.savefig(self.output_dir / f'projection_space_modality_{method}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
                
                # 3. Still create the original damage-only plot
                plt.figure(figsize=(10, 8))
                
                # Plot by damage status (intact vs damaged)
                intact_indices = np.where(damage_labels == 0)[0]
                damaged_indices = np.where(damage_labels == 1)[0]
                
                plt.scatter(
                    reduced_data[intact_indices, 0],
                    reduced_data[intact_indices, 1],
                    color='green',
                    alpha=0.7,
                    label='Intact'
                )
                
                plt.scatter(
                    reduced_data[damaged_indices, 0],
                    reduced_data[damaged_indices, 1],
                    color='red',
                    alpha=0.7,
                    label='Damaged'
                )
                
                plt.legend(loc='best')
                plt.title(f'Projection Space by Damage - {method.upper()}')
                plt.xlabel(f'{method.upper()} Component 1')
                plt.ylabel(f'{method.upper()} Component 2')
                
                # Add point labels only if requested
                if show_labels:
                    for i, (x, y) in enumerate(reduced_data):
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
                    plt.savefig(self.output_dir / f'projection_space_damage_{method}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
            except Exception as e:
                print(f"Error creating {method.upper()} visualization: {e}")

# class VisualizationApp:
#     """Interactive Tkinter application for visualizing model outputs."""
    
#     def __init__(self, model, dataloader, device="cuda", normalize=False):
#         self.model = model
#         self.dataloader = dataloader
#         self.device = device
        
#         # Initialize data containers
#         self.sar_features = None
#         self.optical_features = None
#         self.sar_projected = None
#         self.optical_projected = None
#         self.is_positive_labels = None
        
#         # Collect features first
#         self._collect_features()
        
#         # Create the main window
#         self.root = Tk()
#         self.root.title("Model Output Visualization")
#         self.root.geometry("1200x800")
#         self.normalize = normalize
#         # Create the control panel
#         self.create_control_panel()
        
#         # Create the visualization panel
#         self.create_visualization_panel()
        
#         # Initialize with default visualization
#         self.update_visualization()
    
#     def _collect_features(self, max_batches=10):
#         """Collect features from the model."""
#         self.model.eval()
        
#         # Create empty lists
#         sar_features = []
#         optical_features = []
#         sar_projected = []
#         optical_projected = []
#         is_positive_labels = []
        
#         with torch.no_grad():
#             batch_count = 0
#             for batch in self.dataloader:
#                 pre_patches = batch["pre_patch"].to(self.device)
#                 post_patches = batch["post_patch"].to(self.device)
#                 is_positive = batch["is_positive"].to(self.device)
                
#                 # Forward pass
#                 outputs = self.model(optical=pre_patches, sar=post_patches)
                
#                 # Store features and labels
#                 sar_features.append(outputs["sar_features"].cpu().numpy())
#                 optical_features.append(outputs["optical_features"].cpu().numpy())
#                 if self.normalize:
#                     sar_projected.append(F.normalize(outputs["sar_projected"], dim=1).cpu().numpy())
#                     optical_projected.append(F.normalize(outputs["optical_projected"], dim=1).cpu().numpy())
#                 else:
#                     sar_projected.append(outputs["sar_projected"].cpu().numpy())
#                     optical_projected.append(outputs["optical_projected"].cpu().numpy())
                    
#                 is_positive_labels.append(is_positive.cpu().numpy())
                
#                 batch_count += 1
#                 if batch_count >= max_batches:
#                     break
        
#         # Concatenate all batches
#         self.sar_features = np.concatenate(sar_features)
#         self.optical_features = np.concatenate(optical_features)
#         self.sar_projected = np.concatenate(sar_projected)
#         self.optical_projected = np.concatenate(optical_projected)
#         self.is_positive_labels = np.concatenate(is_positive_labels)
        
#         # Ensure is_positive_labels is flattened to 1D
#         self.is_positive_labels = self.is_positive_labels.flatten()
        
#         print(f"Collected features from {len(self.sar_features)} samples")
    
#     def create_control_panel(self):
#         """Create the control panel with widgets."""
#         control_frame = Frame(self.root, width=300, height=800, padx=10, pady=10)
#         control_frame.pack(side=LEFT, fill=Y)
        
#         # Feature selection
#         Label(control_frame, text="Feature Space:").pack(anchor="w", pady=(10, 5))
#         self.feature_var = StringVar(value="SAR Encoder")
#         feature_options = ["SAR Encoder", "Projection Space"]
#         OptionMenu(control_frame, self.feature_var, *feature_options, command=self.update_visualization).pack(fill=X)
        
#         # Method selection
#         Label(control_frame, text="Dimension Reduction:").pack(anchor="w", pady=(10, 5))
#         self.method_var = StringVar(value="PCA")
#         method_options = ["PCA", "t-SNE"]
#         if UMAP_AVAILABLE:
#             method_options.append("UMAP")
#         OptionMenu(control_frame, self.method_var, *method_options, command=self.update_visualization).pack(fill=X)
        
#         # Parameter controls
#         Label(control_frame, text="t-SNE Perplexity:").pack(anchor="w", pady=(10, 5))
#         self.perplexity_var = Scale(control_frame, from_=5, to=50, orient=HORIZONTAL, resolution=5)
#         self.perplexity_var.set(30)
#         self.perplexity_var.pack(fill=X)
        
#         if UMAP_AVAILABLE:
#             Label(control_frame, text="UMAP Neighbors:").pack(anchor="w", pady=(10, 5))
#             self.neighbors_var = Scale(control_frame, from_=5, to=50, orient=HORIZONTAL, resolution=5)
#             self.neighbors_var.set(15)
#             self.neighbors_var.pack(fill=X)
        
#         # Color mapping
#         Label(control_frame, text="Color By:").pack(anchor="w", pady=(10, 5))
#         self.color_var = StringVar(value="Damage")
#         color_options = ["Damage"]
#         if self.feature_var.get() == "Projection Space":
#             color_options.append("Modality")
#         OptionMenu(control_frame, self.color_var, *color_options, command=self.update_visualization).pack(fill=X)
        
#         # Button to update visualization
#         Button(control_frame, text="Update Visualization", command=self.update_visualization).pack(pady=20, fill=X)
        
#         # Button to save visualization
#         Button(control_frame, text="Save Visualization", command=self.save_visualization).pack(pady=5, fill=X)
        
#         # Checkbox for point labels
#         self.show_labels_var = StringVar(value="1")
#         Label(control_frame, text="Show Point Labels:").pack(anchor="w", pady=(10, 5))
#         Frame(control_frame).pack(anchor="w")
#         self.show_labels_options = ["Yes", "No"]
#         OptionMenu(control_frame, self.show_labels_var, *self.show_labels_options, command=self.update_visualization).pack(fill=X)
    
#     def create_visualization_panel(self):
#         """Create the visualization panel with matplotlib figure."""
#         self.viz_frame = Frame(self.root, bg="white")
#         self.viz_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        
#         # Create matplotlib figure
#         self.fig = plt.figure(figsize=(8, 6), dpi=100)
#         self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
#         self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
#     def update_visualization(self, *args):
#         """Update the visualization based on current settings."""
#         # Clear the figure
#         self.fig.clear()
        
#         # Get parameters
#         feature_space = self.feature_var.get()
#         method = self.method_var.get().lower()
#         perplexity = self.perplexity_var.get()
#         neighbors = self.neighbors_var.get() if UMAP_AVAILABLE else 15
#         color_by = self.color_var.get()
#         show_labels = self.show_labels_var.get() == "Yes"
        
#         # Make sure labels are flattened
#         is_positive_flat = self.is_positive_labels.flatten()
        
#         # Apply dimension reduction
#         if feature_space == "SAR Encoder":
#             # For SAR Encoder visualization
#             data = self.sar_features
            
#             if method == 'pca':
#                 reducer = PCA(n_components=2)
#                 reduced_data = reducer.fit_transform(data)
#             elif method == 't-sne':
#                 # Adjust perplexity if needed
#                 adjusted_perplexity = min(perplexity, data.shape[0] // 3)
#                 adjusted_perplexity = max(adjusted_perplexity, 5)
#                 reducer = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
#                 reduced_data = reducer.fit_transform(data)
#             elif method == 'umap' and UMAP_AVAILABLE:
#                 reducer = UMAP(n_components=2, n_neighbors=min(neighbors, data.shape[0] // 2), random_state=42)
#                 reduced_data = reducer.fit_transform(data)
            
#             # Create plot with clear intact/damaged distinction
#             ax = self.fig.add_subplot(111)
            
#             # Create separate scatter plots for intact and damaged
#             intact_indices = np.where(is_positive_flat == 0)[0]
#             damaged_indices = np.where(is_positive_flat == 1)[0]
            
#             if len(intact_indices) > 0:
#                 ax.scatter(
#                     reduced_data[intact_indices, 0], 
#                     reduced_data[intact_indices, 1], 
#                     color='blue',
#                     marker='o',
#                     alpha=0.7,
#                     label='Intact (0)'
#                 )
            
#             if len(damaged_indices) > 0:
#                 ax.scatter(
#                     reduced_data[damaged_indices, 0], 
#                     reduced_data[damaged_indices, 1], 
#                     color='red',
#                     marker='o',
#                     alpha=0.7,
#                     label='Damaged (1)'
#                 )
            
#             ax.legend(loc='best')
#             ax.set_title(f'SAR Encoder Features - {method.upper()}')
#             ax.set_xlabel(f'{method.upper()} Component 1')
#             ax.set_ylabel(f'{method.upper()} Component 2')
            
#             # Add point labels if enabled
#             if show_labels:
#                 for i, (x, y) in enumerate(reduced_data):
#                     ax.annotate(
#                         str(i),
#                         (x, y),
#                         textcoords="offset points",
#                         xytext=(0, 5),
#                         ha='center',
#                         fontsize=8
#                     )
                
#         else:  # Projection Space
#             # For Projection Space visualization
#             combined_data = np.concatenate([self.optical_projected, self.sar_projected])
#             n_optical = len(self.optical_projected)
#             n_sar = len(self.sar_projected)
            
#             # Create modality labels (optical=0, SAR=1)
#             modality_labels = np.concatenate([
#                 np.zeros(n_optical),  # Optical
#                 np.ones(n_sar)        # SAR
#             ])
            
#             # Create damage labels
#             # Note: all optical are intact (0), SAR can be intact (0) or damaged (1)
#             optical_damage = np.zeros(n_optical)  # Always intact
#             damage_labels = np.concatenate([
#                 optical_damage,
#                 is_positive_flat
#             ])
            
#             # Apply dimension reduction
#             if method == 'pca':
#                 reducer = PCA(n_components=2)
#                 reduced_data = reducer.fit_transform(combined_data)
#             elif method == 't-sne':
#                 adjusted_perplexity = min(perplexity, combined_data.shape[0] // 3)
#                 adjusted_perplexity = max(adjusted_perplexity, 5)
#                 reducer = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
#                 reduced_data = reducer.fit_transform(combined_data)
#             elif method == 'umap' and UMAP_AVAILABLE:
#                 reducer = UMAP(n_components=2, n_neighbors=min(neighbors, combined_data.shape[0] // 2), random_state=42)
#                 reduced_data = reducer.fit_transform(combined_data)
            
#             ax = self.fig.add_subplot(111)
            
#             if color_by == "Damage":
#                 # Separate by modality and damage using indices
#                 optical_indices = np.where(modality_labels == 0)[0]
#                 sar_intact_indices = np.where((modality_labels == 1) & (damage_labels == 0))[0]
#                 sar_damaged_indices = np.where((modality_labels == 1) & (damage_labels == 1))[0]
                
#                 # Plot each category
#                 if len(optical_indices) > 0:
#                     ax.scatter(
#                         reduced_data[optical_indices, 0],
#                         reduced_data[optical_indices, 1],
#                         marker='^',  # Triangle for optical
#                         color='blue',
#                         alpha=0.7,
#                         label='Optical (Intact)'
#                     )
                
#                 if len(sar_intact_indices) > 0:
#                     ax.scatter(
#                         reduced_data[sar_intact_indices, 0],
#                         reduced_data[sar_intact_indices, 1],
#                         marker='o',  # Circle for SAR
#                         color='green',
#                         alpha=0.7,
#                         label='SAR (Intact)'
#                     )
                
#                 if len(sar_damaged_indices) > 0:
#                     ax.scatter(
#                         reduced_data[sar_damaged_indices, 0],
#                         reduced_data[sar_damaged_indices, 1],
#                         marker='o',  # Circle for SAR
#                         color='red',
#                         alpha=0.7,
#                         label='SAR (Damaged)'
#                     )
                
#                 ax.set_title(f'Projection Space by Damage - {method.upper()}')
                
#             else:  # By modality
#                 # Separate by modality using indices
#                 optical_indices = np.where(modality_labels == 0)[0]
#                 sar_indices = np.where(modality_labels == 1)[0]
                
#                 if len(optical_indices) > 0:
#                     ax.scatter(
#                         reduced_data[optical_indices, 0],
#                         reduced_data[optical_indices, 1],
#                         marker='^',  # Triangle for optical
#                         color='blue',
#                         alpha=0.7,
#                         label='Optical'
#                     )
                
#                 if len(sar_indices) > 0:
#                     ax.scatter(
#                         reduced_data[sar_indices, 0],
#                         reduced_data[sar_indices, 1],
#                         marker='o',  # Circle for SAR
#                         color='red',
#                         alpha=0.7,
#                         label='SAR'
#                     )
                
#                 ax.set_title(f'Projection Space by Modality - {method.upper()}')
            
#             ax.legend(loc='best')
#             ax.set_xlabel(f'{method.upper()} Component 1')
#             ax.set_ylabel(f'{method.upper()} Component 2')
            
#             # Add point labels if enabled
#             if show_labels:
#                 for i, (x, y) in enumerate(reduced_data):
#                     if i < n_optical:
#                         label = f"{i}O"  # Optical
#                     else:
#                         label = f"{i-n_optical}S"  # SAR
                    
#                     ax.annotate(
#                         label,
#                         (x, y),
#                         textcoords="offset points",
#                         xytext=(0, 5),
#                         ha='center',
#                         fontsize=8
#                     )
        
#         # Update the canvas
#         self.canvas.draw()
    
#     def save_visualization(self):
#         """Save the current visualization to a file."""
#         feature_space = self.feature_var.get().lower().replace(" ", "_")
#         method = self.method_var.get().lower()
#         color_by = self.color_var.get().lower()
        
#         filename = f"{feature_space}_{color_by}_{method}.png"
#         self.fig.savefig(filename, dpi=300, bbox_inches='tight')
#         print(f"Visualization saved to {filename}")
    
#     def run(self):
#         """Run the application."""
#         self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Visualize model outputs")
    parser.add_argument("--config", type=Path, default="configs/default.yaml", 
                      help="Path to configuration file")
    parser.add_argument("--model_path", type=str, default="output/patch_contrastive/best_model.pth", 
                      help="Path to the saved model")
    parser.add_argument("--config_path", type=str, default="configs/default.yaml", 
                      help="Path to the model config file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for data loading")
    parser.add_argument("--output_dir", type=str, default="output/features_explorer", 
                      help="Directory to save output")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive visualization")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cuda/cpu)")
    parser.add_argument("--max_batches", type=int, default=10, help="Maximum number of batches to process")
    parser.add_argument("--show_labels", action="store_true", help="Show point labels in static visualizations")
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
    parser.add_argument("--target_neg_ratio", type=float, default=0.1,
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
    
    if args.interactive:
        # Launch interactive visualization
        # app = VisualizationApp(model, dataloader, device=args.device, normalize=args.normalize)
        # app.run()
        raise NotImplementedError("Interactive visualization not implemented for running it in kaggle")
    else:
        # Generate static visualizations
        visualizer = ModelVisualizer(
            model, 
            dataloader, 
            device=args.device, 
            output_dir=args.output_dir,
            normalize=args.normalize
        )
        visualizer.collect_features(max_batches=args.max_batches)
        visualizer.create_visualizations(save_to_file=True, show_labels=args.show_labels)
        print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()