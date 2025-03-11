import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


class GradientMonitor:
    """
    Callback to track and visualize gradients and activations during training.
    """

    def __init__(
        self, model, output_dir="output/gradients", log_interval=10, num_points=300
    ):
        """
        Args:
            model: The model to monitor
            output_dir: Directory to save visualizations
            log_interval: Recording frequency (in batches)
            num_points: Number of points to sample from tensors for histograms
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.num_points = num_points

        self.logger = logging.getLogger("GradientMonitor")

        # History for metrics
        self.history = {
            "epoch": [],
            "batch": [],
            "layer_grad_norms": {},  # Average gradient norms by layer
            "layer_activations": {},  # Average activations by layer
            "optical_feature_stats": [],  # Statistics of optical features
            "sar_feature_stats": [],  # Statistics of SAR features
            "projection_stats": [],  # Statistics of projections
            "similarity_stats": [],  # Statistics of similarities
        }

        # Hook registry
        self.grad_hooks = []
        self.activation_hooks = []

        # Important layers to track (adapt according to your model)
        self.tracked_modules = [
            ("optical_encoder.layer2", "Optical Encoder (Layer 2)"),
            ("sar_encoder.layer2", "SAR Encoder (Layer 2)"),
            ("optical_encoder.layer3", "Optical Encoder (Layer 3)"),
            ("sar_encoder.layer3", "SAR Encoder (Layer 3)"),
            ("optical_encoder.layer4", "Optical Encoder (Layer 4)"),
            ("sar_encoder.layer4", "SAR Encoder (Layer 4)"),
            ("optical_projector.layer1.0", "Optical Projector (FC)"),
            ("sar_projector.layer1.0", "SAR Projector (FC)"),
            ("optical_projector.layer2.0", "Optical Projector Output"),
            ("sar_projector.layer2.0", "SAR Projector Output"),
        ]

        # Dictionaries to temporarily store values from current batch
        self.current_grads = {}
        self.current_activations = {}

        # Initialize tracking
        self.setup_hooks()

    def setup_hooks(self):
        """Configure hooks on layers to track"""
        for module_name, _ in self.tracked_modules:
            # Parse module name to access the sub-layer
            parts = module_name.split(".")
            module = self.model
            for part in parts:
                module = getattr(module, part)

            # Hook for activations (forward)
            def get_activation_hook(name):
                def hook(module, input, output):
                    self.current_activations[name] = output.detach()

                return hook

            # Hook for gradients (backward)
            def get_gradient_hook(name):
                def hook(module, grad_input, grad_output):
                    self.current_grads[name] = grad_output[0].detach()

                return hook

            # Register hooks
            self.activation_hooks.append(
                module.register_forward_hook(get_activation_hook(module_name))
            )
            self.grad_hooks.append(
                module.register_full_backward_hook(get_gradient_hook(module_name))
            )

            # Initialize entries in history
            self.history["layer_grad_norms"][module_name] = []
            self.history["layer_activations"][module_name] = []

    def after_batch(self, epoch, batch_idx, outputs, loss):
        """
        Called after processing a batch

        Args:
            epoch: Current epoch
            batch_idx: Batch index
            outputs: Model outputs
            loss: Loss value
        """
        # Only process at regular intervals to avoid slowing down training
        if batch_idx % self.log_interval != 0:
            return

        # Record epoch and batch
        self.history["epoch"].append(epoch)
        self.history["batch"].append(batch_idx)

        # Calculate statistics on gradients
        for module_name in self.current_grads:
            grad = self.current_grads[module_name]
            if grad is not None:
                # Calculate the average norm of gradients
                grad_norm = grad.norm(2, dim=1).mean().item()
                self.history["layer_grad_norms"][module_name].append(grad_norm)

                # Check if NaN or Inf are present
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    self.logger.warning(
                        f"NaN/Inf gradients detected in {module_name} at epoch {epoch}, batch {batch_idx}"
                    )

        # Calculate statistics on activations
        for module_name in self.current_activations:
            activation = self.current_activations[module_name]
            if activation is not None:
                # Calculate the average of activations
                act_mean = activation.abs().mean().item()
                self.history["layer_activations"][module_name].append(act_mean)

                # Check if NaN or Inf are present
                if torch.isnan(activation).any() or torch.isinf(activation).any():
                    self.logger.warning(
                        f"NaN/Inf activations detected in {module_name} at epoch {epoch}, batch {batch_idx}"
                    )

        # Analyze feature and projection representations
        if "optical_features" in outputs and "sar_features" in outputs:
            # Calculate feature statistics
            opt_feat = outputs["optical_features"]
            sar_feat = outputs["sar_features"]

            self.history["optical_feature_stats"].append(
                {
                    "mean": opt_feat.mean().item(),
                    "std": opt_feat.std().item(),
                    "min": opt_feat.min().item(),
                    "max": opt_feat.max().item(),
                }
            )

            self.history["sar_feature_stats"].append(
                {
                    "mean": sar_feat.mean().item(),
                    "std": sar_feat.std().item(),
                    "min": sar_feat.min().item(),
                    "max": sar_feat.max().item(),
                }
            )

            # Calculate projection statistics
            if "optical_projected" in outputs and "sar_projected" in outputs:
                opt_proj = outputs["optical_projected"]
                sar_proj = outputs["sar_projected"]

                self.history["projection_stats"].append(
                    {
                        "optical_norm": torch.norm(opt_proj, dim=1).mean().item(),
                        "sar_norm": torch.norm(sar_proj, dim=1).mean().item(),
                    }
                )

                # Calculate similarity statistics
                if "change_score" in outputs:
                    change_scores = outputs["change_score"]
                    self.history["similarity_stats"].append(
                        {
                            "mean": change_scores.mean().item(),
                            "std": change_scores.std().item(),
                            "min": change_scores.min().item(),
                            "max": change_scores.max().item(),
                        }
                    )

        # At larger intervals, generate visualizations
        if epoch % 5 == 0 and batch_idx % (self.log_interval * 5) == 0:
            self.generate_visualizations(epoch, batch_idx)

    def after_epoch(self, epoch):
        """Generates visualizations at the end of each epoch"""
        self.generate_visualizations(epoch, "end")

    def generate_visualizations(self, epoch, batch_idx):
        """Generates and saves visualizations"""
        # 1. Visualize the evolution of gradient norms
        self._plot_grad_norms(epoch, batch_idx)

        # 2. Visualize the evolution of activations
        self._plot_activations(epoch, batch_idx)

        # 3. Visualize the distributions of features and projections
        if len(self.history["optical_feature_stats"]) > 0:
            self._plot_feature_stats(epoch, batch_idx)

        # 4. Visualize the distributions of similarities
        if len(self.history["similarity_stats"]) > 0:
            self._plot_similarity_stats(epoch, batch_idx)

    def _plot_grad_norms(self, epoch, batch_idx):
        """Visualizes the evolution of gradient norms"""
        plt.figure(figsize=(12, 8))

        for module_name, display_name in self.tracked_modules:
            if (
                module_name in self.history["layer_grad_norms"]
                and len(self.history["layer_grad_norms"][module_name]) > 0
            ):
                values = self.history["layer_grad_norms"][module_name]
                batches = list(range(len(values)))
                plt.plot(batches, values, label=display_name)

        plt.title("Evolution of gradient norms by layer")
        plt.xlabel("Batch")
        plt.ylabel("Average gradient norm")
        plt.yscale("log")  # Often useful for gradients
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the image
        plt.savefig(self.output_dir / f"grad_norms_e{epoch}_b{batch_idx}.png")
        plt.close()

    def _plot_activations(self, epoch, batch_idx):
        """Visualizes the evolution of activations"""
        plt.figure(figsize=(12, 8))

        for module_name, display_name in self.tracked_modules:
            if (
                module_name in self.history["layer_activations"]
                and len(self.history["layer_activations"][module_name]) > 0
            ):
                values = self.history["layer_activations"][module_name]
                batches = list(range(len(values)))
                plt.plot(batches, values, label=display_name)

        plt.title("Evolution of activations by layer")
        plt.xlabel("Batch")
        plt.ylabel("Average of activations (abs)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save the image
        plt.savefig(self.output_dir / f"activations_e{epoch}_b{batch_idx}.png")
        plt.close()

    def _plot_feature_stats(self, epoch, batch_idx):
        """Visualizes statistics of features and projections"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Evolution of optical feature statistics
        ax = axes[0, 0]
        opt_means = [s["mean"] for s in self.history["optical_feature_stats"]]
        opt_stds = [s["std"] for s in self.history["optical_feature_stats"]]
        batches = list(range(len(opt_means)))

        ax.plot(batches, opt_means, label="Mean")
        ax.fill_between(
            batches,
            [m - s for m, s in zip(opt_means, opt_stds)],
            [m + s for m, s in zip(opt_means, opt_stds)],
            alpha=0.3,
            label="±1 σ",
        )
        ax.set_title("Optical Features - Statistics")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Evolution of SAR feature statistics
        ax = axes[0, 1]
        sar_means = [s["mean"] for s in self.history["sar_feature_stats"]]
        sar_stds = [s["std"] for s in self.history["sar_feature_stats"]]

        ax.plot(batches, sar_means, label="Mean")
        ax.fill_between(
            batches,
            [m - s for m, s in zip(sar_means, sar_stds)],
            [m + s for m, s in zip(sar_means, sar_stds)],
            alpha=0.3,
            label="±1 σ",
        )
        ax.set_title("SAR Features - Statistics")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Evolution of projection norms
        if len(self.history["projection_stats"]) > 0:
            ax = axes[1, 0]
            opt_norms = [s["optical_norm"] for s in self.history["projection_stats"]]
            sar_norms = [s["sar_norm"] for s in self.history["projection_stats"]]

            ax.plot(batches, opt_norms, label="Optical")
            ax.plot(batches, sar_norms, label="SAR")
            ax.set_title("Projection Norms")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Average Norm")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Latest cosine similarity distribution
        if len(self.history["similarity_stats"]) > 0:
            ax = axes[1, 1]
            sim_means = [s["mean"] for s in self.history["similarity_stats"]]
            sim_stds = [s["std"] for s in self.history["similarity_stats"]]
            sim_mins = [s["min"] for s in self.history["similarity_stats"]]
            sim_maxs = [s["max"] for s in self.history["similarity_stats"]]

            ax.plot(batches, sim_means, label="Mean")
            ax.fill_between(
                batches,
                [m - s for m, s in zip(sim_means, sim_stds)],
                [m + s for m, s in zip(sim_means, sim_stds)],
                alpha=0.3,
                label="±1 σ",
            )
            ax.plot(batches, sim_mins, "r--", alpha=0.5, label="Min")
            ax.plot(batches, sim_maxs, "g--", alpha=0.5, label="Max")
            ax.set_title("Change Scores")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"feature_stats_e{epoch}_b{batch_idx}.png")
        plt.close()

    def _plot_similarity_stats(self, epoch, batch_idx):
        """Visualizes similarity statistics in more detail"""
        if len(self.history["similarity_stats"]) == 0:
            return

        plt.figure(figsize=(10, 6))
        recent_stats = self.history["similarity_stats"][
            -1
        ]  # Take the most recent stats

        # Create a synthetic histogram based on mean and standard deviation
        mean = recent_stats["mean"]
        std = recent_stats["std"]
        min_val = recent_stats["min"]
        max_val = recent_stats["max"]

        # Generate a normal distribution with these stats for visualization
        x = np.linspace(min_val - 0.5, max_val + 0.5, 100)
        y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

        plt.plot(
            x, y, "r-", label=f"Estimated distribution (µ={mean:.3f}, σ={std:.3f})"
        )
        plt.axvline(x=0.5, color="k", linestyle="--", label="Typical threshold (0.5)")
        plt.title(f"Estimated distribution of change scores (Epoch {epoch})")
        plt.xlabel("Change score")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(self.output_dir / f"similarity_dist_e{epoch}_b{batch_idx}.png")
        plt.close()

    def cleanup(self):
        """Cleans up hooks to avoid memory leaks"""
        for hook in self.activation_hooks + self.grad_hooks:
            hook.remove()
        self.activation_hooks.clear()
        self.grad_hooks.clear()
