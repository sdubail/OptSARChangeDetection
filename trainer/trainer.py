import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from utils.gradient_monitor import GradientMonitor


class ContrastiveTrainer:
    """Trainer for patch-based contrastive learning for change detection."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device="cuda",
        num_epochs=100,
        output_dir="output",
        save_best=True,
        log_interval=10,
        loading_checkpoint=False,
        monitor_gradients=False,
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader with patch pairs
            val_loader: Validation data loader with patch pairs
            criterion: Contrastive loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            num_epochs: Number of epochs to train
            output_dir: Directory to save outputs
            save_best: Whether to save only the best model
            log_interval: How often to log batch results
            monitor_gradients: Use callbacks for gradient flow inspection
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.save_best = save_best
        self.log_interval = log_interval

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "training.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # if load_path is provided, resume training
        if loading_checkpoint:
            checkpoint_path = (
                self.output_dir / "best_model.pth"
            )  # Or the correct checkpoint file
            if checkpoint_path.exists():
                self.start_epoch, best_val_loss, best_val_acc = self.load_checkpoint(
                    checkpoint_path
                )
                self.logger.info(
                    f"Resumed training from epoch {self.start_epoch}, best val loss: {best_val_loss}, best val acc: {best_val_acc}"
                )
            else:
                self.logger.warning(
                    f"No checkpoint found at {checkpoint_path}, starting training from scratch."
                )

        self.monitor_gradients = monitor_gradients
        if self.monitor_gradients:
            self.gradient_monitor = GradientMonitor(
                model=model,
                output_dir=f"{output_dir}/gradients",
                log_interval=log_interval,
            )

    def train(self):
        """Train the model."""
        best_val_loss = float("inf")

        # Metrics tracking
        train_losses = []
        train_losses_pos = []
        train_losses_neg = []
        val_losses = []
        val_losses_pos = []
        val_losses_neg = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(self.num_epochs):
            # Update the sampler with current epoch
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # Training
            train_loss, train_acc, train_loss_pos, train_loss_neg = self._train_epoch(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            train_losses_pos.append(train_loss_pos)
            train_losses_neg.append(train_loss_neg)

            # Validation
            val_loss, val_acc, val_loss_pos, val_loss_neg = self._validate_epoch(epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            val_losses_pos.append(val_loss_pos)
            val_losses_neg.append(val_loss_neg)

            if self.monitor_gradients:
                self.gradient_monitor.after_epoch(epoch)

            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Log training progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.logger.info(f"New best model with validation loss: {val_loss:.4f}")
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)
            elif not self.save_best:
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=False)

        # Plot training curves
        self._plot_training_curves(
            train_losses, val_losses, train_accuracies, val_accuracies, train_losses_pos, train_losses_neg, val_losses_pos, val_losses_neg
        )
        # save training curves as numpy arrays
        np.save(self.output_dir / "train_losses.npy", np.array(train_losses))
        np.save(self.output_dir / "val_losses.npy", np.array(val_losses))
        np.save(self.output_dir / "train_accuracies.npy", np.array(train_accuracies))
        np.save(self.output_dir / "val_accuracies.npy", np.array(val_accuracies))
        np.save(self.output_dir / "train_losses_pos.npy", np.array(train_losses_pos))
        np.save(self.output_dir / "train_losses_neg.npy", np.array(train_losses_neg))
        np.save(self.output_dir / "val_losses_pos.npy", np.array(val_losses_pos))
        np.save(self.output_dir / "val_losses_neg.npy", np.array(val_losses_neg))

        # Save
        # log best validation loss
        self.logger.info(
            f"Training completed. Best validation loss: {best_val_loss:.4f}"
        )

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        epoch_loss_pos = 0
        epoch_loss_neg = 0
        all_preds = []
        all_targets = []

        with tqdm(
            self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Train"
        ) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                pre_patches = batch["pre_patch"].to(self.device)
                post_patches = batch["post_patch"].to(self.device)
                is_positive = batch["is_positive"].to(self.device)

                # Forward pass
                outputs = self.model(optical=pre_patches, sar=post_patches)

                # Compute loss
                pos_loss, neg_loss = self.criterion(
                    outputs["pre_projected"], outputs["post_projected"], is_positive
                )
                loss = pos_loss + neg_loss  # CAREFUL
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update statistics
                epoch_loss += loss.item()
                epoch_loss_pos += pos_loss.item()
                epoch_loss_neg += neg_loss.item()
                
                # Call gradient monitor after backward pass (if enabled)
                if self.monitor_gradients:
                    self.gradient_monitor.after_batch(epoch, batch_idx, outputs, loss)

                # Collect predictions for evaluation
                if "change_score" in outputs:
                    predictions = (
                        outputs["change_score"]
                        < 1.0  # change score is between 0 and 2.
                    ).float()  # Low score = positive pair
                    all_preds.extend(predictions.cpu().numpy())
                    all_targets.extend(is_positive.cpu().numpy())

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": loss.item(),
                        "pos_loss": pos_loss.item(),
                        "neg_loss": neg_loss.item(),
                    }
                )

                # Log batch metrics at intervals
                if batch_idx % self.log_interval == 0:
                    self.logger.info(
                        f"""Train Batch {batch_idx}/{len(self.train_loader)} 
                        - Loss: {loss.item():.4f} 
                        - Pos Loss: {pos_loss.item():.4f}  
                        - Neg Loss: {neg_loss.item():.4f}"""
                    )

        # Calculate average loss and accuracy
        epoch_loss /= len(self.train_loader)
        epoch_loss_pos /= len(self.train_loader)
        epoch_loss_neg /= len(self.train_loader)

        # Calculate accuracy if we have predictions
        epoch_acc = 0
        if all_preds and all_targets:
            epoch_acc = balanced_accuracy_score(all_targets, all_preds)

        return epoch_loss, epoch_acc, epoch_loss_pos, epoch_loss_neg

    def _validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0
        epoch_loss_pos = 0
        epoch_loss_neg = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            with tqdm(
                self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Val"
            ) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Move data to device
                    pre_patches = batch["pre_patch"].to(self.device)
                    post_patches = batch["post_patch"].to(self.device)
                    is_positive = batch["is_positive"].to(self.device)

                    # Forward pass
                    outputs = self.model(optical=pre_patches, sar=post_patches)

                    # Compute loss
                    pos_loss, neg_loss = self.criterion(
                        outputs["pre_projected"], outputs["post_projected"], is_positive
                    )

                    # Update statistics
                    loss = pos_loss + neg_loss
                    epoch_loss += loss.item()
                    epoch_loss_pos += pos_loss.item()
                    epoch_loss_neg += neg_loss.item()
                    
                    # Collect predictions for evaluation
                    if "change_score" in outputs:
                        predictions = (
                            outputs["change_score"] < 1.0
                        ).float()  # Low score = positive pair
                        all_preds.extend(predictions.cpu().numpy())
                        all_targets.extend(is_positive.cpu().numpy())

                    # Update progress bar
                    pbar.set_postfix(
                        {
                            "loss": loss.item(),
                            "pos_loss": pos_loss.item(),
                            "neg_loss": neg_loss.item(),
                        }
                    )

        # Calculate average loss and accuracy
        epoch_loss /= len(self.val_loader)
        epoch_loss_pos /= len(self.val_loader)
        epoch_loss_neg /= len(self.val_loader)
        

        # Calculate accuracy if we have predictions
        epoch_acc = 0
        if all_preds and all_targets:
            epoch_acc = balanced_accuracy_score(all_targets, all_preds)

        return epoch_loss, epoch_acc, epoch_loss_pos, epoch_loss_neg

    def _save_checkpoint(self, epoch, val_loss, val_acc, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / "best_model.pth")
        else:
            torch.save(checkpoint, self.output_dir / f"checkpoint_epoch_{epoch+1}.pth")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Returns epoch, val_loss, val_acc
        return (
            checkpoint.get("epoch", 0),
            checkpoint.get("val_loss", None),
            checkpoint.get("val_acc", None),
        )

    def _plot_training_curves(self, train_losses, val_losses, train_accs, val_accs, train_losses_pos, train_losses_neg, val_losses_pos, val_losses_neg):
        """Plot training curves."""
        plt.figure(figsize=(12, 5))

        # Loss curve
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        

        # Loss curve subterms
        plt.subplot(1, 3, 2)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.plot(train_losses_pos, label="Train Pos Loss")
        plt.plot(train_losses_neg, label="Train Neg Loss")
        plt.plot(val_losses_pos, label="Val Pos Loss")
        plt.plot(val_losses_neg, label="Val Neg Loss")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        
        # Accuracy curve
        plt.subplot(1, 3, 3)
        plt.plot(train_accs, label="Train Accuracy")
        plt.plot(val_accs, label="Val Accuracy")
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png")
        plt.close()
