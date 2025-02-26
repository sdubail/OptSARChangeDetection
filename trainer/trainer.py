# trainer/trainer.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging

class Trainer:
    """Trainer for multimodal damage assessment model with supervised contrastive learning."""
    
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, 
                 scheduler=None, device='cuda', num_epochs=100, 
                 output_dir='output', save_best=True):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            num_epochs: Number of epochs to train
            output_dir: Directory to save outputs
            save_best: Whether to save only the best model
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
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train(self):
        """Train the model."""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training
            train_loss, train_metrics = self._train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self._validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log training progress
            self.logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )
            
            # Save model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.logger.info(f"New best model with validation loss: {val_loss:.4f}")
                self._save_checkpoint(epoch, val_loss, is_best=True)
            elif not self.save_best:
                self._save_checkpoint(epoch, val_loss, is_best=False)
        
        self.logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
        
    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        metrics = {}
        
        with tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Train") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                pre_images = batch['pre_image'].to(self.device)
                post_images = batch['post_image'].to(self.device)
                labels = batch['label'].to(self.device)
                damage_labels = batch['damage_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(pre_images, post_images)
                
                # Compute loss
                loss, batch_metrics = self.criterion(
                    outputs, 
                    {'label': labels, 'damage_label': damage_labels}
                )
                
                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                
                # Update metrics
                for key, value in batch_metrics.items():
                    if key not in metrics:
                        metrics[key] = 0
                    metrics[key] += value
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
        # Calculate average metrics
        epoch_loss /= len(self.train_loader)
        for key in metrics:
            metrics[key] /= len(self.train_loader)
            
        return epoch_loss, metrics
    
    def _validate_epoch(self, epoch):
        """Validate for one epoch."""
        self.model.eval()
        epoch_loss = 0
        metrics = {}
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} - Val") as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Move data to device
                    pre_images = batch['pre_image'].to(self.device)
                    post_images = batch['post_image'].to(self.device)
                    labels = batch['label'].to(self.device)
                    damage_labels = batch['damage_label'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(pre_images, post_images)
                    
                    # Compute loss
                    loss, batch_metrics = self.criterion(
                        outputs, 
                        {'label': labels, 'damage_label': damage_labels}
                    )
                    
                    # Update statistics
                    epoch_loss += loss.item()
                    
                    # Update metrics
                    for key, value in batch_metrics.items():
                        if key not in metrics:
                            metrics[key] = 0
                        metrics[key] += value
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average metrics
        epoch_loss /= len(self.val_loader)
        for key in metrics:
            metrics[key] /= len(self.val_loader)
            
        return epoch_loss, metrics
    
    def _save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
        else:
            torch.save(checkpoint, self.output_dir / f'checkpoint_epoch_{epoch+1}.pth')