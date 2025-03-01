# main.py
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from data.dataset import MultimodalDamageDataset
from data.transforms import get_transform
from models.pseudo_siamese import MultimodalDamageNet
from losses.contrastive_loss import SupervisedContrastiveLoss
from trainer.trainer import Trainer

def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_transform = get_transform('train', crop_size=config['data']['crop_size'])
    val_transform = get_transform('val')
    
    train_dataset = MultimodalDamageDataset(
        root_dir=config['data']['root_dir'],
        split='train',
        transform=train_transform,
        crop_size=config['data']['crop_size']
    )

    val_dataset = MultimodalDamageDataset(
        root_dir=config['data']['root_dir'],
        split='val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Create model
    model = MultimodalDamageNet(
        optical_channels=config['model']['optical_channels'],
        sar_channels=config['model']['sar_channels'],
        projection_dim=config['model']['projection_dim']
    )
    
    # Create loss function
    criterion = SupervisedContrastiveLoss(
        # contrastive_weight=config['training']['contrastive_weight'],
        temperature=config['training']['temperature']
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['training']['num_epochs'],
        output_dir=config['training']['output_dir'],
        save_best=True
    )
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multimodal damage assessment model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    main(args)