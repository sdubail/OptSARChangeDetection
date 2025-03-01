# main_contrastive.py
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the preprocessed patch dataset
from data.dataset_patches import PreprocessedPatchDataset
from data.transforms import get_transform
from models.pseudo_siamese import MultimodalDamageNet  # Using your original model with minimal changes
from losses.contrastive_loss import SupervisedContrastiveLoss
from trainer.trainer import ContrastiveTrainer

def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets using preprocessed patches
    train_transform = get_transform('train') if args.use_transforms else None
    val_transform = get_transform('val') if args.use_transforms else None
    
    print("Loading preprocessed patch datasets...")
    train_dataset = PreprocessedPatchDataset(
        patch_dir=args.patch_dir,
        split='train',
        transform=train_transform,
        cache_size=args.cache_size
    )

    val_dataset = PreprocessedPatchDataset(
        patch_dir=args.patch_dir,
        split='val',
        transform=val_transform,
        cache_size=args.cache_size
    )
    
    # Print dataset statistics
    print(f"Train dataset: {len(train_dataset)} patch pairs")
    print(f"  - Positive ratio: {train_dataset.get_pos_neg_ratio():.2f}")
    print(f"Val dataset: {len(val_dataset)} patch pairs")
    print(f"  - Positive ratio: {val_dataset.get_pos_neg_ratio():.2f}")
    
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
    
    # Create model with your original architecture (minus classification head)
    model = MultimodalDamageNet(
        optical_channels=config['model']['optical_channels'],
        sar_channels=config['model']['sar_channels'],
        projection_dim=config['model']['projection_dim']
    )
    
    # Create simplified contrastive loss function
    criterion = SupervisedContrastiveLoss(
        temperature=config['training'].get('temperature', 0.07),
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
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['training']['num_epochs'],
        output_dir=config['training']['output_dir'],
        save_best=True,
        log_interval=args.log_interval
    )
    
    # Train model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train contrastive learning model with preprocessed patches')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--patch_dir', type=str, default='data/processed_patches',
                        help='Directory containing preprocessed patches')
    parser.add_argument('--use_transforms', action='store_true',
                        help='Whether to apply data augmentation transforms')
    parser.add_argument('--cache_size', type=int, default=1000,
                        help='Number of patches to cache in memory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='How often to log training metrics (in batches)')
    args = parser.parse_args()
    
    main(args)