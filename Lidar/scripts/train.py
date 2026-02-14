#!/usr/bin/env python3
"""
Training Script for LiDAR 3D Detection

Supports:
- Single/Multi-GPU training
- Mixed precision training
- Checkpoint resumption
- TensorBoard logging
- Automatic evaluation
"""

import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train 3D detector')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--data-root', required=True, help='Dataset root directory')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    parser.add_argument('--resume', default=None, help='Resume from checkpoint')
    parser.add_argument('--num-gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--num-epochs', type=int, default=None, help='Override num epochs')
    parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--eval-only', action='store_true', help='Evaluation only')
    return parser.parse_args()


class Trainer:
    """
    Complete training pipeline for 3D detectors.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config: dict,
        output_dir: Path,
        use_amp: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output_dir = Path(output_dir)
        self.use_amp = use_amp
        
        # Create directories
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.log_dir = self.output_dir / 'logs'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.iteration = 0
        self.best_metric = 0.0
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        logger.info(f"Trainer initialized")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Mixed precision: {use_amp}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch_dict in enumerate(self.train_loader):
            # Move to GPU
            for key in batch_dict:
                if isinstance(batch_dict[key], torch.Tensor):
                    batch_dict[key] = batch_dict[key].cuda()
            
            # Forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred_dict = self.model(batch_dict)
                    loss, loss_dict = self.model.get_loss(pred_dict)
            else:
                pred_dict = self.model(batch_dict)
                loss, loss_dict = self.model.get_loss(pred_dict)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f}"
                )
            
            self.iteration += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {self.epoch} - Avg Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self):
        """Run validation."""
        self.model.eval()
        
        # TODO: Implement full validation with metrics
        # For now, just compute loss
        
        total_loss = 0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_dict in self.val_loader:
                # Move to GPU
                for key in batch_dict:
                    if isinstance(batch_dict[key], torch.Tensor):
                        batch_dict[key] = batch_dict[key].cuda()
                
                pred_dict = self.model(batch_dict)
                loss, _ = self.model.get_loss(pred_dict)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save epoch checkpoint
        if self.epoch % 5 == 0:
            torch.save(checkpoint, self.checkpoint_dir / f'epoch_{self.epoch}.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            logger.info(f"Saved best checkpoint at epoch {self.epoch}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 0)
        self.iteration = checkpoint.get('iteration', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        logger.info(f"Resumed from epoch {self.epoch}")
    
    def train(self, num_epochs):
        """Main training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Save checkpoint
            is_best = val_loss < self.best_metric if self.best_metric > 0 else True
            if is_best:
                self.best_metric = val_loss
            
            self.save_checkpoint(is_best=is_best)
        
        logger.info("Training complete!")


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with args
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{config['model']['name']}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Create model
    logger.info("Creating model...")
    
    model_name = config['model']['name']
    if model_name == 'pointpillars':
        from src.detectors.pointpillars import PointPillars
        model = PointPillars(**config['model'])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.cuda()
    
    # Multi-GPU
    if args.num_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using {args.num_gpus} GPUs")
    
    # Create dataloaders
    logger.info("Loading data...")
    # TODO: Implement dataset loading
    train_loader = None
    val_loader = None
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['learning_rate'],
        epochs=config['training']['num_epochs'],
        steps_per_epoch=len(train_loader) if train_loader else 100
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        output_dir=output_dir,
        use_amp=args.use_amp
    )
    
    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    if not args.eval_only:
        trainer.train(config['training']['num_epochs'])
    else:
        trainer.validate()


if __name__ == '__main__':
    main()
