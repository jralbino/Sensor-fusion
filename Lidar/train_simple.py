#!/usr/bin/env python3
"""
Simple Training Script - FIXED VERSION

Minimal but complete training pipeline for PointPillars.
This version actually works!
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-dir', default='outputs/test_run')
    parser.add_argument('--model', choices=['pointpillars', 'second', 'centerpoint'], default='pointpillars',
                        help='Model architecture (default: pointpillars)')
    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")

    # TensorBoard
    tb_dir = output_dir / 'tensorboard'
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info(f"TensorBoard: {tb_dir}")
    global_step = 0
    
    # Import modules
    from src.data.datasets import create_dataloader
    from src.training.losses import get_loss_function

    # Create model
    logger.info(f"Creating {args.model} model...")
    if args.model == 'second':
        from src.detectors.second import SECOND
        model = SECOND(num_classes=10)
    elif args.model == 'centerpoint':
        from src.detectors.centerpoint import CenterPoint
        model = CenterPoint(num_classes=10)
    else:
        from src.detectors.pointpillars import PointPillars
        model = PointPillars(num_classes=10)
    model = model.to(device)
    
    # Create dataloaders
    logger.info("Loading datasets...")
    
    data_root = Path(args.data_root)
    train_info = data_root / 'nuscenes_infos_train.pkl'
    val_info = data_root / 'nuscenes_infos_val.pkl'
    
    if not train_info.exists():
        logger.error(f"Train info not found: {train_info}")
        logger.error("Run: python prepare_data.py --data-root <path>")
        return
    
    train_loader = create_dataloader(
        data_root=str(data_root),
        info_path=str(train_info),
        batch_size=args.batch_size,
        num_workers=2,
        split='train'
    )
    
    val_loader = create_dataloader(
        data_root=str(data_root),
        info_path=str(val_info),
        batch_size=args.batch_size,
        num_workers=2,
        split='val'
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Derive feature map size from the model grid (backbone downsamples by 2)
    fm_h = model.grid_size[1] // 2   # 496 // 2 = 248
    fm_w = model.grid_size[0] // 2   # 432 // 2 = 216

    # Create loss and optimizer
    if args.model == 'centerpoint':
        from src.training.centerpoint_loss import CenterPointLoss
        criterion = CenterPointLoss(
            num_classes=10,
            feature_map_size=(fm_h, fm_w),
            point_cloud_range=model.point_cloud_range,
            voxel_size=model.voxel_size,
        )
    else:
        criterion = get_loss_function(
            num_classes=10,
            feature_map_size=(fm_h, fm_w),
            point_cloud_range=model.point_cloud_range,
            voxel_size=model.voxel_size,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_epochs = 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.num_epochs - warmup_epochs, 1), eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        model.train()
        train_loss = 0
        train_steps = 0
        
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch_dict in enumerate(pbar):
            try:
                # Linear warmup during first epoch
                if epoch < warmup_epochs:
                    warmup_lr = args.lr * (batch_idx + 1) / num_batches
                    for pg in optimizer.param_groups:
                        pg['lr'] = warmup_lr

                # Move to device
                for key in batch_dict:
                    if isinstance(batch_dict[key], torch.Tensor):
                        batch_dict[key] = batch_dict[key].to(device)

                # Forward
                pred_dict = model(batch_dict)

                # Loss
                loss, loss_dict = criterion(batch_dict, pred_dict)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (returns the total norm before clipping)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

                optimizer.step()

                # Log
                train_loss += loss.item()
                train_steps += 1
                global_step += 1

                # TensorBoard per-step
                writer.add_scalar('train/loss', loss.item(), global_step)
                for lk, lv in loss_dict.items():
                    writer.add_scalar(f'train/{lk}', lv, global_step)
                writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg': f"{train_loss/train_steps:.4f}"
                })
            
            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_train_loss = train_loss / max(train_steps, 1)
        
        # Validate
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_dict in pbar:
                try:
                    # Move to device
                    for key in batch_dict:
                        if isinstance(batch_dict[key], torch.Tensor):
                            batch_dict[key] = batch_dict[key].to(device)
                    
                    # Forward
                    pred_dict = model(batch_dict)
                    
                    # Loss
                    loss, _ = criterion(batch_dict, pred_dict)
                    
                    val_loss += loss.item()
                    val_steps += 1
                    
                    pbar.set_postfix({'val_loss': f"{val_loss/val_steps:.4f}"})
                
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_val_loss = val_loss / max(val_steps, 1)
        
        # Learning rate step (only after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # TensorBoard epoch-level
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch + 1)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch + 1)
        writer.add_scalar('epoch/lr', scheduler.get_last_lr()[0], epoch + 1)

        # Log epoch summary
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:   {avg_val_loss:.4f}")
        logger.info(f"  LR:         {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_type': args.model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / 'latest.pth')
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, output_dir / 'best.pth')
            logger.info(f"  ✅ New best model saved!")
        
        # Save periodic
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, output_dir / f'epoch_{epoch+1}.pth')
    
    writer.close()
    logger.info(f"\n🎉 Training complete!")
    logger.info(f"   Best val loss: {best_val_loss:.4f}")
    logger.info(f"   Models saved in: {output_dir}")
    logger.info(f"   TensorBoard: tensorboard --logdir {tb_dir}")


if __name__ == '__main__':
    main()
