import sys
import torch
import pytorch_lightning as pl
import os
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# Updated Import
from Models.Unified_Glass_Segmentor import UnifiedGlassDetectionModel
from Data.PLdataModule import SimpleDermoscopicDataModule

def parse_args():
    parser = argparse.ArgumentParser(description='Progressive Coarse-to-Fine Glass Detection')
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default="/mnt/tmp/Onpzs_gsd_t/", help='Dataset root')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model Architecture
    parser.add_argument('--module_name', type=str, default='sam1', help='Backbone prompt module')
    parser.add_argument('--sam_model', type=str, default='vit_h')
    parser.add_argument('--sam_checkpoint', type=str, default='rms/checkpoint/sam_vit_h_4b8939.pth')
    
    # Curriculum Learning Parameters
    parser.add_argument('--refinement_warmup', type=int, default=20, 
                        help='Epoch to switch from semantic learning to detail refinement')
    parser.add_argument('--refiner_lr_scale', type=float, default=0.5,
                        help='Learning rate scaling factor for the refinement module')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lora_rank', type=int, default=4)
    
    # Checkpointing
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/')
    parser.add_argument('--save_lora_checkpoint', action='store_true')
    parser.add_argument('--load_lora_checkpoint', type=str, default=None)
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--lora_layers', type=str, default=None)
    
    # Legacy args support (kept for compatibility)
    parser.add_argument('--ft_dec', action='store_true')
    parser.add_argument('--prompt_lr_factor', type=float, default=1.0)
    parser.add_argument('--bce_weight', type=float, default=1.0)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--focal_weight', type=float, default=1.0)
    parser.add_argument('--precision', type=int, default=16)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    
    # System Setup
    torch.set_float32_matmul_precision('high')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    # Data
    data_module = SimpleDermoscopicDataModule(
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Model Initialization
    lora_layers = [int(i) for i in args.lora_layers.split(',')] if args.lora_layers else None
    
    model = UnifiedGlassDetectionModel(
        in_channels=3, 
        out_channels=1, 
        lr=args.lr, 
        sam_model_name=args.sam_model, 
        sam_checkpoint=args.sam_checkpoint,
        module_name=args.module_name,
        lora_rank=args.lora_rank,
        lora_layers=lora_layers,
        # Curriculum Params
        refinement_warmup_epoch=args.refinement_warmup,
        refiner_lr_scale=args.refiner_lr_scale,
        # Loss Params
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight
    )
    
    if args.load_lora_checkpoint:
        model.sam_backbone.load_lora_parameters(args.load_lora_checkpoint)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/iou',
            dirpath=args.ckpt_dir,
            filename=f'{args.module_name}-unified-best',
            save_top_k=1,
            mode='max'
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val_loss', patience=10, mode='min')
    ]
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        gpus=[args.gpu],
        default_root_dir="logs"
    )
    
    print("\n" + "="*60)
    print(f"Unified Progressive Glass Detection Framework")
    print(f"Curriculum Strategy: Semantic Warmup ({args.refinement_warmup} eps) -> Boundary Refinement")
    print("="*60 + "\n")
    
    trainer.fit(model, datamodule=data_module)