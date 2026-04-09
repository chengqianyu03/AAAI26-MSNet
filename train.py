import sys
import torch
import pytorch_lightning as pl
import os
import argparse
import json
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from Models.MSNet import BaselineSAMModel
from Data.PLdataModule import SimpleDermoscopicDataModule

import Models.reflection


def parse_args():
    parser = argparse.ArgumentParser(
        description='AAAI MSNet: SAM ViT-H + Multi-Semantic Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Dataset ──
    data = parser.add_argument_group('Dataset')
    data.add_argument('--data_dir', type=str, default='/mnt/tmp/Onpzs_gsd_t')
    data.add_argument('--batch_size', type=int, default=1)
    data.add_argument('--num_workers', type=int, default=4)

    # ── SAM Backbone ──
    sam = parser.add_argument_group('SAM Backbone')
    sam.add_argument('--sam_checkpoint', type=str, default='/mnt/tmp/checkpoints/sam_vit_h_4b8939.pth')
    sam.add_argument('--sam_model_type', type=str, default='vit_h',
                     choices=['vit_h', 'vit_l', 'vit_b'])
    sam.add_argument('--lora_rank', type=int, default=512,
                     help='SAM LoRA rank (AAAI: 512)')
    sam.add_argument('--ft_dec', action='store_true', default=True,
                     help='Finetune SAM mask decoder')
    sam.add_argument('--no_ft_dec', action='store_false', dest='ft_dec')

    # ── CLIP LoRA ──
    clip_g = parser.add_argument_group('CLIP LoRA')
    clip_g.add_argument('--clip_lora_rank', type=int, default=128,
                        help='CLIP LoRA rank (AAAI: 128)')
    clip_g.add_argument('--clip_lora_alpha', type=int, default=256,
                        help='CLIP LoRA alpha (AAAI: 256)')
    clip_g.add_argument('--clip_layer_idx', type=int, default=10,
                        help='CLIP feature extraction layer index')

    # ── Reflection Estimator ──
    refl = parser.add_argument_group('Reflection Estimator')
    refl.add_argument('--reflection_estimator', type=str, default='lrm',
                      help='Which reflection estimator to use')
    refl.add_argument('--reflection_checkpoint', type=str, default=None)
    refl.add_argument('--reflection_checkpoint_det', type=str, default=None,
                      help='Detection checkpoint (for cvpr2024 two-stage)')
    refl.add_argument('--reflection_proc_size', type=int, default=256)
    refl.add_argument('--reflection_n_iters', type=int, default=3,
                      help='Number of iterations (for LRM)')
    refl.add_argument('--reflection_finetune', action='store_true',
                      help='Unfreeze reflection estimator for fine-tuning')
    refl.add_argument('--reflection_kwargs_json', type=str, default=None,
                      help='JSON string of kwargs (overrides all reflection args)')

    # ── Architecture ──
    arch = parser.add_argument_group('Architecture')
    arch.add_argument('--multi_scale_weight', type=float, default=0.3)
    arch.add_argument('--boundary_warmup_epoch', type=int, default=50)

    # ── Training ──
    train_g = parser.add_argument_group('Training')
    train_g.add_argument('--lr', type=float, default=1e-5,
                         help='Base learning rate (AAAI: 1e-5)')
    train_g.add_argument('--weight_decay', type=float, default=5e-4,
                         help='Weight decay (AAAI: 5e-4)')
    train_g.add_argument('--max_epochs', type=int, default=50)
    train_g.add_argument('--gradient_clip_val', type=float, default=1.0)
    train_g.add_argument('--accumulate_grad_batches', type=int, default=1)

    # ── System ──
    sys_g = parser.add_argument_group('System')
    sys_g.add_argument('--gpu', type=int, default=0)
    sys_g.add_argument('--seed', type=int, default=42)

    import pytorch_lightning as _pl
    _pl_version = tuple(int(x) for x in _pl.__version__.split('.')[:2])
    if _pl_version >= (2, 0):
        _precision_choices = ['32', '16-mixed', 'bf16-mixed']
        _precision_default = 'bf16-mixed'
    else:
        _precision_choices = ['64', '32', '16', 'bf16']
        _precision_default = '16'

    sys_g.add_argument('--precision', type=str, default=_precision_default, choices=_precision_choices)
    sys_g.add_argument('--ckpt_dir', type=str, default='/mnt/tmp/AAAIFinalT/MSNet')
    sys_g.add_argument('--log_dir', type=str, default='logs_baseline')
    sys_g.add_argument('--resume_from', type=str, default=None,
                       help='Resume training from checkpoint')
    sys_g.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')

    return parser.parse_args()


def build_reflection_kwargs(args):
    """Build reflection estimator kwargs from CLI args."""
    if args.reflection_kwargs_json is not None:
        try:
            return json.loads(args.reflection_kwargs_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid --reflection_kwargs_json: {e}")

    name = args.reflection_estimator

    if name == 'cvpr2024' and args.reflection_checkpoint_det:
        return {
            'checkpoint_removal': args.reflection_checkpoint,
            'checkpoint_detection': args.reflection_checkpoint_det,
            'proc_size': args.reflection_proc_size,
            'finetune': args.reflection_finetune,
        }

    return None


if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    torch.set_float32_matmul_precision('high')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    data_module = SimpleDermoscopicDataModule(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    refl_kwargs = build_reflection_kwargs(args)

    model = BaselineSAMModel(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        lr=args.lr,
        lora_rank=args.lora_rank,
        ft_dec=args.ft_dec,
        clip_lora_rank=args.clip_lora_rank,
        clip_lora_alpha=args.clip_lora_alpha,
        clip_layer_idx=args.clip_layer_idx,
        multi_scale_weight=args.multi_scale_weight,
        boundary_warmup_epoch=args.boundary_warmup_epoch,
        reflection_estimator_name=args.reflection_estimator,
        reflection_estimator_kwargs=refl_kwargs,
        reflection_checkpoint=args.reflection_checkpoint,
        reflection_proc_size=args.reflection_proc_size,
        reflection_n_iters=args.reflection_n_iters,
        reflection_finetune=args.reflection_finetune,
    )

    callbacks = [
        ModelCheckpoint(
            monitor='val/iou',
            dirpath=args.ckpt_dir,
            filename='msnet-sam-{epoch:02d}-{val/iou:.4f}',
            save_top_k=2,
            mode='max',
            save_last=True,
        ),
        ModelCheckpoint(
            monitor='val/loss',
            dirpath=args.ckpt_dir,
            filename='msnet-sam-best-loss-{epoch:02d}-{val/loss:.4f}',
            save_top_k=1,
            mode='min',
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(monitor='val/loss', patience=args.patience, mode='min', verbose=True),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        precision=args.precision,
        callbacks=callbacks,
        devices=[args.gpu],
        accelerator='gpu',
        default_root_dir=args.log_dir,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    est_name = args.reflection_estimator
    print("\n" + "=" * 70)
    print("  AAAI MSNet Training: SAM ViT-H + Multi-Semantic Detection")
    print("=" * 70)
    print(f"  Dataset:          {args.data_dir}")
    print(f"  SAM:              {args.sam_model_type} + LoRA(r={args.lora_rank})")
    print(f"  ft_dec:           {args.ft_dec}")
    print(f"  CLIP:             LoRA(r={args.clip_lora_rank}, a={args.clip_lora_alpha})")
    print(f"  Reflection:       {est_name}")
    print(f"  Refl checkpoint:  {args.reflection_checkpoint}")
    print(f"  Refl finetune:    {args.reflection_finetune}")
    print(f"  LR:               {args.lr}")
    print(f"  Weight decay:     {args.weight_decay}")
    print(f"  Precision:        {args.precision}")
    print(f"  Grad accum:       {args.accumulate_grad_batches}")
    print(f"  Grad clip:        {args.gradient_clip_val}")
    print(f"  Max epochs:       {args.max_epochs}")
    print(f"  Patience:         {args.patience}")
    print(f"  Checkpoint dir:   {args.ckpt_dir}")
    if args.resume_from:
        print(f"  Resume from:      {args.resume_from}")
    print("=" * 70 + "\n")

    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from)

    best_path = trainer.checkpoint_callback.best_model_path
    best_score = trainer.checkpoint_callback.best_model_score
    if best_path:
        print(f"\n{'=' * 70}")
        print(f"  Training complete!")
        print(f"  Best checkpoint: {best_path}")
        print(f"  Best val/iou:    {best_score:.4f}")
        print(f"{'=' * 70}\n")