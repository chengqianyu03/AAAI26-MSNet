#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSNet Inference & Mask Generation Script.

This script loads a trained SAM-based model and generates binary segmentation masks
for a test dataset across multiple thresholds. It saves the masks as PNG images 
(0 for background, 255 for foreground) and computes IoU metrics.

Intended for generating qualitative results or competition submissions.
"""

import argparse
import datetime
import os
import random
import sys
import traceback
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure project root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from Models.Unified_Glass_Segmentor import UnifiedGlassDetectionModel
from Data.PLdataModule import SimpleDermoscopicDataModule


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MSNet Inference: Generate Masks at Multiple Thresholds')
    
    # Path configurations
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing data (with train/val/test folders)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='/mnt/tmp/inference2/',
                        help='Directory to save results')
    
    # Model configurations
    parser.add_argument('--sam_model_type', type=str, default='vit_h',
                        help='SAM model type')
    parser.add_argument('--sam_checkpoint', type=str, 
                        default='checkpoint/sam_vit_h_4b8939.pth',
                        help='Path to SAM checkpoint')
    
    # Inference parameters
    parser.add_argument('--vis_samples', type=int, default=605,
                        help='Number of samples to process')
    parser.add_argument('--vis_thresholds', type=float, nargs='+', 
                        default=[0.2, 0.3, 0.4, 0.5],
                        help='Thresholds for binary masks')
    parser.add_argument('--use_original_size', action='store_true', default=True,
                        help='Use original size masks (Stage 2) instead of 512x512 (Stage 1)')
    
    # System settings
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use for computation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def load_model(
    ckpt_path: str, 
    device: torch.device, 
    sam_model_name: str = "vit_b", 
    sam_checkpoint: str = "checkpoint/sam_vit_b_01ec64.pth"
) -> Tuple[torch.nn.Module, torch.device]:
    """
    Load trained model with robustness against GPU unavailability.
    """
    try:
        print(f"Loading model from {ckpt_path}...")
        model = UnifiedGlassDetectionModel.load_from_checkpoint(
            ckpt_path,
            sam_model_name=sam_model_name,
            sam_checkpoint=sam_checkpoint,
            strict=False
        )
        
        model.eval()
        
        try:
            model = model.to(device)
            print(f"Successfully loaded model to {device}")
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"\nWARNING: {str(e)}")
                print("GPU appears to be busy or unavailable. Falling back to CPU.")
                device = torch.device("cpu")
                model = model.to(device)
            else:
                raise e
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
        
    return model, device


def get_original_filename(batch: Dict, index: int) -> str:
    """
    Safely extract original filename or image ID from batch data.
    Handles various data formats (List, Tensor, Scalar).
    """
    if 'image_id' in batch:
        val = batch['image_id']
        # Handle list wrapping
        if isinstance(val, list):
            return str(val[0])
        # Handle tensor wrapping
        elif isinstance(val, torch.Tensor):
            if val.numel() == 1:
                return str(val.item())
            try:
                # Try to extract first item if it's a sequence
                return str(val[0].item() if isinstance(val[0], torch.Tensor) else val[0])
            except Exception:
                pass
        # Handle scalar
        else:
            return str(val)
            
    # Fallbacks
    if 'filename' in batch:
        return str(batch['filename'])
    if 'name' in batch:
        return str(batch['name'])
        
    return f"sample_{index}"


def calculate_iou(
    gt_mask: Union[np.ndarray, torch.Tensor], 
    pred_mask: Union[np.ndarray, torch.Tensor], 
    threshold: float = 0.5
) -> float:
    """
    Calculate Intersection over Union (IoU) between ground truth and prediction.
    """
    # 1. Process Ground Truth
    if gt_mask is not None:
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.detach().cpu().numpy()
        
        if gt_mask.ndim > 2:
            gt_mask = gt_mask.squeeze()
        
        # Normalize to 0-1
        if gt_mask.max() > 1.0:
            gt_mask = gt_mask / 255.0
        
        gt_binary = gt_mask > 0.5
    else:
        # Fallback if no GT
        shape = pred_mask.shape[-2:] if pred_mask is not None else (512, 512)
        gt_binary = np.zeros(shape, dtype=bool)

    # 2. Process Prediction
    if pred_mask is not None:
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
            
        if pred_mask.ndim > 2:
            pred_mask = pred_mask.squeeze()
        
        pred_binary = pred_mask > threshold
    else:
        pred_binary = np.zeros_like(gt_binary, dtype=bool)
    
    # 3. Handle extra dimensions if any remained (batch dim)
    if gt_binary.ndim > 2:
        gt_binary = gt_binary[0] if gt_binary.shape[0] == 1 else gt_binary.mean(axis=0) > 0.5
    if pred_binary.ndim > 2:
        pred_binary = pred_binary[0] if pred_binary.shape[0] == 1 else pred_binary.mean(axis=0) > 0.5
    
    # 4. Compute Metrics
    tp = np.sum((gt_binary == True) & (pred_binary == True))
    fp = np.sum((gt_binary == False) & (pred_binary == True))
    fn = np.sum((gt_binary == True) & (pred_binary == False))
    
    iou = tp / (tp + fp + fn + 1e-8)
    return float(iou)


def generate_masks(
    model: torch.nn.Module, 
    data_module: SimpleDermoscopicDataModule, 
    output_dir: str, 
    device: torch.device, 
    vis_samples: int = 100, 
    use_original_size: bool = True, 
    vis_thresholds: List[float] = [0.4], 
    seed: int = 42
) -> Dict[float, List[float]]:
    """
    Main loop to generate binary masks and calculate metrics.
    """
    # Set reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Setup directories
    thresh_dirs = {}
    summary_files = {}
    
    for threshold in vis_thresholds:
        t_dir = os.path.join(output_dir, f"threshold_{threshold:.2f}")
        os.makedirs(t_dir, exist_ok=True)
        thresh_dirs[threshold] = t_dir
        
        s_path = os.path.join(t_dir, 'iou_summary.csv')
        with open(s_path, 'w') as f:
            f.write("Image_ID,IoU\n")
        summary_files[threshold] = s_path

    # Prepare Data
    data_module.setup("test")
    test_dataset = data_module.test_dataset
    total_samples = len(test_dataset)
    
    # Sample selection
    if vis_samples >= total_samples:
        sample_indices = list(range(total_samples))
    else:
        sample_indices = random.sample(range(total_samples), vis_samples)
    
    # Metadata
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_user = os.environ.get('USER', 'unknown')
    
    print("\n" + "="*60)
    print(f"MSNet Mask Generation")
    print(f"Date/Time: {current_time}")
    print(f"User: {current_user}")
    print(f"Device: {device}")
    print(f"Mode: {'ORIGINAL Resolution (Stage 2)' if use_original_size else 'STANDARD 512x512 (Stage 1)'}")
    print(f"Processing {len(sample_indices)} samples")
    print(f"Thresholds: {vis_thresholds}")
    print("="*60 + "\n")
    
    # Results container
    all_ious = {t: [] for t in vis_thresholds}
    
    with torch.no_grad():
        for idx, sample_idx in enumerate(tqdm(sample_indices, desc="Generating Masks")):
            # 1. Load Sample
            sample = test_dataset[sample_idx]
            
            # Normalize format to dict
            if not isinstance(sample, dict):
                # Handle cases where dataset returns tuples (img, label)
                sample = {'data': sample[0], 'label': sample[1]}
            
            # Create batch (add batch dim)
            batch = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.unsqueeze(0)
                else:
                    batch[k] = v
            
            # Extract Metadata
            image_id = get_original_filename(batch, sample_idx)
            
            # Move to Device
            batch_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            
            # 2. Inference
            outputs = model(batch_device)
            
            # Handle Output Format
            if isinstance(outputs, dict) and 'stage_one' in outputs and 'stage_two' in outputs:
                logits = outputs['stage_two'] if use_original_size else outputs['stage_one']
            else:
                logits = outputs
            
            # Activation
            probs = model._apply_activation(logits, is_training=False)
            
            # 3. Get Ground Truth
            if use_original_size and 'label_orig' in batch:
                gt_mask = batch['label_orig']
            else:
                gt_mask = batch['label']
            
            # 4. Process per Threshold
            # Convert to numpy for processing
            probs_np = probs.cpu().numpy() if isinstance(probs, torch.Tensor) else probs
            if probs_np.ndim > 3: probs_np = probs_np.squeeze(0) # Remove batch dim
            if probs_np.ndim > 2: probs_np = probs_np.squeeze(0) # Remove channel dim if 1
            
            for threshold in vis_thresholds:
                # Calculate IoU
                iou = calculate_iou(gt_mask, probs, threshold=threshold)
                all_ious[threshold].append(iou)
                
                # Generate Binary Mask Image (0 or 255)
                binary_mask = (probs_np > threshold).astype(np.uint8) * 255
                
                # Save Image
                save_name = f"{image_id}_{threshold:.2f}_MSNet.png"
                save_path = os.path.join(thresh_dirs[threshold], save_name)
                cv2.imwrite(save_path, binary_mask)
                
                # Log Result
                with open(summary_files[threshold], 'a') as f:
                    f.write(f"{image_id},{iou:.6f}\n")
                
            # Optional console log (per 10 samples to avoid clutter)
            if idx % 10 == 0:
                avg_current_iou = np.mean([all_ious[t][-1] for t in vis_thresholds])
                # print(f"Processed {image_id} | IoU: {avg_current_iou:.4f}")

    # Final Summary
    summary_csv = os.path.join(output_dir, 'threshold_comparison.csv')
    print("\n" + "="*30)
    print(f"{'Threshold':<10} {'Avg IoU':<12}")
    print("-" * 30)
    
    with open(summary_csv, 'w') as f:
        f.write("Threshold,Avg_IoU\n")
        for threshold in vis_thresholds:
            avg_iou = np.mean(all_ious[threshold]) if all_ious[threshold] else 0.0
            print(f"{threshold:<10.2f} {avg_iou:<12.4f}")
            f.write(f"{threshold:.2f},{avg_iou:.6f}\n")
            
    print("="*30)
    print(f"Masks saved to: {output_dir}")
    print(f"Summary saved to: {summary_csv}")
    
    return all_ious


def main():
    """Main entry point."""
    args = parse_args()
    
    # GPU Setup with Fallback
    try:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {args.gpu} - {torch.cuda.get_device_name(args.gpu)}")
        else:
            print("CUDA not available, using CPU")
    except Exception as e:
        print(f"Error setting up GPU: {e}, falling back to CPU.")
        device = torch.device("cpu")
        
    # Data Module
    data_module = SimpleDermoscopicDataModule(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Model Loading
    try:
        model, device = load_model(
            args.checkpoint_path, 
            device=device, 
            sam_model_name=args.sam_model_type,
            sam_checkpoint=args.sam_checkpoint
        )
    except Exception:
        traceback.print_exc()
        sys.exit(1)
        
    # Execution
    generate_masks(
        model=model,
        data_module=data_module,
        output_dir=args.output_dir,
        device=device,
        vis_samples=args.vis_samples,
        use_original_size=args.use_original_size,
        vis_thresholds=args.vis_thresholds,
        seed=args.seed
    )


if __name__ == "__main__":
    main()