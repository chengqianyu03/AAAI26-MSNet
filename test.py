import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import datetime
from Models.Unified_Glass_Segmentor import UnifiedGlassDetectionModel
from Data.PLdataModule import SimpleDermoscopicDataModule

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation: Progressive Glass Detection (Multi-Threshold)')
    
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_refined_mask', action='store_true', default=True,
                        help='Use high-resolution refined mask (original size) for evaluation')
    parser.add_argument('--sam_model_type', type=str, default='vit_h')
    
    # Add other necessary args for data module
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    
    return parser.parse_args()

def calculate_metrics(pred_mask, gt_mask):
    """
    Standard Segmentation Metrics including IoU, MAE, F-measure (Fb), BER, ACC
    """
    pred_binary = pred_mask.astype(np.float32)
    gt_binary = gt_mask.astype(np.float32)
    
    # Calculate TP, FP, TN, FN
    tp = np.sum(pred_binary * gt_binary)
    fp = np.sum(pred_binary * (1 - gt_binary))
    fn = np.sum((1 - pred_binary) * gt_binary)
    tn = np.sum((1 - pred_binary) * (1 - gt_binary))
    
    # 1. IoU
    iou = tp / (tp + fp + fn + 1e-8)
    
    # 2. Accuracy
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    
    # 3. MAE
    mae = np.mean(np.abs(pred_binary - gt_binary))
    
    # 4. F-measure (F_beta) with beta^2 = 0.3 (standard for glass/salient object detection)
    # Note: Some implementations use beta=0.3 directly, here we follow the standard convention
    # where beta^2 is often cited as 0.3. If you specifically need beta=0.3 (beta^2=0.09), check your requirement.
    # Here using the definition from common benchmarks.
    # If following previous realtest.py: beta = 0.3 -> beta_square = 0.09
    beta = 0.3
    beta_square = beta * beta
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f_beta = (1 + beta_square) * precision * recall / (beta_square * precision + recall + 1e-8)
    
    # 5. BER (Balanced Error Rate)
    specificity = tn / (tn + fp + 1e-8)
    ber = 1 - 0.5 * (recall + specificity)
    
    return {
        'IoU': iou, 
        'ACC': acc, 
        'MAE': mae, 
        'Fb': f_beta, 
        'BER': ber,
        'Recall': recall,
        'Precision': precision
    }

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    print(f"Loading model from {args.checkpoint_path}...")
    model = UnifiedGlassDetectionModel.load_from_checkpoint(
        args.checkpoint_path,
        sam_model_name=args.sam_model_type,
        strict=False
    )
    model.eval().to(device)
    
    # Load Data
    data_module = SimpleDermoscopicDataModule(args.data_dir, batch_size=1, num_workers=args.num_workers)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    # Define thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    
    # Initialize results container
    # Structure: {'0.5': {'IoU': [], 'MAE': [], ...}, '0.1': ...}
    all_results = {str(t): {'IoU': [], 'MAE': [], 'Fb': [], 'BER': [], 'ACC': []} for t in thresholds}
    
    mask_type_str = 'REFINED (High-Res)' if args.use_refined_mask else 'COARSE (512x512)'
    print(f"\nStarting Evaluation using {mask_type_str} masks...")
    print(f"Evaluating Thresholds: {thresholds}")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            
            # Inference
            outputs = model(batch)
            
            # Select Output based on mode
            if args.use_refined_mask:
                logits = outputs['refined']
                # Use Original Label
                if 'label_orig' in batch:
                    gt = batch['label_orig'].cpu().numpy().squeeze()
                else:
                    gt = batch['label'].cpu().numpy().squeeze() # Fallback
            else:
                logits = outputs['coarse']
                gt = batch['label'].cpu().numpy().squeeze()
                
            # Get Probabilities (Sigmoid) - Do this ONCE per image
            probs = model._apply_activation(logits).cpu().numpy().squeeze()
            
            # Resize handling if shapes mismatch (Robustness)
            if probs.shape != gt.shape:
                import cv2
                # Note: Resize probs, not binary mask, to keep precision
                probs = cv2.resize(probs, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Loop through thresholds
            for t in thresholds:
                pred_mask = (probs > t).astype(np.float32)
                metrics = calculate_metrics(pred_mask, gt)
                
                # Store metrics
                for k in all_results[str(t)]:
                    all_results[str(t)][k].append(metrics[k])
            
    # --- Summary & Reporting ---
    print("\n" + "="*80)
    print(f"{'Threshold':<10} | {'mIoU':<10} | {'MAE':<10} | {'Fb':<10} | {'BER':<10} | {'ACC':<10}")
    print("-" * 80)
    
    summary_data = []
    
    for t in thresholds:
        key = str(t)
        res = all_results[key]
        
        mIoU = np.mean(res['IoU'])
        mMAE = np.mean(res['MAE'])
        mFb = np.mean(res['Fb'])
        mBER = np.mean(res['BER'])
        mACC = np.mean(res['ACC'])
        
        print(f"{t:<10.2f} | {mIoU:<10.4f} | {mMAE:<10.4f} | {mFb:<10.4f} | {mBER:<10.4f} | {mACC:<10.4f}")
        
        summary_data.append({
            'Threshold': t,
            'mIoU': mIoU,
            'MAE': mMAE,
            'Fb': mFb,
            'BER': mBER,
            'ACC': mACC
        })
    print("="*80 + "\n")
    
    # Save results to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Summary CSV
    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, 'evaluation_summary_multithreshold.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    # 2. Detailed CSV (Optional: Save detailed results for best threshold, e.g., 0.5)
    best_t = 0.5
    df_detail = pd.DataFrame(all_results[str(best_t)])
    detail_path = os.path.join(args.output_dir, f'results_threshold_{best_t}.csv')
    df_detail.to_csv(detail_path, index=False)
    print(f"Detailed results (t={best_t}) saved to: {detail_path}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)