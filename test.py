import os
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
import pandas as pd

from Models.MSNet import BaselineSAMModel
from Data.PLdataModule import SimpleDermoscopicDataModule

import Models.reflection


def calculate_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(np.float32)
    gt = gt_mask.astype(np.float32)

    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)
    tn = np.sum((1 - pred) * (1 - gt))

    iou = tp / (tp + fp + fn + 1e-8)
    mae = np.mean(np.abs(pred - gt))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    beta = 0.3
    beta_sq = beta * beta
    f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-8)

    specificity = tn / (tn + fp + 1e-8)
    ber = 1 - 0.5 * (recall + specificity)

    return {'IoU': iou, 'MAE': mae, 'Fb': f_beta, 'BER': ber}


def parse_args():
    parser = argparse.ArgumentParser(description='AAAI MSNet Evaluation (SAM ViT-H)')

    parser.add_argument('--data_dir', type=str, default="/mnt/tmp/Onpzs_gsd_t")
    parser.add_argument('--checkpoint_path', type=str, default="/mnt/tmp/AAAIFinalT/MSNet/msnet-sam-epoch=02-val/iou=0.6947.ckpt")
    parser.add_argument('--output_dir', type=str, default='AAAI26/evaluation_results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--multi_threshold', action='store_true', default=True,
                        help='Evaluate across multiple thresholds')

    return parser.parse_args()


def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.checkpoint_path}...")
    model = BaselineSAMModel.load_from_checkpoint(
        args.checkpoint_path,
        strict=False
    )
    model.eval().to(device)

    data_module = SimpleDermoscopicDataModule(
        args.data_dir, batch_size=1, num_workers=args.num_workers
    )
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    if args.multi_threshold:
        thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]
    else:
        thresholds = [args.threshold]

    all_results = {
        str(t): {'IoU': [], 'MAE': [], 'Fb': [], 'BER': []}
        for t in thresholds
    }

    print(f"\nEvaluating on {len(test_loader)} samples, thresholds={thresholds}\n")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs = model(batch)

            logits = outputs['coarse']
            gt = batch.get('label_orig', batch['label']).cpu().numpy().squeeze()

            probs = torch.sigmoid(logits).cpu().numpy().squeeze()

            if probs.shape != gt.shape:
                probs = cv2.resize(probs, (gt.shape[1], gt.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

            for t in thresholds:
                pred_mask = (probs > t).astype(np.float32)
                metrics = calculate_metrics(pred_mask, gt)
                for k in all_results[str(t)]:
                    all_results[str(t)][k].append(metrics[k])

    print("\n" + "=" * 70)
    print(f"{'Threshold':<12} | {'mIoU':<10} | {'MAE':<10} | {'Fb':<10} | {'BER':<10}")
    print("-" * 70)

    summary_data = []
    for t in thresholds:
        key = str(t)
        res = all_results[key]
        mIoU = np.mean(res['IoU'])
        mMAE = np.mean(res['MAE'])
        mFb  = np.mean(res['Fb'])
        mBER = np.mean(res['BER'])

        print(f"{t:<12.2f} | {mIoU:<10.4f} | {mMAE:<10.4f} | {mFb:<10.4f} | {mBER:<10.4f}")
        summary_data.append({
            'Threshold': t, 'mIoU': mIoU, 'MAE': mMAE, 'Fb': mFb, 'BER': mBER
        })

    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    if len(thresholds) > 1:
        best_idx = np.argmax([s['mIoU'] for s in summary_data])
        best = summary_data[best_idx]
        print(f"\nBest threshold: {best['Threshold']:.2f}  "
              f"(mIoU={best['mIoU']:.4f}, MAE={best['MAE']:.4f}, "
              f"Fb={best['Fb']:.4f}, BER={best['BER']:.4f})")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)