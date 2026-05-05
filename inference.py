"""
AAAI MSNet Inference (SAM ViT-H)

Usage:
  python inference_base.py \
      --checkpoint_path /mnt/tmp/AAAI/MSNet/best.ckpt \
      --input_image /path/to/image.png \
      --output_dir results/
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from Models.MSNet import BaselineSAMModel

import Models.reflection


def load_image(image_path, image_size=512):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_orig = img.copy()

    img_resized = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(
        img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, H, W)

    return img_tensor, img_orig


def save_results(img_orig, pred_prob, pred_binary, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)

    # Probability map
    cv2.imwrite(os.path.join(output_dir, f"{name}_prob.png"),
                (pred_prob * 255).astype(np.uint8))

    # Binary mask
    mask_uint8 = (pred_binary * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, f"{name}_mask.png"), mask_uint8)

    # Overlay (red tint + green contour)
    img_display = cv2.resize(img_orig, (pred_binary.shape[1], pred_binary.shape[0]),
                             interpolation=cv2.INTER_LINEAR) if img_orig.shape[:2] != pred_binary.shape else img_orig.copy()
    overlay = img_display.astype(np.float32)
    glass = pred_binary[:, :, None]
    red = np.zeros_like(overlay); red[:, :, 0] = 255
    overlay = (overlay * (1 - 0.4 * glass) + red * 0.4 * glass).astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{name}_overlay.png"), overlay_bgr)

    print(f"Saved: {name}_prob.png, {name}_mask.png, {name}_overlay.png")


def parse_args():
    p = argparse.ArgumentParser(description='AAAI MSNet Inference (SAM ViT-H)')
    p.add_argument('--input_image', type=str, default="AAAI26/TestImg/95ed3544c2da3a923eba77ac74478ef.jpg")
    p.add_argument('--checkpoint_path', type=str, default="/mnt/tmp/AAAIFinalT/MSNet/msnet-sam-epoch=08-val/iou=0.7991.ckpt")
    #/mnt/tmp/AAAIFinalT/MSNet/msnet-sam-epoch=08-val/iou=0.7991.ckpt
    #/mnt/tmp/AAAIFinalT/MSNet/msnet-sam-epoch=13-val/iou=0.7974.ckpt
    p.add_argument('--output_dir', type=str, default='AAAI26/inference_results')
    p.add_argument('--threshold', type=float, default=0.2)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--image_size', type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")

    print(f"Loading model from: {args.checkpoint_path}")
    model = BaselineSAMModel.load_from_checkpoint(args.checkpoint_path, strict=False)
    model.eval().to(device)

    print(f"Loading image: {args.input_image}")
    img_tensor, img_orig = load_image(args.input_image, args.image_size)
    batch = {'data': img_tensor.to(device)}

    with torch.no_grad():
        outputs = model(batch)

    probs = torch.sigmoid(outputs['coarse']).cpu().numpy().squeeze()
    pred_binary = (probs > args.threshold).astype(np.float32)

    name = os.path.splitext(os.path.basename(args.input_image))[0]
    save_results(img_orig, probs, pred_binary, args.output_dir, name)
    print("Done.")


if __name__ == "__main__":
    main()