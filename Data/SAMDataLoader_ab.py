import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from datetime import datetime

class SAMDataLoader(Dataset):
    def __init__(self,
                 npz_dir,
                 is_train=False,
                 enable_mixup=True,       # New parameter to control mixup on/off
                 mixup_alpha=0.2,
                 mixup_prob=1.0,
                 image_size=(256, 256),
                 seed=None,
                 save_aug_preview_prob=0.001,
                 preview_dir="augmentation_vis"):
        """
        Args:
            npz_dir: directory with .npz files
            is_train: whether to apply augmentations + mixup
            enable_mixup: master switch to enable/disable mixup operations
            mixup_alpha: alpha parameter for Beta distribution
            mixup_prob: probability of applying mixup when enabled
            image_size: (H, W) for Resize
            seed: random seed (kept for API compatibility, global seed is set by pl.seed_everything)
            save_aug_preview_prob: probability of saving augmentation preview (0-1)
            preview_dir: directory for saving preview images
        """
        self.files = [os.path.join(npz_dir, f)
                      for f in os.listdir(npz_dir)
                      if f.endswith('.npz')]
        self.is_train = is_train
        self.enable_mixup = enable_mixup  # Master switch for mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        h, w = image_size
        
        # Augmentation preview parameters
        self.save_aug_preview_prob = save_aug_preview_prob
        self.preview_dir = preview_dir
        if self.save_aug_preview_prob > 0:
            os.makedirs(self.preview_dir, exist_ok=True)

        # minimal augmentation: horizontal flip, small rotation, and random scale
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),  # ±20% scaling
                A.Resize(h, w),
            ], additional_targets={
                'mask': 'mask',
                'reflection_map': 'image',
                'removal_reflection': 'image',
                'heatmap_diff': 'mask'
            })
        else:
            self.transform = A.Compose([
                A.Resize(h, w),
            ], additional_targets={
                'mask': 'mask',
                'reflection_map': 'image',
                'removal_reflection': 'image',
                'heatmap_diff': 'mask'
            })

    def __len__(self):
        return len(self.files)

    def _load_raw(self, idx):
        """Load raw data dict from npz (no transforms, no tensor)"""

        data = np.load(self.files[idx], allow_pickle=True)
        
        # Convert from BCHW to HWC format
        img = data['data'][0].transpose(1, 2, 0)  # (3, 512, 512) -> (512, 512, 3)
        mask = data['label'][0].transpose(1, 2, 0)  # (1, 512, 512) -> (512, 512, 1)
        refl = data['reflection_map'][0].transpose(1, 2, 0)  # (3, 512, 512) -> (512, 512, 3)
        rem = data['removal_reflection'][0].transpose(1, 2, 0)  # (3, 512, 512) -> (512, 512, 3)
        
        # Special handling for heatmap_diff with extra dimensions (1, 1, 512, 512, 1)
        heat = None
        if 'heatmap_diff' in data and data['heatmap_diff'] is not None:
            # Handle extra dimensions
            if data['heatmap_diff'].shape == (1, 1, 512, 512, 1):
                heat = data['heatmap_diff'][0, 0]  # Get (512, 512, 1)
            else:
                heat = data['heatmap_diff'][0].transpose(1, 2, 0)  # Standard BCHW to HWC
        
        # Keep image_embeddings as is, no transposition needed
        emb = None
        if 'image_embeddings' in data:
            emb = data['image_embeddings']  # Keep as is
            
        sample = {
            'image': img,
            'mask': mask,
            'reflection_map': refl,
            'removal_reflection': rem
        }
        
        if heat is not None:
            sample['heatmap_diff'] = heat
        if emb is not None:
            sample['image_embeddings'] = emb
            
        return sample
        

    def save_augmentation_preview(self, raw_sample, aug_sample, idx):
        """Save comparison images showing before and after data augmentation"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aug_preview_{timestamp}_idx{idx}.jpg"
        filepath = os.path.join(self.preview_dir, filename)
        
        # Create a 4x2 figure to display original and augmented images and masks
        fig = plt.figure(figsize=(15, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 4)
        
        # First row: Original images
        # Main image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(raw_sample['image'])
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Mask
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(raw_sample['mask'].squeeze(), cmap='gray')
        ax2.set_title('Original Mask')
        ax2.axis('off')
        
        # Reflection map
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(raw_sample['reflection_map'])
        ax3.set_title('Original Reflection Map')
        ax3.axis('off')
        
        # Removal reflection
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(raw_sample['removal_reflection'])
        ax4.set_title('Original Removal Reflection')
        ax4.axis('off')
        
        # Second row: Augmented images
        # Main image
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(aug_sample['image'])
        ax5.set_title('Augmented Image')
        ax5.axis('off')
        
        # Mask
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(aug_sample['mask'].squeeze(), cmap='gray')
        ax6.set_title('Augmented Mask')
        ax6.axis('off')
        
        # Reflection map
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(aug_sample['reflection_map'])
        ax7.set_title('Augmented Reflection Map')
        ax7.axis('off')
        
        # Removal reflection
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.imshow(aug_sample['removal_reflection'])
        ax8.set_title('Augmented Removal Reflection')
        ax8.axis('off')
        
        # Add top title
        dataset_type = "Training Set" if self.is_train else "Validation/Test Set"
        fig.suptitle(f'Data Augmentation Preview - {dataset_type} - Sample {idx}', fontsize=16)
        
        # Save the image
        plt.savefig(filepath)
        plt.close(fig)

    def __getitem__(self, idx):
        # 1. Load and augment first sample
        raw1 = self._load_raw(idx)
        
        # Check input shape consistency and print any issues
        shapes = {k: v.shape for k, v in raw1.items() if isinstance(v, np.ndarray) and k != 'image_embeddings'}
        heights = [s[0] for s in shapes.values()]
        widths = [s[1] for s in shapes.values()]
        
        if len(set(heights)) > 1 or len(set(widths)) > 1:
            print(f"Warning: Inconsistent data shapes (file {self.files[idx]}):")
            for k, v in shapes.items():
                print(f"  - {k}: {v}")
        
        # Apply transforms
        aug1 = self.transform(**raw1)
        
        # Randomly decide whether to save augmentation preview
        save_preview = random.random() < self.save_aug_preview_prob
        if save_preview:
            self.save_augmentation_preview(raw1, aug1, idx)

        def to_tensor(aug):
            out = {
                'data': torch.from_numpy(aug['image'].transpose(2, 0, 1)),
                'label': torch.from_numpy(aug['mask'].transpose(2, 0, 1)),
                'reflection_map': torch.from_numpy(aug['reflection_map'].transpose(2, 0, 1)),
                'removal_reflection': torch.from_numpy(aug['removal_reflection'].transpose(2, 0, 1)),
            }
            if 'heatmap_diff' in aug:
                out['heatmap_diff'] = torch.from_numpy(aug['heatmap_diff'].transpose(2, 0, 1))
            if 'image_embeddings' in aug:
                emb = np.array(aug['image_embeddings'])
                if emb.ndim == 3:
                    emb = emb[np.newaxis, ...]
                out['image_embeddings'] = torch.from_numpy(emb)
            return out

        sample1 = to_tensor(aug1)

        # 2. MixUp if enabled - check both the master switch and probability
        if self.is_train and self.enable_mixup and random.random() < self.mixup_prob:
            idx2 = random.randrange(len(self.files))
            raw2 = self._load_raw(idx2)
            aug2 = self.transform(**raw2)
            sample2 = to_tensor(aug2)

            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1 - lam)

            mixed = {}
            for key in ['data', 'reflection_map', 'removal_reflection']:
                mixed[key] = lam * sample1[key] + (1 - lam) * sample2[key]
            mixed['label'] = lam * sample1['label'] + (1 - lam) * sample2['label']
            if 'heatmap_diff' in sample1 and 'heatmap_diff' in sample2:
                mixed['heatmap_diff'] = lam * sample1['heatmap_diff'] + (1 - lam) * sample2['heatmap_diff']
            if 'image_embeddings' in sample1 and 'image_embeddings' in sample2:
                mixed['image_embeddings'] = lam * sample1['image_embeddings'] + (1 - lam) * sample2['image_embeddings']

            # If preview is enabled and mixup is performed, also save mixup result preview
            if save_preview:
                # Convert mixed results back to HWC format for visualization
                mixed_vis = {
                    'image': mixed['data'].permute(1, 2, 0).cpu().numpy(),
                    'mask': mixed['label'].permute(1, 2, 0).cpu().numpy(),
                    'reflection_map': mixed['reflection_map'].permute(1, 2, 0).cpu().numpy(),
                    'removal_reflection': mixed['removal_reflection'].permute(1, 2, 0).cpu().numpy()
                }
                
                # Save mixup result preview
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"mixup_preview_{timestamp}_idx{idx}_{idx2}.jpg"
                filepath = os.path.join(self.preview_dir, filename)
                
                # Create a 2x2 figure to display only the mixed results
                fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                
                # Main image
                axs[0, 0].imshow(mixed_vis['image'])
                axs[0, 0].set_title(f'Mixup Image (λ={lam:.2f})')
                axs[0, 0].axis('off')
                
                # Mask
                axs[0, 1].imshow(mixed_vis['mask'].squeeze(), cmap='gray')
                axs[0, 1].set_title('Mixup Mask')
                axs[0, 1].axis('off')
                
                # Reflection map
                axs[1, 0].imshow(mixed_vis['reflection_map'])
                axs[1, 0].set_title('Mixup Reflection Map')
                axs[1, 0].axis('off')
                
                # Removal reflection
                axs[1, 1].imshow(mixed_vis['removal_reflection'])
                axs[1, 1].set_title('Mixup Removal Reflection')
                axs[1, 1].axis('off')
                
                # Add top title
                plt.suptitle(f'Mixup Preview - Sample {idx} mixed with Sample {idx2} (λ={lam:.2f})', fontsize=16)
                
                # Save image
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close(fig)

            return mixed

        return sample1