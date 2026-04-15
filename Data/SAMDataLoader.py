import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import albumentations as A

class SAMDataLoader(Dataset):
    def __init__(self,
                 npz_dir,
                 is_train=False,
                 enable_mixup=True,       
                 mixup_alpha=0.2,
                 mixup_prob=0.5,
                 image_size=(512, 512),
                 seed=42,
                 save_aug_preview_prob=0.0):
        """
        High-Efficiency SAM DataLoader
        - Lazy loading to prevent RAM explosion
        - Optimized MixUp logic to reduce IO
        """
        self.files = [os.path.join(npz_dir, f)
                      for f in os.listdir(npz_dir)
                      if f.endswith('.npz')]
        self.is_train = is_train
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.image_size = image_size
        self.save_aug_preview_prob = save_aug_preview_prob
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        h, w = image_size
        print(f"SAMDataLoader initialized: {len(self.files)} samples. Train: {is_train}")

        # Augmentation pipeline
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
                A.Resize(h, w),
            ], additional_targets={
                'mask': 'mask',
                'heatmap_diff': 'mask',
            })
        else:
            self.transform = A.Compose([
                A.Resize(h, w),
            ], additional_targets={
                'mask': 'mask',
                'heatmap_diff': 'mask'
            })
            
        # Check data format once
        self.has_dual_masks = False
        if len(self.files) > 0:
            try:
                with np.load(self.files[0], allow_pickle=True) as sample_data:
                    self.has_dual_masks = 'label_512' in sample_data and 'label_orig' in sample_data
                    if self.has_dual_masks:
                        print("Format detected: Dual Masks (label_512 + label_orig)")
            except Exception as e:
                print(f"Warning checking first file: {e}")

    def __len__(self):
        return len(self.files)

    def _load_data_item(self, idx):
        """Reads a single .npz file safely and returns numpy arrays."""
        filepath = self.files[idx]
        try:
            with np.load(filepath, allow_pickle=True) as data:
                # 1. Image: Transpose to HWC and copy to release file handle
                img = data['data'][0].transpose(1, 2, 0).copy()
                
                # 2. Masks
                if self.has_dual_masks and 'label_512' in data:
                    mask_512 = data['label_512'][0].transpose(1, 2, 0).copy()
                    mask_orig = data['label_orig'][0].transpose(1, 2, 0).copy()
                else:
                    raw_mask = data['label'][0].transpose(1, 2, 0).copy()
                    mask_512 = raw_mask
                    mask_orig = raw_mask.copy()

                # 3. Embeddings (Only load if present)
                emb = None
                if 'image_embeddings' in data:
                    emb = data['image_embeddings'].copy()

                # 4. Heatmap
                heat = None
                if 'heatmap_diff' in data:
                    h_raw = data['heatmap_diff']
                    if h_raw is not None:
                        # Handle varied dimensions: (1,1,H,W), (1,H,W), (H,W)
                        if isinstance(h_raw, np.ndarray):
                            if h_raw.ndim >= 4: 
                                heat = h_raw[0].transpose(1, 2, 0).copy()
                            elif h_raw.ndim == 3 and h_raw.shape[0] == 1: 
                                heat = h_raw.transpose(1, 2, 0).copy()
                            elif h_raw.ndim == 2:
                                heat = h_raw[..., None].copy()
                            else:
                                heat = h_raw.copy()

                # 5. Metadata
                orig_shape = mask_orig.shape[:2]
                if 'orig_height' in data and 'orig_width' in data:
                     orig_shape = (int(data['orig_height'][0]), int(data['orig_width'][0]))
                
                image_id = None
                if 'image_id' in data:
                    image_id = data['image_id']
                    if isinstance(image_id, np.ndarray) and image_id.size == 1:
                        image_id = image_id.item()

                return {
                    'image': img,
                    'mask': mask_512,
                    'mask_orig': mask_orig,
                    'heatmap_diff': heat,
                    'image_embeddings': emb,
                    'orig_shape': orig_shape,
                    'image_id': image_id
                }

        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            # Fallback to random sample
            new_idx = random.randint(0, len(self.files) - 1)
            if new_idx != idx:
                return self._load_data_item(new_idx)
            return None

    def __getitem__(self, idx):
        # 1. Load Primary Sample
        raw1 = self._load_data_item(idx)
        if raw1 is None:
             return self.__getitem__(random.randint(0, len(self.files) - 1))

        # Extract items that should NOT be transformed/resized
        mask_orig_1 = raw1.pop('mask_orig')
        image_id_1 = raw1.pop('image_id')
        orig_shape_1 = raw1.pop('orig_shape')
        emb_1 = raw1.pop('image_embeddings')
        
        # Ensure heatmap exists for albumentations
        has_heat = raw1['heatmap_diff'] is not None
        if not has_heat:
            raw1['heatmap_diff'] = np.zeros((raw1['image'].shape[0], raw1['image'].shape[1], 1), dtype=np.float32)

        # Apply Transform
        aug1 = self.transform(**raw1)

        # 2. Mixup Logic (Optimized: Check probability BEFORE loading second file)
        do_mixup = self.is_train and self.enable_mixup and (random.random() < self.mixup_prob)
        
        if do_mixup:
            idx2 = random.randint(0, len(self.files) - 1)
            raw2 = self._load_data_item(idx2)
            
            if raw2 is not None:
                 # Remove unused items
                 raw2.pop('mask_orig') 
                 raw2.pop('image_id')
                 raw2.pop('orig_shape')
                 emb_2 = raw2.pop('image_embeddings')
                 
                 if raw2['heatmap_diff'] is None:
                     raw2['heatmap_diff'] = np.zeros_like(raw2['image'][..., :1])

                 # Transform second sample
                 aug2 = self.transform(**raw2)
                 
                 # Mixup Weights
                 lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                 lam = max(lam, 1 - lam) # bias towards primary

                 # Mix Image & Mask
                 aug1['image'] = lam * aug1['image'] + (1 - lam) * aug2['image']
                 aug1['mask'] = lam * aug1['mask'] + (1 - lam) * aug2['mask']
                 
                 # Mix Heatmap
                 if has_heat or raw2['heatmap_diff'] is not None:
                     aug1['heatmap_diff'] = lam * aug1['heatmap_diff'] + (1 - lam) * aug2['heatmap_diff']

                 # Mix Embeddings (if both exist)
                 if emb_1 is not None and emb_2 is not None:
                     if emb_1.shape == emb_2.shape:
                         emb_1 = lam * emb_1 + (1 - lam) * emb_2
                 elif emb_1 is None and emb_2 is not None:
                     emb_1 = emb_2
        
        # 3. Convert to Tensor
        def to_tensor(x):
            if x.ndim == 2: x = x[..., None]
            # .copy() ensures positive strides for torch
            return torch.from_numpy(x.transpose(2, 0, 1).copy()).float()

        sample = {
            'data': to_tensor(aug1['image']),
            'label': to_tensor(aug1['mask']),
            'label_orig': to_tensor(mask_orig_1), # Original High-Res Mask
            'orig_shape': torch.tensor(orig_shape_1),
        }
        
        if image_id_1 is not None:
            sample['image_id'] = image_id_1
            
        if has_heat or do_mixup:
             sample['heatmap_diff'] = to_tensor(aug1['heatmap_diff'])
             
        if emb_1 is not None:
             if isinstance(emb_1, np.ndarray):
                 sample['image_embeddings'] = torch.from_numpy(emb_1).float()
             else:
                 sample['image_embeddings'] = emb_1.float()

        return sample
