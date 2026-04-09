import pytorch_lightning as pl
import os
import torch
from torch.utils.data import DataLoader
from Data.SAMDataLoader import SAMDataLoader


class SimpleDermoscopicDataModule(pl.LightningDataModule):
    """
    Improved DataModule with optimized worker settings and caching.
    """
    
    def __init__(self, data_dir, batch_size=4, num_workers=4, seed=42):
        """
        Args:
            data_dir: Root directory containing 'train' and 'test' subdirs.
            batch_size: Batch size.
            num_workers: Number of workers. Set to 0 if debugging, 4-8 for training.
            seed: Random seed.
        """
        super().__init__()
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        
        # Performance tuning: Pin memory speeds up host-to-device copy
        self.pin_memory = True if torch.cuda.is_available() else False
        
        print(f"DataModule Configured: Batch={batch_size}, Workers={num_workers}, PinMem={self.pin_memory}")
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Training Dataset: Enable MixUp and Augmentations
            # cache_mode='full' loads all data to RAM. 
            # If you run out of memory, change to 'part' or 'none'.
            self.train_dataset = SAMDataLoader(
                self.train_dir,
                is_train=True,
                seed=self.seed,
                enable_mixup=True,
                mixup_prob=0.5,
            )
            
            # Validation Dataset: No Augmentations
            self.val_dataset = SAMDataLoader(
                self.test_dir,
                is_train=False,
                seed=self.seed,
                enable_mixup=False,
            )
            
            print(f"Datasets ready: {len(self.train_dataset)} Train, {len(self.val_dataset)} Val")
        
        if stage == 'test' or stage is None:
            self.test_dataset = SAMDataLoader(
                self.test_dir, 
                is_train=False, 
                seed=self.seed,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True # Good for training stability
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )