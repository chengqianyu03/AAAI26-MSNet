import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm

class SAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initialize the SAMDataset
        
        Args:
            image_dir: Directory containing original images
            mask_dir: Directory containing mask images
            transform: Optional transforms to apply to images
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        
        print(f"SAMDataset: Found {len(self.image_files)} image files")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get file paths
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, image_file.replace('.jpg', '.png'))

        # Load images
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Image '{image_path}' is empty or unreadable")
        if mask is None:
            raise ValueError(f"Mask '{mask_path}' is empty or unreadable")
        
        # Convert to RGB format
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(mask)
        
        # Apply transform if available
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        # Return sample dictionary
        sample = {
            'data': img,
            'label': mask
        }
        
        return sample

def save_to_npz(dataloader, output_dir):
    """Save samples from dataloader to NPZ files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Saving data to NPZ files in {output_dir}...")
    
    for i, batch in enumerate(tqdm(dataloader, desc="Saving NPZs")):
        data = batch['data'].numpy()
        label = batch['label'].numpy()
        
        # Save to NPZ file
        np.savez(
            os.path.join(output_dir, f'{i}.npz'), 
            data=data, 
            label=label
        )

# Example usage
if __name__ == "__main__":
    image_dir = '/mnt/tmp/T-imgs/GSD-T/test/image'
    mask_dir = '/mnt/tmp/T-imgs/GSD-T/test/mask'
    output_dir = '/mnt/tmp/npzs_gsd_t/test'
    
    # Create transform (only ToTensor, no resize)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = SAMDataset(
        image_dir, 
        mask_dir, 
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Save to NPZ files
    print("Saving data to NPZ files...")
    save_to_npz(dataloader, output_dir)
    
    print("Processing complete!")
