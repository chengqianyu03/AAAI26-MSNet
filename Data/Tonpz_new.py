import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from tqdm import tqdm
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

class SAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, reflection_map_dir, removal_reflection_dir, transform=None):
        """
        Initialize the SAMDataset
        
        Args:
            image_dir: Directory containing original images
            mask_dir: Directory containing mask images
            reflection_map_dir: Directory containing reflection map images
            removal_reflection_dir: Directory containing reflection removal images
            transform: Optional transforms to apply to images
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.reflection_map_dir = reflection_map_dir
        self.removal_reflection_dir = removal_reflection_dir
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
        reflection_map_path = os.path.join(self.reflection_map_dir, image_file.replace('.jpg', '_fake_Rs_03.png'))
        removal_reflection_path = os.path.join(self.removal_reflection_dir, image_file.replace('.jpg', '_fake_Ts_03.png'))

        # Save image ID (filename without extension)
        image_id = os.path.splitext(image_file)[0]

        # Load images
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_path)
        
        if not os.path.exists(reflection_map_path):
            raise FileNotFoundError(f"Reflection map '{reflection_map_path}' not found")
            
        reflection_map = cv2.imread(reflection_map_path)
        if reflection_map is None:
            raise ValueError(f"Reflection map '{reflection_map_path}' is empty or unreadable")
        
        # Load reflection removal image
        if not os.path.exists(removal_reflection_path):
            print(f"Warning: Reflection removal image '{removal_reflection_path}' doesn't exist, using original image instead")
            removal_reflection = img.copy()
        else:
            removal_reflection = cv2.imread(removal_reflection_path)
            if removal_reflection is None:
                print(f"Warning: Failed to read reflection removal image '{removal_reflection_path}', using original image instead")
                removal_reflection = img.copy()
        
        # Convert to RGB format
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(mask)
        reflection_map = Image.fromarray(cv2.cvtColor(reflection_map, cv2.COLOR_BGR2RGB))
        removal_reflection = Image.fromarray(cv2.cvtColor(removal_reflection, cv2.COLOR_BGR2RGB))
        
        # Apply transform if available
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
            reflection_map = self.transform(reflection_map)
            removal_reflection = self.transform(removal_reflection)
        
        # Return sample dictionary
        sample = {
            'data': img,
            'label': mask,
            'reflection_map': reflection_map,
            'removal_reflection': removal_reflection,
            'image_id': image_id
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
        reflection_map = batch['reflection_map'].numpy()
        removal_reflection = batch['removal_reflection'].numpy()
        image_id = batch['image_id']
        
        # Save to NPZ file
        np.savez(
            os.path.join(output_dir, f'{i}.npz'), 
            data=data, 
            label=label, 
            reflection_map=reflection_map,
            removal_reflection=removal_reflection,
            image_id=image_id
        )

# Example usage
if __name__ == "__main__":
    image_dir = '/mnt/tmp/T-imgs/GSD-T/test/image'
    mask_dir = '/mnt/tmp/T-imgs/GSD-T/test/mask'
    reflection_map_dir = '/mnt/tmp/T-imgs/GSD-T/test/reflectionmaps'
    removal_reflection_dir = '/mnt/tmp/T-imgs/GSD-T/test/removedimg'
    output_dir = '/mnt/tmp/npzs_gsd_t/test'
    
    # Create removal reflection directory if it doesn't exist
    os.makedirs(removal_reflection_dir, exist_ok=True)
    
    # Create transform (only ToTensor, no resize)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = SAMDataset(
        image_dir, 
        mask_dir, 
        reflection_map_dir,
        removal_reflection_dir,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Save to NPZ files
    print("Saving data to NPZ files...")
    save_to_npz(dataloader, output_dir)
    
    print("Processing complete!")