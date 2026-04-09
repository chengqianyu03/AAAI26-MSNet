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

class DualMaskDataset(Dataset):
    def __init__(self, 
                 image_dir, 
                 mask_dir_512,     # 512×512掩码目录
                 mask_dir_orig,    # 原始尺寸掩码目录
                 reflection_map_dir, 
                 removal_reflection_dir, 
                 transform=None):
        """
        从两个不同的掩码文件夹加载数据的数据集类
        
        Args:
            image_dir: 原始图像目录
            mask_dir_512: 512×512掩码目录
            mask_dir_orig: 原始尺寸掩码目录
            reflection_map_dir: 反射图目录
            removal_reflection_dir: 去除反射后图像目录
            transform: 可选的变换函数
        """
        self.image_dir = image_dir
        self.mask_dir_512 = mask_dir_512
        self.mask_dir_orig = mask_dir_orig
        self.reflection_map_dir = reflection_map_dir
        self.removal_reflection_dir = removal_reflection_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        
        print(f"DualMaskDataset: 找到 {len(self.image_files)} 个图像文件")
        print(f"使用两个掩码源：\n  - 512×512: {mask_dir_512}\n  - 原始尺寸: {mask_dir_orig}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取文件路径
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        
        # 从两个不同的目录加载掩码
        mask_path_512 = os.path.join(self.mask_dir_512, image_file.replace('.jpg', '.png'))
        mask_path_orig = os.path.join(self.mask_dir_orig, image_file.replace('.jpg', '.png'))
        
        reflection_map_path = os.path.join(self.reflection_map_dir, image_file.replace('.jpg', '_fake_Rs_03.png'))
        removal_reflection_path = os.path.join(self.removal_reflection_dir, image_file.replace('.jpg', '_fake_Ts_03.png'))

        # 保存图像ID（不带扩展名的文件名）
        image_id = os.path.splitext(image_file)[0]

        # 加载图像和掩码
        mask_512 = cv2.imread(mask_path_512, cv2.IMREAD_GRAYSCALE)
        mask_orig = cv2.imread(mask_path_orig, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_path)
        
        # 存储原始尺寸信息
        orig_height, orig_width = mask_orig.shape[:2]
        
        # 检查反射图路径
        if not os.path.exists(reflection_map_path):
            raise FileNotFoundError(f"反射图'{reflection_map_path}'未找到")
            
        reflection_map = cv2.imread(reflection_map_path)
        if reflection_map is None:
            raise ValueError(f"反射图'{reflection_map_path}'为空或无法读取")
        
        # 加载去除反射后的图像
        if not os.path.exists(removal_reflection_path):
            print(f"警告：去除反射后的图像'{removal_reflection_path}'不存在，使用原始图像替代")
            removal_reflection = img.copy()
        else:
            removal_reflection = cv2.imread(removal_reflection_path)
            if removal_reflection is None:
                print(f"警告：无法读取去除反射后的图像'{removal_reflection_path}'，使用原始图像替代")
                removal_reflection = img.copy()
        
        # 转换为RGB格式
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask_512 = Image.fromarray(mask_512)
        mask_orig = Image.fromarray(mask_orig)
        reflection_map = Image.fromarray(cv2.cvtColor(reflection_map, cv2.COLOR_BGR2RGB))
        removal_reflection = Image.fromarray(cv2.cvtColor(removal_reflection, cv2.COLOR_BGR2RGB))
        
        # 应用变换（如果有）
        if self.transform:
            img = self.transform(img)
            mask_512 = self.transform(mask_512)
            mask_orig = self.transform(mask_orig)
            reflection_map = self.transform(reflection_map)
            removal_reflection = self.transform(removal_reflection)
        else:
            # 如果没有提供变换，手动转换为tensor
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
            mask_512 = to_tensor(mask_512)
            mask_orig = to_tensor(mask_orig)
            reflection_map = to_tensor(reflection_map)
            removal_reflection = to_tensor(removal_reflection)
        
        # 返回包含两种尺寸掩码的样本字典
        sample = {
            'data': img,
            'label_512': mask_512,      # 512×512掩码
            'label_orig': mask_orig,    # 原始尺寸掩码
            'reflection_map': reflection_map,
            'removal_reflection': removal_reflection,
            'image_id': image_id,
            'orig_height': orig_height,
            'orig_width': orig_width
        }
        
        return sample

def save_to_npz(dataloader, output_dir):
    """将包含两种尺寸掩码的样本保存为NPZ文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"将数据保存为NPZ文件，存储在 {output_dir}...")
    
    for i, batch in enumerate(tqdm(dataloader, desc="保存NPZ文件")):
        data = batch['data'].numpy()
        label_512 = batch['label_512'].numpy()  # 512×512掩码
        label_orig = batch['label_orig'].numpy() # 原始尺寸掩码
        reflection_map = batch['reflection_map'].numpy()
        removal_reflection = batch['removal_reflection'].numpy()
        image_id = batch['image_id']
        orig_height = batch['orig_height'].numpy()
        orig_width = batch['orig_width'].numpy()
        
        # 保存为NPZ文件，包含两种尺寸的掩码
        np.savez(
            os.path.join(output_dir, f'{i}.npz'), 
            data=data, 
            label_512=label_512,        # 512×512掩码
            label_orig=label_orig,      # 原始尺寸掩码
            reflection_map=reflection_map,
            removal_reflection=removal_reflection,
            image_id=image_id,
            orig_height=orig_height,
            orig_width=orig_width
        )

# 示例用法
if __name__ == "__main__":
    image_dir = '/mnt/tmp/T-imgs/GSD-T/test/image'
    mask_dir_512 = '/mnt/tmp/T-imgs/GSD-T/test/mask'     # 512×512掩码目录
    mask_dir_orig = "/mnt/tmp/O-imgs_GSD_S/test/mask/"   # 原始尺寸掩码目录
    reflection_map_dir = '/mnt/tmp/T-imgs/GSD-T/test/reflectionmaps'
    removal_reflection_dir = '/mnt/tmp/T-imgs/GSD-T/test/removedimg'
    output_dir = '/mnt/tmp/npzs_gsd_T_B/test'
    
    # 确保输出目录存在
    os.makedirs(removal_reflection_dir, exist_ok=True)
    
    # 创建仅包含ToTensor的变换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 创建数据集和数据加载器
    print("创建数据集...")
    dataset = DualMaskDataset(
        image_dir, 
        mask_dir_512,
        mask_dir_orig,
        reflection_map_dir,
        removal_reflection_dir,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 保存为NPZ文件
    print("将数据保存为NPZ文件...")
    save_to_npz(dataloader, output_dir)
    
    print("处理完成！")