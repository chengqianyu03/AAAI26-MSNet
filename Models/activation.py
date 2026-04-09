import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledSigmoid(nn.Module):
    """带温度缩放的Sigmoid激活函数，用于产生更锐利的边界
    
    Args:
        temp_scaling (float): 温度缩放系数。较小的值(如0.5)会产生更锐利的边界，较大的值会产生更平滑的边界。
    """
    def __init__(self, temp_scaling=0.5):
        super().__init__()
        self.temp_scaling = temp_scaling
        
    def forward(self, x):
        return torch.sigmoid(x / self.temp_scaling)

class LeakyReluClamped(nn.Module):
    """带截断的Leaky ReLU，输出限制在[0,1]区间
    
    Args:
        negative_slope (float): 负半轴的斜率
    """
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
        
    def forward(self, x):
        return torch.clamp(F.leaky_relu(x, negative_slope=self.negative_slope), 0, 1)

class GeluSigmoid(nn.Module):
    """GELU+Sigmoid组合激活函数"""
    def forward(self, x):
        return torch.sigmoid(F.gelu(x))

class SwishClamped(nn.Module):
    """带截断的Swish/SiLU激活函数"""
    def forward(self, x):
        return torch.clamp(F.silu(x), 0, 1)

class MishClamped(nn.Module):
    """带截断的Mish激活函数"""
    def forward(self, x):
        return torch.clamp(F.mish(x), 0, 1)

class BoundaryEnhancer(nn.Module):
    """边界增强器：增强分割掩码的边界清晰度
    
    Args:
        sharpness (float): 锐化系数。大于1会增强边界锐度，小于1会使边界更平滑。
        apply_in_training (bool): 是否在训练期间应用。
    """
    def __init__(self, sharpness=1.5, apply_in_training=False):
        super().__init__()
        self.sharpness = sharpness
        self.apply_in_training = apply_in_training
        
    def forward(self, x, is_training=False):
        if not is_training or (is_training and self.apply_in_training):
            return torch.pow(x, self.sharpness)
        return x

def get_activation(activation_type, temp_scaling=0.5, sharpness=1.1, apply_sharpening=True):
    """获取指定类型的激活函数
    
    Args:
        activation_type (str): 激活函数类型
        temp_scaling (float): 温度缩放系数(用于scaled_sigmoid)
        sharpness (float): 边界锐化系数
        apply_sharpening (bool): 是否应用边界锐化
        
    Returns:
        activation_fn: 激活函数模块
        enhancer: 边界增强器(如果启用)
    """
    # 创建边界增强器
    enhancer = BoundaryEnhancer(sharpness=sharpness) if apply_sharpening else None
    
    # 创建激活函数
    if activation_type == 'sigmoid':
        activation_fn = nn.Sigmoid()
    elif activation_type == 'leaky_relu':
        activation_fn = LeakyReluClamped(negative_slope=0.01)
    elif activation_type == 'gelu':
        activation_fn = GeluSigmoid()
    elif activation_type == 'swish':
        activation_fn = SwishClamped()
    elif activation_type == 'mish':
        activation_fn = MishClamped()
    elif activation_type == 'scaled_sigmoid':
        activation_fn = ScaledSigmoid(temp_scaling=temp_scaling)
    else:
        activation_fn = nn.Sigmoid()  # 默认
        
    return activation_fn, enhancer