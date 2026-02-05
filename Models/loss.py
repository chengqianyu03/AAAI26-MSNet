import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedBCEDiceFocalLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, focal_weight=1.0, 
                 focal_gamma=2.0, focal_alpha=0.25, smooth=1e-6):
        super(CombinedBCEDiceFocalLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (logits, before sigmoid) (B, C, H, W)
            targets: Ground truth labels (B, H, W) or (B, 1, H, W)
        """
        # Ensure targets have proper dimensions
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        
        # Convert targets to float for calculations
        targets = targets.float()
        
        # Flatten inputs and targets for loss calculations
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # BCE Loss - correct, using with_logits
        bce_loss = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='mean')
        
        # Get probabilities for Dice and Focal Loss
        inputs_sigmoid = torch.sigmoid(inputs_flat)
        
        # Dice Loss - now uses probabilities
        intersection = (inputs_sigmoid * targets_flat).sum()
        union = inputs_sigmoid.sum() + targets_flat.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice
        
        # Focal Loss - now uses proper probabilities for pt
        pt = torch.where(targets_flat == 1, inputs_sigmoid, 1 - inputs_sigmoid)  # p_t in the paper
        focal_weight = (1 - pt) ** self.focal_gamma
        alpha_factor = torch.where(targets_flat == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_bce = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='none')
        focal_loss = (alpha_factor * focal_weight * focal_bce).mean()
        
        # Combine all losses with their weights
        combined_loss = (
            self.bce_weight * bce_loss + 
            self.dice_weight * dice_loss + 
            self.focal_weight * focal_loss
        )
        
        return combined_loss