import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.swin_transformer import swin_v2_b

class SimpleGlassPromptGenerator(nn.Module):

    def __init__(self, num_points=20, d_model=256, dropout_rate=0.5):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model
        
    def forward(self, imageebd, orimg, reflection, sparse_embeddings2, dense_embeddings2):
        return sparse_embeddings2, dense_embeddings2