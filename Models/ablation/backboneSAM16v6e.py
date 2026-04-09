import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from torchvision import transforms
import torch.utils.checkpoint as checkpoint
import math

class MultiheadAttentionLoRA(nn.Module):
    """LoRA适配器用于MultiheadAttention"""
    def __init__(self, mha_module, embed_dim, num_heads, rank=8, alpha=16):
        super().__init__()
        self.original_module = mha_module
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.rank = rank
        self.scaling = alpha / rank
        
        # 仅为QV创建LoRA权重
        self.q_lora_A = nn.Parameter(torch.zeros(embed_dim, rank))
        self.q_lora_B = nn.Parameter(torch.zeros(rank, embed_dim))
        self.v_lora_A = nn.Parameter(torch.zeros(embed_dim, rank))
        self.v_lora_B = nn.Parameter(torch.zeros(rank, embed_dim))
        
        # 初始化
        nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.q_lora_B)
        nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.v_lora_B)
        
        # 替换前向传播
        self.original_forward = mha_module.forward
        mha_module.forward = self.forward
        
    def forward(self, query, key, value, key_padding_mask=None, 
               need_weights=True, attn_mask=None):
        # 推理模式直接使用原始模块
        if not self.training:
            return self.original_forward(
                query, key, value, 
                key_padding_mask=key_padding_mask,
                need_weights=need_weights, 
                attn_mask=attn_mask
            )
        
        # 训练模式应用LoRA
        seq_len, batch_size, _ = query.shape
        
        # 保存原始输入
        original_query = query.clone()
        original_value = value.clone()
        
        # 应用LoRA
        flat_query = query.reshape(-1, self.embed_dim)
        q_lora = (flat_query @ self.q_lora_A @ self.q_lora_B) * self.scaling
        q_lora = q_lora.reshape(seq_len, batch_size, self.embed_dim)
        query = query + q_lora
        
        flat_value = value.reshape(-1, self.embed_dim)
        v_lora = (flat_value @ self.v_lora_A @ self.v_lora_B) * self.scaling
        v_lora = v_lora.reshape(seq_len, batch_size, self.embed_dim)
        value = value + v_lora
        
        # 调用原始forward
        output, attn_weights = self.original_forward(
            query, key, value, 
            key_padding_mask=key_padding_mask,
            need_weights=need_weights, 
            attn_mask=attn_mask
        )
        
        # 恢复原始输入
        query.copy_(original_query)
        value.copy_(original_value)
        
        return output, attn_weights

class CLIPExtractorWithLoRA(nn.Module):
    """简化的CLIP特征提取器"""
    def __init__(self, clip_model, layer_idx=9, lora_r=8, lora_alpha=16):
        super().__init__()
        self.visual = clip_model.visual
        self.layer_idx = layer_idx
        
        # 冻结CLIP参数
        for param in self.visual.parameters():
            param.requires_grad = False
        
        # 添加LoRA
        self.lora_layers = nn.ModuleList()
        lora_count = 0
        
        # 使用简化逻辑添加LoRA层
        if hasattr(self.visual, 'transformer'):
            for i, block in enumerate(self.visual.transformer.resblocks):
                if i <= self.layer_idx and hasattr(block, 'attn'):
                    if hasattr(block.attn, 'embed_dim'):
                        embed_dim = block.attn.embed_dim
                    else:
                        embed_dim = 768
                    
                    if hasattr(block.attn, 'num_heads'):
                        num_heads = block.attn.num_heads
                    else:
                        num_heads = embed_dim // 64
                        
                    lora_adapter = MultiheadAttentionLoRA(
                        block.attn, 
                        embed_dim, 
                        num_heads, 
                        lora_r, 
                        lora_alpha
                    )
                    self.lora_layers.append(lora_adapter)
                    lora_count += 1
    
    def forward(self, x):
        # ViT特征提取
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + 
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        for i, block in enumerate(self.visual.transformer.resblocks):
            if i <= self.layer_idx:
                x = block(x)
            else:
                break

        x = x.permute(1, 0, 2)  # LND -> NLD
        
        grid_size = int(np.sqrt(x.shape[1] - 1))
        visual_features = x[:, 1:].reshape(x.shape[0], grid_size, grid_size, x.shape[2])
        visual_features = visual_features.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return visual_features

class DynamicWeightPredictor(nn.Module):
    """
    ASFM的核心组件：动态权重预测器
    Implementation of Eq. (7): w_j = F_weight(F_j)
    使用 Global Average Pooling + MLP 来预测每个特征图的标量权重
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, 1, bias=False)
        )

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.mlp(y) # [B, 1]
        return y

class SemanticAttentionBlock(nn.Module):
    """
    SEB的核心组件：语义注意力生成
    用于计算 Attn(F_I, F_r) 或 Attn(F_I, F_t)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        # Concatenate features along channel dimension
        cat_feat = torch.cat([feat1, feat2], dim=1)
        # Generate attention map [B, 1, H, W]
        attn = self.conv(cat_feat)
        return attn

class CompactMultiSemanticFusion(nn.Module):
    """
    Revised Semantic Fusion Module implementing AAAI/TPAMI paper logic:
    1. Attention-based Semantic Elimination (SEB) - Eqs 4 & 5
    2. Adaptive (Dynamic) Semantic Fusion (ASFM) - Eqs 7 & 8
    """
    def __init__(self, clip_dim, sam_dim, d_model=256):
        super().__init__()
        
        # --- 1. Feature Projections (Equation 6) ---
        self.rgb_projection = nn.Conv2d(clip_dim, d_model, kernel_size=1)
        self.reflection_projection = nn.Conv2d(clip_dim, d_model, kernel_size=1)
        self.sam_projection = nn.Conv2d(sam_dim, d_model, kernel_size=1)
        
        # --- 2. Semantic Elimination Block (SEB) Components ---
        # 用于生成反射区域的注意力: Attn(F_I, F_r)
        self.seb_attn_refl = SemanticAttentionBlock(d_model)
        
        # 用于生成透射区域的注意力: Attn(F_I, F_t)
        self.seb_attn_trans = SemanticAttentionBlock(d_model)
        
        # --- 3. Adaptive Semantic Fusion (ASFM) Components ---
        # 动态权重预测器 (不再是静态参数)
        self.weight_pred_rgb = DynamicWeightPredictor(d_model)
        self.weight_pred_refl = DynamicWeightPredictor(d_model)
        self.weight_pred_trans = DynamicWeightPredictor(d_model)
        self.weight_pred_surr = DynamicWeightPredictor(d_model)
        self.weight_pred_sam = DynamicWeightPredictor(d_model)
        
        # 融合后的细化层 (Equation 9)
        self.fusion_processor = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_features, reflection_features, sam_features):
        # Resize SAM features to match
        target_size = (min(48, sam_features.shape[2]), min(48, sam_features.shape[3]))
        sam_features_resized = F.interpolate(
            sam_features, 
            size=target_size,
            mode='bilinear', 
            align_corners=False
        )
            
        # 1. Feature Projection (Eq. 6)
        # F_I (Image)
        rgb_proj = self.rgb_projection(rgb_features)
        # F_r (Reflection)
        refl_proj = self.reflection_projection(reflection_features)
        # F_g (Glass/SAM)
        sam_proj = self.sam_projection(sam_features_resized)
        
        # Align spatial dimensions if necessary
        if rgb_proj.shape[2:] != sam_proj.shape[2:]:
            rgb_proj = F.interpolate(rgb_proj, size=sam_proj.shape[2:], mode='bilinear', align_corners=False)
        if refl_proj.shape[2:] != sam_proj.shape[2:]:
            refl_proj = F.interpolate(refl_proj, size=sam_proj.shape[2:], mode='bilinear', align_corners=False)
        
        # --- 2. Semantic Elimination Block (SEB) Logic ---
        
        # Calculate Transmission Features F_t (Equation 4)
        # F_t = F_I * Attn(F_I, F_r) - F_r
        # Attention highlighting reflection regions
        attn_refl = self.seb_attn_refl(rgb_proj, refl_proj) 
        # Remove reflection semantics from image semantics
        trans_features = rgb_proj * attn_refl - refl_proj
        
        # Calculate Surrounding Features F_s (Equation 5)
        # F_s = F_I - Attn(F_I, F_t) * F_t
        # Attention highlighting transmission regions
        attn_trans = self.seb_attn_trans(rgb_proj, trans_features)
        # Remove transmission semantics from image to get surroundings
        surrounding_features = rgb_proj - (attn_trans * trans_features)
        
        # --- 3. Adaptive Semantic Fusion Module (ASFM) Logic ---
        
        # Calculate Dynamic Weights (Equation 7)
        # Weights depend on the input content itself (B, 1)
        w_rgb = self.weight_pred_rgb(rgb_proj)
        w_refl = self.weight_pred_refl(refl_proj)
        w_trans = self.weight_pred_trans(trans_features)
        w_surr = self.weight_pred_surr(surrounding_features)
        w_sam = self.weight_pred_sam(sam_proj)
        
        # Stack weights: [B, 5, 1, 1] for broadcasting
        stacked_weights = torch.stack([w_rgb, w_refl, w_trans, w_surr, w_sam], dim=1).unsqueeze(-1)
        
        # Apply Softmax to weights (Equation 8 part 1)
        norm_weights = F.softmax(stacked_weights, dim=1) # [B, 5, 1, 1]
        
        # Weighted Fusion (Equation 8 part 2)
        # Expand weights to match spatial dimensions [B, 1, 1, 1]
        fused_features = (
            norm_weights[:, 0] * rgb_proj + 
            norm_weights[:, 1] * refl_proj + 
            norm_weights[:, 2] * trans_features + 
            norm_weights[:, 3] * surrounding_features +
            norm_weights[:, 4] * sam_proj
        )
        
        # Refinement (Equation 9)
        processed = self.fusion_processor(fused_features)
        
        # Logging weights for analysis
        # Using mean across batch for logging purposes
        weights_log = {
            "rgb_weight": norm_weights[:, 0].mean().item(),
            "reflection_weight": norm_weights[:, 1].mean().item(),
            "transmission_weight": norm_weights[:, 2].mean().item(),
            "surrounding_weight": norm_weights[:, 3].mean().item(),
            "sam_weight": norm_weights[:, 4].mean().item()
        }
        
        # Note: diff_attention is just returned for visualization compatibility
        return processed, attn_refl, weights_log


class MultiSemanticPromptGenerator(nn.Module):
    """使用多语义分解的简化提示生成器"""
    def __init__(self, num_points=20, d_model=256, sam_embed_dim=256, 
                 clip_model_name="ViT-B/16", lora_r=128, lora_alpha=256, layer_idx = 10):
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model
        self.use_checkpointing = True
        
        # 加载CLIP模型
        self.clip_model, self.clip_preprocess = clip.load(
            clip_model_name, 
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 创建CLIP特征提取器
        self.clip_extractor = CLIPExtractorWithLoRA(
            self.clip_model, 
            layer_idx=layer_idx,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        # 获取CLIP特征维度
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224).to(next(self.clip_model.parameters()).device)
            if hasattr(self.clip_model.visual, 'transformer'):
                x = self.clip_model.visual.conv1(test_input)
                if hasattr(self.clip_model.visual, 'width'):
                    clip_dim = self.clip_model.visual.width
                else:
                    clip_dim = self.clip_model.visual.transformer.width
            else:
                clip_dim = self.clip_model.visual.output_dim
        
        # 使用多语义融合模块 (UPDATED Class)
        self.fusion_module = CompactMultiSemanticFusion(clip_dim, sam_embed_dim, d_model)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, d_model, 32, 32))
        
        # 简化稀疏提示生成
        self.sparse_processor = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(d_model, d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4, 4))
        )
        
        fc_input_size = d_model * 4 * 4
        self.fc = nn.Linear(fc_input_size, d_model * num_points)
        
        # 密集提示生成
        self.dense_prompt_generator = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        )

    def _extract_clip_features_wrapper(self, image):
        return self.extract_clip_features(image)
    
    def _fusion_module_wrapper(self, rgb_features, reflection_features, sam_features):
        return self.fusion_module(rgb_features, reflection_features, sam_features)
    
    def _sparse_process_wrapper(self, features):
        return self.sparse_processor(features)
    
    def _dense_process_wrapper(self, features):
        return self.dense_prompt_generator(features)
        
    def extract_clip_features(self, image):
        """从图像中提取CLIP特征"""
        device = next(self.parameters()).device
        
        # 处理输入图像
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = F.interpolate(image, size=(224, 224), mode='bilinear')
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
            image = (image - mean) / std
        else:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                    (0.26862954, 0.26130258, 0.27577711))
            ])
            image = transform(image).unsqueeze(0).to(device)
            
        # 提取特征
        features = self.clip_extractor(image)
        return features
            
    def forward(self, image_embeddings, original_image, reflection_map=None, sparse_embeddings2=None, dense_embeddings2=None):
        """
        前向传播
        """
        batch_size = image_embeddings.shape[0]
        device = image_embeddings.device
        
        # 处理反射图，如果未提供则创建空图
        if reflection_map is None:
            reflection_map = torch.zeros_like(original_image)
        
        # 提取特征
        if self.use_checkpointing and self.training:
            rgb_features = checkpoint.checkpoint(
                self._extract_clip_features_wrapper,
                original_image.detach() if isinstance(original_image, torch.Tensor) else original_image
            )
            reflection_features = checkpoint.checkpoint(
                self._extract_clip_features_wrapper,
                reflection_map.detach() if isinstance(reflection_map, torch.Tensor) else reflection_map
            )
        else:
            rgb_features = self.extract_clip_features(original_image)
            reflection_features = self.extract_clip_features(reflection_map)
            
        # 确保批次大小一致
        if rgb_features.size(0) != batch_size:
            rgb_features = rgb_features.expand(batch_size, -1, -1, -1)
        if reflection_features.size(0) != batch_size:
            reflection_features = reflection_features.expand(batch_size, -1, -1, -1)
        
        # 特征融合
        if self.use_checkpointing and self.training:
            fused_features, diff_attention, fusion_weights = checkpoint.checkpoint(
                self._fusion_module_wrapper,
                rgb_features,
                reflection_features,
                image_embeddings
            )
        else:
            fused_features, diff_attention, fusion_weights = self.fusion_module(
                rgb_features, 
                reflection_features,
                image_embeddings
            )
            
        # 记录融合权重
        if hasattr(self, 'log') and self.training:
            for weight_name, weight_value in fusion_weights.items():
                self.log(f"fusion_{weight_name}", weight_value, on_step=False, on_epoch=True)
                
        # 添加位置编码
        pos_encoding = F.interpolate(
            self.pos_encoder,
            size=fused_features.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        enhanced_features = fused_features + pos_encoding
        
        # 生成稀疏提示
        if self.use_checkpointing and self.training:
            sparse_features = checkpoint.checkpoint(
                self._sparse_process_wrapper,
                enhanced_features
            )
        else:
            sparse_features = self.sparse_processor(enhanced_features)
            
        sparse_flat = sparse_features.flatten(start_dim=1)
        sparse_embeddings = self.fc(sparse_flat).view(batch_size, self.num_points, self.d_model)
        
        # 生成密集提示
        if self.use_checkpointing and self.training:
            dense_embeddings = checkpoint.checkpoint(
                self._dense_process_wrapper,
                enhanced_features
            )
        else:
            dense_embeddings = self.dense_prompt_generator(enhanced_features)
            
        # 调整密集提示尺寸
        dense_embeddings = F.interpolate(
            dense_embeddings,
            size=image_embeddings.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        return sparse_embeddings, dense_embeddings