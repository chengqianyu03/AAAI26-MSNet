import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
import clip
import numpy as np
from segment_anything import sam_model_registry
from Models.Base import BaseModel
from Models.loss import CombinedBCEDiceFocalLoss
from Models.reflection import get_reflection_estimator
from torch.nn.parameter import Parameter
import torch.utils.checkpoint as cp

# ==========================================
# Part 1: SAM ViT-H LoRA — FIXED: 使用原版 _LoRA_qkv (分离 Q/V)
# ==========================================

class _LoRA_qkv(nn.Module):
    """原版 LoRA：对 fused QKV 的 Q 和 V 分别加独立的低秩 delta，K 不变。"""
    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)                              # [B, N, N, 3*dim]
        new_q = self.linear_b_q(self.linear_a_q(x))    # Q 独立 delta
        new_v = self.linear_b_v(self.linear_a_v(x))    # V 独立 delta
        qkv[:, :, :, :self.dim] += new_q               # 只改 Q
        qkv[:, :, :, -self.dim:] += new_v              # 只改 V，K 不变
        return qkv


class LoRA_Sam(nn.Module):
    """原版 LoRA 注入：为每个 block 的 attn.qkv 创建独立的 Q/V 低秩适配器。"""
    def __init__(self, sam_model, r=4, lora_layer=None):
        super().__init__()

        assert r > 0
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.image_encoder.blocks)))

        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

        # Freeze image encoder
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q, w_b_linear_q,
                w_a_linear_v, w_b_linear_v,
            )

        self.reset_parameters()
        self.sam = sam_model

    @property
    def lora_params(self):
        params = []
        for w in self.w_As:
            params.extend(list(w.parameters()))
        for w in self.w_Bs:
            params.extend(list(w.parameters()))
        return params

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, *args, **kwargs):
        return self.sam(*args, **kwargs)


# ==========================================
# Part 2: CLIP LoRA — MHA-level injection
# ==========================================

class CLIPLoRAAdapter(nn.Module):
    def __init__(self, mha_module, embed_dim, rank=64, alpha=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.scaling = alpha / rank
        self.q_lora_A = nn.Parameter(torch.zeros(embed_dim, rank))
        self.q_lora_B = nn.Parameter(torch.zeros(rank, embed_dim))
        self.v_lora_A = nn.Parameter(torch.zeros(embed_dim, rank))
        self.v_lora_B = nn.Parameter(torch.zeros(rank, embed_dim))
        nn.init.kaiming_uniform_(self.q_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.q_lora_B)
        nn.init.kaiming_uniform_(self.v_lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.v_lora_B)
        self._original_forward = mha_module.forward

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        L, B, D = query.shape
        q_delta = (query.reshape(-1, D) @ self.q_lora_A @ self.q_lora_B).reshape(L, B, D)
        query_aug = query + q_delta * self.scaling
        v_delta = (value.reshape(-1, D) @ self.v_lora_A @ self.v_lora_B).reshape(L, B, D)
        value_aug = value + v_delta * self.scaling
        return self._original_forward(
            query_aug, key, value_aug,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask
        )


class CLIPExtractorWithLoRA(nn.Module):
    def __init__(self, clip_model, layer_idx=10, lora_rank=64, lora_alpha=128):
        super().__init__()
        self.visual = clip_model.visual
        self.layer_idx = layer_idx
        for param in self.visual.parameters():
            param.requires_grad = False
        self.lora_adapters = nn.ModuleList()
        if hasattr(self.visual, 'transformer'):
            for i, block in enumerate(self.visual.transformer.resblocks):
                if i <= self.layer_idx and hasattr(block, 'attn'):
                    if hasattr(block.attn, 'embed_dim'):
                        embed_dim = block.attn.embed_dim
                    elif hasattr(self.visual, 'width'):
                        embed_dim = self.visual.width
                    else:
                        embed_dim = 768
                    adapter = CLIPLoRAAdapter(
                        block.attn, embed_dim,
                        rank=lora_rank, alpha=lora_alpha
                    )
                    block.attn.forward = adapter.forward
                    self.lora_adapters.append(adapter)
        print(f"  CLIP LoRA: {len(self.lora_adapters)} MHA layers adapted "
              f"(rank={lora_rank}, alpha={lora_alpha}, layers 0-{layer_idx})")

    def forward(self, x):
        x = self.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls = self.visual.class_embedding.to(x.dtype) + \
              torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        for i, block in enumerate(self.visual.transformer.resblocks):
            if i > self.layer_idx:
                break
            x = block(x)
        x = x.permute(1, 0, 2)
        spatial = x[:, 1:]
        B, L, C = spatial.shape
        grid = int(math.sqrt(L))
        return spatial.permute(0, 2, 1).reshape(B, C, grid, grid).float()


# ==========================================
# Part 3: Multi-Scale Boundary Refinement
# ==========================================

class MultiScaleBoundaryRefinementHead(nn.Module):
    def __init__(self, fpn0_dim=32, fpn1_dim=64, hidden_dim=64):
        super().__init__()
        self.fpn0_dim = fpn0_dim
        self.fpn1_dim = fpn1_dim
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(fpn0_dim + fpn1_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.refine_conv = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        nn.init.zeros_(self.refine_conv[-1].weight)
        nn.init.zeros_(self.refine_conv[-1].bias)

    def forward(self, coarse_logits_512, fpn0_feat=None, fpn1_feat=None, target_size=None):
        if target_size is None:
            target_size = coarse_logits_512.shape[-2:]
        coarse_up = F.interpolate(coarse_logits_512, size=target_size,
                                   mode='bilinear', align_corners=False)

        if fpn0_feat is None or fpn1_feat is None:
            context = torch.zeros(
                coarse_up.shape[0],
                self.refine_conv[0].in_channels - 1,
                coarse_up.shape[2],
                coarse_up.shape[3],
                device=coarse_up.device,
                dtype=coarse_up.dtype,
            )
            residual = self.refine_conv(torch.cat([coarse_up, context], dim=1))
            return coarse_up + 0.1 * residual

        fpn0_up = F.interpolate(fpn0_feat, size=target_size, mode='bilinear', align_corners=False)
        fpn1_up = F.interpolate(fpn1_feat, size=target_size, mode='bilinear', align_corners=False)
        fpn_cat = torch.cat([fpn0_up, fpn1_up], dim=1)
        fpn_fused = self.fpn_fuse(fpn_cat)
        combined = torch.cat([coarse_up, fpn_fused], dim=1)
        residual = self.refine_conv(combined)
        return coarse_up + residual


# ==========================================
# Part 4: SDM & ASFM Modules
# ==========================================

class DynamicWeightPredictor(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, 1, bias=False)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        return self.mlp(self.avg_pool(x).view(b, c))


class SemanticAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat1, feat2):
        return self.conv(torch.cat([feat1, feat2], dim=1))


class AAAI_FusionModule(nn.Module):
    def __init__(self, clip_dim, sam_dim, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.rgb_proj = nn.Conv2d(clip_dim, d_model, 1)
        self.refl_proj = nn.Conv2d(clip_dim, d_model, 1)
        self.sam_proj = nn.Conv2d(sam_dim, d_model, 1)
        self.seb_attn_refl = SemanticAttentionBlock(d_model)
        self.seb_attn_trans = SemanticAttentionBlock(d_model)
        self.w_rgb = DynamicWeightPredictor(d_model)
        self.w_refl = DynamicWeightPredictor(d_model)
        self.w_trans = DynamicWeightPredictor(d_model)
        self.w_surr = DynamicWeightPredictor(d_model)
        self.w_sam = DynamicWeightPredictor(d_model)
        self.fusion_processor = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU(True),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model), nn.ReLU(True),
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.ReLU(True)
        )

    def forward(self, rgb_feat, refl_feat, sam_feat):
        F_I = self.rgb_proj(rgb_feat)
        F_r = self.refl_proj(refl_feat)
        if sam_feat.shape[-2:] != F_I.shape[-2:]:
            sam_feat = F.interpolate(sam_feat, size=F_I.shape[-2:],
                                     mode='bilinear', align_corners=False)
        F_g = self.sam_proj(sam_feat)
        attn_refl = self.seb_attn_refl(F_I, F_r)
        F_t = F_I * attn_refl - F_r
        attn_trans = self.seb_attn_trans(F_I, F_t)
        F_s = F_I - (attn_trans * F_t)
        weights = torch.cat([
            self.w_rgb(F_I), self.w_refl(F_r), self.w_trans(F_t),
            self.w_surr(F_s), self.w_sam(F_g)
        ], dim=1)
        norm_w = F.softmax(weights, dim=1).unsqueeze(-1).unsqueeze(-1)
        F_fused = (norm_w[:, 0] * F_I + norm_w[:, 1] * F_r +
                   norm_w[:, 2] * F_t + norm_w[:, 3] * F_s +
                   norm_w[:, 4] * F_g)
        processed = self.fusion_processor(F_fused)
        weights_log = {
            "w_rgb": norm_w[:, 0].mean().item(),
            "w_refl": norm_w[:, 1].mean().item(),
            "w_trans": norm_w[:, 2].mean().item(),
            "w_surr": norm_w[:, 3].mean().item(),
            "w_sam": norm_w[:, 4].mean().item(),
        }
        return processed, attn_refl, weights_log


# ==========================================
# Part 5: Prompt Generator — FIXED: 接收 SAM 默认 embeddings
# ==========================================

class AdvancedAAAI_PromptGenerator(nn.Module):
    def __init__(self, num_points=20, d_model=256, sam_dim=256,
                 clip_model_name="ViT-B/16", clip_lora_rank=64,
                 clip_lora_alpha=128, clip_layer_idx=10):
        # NOTE 20260504: switched default from 'CS-ViT-B/16' (CLIP-Surgery)
        # to standard 'ViT-B/16' to match the rms repo configuration
        # which reaches ~0.81 IoU on GSD-S. CLIP-Surgery has different attention
        # behavior; rms's MultiSemanticPromptGenerator was tuned for vanilla CLIP.
        super().__init__()
        self.num_points = num_points
        self.d_model = d_model
        print("Initializing CLIP with MHA-level LoRA (matching AAAI)...")
        clip_model, _ = clip.load(clip_model_name, device="cpu")
        self.clip_extractor = CLIPExtractorWithLoRA(
            clip_model,
            layer_idx=clip_layer_idx,
            lora_rank=clip_lora_rank,
            lora_alpha=clip_lora_alpha
        )
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            try:
                feat = self.clip_extractor(dummy)
                self.clip_dim = feat.shape[1]
                print(f"  CLIP output dim: {self.clip_dim}")
            except Exception:
                self.clip_dim = 768
                print(f"  CLIP dim detection failed, defaulting to {self.clip_dim}")
        self.fusion_module = AAAI_FusionModule(self.clip_dim, sam_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, d_model, 32, 32))
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
        self.dense_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2.2857, mode='bilinear', align_corners=False),
            nn.Conv2d(d_model, d_model, 3, padding=1)
        )

    def _extract_clip_feat(self, img):
        img_224 = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        device = img_224.device
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
        img_norm = (img_224 - mean) / std
        return self.clip_extractor(img_norm)

    def extract_clip_feat(self, img):
        if self.training:
            return cp.checkpoint(self._extract_clip_feat, img, use_reentrant=False)
        return self._extract_clip_feat(img)

    def forward(self, image_embeddings, images, reflection_map, sam_features,
                sparse_embeddings2=None, dense_embeddings2=None):
        """
        FIXED: 接口匹配原版 AAAI prompt module。
        
        Args:
            image_embeddings: SAM 图像嵌入 [B, C, H, W]（用于确定 dense 输出尺寸）
            images: 原始 RGB 图像 [B, 3, 1024, 1024]
            reflection_map: 反射图 [B, 3, H, W] 或 None
            sam_features: SAM 嵌入特征 [B, C, H, W]（送入 fusion module）
            sparse_embeddings2: SAM prompt_encoder 默认稀疏嵌入（保留 API 兼容性）
            dense_embeddings2: SAM prompt_encoder 默认密集嵌入（保留 API 兼容性）
        
        Returns:
            sparse: [B, num_points, d_model]
            dense: [B, d_model, H_pe, W_pe] 尺寸匹配 image_embeddings
            weights_log: dict
        """
        rgb_feat = self.extract_clip_feat(images)
        if reflection_map is not None:
            refl_feat = self.extract_clip_feat(reflection_map)
        else:
            refl_feat = torch.zeros_like(rgb_feat)
        fused_feat, attn_map, weights_log = self.fusion_module(rgb_feat, refl_feat, sam_features)
        pos_enc = F.interpolate(
            self.pos_encoder, size=fused_feat.shape[2:],
            mode='bilinear', align_corners=False
        )
        enhanced = fused_feat + pos_enc
        sparse_features = self.sparse_processor(enhanced)
        sparse_flat = sparse_features.flatten(start_dim=1)
        sparse = self.fc(sparse_flat).view(-1, self.num_points, self.d_model)
        dense = self.dense_head(enhanced)
        # FIXED: 动态匹配 image_embeddings 的空间尺寸，而不是硬编码 64
        target_dense_size = image_embeddings.shape[2:]  # e.g. (64, 64) for SAM ViT-H
        if dense.shape[2:] != target_dense_size:
            dense = F.interpolate(dense, size=target_dense_size,
                                  mode='bilinear', align_corners=False)
        return sparse, dense, weights_log


# ==========================================
# Part 6: BaselineSAMModel
# ==========================================

class BaselineSAMModel(BaseModel):
    def __init__(
        self,
        sam_checkpoint="checkpoints/sam_vit_h.pth",
        sam_model_type="vit_h",
        lr=5e-5,
        lora_rank=4,
        multi_scale_weight=0.3,
        boundary_warmup_epoch=50,
        clip_lora_rank=64,
        clip_lora_alpha=128,
        clip_layer_idx=10,
        ft_dec=True,
        reflection_estimator_name="lrm",
        reflection_estimator_kwargs=None,
        reflection_checkpoint="/mnt/tmp/checkpoints/model.pth",
        reflection_proc_size=256,
        reflection_n_iters=3,
        reflection_finetune=False,
        weight_decay=5e-5,
        **kwargs
    ):
        super().__init__(in_channels=3, out_channels=1, lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.multi_scale_weight = multi_scale_weight
        self.boundary_warmup_epoch = boundary_warmup_epoch

        # --- Load SAM ViT-H ---
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)

        for param in sam_model.parameters():
            param.requires_grad = False

        # FIXED: 使用原版 _LoRA_qkv 的 LoRA_Sam（分离 Q/V delta）
        self.sam_backbone = LoRA_Sam(sam_model, r=lora_rank)

        if ft_dec:
            for param in self.sam_backbone.sam.mask_decoder.parameters():
                param.requires_grad = True
            print(f"  SAM Mask Decoder: FINETUNING (ft_dec=True)")
        else:
            print(f"  SAM Mask Decoder: FROZEN (ft_dec=False)")

        sam_embed_dim = self.sam_backbone.sam.image_encoder.neck[-2].out_channels
        print(f"SAM image encoder output dim: {sam_embed_dim}")

        # --- Reflection Estimator ---
        self.reflection_estimator = self._build_reflection_estimator(
            name=reflection_estimator_name,
            explicit_kwargs=reflection_estimator_kwargs,
            legacy_checkpoint=reflection_checkpoint,
            legacy_proc_size=reflection_proc_size,
            legacy_n_iters=reflection_n_iters,
            legacy_finetune=reflection_finetune,
        )

        # --- Prompt Generator ---
        self.prompt_gen = AdvancedAAAI_PromptGenerator(
            sam_dim=sam_embed_dim,
            clip_lora_rank=clip_lora_rank,
            clip_lora_alpha=clip_lora_alpha,
            clip_layer_idx=clip_layer_idx,
        )

        # --- Multi-Scale Boundary Refinement ---
        self.boundary_head = MultiScaleBoundaryRefinementHead(
            fpn0_dim=32,
            fpn1_dim=64,
            hidden_dim=64
        )

        # NOTE 20260505: align loss weights with rms (which reaches ~0.81 IoU on GSD-S).
        # rms uses bce=1, dice=1, focal=0.5, focal_alpha=0.5; AAAI26 previously
        # used the defaults (focal=1.0, focal_alpha=0.25) which over-weighted the
        # focal term and biased toward foreground hard pixels at the cost of IoU.
        self.loss_fn = CombinedBCEDiceFocalLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            focal_weight=0.5,
            focal_gamma=2.0,
            focal_alpha=0.5,
        )

        # --- Collect trainable params ---
        self.trainable_params = self.sam_backbone.lora_params.copy()
        if ft_dec:
            self.trainable_params += list(self.sam_backbone.sam.mask_decoder.parameters())
        self.trainable_params += list(self.prompt_gen.parameters())
        self.trainable_params += list(self.boundary_head.parameters())
        self.trainable_params += [p for p in self.reflection_estimator.parameters() if p.requires_grad]

        total_trainable = sum(p.numel() for p in self.trainable_params if p.requires_grad)
        clip_lora_params = sum(
            p.numel() for p in self.prompt_gen.clip_extractor.lora_adapters.parameters()
        )
        boundary_params = sum(p.numel() for p in self.boundary_head.parameters())
        refl_train = sum(p.numel() for p in self.reflection_estimator.parameters() if p.requires_grad)
        est_name = getattr(self.reflection_estimator, '_registry_name', type(self.reflection_estimator).__name__)

        print(f"\nModel Ready: SAM ViT-H + LoRA(_LoRA_qkv) + CLIP-LoRA(MHA-level) + AAAI Fusion")
        print(f"  Total trainable params: {total_trainable:,}")
        print(f"  CLIP LoRA params: {clip_lora_params:,}")
        print(f"  Boundary head params: {boundary_params:,} (warmup={boundary_warmup_epoch}, zero-context fallback)")
        print(f"  Reflection estimator: '{est_name}' (trainable: {refl_train:,})")
        print(f"  ft_dec: {ft_dec}")

    @staticmethod
    def _build_reflection_estimator(name, explicit_kwargs, legacy_checkpoint,
                                     legacy_proc_size, legacy_n_iters, legacy_finetune):
        if explicit_kwargs is not None:
            re_kwargs = dict(explicit_kwargs)
        else:
            re_kwargs = {'proc_size': legacy_proc_size, 'finetune': legacy_finetune}
            if name == 'lrm':
                re_kwargs['checkpoint_path'] = legacy_checkpoint
                re_kwargs['n_iters'] = legacy_n_iters
            elif name == 'cvpr2024':
                re_kwargs['checkpoint_removal'] = legacy_checkpoint
            elif name == 'rdnet':
                re_kwargs['checkpoint_main'] = legacy_checkpoint
            elif name in ('identity', 'precomputed'):
                pass
            elif legacy_checkpoint:
                re_kwargs['checkpoint_path'] = legacy_checkpoint
        return get_reflection_estimator(name, **re_kwargs)

    def _estimate_reflection(self, images):
        no_grad = (not self.training or
                   not any(p.requires_grad for p in self.reflection_estimator.parameters()))
        if no_grad:
            with torch.no_grad():
                output = self.reflection_estimator(images)
        else:
            output = self.reflection_estimator(images)
        return output.reflection, output.transmission, output.extras

    def _encode_and_decode(self, batch):
        images = batch['data'].squeeze(1)
        if images.shape[-1] != 1024:
            images = F.interpolate(images, size=(1024, 1024), mode='bilinear')

        img_embeds = self.sam_backbone.sam.image_encoder(images)

        reflection_map, transmission_map, refl_extras = self._estimate_reflection(images)

        # FIXED: 获取 SAM 默认 prompt embeddings 并传给 prompt_gen
        sparse_embeddings2, dense_embeddings2 = self.sam_backbone.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        sparse, dense, weights_log = self.prompt_gen(
            img_embeds, images, reflection_map, img_embeds,
            sparse_embeddings2=sparse_embeddings2,
            dense_embeddings2=dense_embeddings2
        )

        if self.training and hasattr(self, 'log'):
            for wname, wval in weights_log.items():
                self.log(f'fusion/{wname}', wval, on_step=False, on_epoch=True)

        low_res_masks, _ = self.sam_backbone.sam.mask_decoder(
            image_embeddings=img_embeds,
            image_pe=self.sam_backbone.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        return low_res_masks

    def forward(self, batch):
        low_res_masks = self._encode_and_decode(batch)
        coarse_512 = F.interpolate(low_res_masks, size=(512, 512),
                                    mode='bilinear', align_corners=False)
        refined = coarse_512
        target_size = None
        if 'label_orig' in batch:
            target_size = batch['label_orig'].shape[-2:]
        elif 'orig_shape' in batch:
            orig_shape = batch['orig_shape']
            if isinstance(orig_shape, torch.Tensor):
                target_shape = orig_shape[0] if orig_shape.dim() > 1 else orig_shape
                target_size = tuple(int(v) for v in target_shape.detach().cpu().tolist())
            else:
                target_size = tuple(int(v) for v in orig_shape[0])

        if target_size is not None and tuple(target_size) != tuple(coarse_512.shape[-2:]):
            refined = self.boundary_head(coarse_512, target_size=target_size)

        return {
            'coarse': coarse_512,
            'refined': refined,
            'stage_one': coarse_512,
            'stage_two': refined,
        }

    def training_step(self, batch, batch_idx):
        low_res_masks = self._encode_and_decode(batch)
        coarse_512 = F.interpolate(low_res_masks, size=(512, 512),
                                    mode='bilinear', align_corners=False)

        gt_512 = batch['label'].squeeze(1)
        loss_coarse = self.loss_fn(coarse_512, gt_512)

        loss_refined = torch.tensor(0.0, device=coarse_512.device)
        use_boundary = (self.current_epoch >= self.boundary_warmup_epoch)

        if use_boundary and 'label_orig' in batch:
            gt_orig = batch['label_orig'].squeeze(1)
            target_h, target_w = gt_orig.shape[-2], gt_orig.shape[-1]
            if (target_h, target_w) != (512, 512):
                refined = self.boundary_head(
                    coarse_512.detach(), target_size=(target_h, target_w)
                )
                loss_refined = self.loss_fn(refined, gt_orig)

        alpha = self.multi_scale_weight if use_boundary else 0.0
        total_loss = (1.0 - alpha) * loss_coarse + alpha * loss_refined

        self.log('train/loss', total_loss, on_step=True, prog_bar=True)
        self.log('train/loss_coarse', loss_coarse, on_step=False, on_epoch=True)
        if use_boundary:
            self.log('train/loss_refined', loss_refined, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)

        gt_512 = batch['label'].squeeze(1)
        loss = self.loss_fn(outputs['coarse'], gt_512)

        gt_orig = batch.get('label_orig', batch['label']).squeeze(1)
        gt_flat = gt_orig.squeeze(1) if gt_orig.dim() > 2 else gt_orig

        # FIXED: 先 bilinear 上采样 logits，再 sigmoid 二值化
        logits = outputs.get('refined', outputs['coarse'])
        if logits.shape[-2:] != gt_flat.shape[-2:]:
            logits = F.interpolate(logits, size=gt_flat.shape[-2:],
                                   mode='bilinear', align_corners=False)
        pred_bin = (torch.sigmoid(logits) > 0.5).float().squeeze(1)

        intersection = (pred_bin * gt_flat).sum()
        union = pred_bin.sum() + gt_flat.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)

        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/iou', iou, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss)
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'iou': iou.detach()
        })
        return loss

    def configure_optimizers(self):
        sam_lora_params = self.sam_backbone.lora_params
        decoder_params = [p for p in self.sam_backbone.sam.mask_decoder.parameters()
                          if p.requires_grad]
        prompt_params = list(self.prompt_gen.parameters())
        boundary_params = list(self.boundary_head.parameters())
        refl_params = [p for p in self.reflection_estimator.parameters() if p.requires_grad]

        # NOTE 20260504: rms uses a single param group at base lr.
        # Previously decoder was at 0.5x and boundary at 2.0x; this slowed the
        # SAM mask decoder (the most important trainable head) and over-trained
        # an essentially-untrained boundary head. Align to rms (all at base lr,
        # except keep reflection estimator down-weighted since it is a frozen
        # pretrained network when finetuned).
        param_groups = [
            {'params': sam_lora_params, 'lr': self.learning_rate, 'name': 'sam_lora'},
        ]
        if decoder_params:
            param_groups.append(
                {'params': decoder_params, 'lr': self.learning_rate, 'name': 'sam_decoder'}
            )
        param_groups.append(
            {'params': prompt_params, 'lr': self.learning_rate, 'name': 'prompt_gen'}
        )
        param_groups.append(
            {'params': boundary_params, 'lr': self.learning_rate, 'name': 'boundary'}
        )
        if refl_params:
            param_groups.append(
                {'params': refl_params, 'lr': self.learning_rate * 0.1, 'name': 'reflection_estimator'}
            )

        optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)

        # NOTE 20260505: cosine min factor 0.01 -> 0.05 to match rms.
        # With base_lr=2e-5, the previous floor was 2e-7 (effectively frozen in
        # the last epochs); rms's 0.05 floor keeps lr at 1e-6 so SAM-LoRA / mask
        # decoder can keep refining. 5x more effective lr in the tail.
        def lr_lambda(epoch):
            if epoch < 3:
                return 0.1 + 0.9 * (epoch / 3.0)
            progress = (epoch - 3) / max(1, 50 - 3)
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        avg_iou = torch.stack([x['iou'] for x in self.validation_step_outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        est_name = getattr(self.reflection_estimator, '_registry_name', type(self.reflection_estimator).__name__)
        print(
            f"\nEPOCH {self.current_epoch} | "
            f"Val Loss: {avg_loss:.4f} | Val IoU: {avg_iou:.4f} | "
            f"Refl: {est_name}\n",
            flush=True
        )
        self.validation_step_outputs.clear()

    def _apply_activation(self, logits, is_training=False):
        return torch.sigmoid(logits)