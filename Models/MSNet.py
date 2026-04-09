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
# Part 1: SAM ViT-H LoRA
# ==========================================

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.scaling = alpha / r
        self.w_a = nn.Linear(original_linear.in_features, r, bias=False)
        self.w_b = nn.Linear(r, original_linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.w_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_b.weight)

    def forward(self, x):
        return self.original_linear(x) + self.w_b(self.w_a(x)) * self.scaling


class LoRA_Sam(nn.Module):
    def __init__(self, sam_model, r=4):
        super().__init__()
        self.sam = sam_model
        self.lora_params = []
        self._inject_lora(self.sam.image_encoder, r)

    def _inject_lora(self, module, r):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in ('qkv', 'q_proj', 'k_proj', 'v_proj'):
                lora_layer = LoRALinear(child, r)
                setattr(module, name, lora_layer)
                self.lora_params.extend(
                    list(lora_layer.w_a.parameters()) + list(lora_layer.w_b.parameters())
                )
            else:
                self._inject_lora(child, r)

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
        if not self.training:
            return self._original_forward(
                query, key, value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask
            )
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
# NOTE: This module is NOT used during AAAI training.
# boundary_warmup_epoch=50 but max_epochs=50, so this head
# is never activated. Kept here for completeness only.
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

    def forward(self, coarse_logits_512, fpn0_feat, fpn1_feat, target_size):
        coarse_up = F.interpolate(coarse_logits_512, size=target_size,
                                   mode='bilinear', align_corners=False)
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
# Part 5: Prompt Generator
# ==========================================

class AdvancedAAAI_PromptGenerator(nn.Module):
    def __init__(self, num_points=20, d_model=256, sam_dim=256,
                 clip_model_name="ViT-B/16", clip_lora_rank=64,
                 clip_lora_alpha=128, clip_layer_idx=10):
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

    def forward(self, images, reflection_map, sam_features):
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
        if dense.shape[-1] != 64:
            dense = F.interpolate(dense, size=(64, 64), mode='bilinear', align_corners=False)
        return sparse, dense, weights_log


# ==========================================
# Part 6: BaselineSAMModel (AAAI MSNet)
# Uses original SAM ViT-H backbone with LoRA,
# integrates LRM reflection module end-to-end.
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
        reflection_checkpoint=None,
        reflection_proc_size=256,
        reflection_n_iters=3,
        reflection_finetune=False,
        **kwargs
    ):
        super().__init__(in_channels=3, out_channels=1, lr=lr, weight_decay=1e-4)
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.multi_scale_weight = multi_scale_weight
        self.boundary_warmup_epoch = boundary_warmup_epoch

        # --- Load SAM ViT-H ---
        sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)

        for param in sam_model.parameters():
            param.requires_grad = False

        self.sam_backbone = LoRA_Sam(sam_model, r=lora_rank)

        if ft_dec:
            for param in self.sam_backbone.sam.mask_decoder.parameters():
                param.requires_grad = True
            print(f"  SAM Mask Decoder: FINETUNING (ft_dec=True)")
        else:
            print(f"  SAM Mask Decoder: FROZEN (ft_dec=False)")

        sam_embed_dim = self.sam_backbone.sam.image_encoder.neck[-2].out_channels
        print(f"SAM image encoder output dim: {sam_embed_dim}")

        # --- Reflection Estimator (end-to-end integrated, like VideoModel) ---
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
        # NOTE: This module is NOT used during AAAI training.
        # boundary_warmup_epoch=50 but max_epochs=50, so the boundary
        # refinement head is never activated. Kept for completeness.
        self.boundary_head = MultiScaleBoundaryRefinementHead(
            fpn0_dim=32,
            fpn1_dim=64,
            hidden_dim=64
        )

        self.loss_fn = CombinedBCEDiceFocalLoss()

        # --- Collect trainable params ---
        self.trainable_params = self.sam_backbone.lora_params.copy()
        if ft_dec:
            self.trainable_params += list(self.sam_backbone.sam.mask_decoder.parameters())
        self.trainable_params += list(self.prompt_gen.parameters())
        # NOTE: boundary_head params are registered but never trained
        # because boundary refinement is never activated (see above).
        self.trainable_params += list(self.boundary_head.parameters())
        self.trainable_params += [p for p in self.reflection_estimator.parameters() if p.requires_grad]

        total_trainable = sum(p.numel() for p in self.trainable_params if p.requires_grad)
        clip_lora_params = sum(
            p.numel() for p in self.prompt_gen.clip_extractor.lora_adapters.parameters()
        )
        boundary_params = sum(p.numel() for p in self.boundary_head.parameters())
        refl_train = sum(p.numel() for p in self.reflection_estimator.parameters() if p.requires_grad)
        est_name = getattr(self.reflection_estimator, '_registry_name', type(self.reflection_estimator).__name__)

        print(f"\nModel Ready: SAM ViT-H + LoRA + CLIP-LoRA(MHA-level) + AAAI Fusion")
        print(f"  Total trainable params: {total_trainable:,}")
        print(f"  CLIP LoRA params: {clip_lora_params:,}")
        print(f"  Boundary head params: {boundary_params:,} (NOT used, warmup={boundary_warmup_epoch})")
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

        sparse, dense, weights_log = self.prompt_gen(images, reflection_map, img_embeds)

        if self.training and hasattr(self, 'log'):
            for wname, wval in weights_log.items():
                self.log(f'fusion/{wname}', wval, on_step=False, on_epoch=True)

        sparse_embeddings, dense_embeddings = self.sam_backbone.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )

        low_res_masks, _, = self.sam_backbone.sam.mask_decoder(
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
        return {
            'coarse': coarse_512,
            'refined': coarse_512,
            'stage_one': coarse_512,
            'stage_two': coarse_512,
        }

    def training_step(self, batch, batch_idx):
        low_res_masks = self._encode_and_decode(batch)
        coarse_512 = F.interpolate(low_res_masks, size=(512, 512),
                                    mode='bilinear', align_corners=False)

        gt_512 = batch['label'].squeeze(1)
        loss_coarse = self.loss_fn(coarse_512, gt_512)

        # NOTE: Boundary refinement is never activated because
        # boundary_warmup_epoch=50 >= max_epochs=50. Only coarse loss is used.
        loss_refined = torch.tensor(0.0, device=coarse_512.device)
        use_boundary = (self.current_epoch >= self.boundary_warmup_epoch)

        if use_boundary and 'label_orig' in batch:
            gt_orig = batch['label_orig'].squeeze(1)
            target_h, target_w = gt_orig.shape[-2], gt_orig.shape[-1]
            if (target_h, target_w) != (512, 512):
                refined = self.boundary_head(
                    coarse_512.detach(), None, None, (target_h, target_w)
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

        pred_bin = (torch.sigmoid(outputs['coarse']) > 0.5).float().squeeze(1)
        gt_flat = gt_orig.squeeze(1) if gt_orig.dim() > 2 else gt_orig
        if pred_bin.shape[-2:] != gt_flat.shape[-2:]:
            pred_bin = F.interpolate(pred_bin.unsqueeze(1), size=gt_flat.shape[-2:],
                                      mode='nearest').squeeze(1)
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

        param_groups = [
            {'params': sam_lora_params, 'lr': self.learning_rate, 'name': 'sam_lora'},
        ]
        if decoder_params:
            param_groups.append(
                {'params': decoder_params, 'lr': self.learning_rate * 0.5, 'name': 'sam_decoder'}
            )
        param_groups.append(
            {'params': prompt_params, 'lr': self.learning_rate * 1.0, 'name': 'prompt_gen'}
        )
        # NOTE: boundary_head params are included but never activated
        param_groups.append(
            {'params': boundary_params, 'lr': self.learning_rate * 2.0, 'name': 'boundary'}
        )
        if refl_params:
            param_groups.append(
                {'params': refl_params, 'lr': self.learning_rate * 0.1, 'name': 'reflection_estimator'}
            )

        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

        def lr_lambda(epoch):
            if epoch < 3:
                return 0.1 + 0.9 * (epoch / 3.0)
            progress = (epoch - 3) / max(1, 50 - 3)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

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
