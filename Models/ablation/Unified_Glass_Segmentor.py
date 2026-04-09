import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import math
import os
import datetime
import time

from segment_anything import sam_model_registry
from Models.Base import BaseModel
from Models.loss import CombinedBCEDiceFocalLoss
from Models.modulezoo import get_prompt_module
from Models.activation import get_activation
from torch.nn.parameter import Parameter

class _LoRA_qkv(nn.Module):
    """Implementation of LoRA QKV layer in SAM"""
    def __init__(self, qkv: nn.Module, linear_a_q: nn.Module, linear_b_q: nn.Module, 
                 linear_a_v: nn.Module, linear_b_v: nn.Module):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv

class LoRA_Sam(nn.Module):
    """Low Rank Adaptation wrapper for SAM image encoder"""
    def __init__(self, sam_model, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()
        assert r > 0
        self.lora_layer = lora_layer if lora_layer else list(range(len(sam_model.image_encoder.blocks)))
        self.w_As = nn.ModuleList()
        self.w_Bs = nn.ModuleList()

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
            
            self.w_As.extend([w_a_linear_q, w_a_linear_v])
            self.w_Bs.extend([w_b_linear_q, w_b_linear_v])
            
            blk.attn.qkv = _LoRA_qkv(w_qkv_linear, w_a_linear_q, w_b_linear_q, w_a_linear_v, w_b_linear_v)
        
        self.reset_parameters()
        self.sam = sam_model

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            
    def save_lora_parameters(self, filename: str) -> None:
        state_dict = self.sam.module.state_dict() if isinstance(self.sam, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else self.sam.state_dict()
        lora_dict = {}
        # Save LoRA weights
        for i, (wa, wb) in enumerate(zip(self.w_As, self.w_Bs)):
            lora_dict[f"w_a_{i:03d}"] = wa.weight
            lora_dict[f"w_b_{i:03d}"] = wb.weight
        # Save prompt encoder and mask decoder
        for k, v in state_dict.items():
            if 'prompt_encoder' in k or 'mask_decoder' in k:
                lora_dict[k] = v
        torch.save(lora_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        state_dict = torch.load(filename)
        for i, (wa, wb) in enumerate(zip(self.w_As, self.w_Bs)):
            if f"w_a_{i:03d}" in state_dict: wa.weight = Parameter(state_dict[f"w_a_{i:03d}"])
            if f"w_b_{i:03d}" in state_dict: wb.weight = Parameter(state_dict[f"w_b_{i:03d}"])
        self.sam.load_state_dict(state_dict, strict=False)

    def forward(self, *args, **kwargs):
        return self.sam(*args, **kwargs)


class DetailRefinementModule(nn.Module):
    def __init__(self, channels=1, hidden_dim=32):
        super().__init__()
        self.refinement_conv = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)
        )
        nn.init.zeros_(self.refinement_conv[-1].weight)
        nn.init.zeros_(self.refinement_conv[-1].bias)
    
    def forward(self, coarse_mask, target_size):
        base_high_res = F.interpolate(coarse_mask, size=target_size, mode='bicubic', align_corners=False)
        residual_details = self.refinement_conv(base_high_res)
        refined_mask = base_high_res + 0.1 * residual_details
        return refined_mask

class UnifiedGlassDetectionModel(BaseModel):

    def __init__(
        self,
        in_channels, out_channels,
        lr=1e-6, prompt_lr_factor=1, weight_decay=1e-4,
        sam_model_name="vit_l", sam_checkpoint="checkpoint/sam_vit_l_0b3195.pth",
        ft_dec=False, num_points=20,
        bce_weight=1.0, dice_weight=1.0, focal_weight=0.5, focal_gamma=2.0, focal_alpha=0.5,
        module_name='sam1', activation_type='scaled_sigmoid', temp_scaling=0.5,
        lora_rank=4, lora_layers=None, clip_module_name='CS-ViT-B/16',
        refiner_lr_scale=0.5,      
        refinement_warmup_epoch=20, 
        stage_two_lr_factor=0.5,
        adapter_start_epoch=None
    ):
        super().__init__(in_channels, out_channels, lr, weight_decay)
        self.save_hyperparameters()
        self.num_points = num_points
        self.prompt_lr_factor = prompt_lr_factor
        self.refiner_lr_scale = refiner_lr_scale
    
        if adapter_start_epoch is not None:
             self.refinement_warmup_epoch = adapter_start_epoch
        else:
             self.refinement_warmup_epoch = refinement_warmup_epoch
        
        # State tracking
        self.training_phase = "semantic" # "semantic" or "refinement"
        self.sam_params = []
        self.refiner_params = []
        self.validation_step_outputs = []
        
        # --- Architecture Initialization ---
        
        # 1. Activation & Loss
        self.activation_fct, _ = get_activation(activation_type, temp_scaling=temp_scaling, apply_sharpening=False)
        self.loss_fn = CombinedBCEDiceFocalLoss(bce_weight, dice_weight, focal_weight, focal_gamma, focal_alpha)

        # 2. Prompt Encoder Backbone
        self.prompt_generator = get_prompt_module(module_name=module_name, num_points=self.num_points, 
                                                d_model=256, clip_module_name=clip_module_name)
        
        # 3. LoRA-SAM Encoder & Decoder
        original_sam = sam_model_registry[sam_model_name](checkpoint=sam_checkpoint)
        for name, param in original_sam.named_parameters():
            param.requires_grad = ('mask_decoder' in name) if ft_dec else False
        self.sam_backbone = LoRA_Sam(original_sam, r=lora_rank, lora_layer=lora_layers)
        
        # 4. Detail Refinement Module
        self.detail_refiner = DetailRefinementModule(channels=1, hidden_dim=32)
        

        self.sam_params.extend([p for p in self.prompt_generator.parameters() if p.requires_grad])
        for w in self.sam_backbone.w_As: self.sam_params.extend([p for p in w.parameters() if p.requires_grad])
        for w in self.sam_backbone.w_Bs: self.sam_params.extend([p for p in w.parameters() if p.requires_grad])
        self.sam_params.extend([p for n, p in self.sam_backbone.sam.named_parameters() if 'mask_decoder' in n and p.requires_grad])
        
        self.refiner_params = [p for p in self.detail_refiner.parameters()]
        
        print(f"Model Initialized: Unified Progressive Framework")
        print(f"- Semantic Warmup: Epochs 0-{self.refinement_warmup_epoch}")
        print(f"- Boundary Refinement: Epochs {self.refinement_warmup_epoch}+")

    def _apply_activation(self, logits, is_training=False):
        return self.activation_fct(logits).float()

    def on_train_epoch_start(self):
        """Dynamic curriculum learning scheduler"""
        is_refinement_phase = (self.current_epoch >= self.refinement_warmup_epoch)
        
        if not is_refinement_phase:
            if self.training_phase != "semantic":
                self.training_phase = "semantic"
                print(f"\n[Epoch {self.current_epoch}] Phase: Semantic Structure Learning (Standard Resolution)")
                self.prompt_generator.train()
                self.sam_backbone.train()
                self.detail_refiner.eval()
                # Optimize Backbone, Freeze Refiner
                for p in self.sam_params: p.requires_grad = True
                for p in self.refiner_params: p.requires_grad = False
        else:
            if self.training_phase != "refinement":
                self.training_phase = "refinement"
                print(f"\n[Epoch {self.current_epoch}] Phase: High-Resolution Boundary Refinement")
                self.prompt_generator.eval()
                self.sam_backbone.eval()
                self.detail_refiner.train()
                # Freeze Backbone, Optimize Refiner
                for p in self.sam_params: p.requires_grad = False
                for p in self.refiner_params: p.requires_grad = True
    
    def forward(self, batch):
        """Unified forward pass with conditional refinement"""
        # 1. Semantic Inference (Standard 512x512)
        # Check if we should compute gradients for SAM part based on current curriculum phase
        is_refinement_phase = (self.current_epoch >= self.refinement_warmup_epoch) if hasattr(self, 'current_epoch') else True
        
        # Inference SAM Backbone
        if is_refinement_phase and not self.training:
             # Pure evaluation mode during refinement phase checks
             with torch.no_grad():
                 coarse_masks = self._inference_backbone(batch)
        elif is_refinement_phase and self.training:
             # Training refinement: freeze SAM grad
             with torch.no_grad():
                 coarse_masks = self._inference_backbone(batch)
        else:
             # Training semantic: enable SAM grad
             coarse_masks = self._inference_backbone(batch)
        
        # Determine target resolution
        if 'label_orig' in batch:
            target_h, target_w = batch['label_orig'].shape[-2:]
        elif 'orig_shape' in batch:
            target_h, target_w = batch['orig_shape'][0].cpu().numpy()
        else:
            target_h, target_w = 512, 512
        
        # 2. Detail Refinement (Adaptive Resolution)
        if (target_h, target_w) != (512, 512):
            refined_masks = self.detail_refiner(coarse_masks, (target_h, target_w))
        else:
            refined_masks = coarse_masks
        
        return {
            'coarse': coarse_masks,
            'refined': refined_masks,
            # For compatibility with older testing scripts that look for stage_one/two
            'stage_one': coarse_masks,
            'stage_two': refined_masks
        }
    
    def _inference_backbone(self, batch):
        """Core SAM backbone inference logic"""
        # Input processing - Try standard then backup keys
        image_embeddings = None
        if 'image_embeddings' in batch:
             image_embeddings = batch['image_embeddings']
        elif 'image_embeddings2' in batch:
             image_embeddings = batch['image_embeddings2']
        
        if image_embeddings is not None:
             image_embeddings = image_embeddings.squeeze(1)
        else:
            # Fallback if precomputed embeddings not found
            data = batch['data'].squeeze(1)
            if data.shape[-1] != 1024:
                 data = F.interpolate(data, size=(1024, 1024), mode='bilinear', align_corners=False)
            image_embeddings = self.sam_backbone.sam.image_encoder(data)

        # Prompt Generation
        orimg = batch['data'].squeeze(1)
        refl = batch.get('reflection_map', torch.zeros_like(orimg)).squeeze(1)
        
        sparse, dense = self.prompt_generator(
            image_embeddings, orimg, refl, None, None
        )

        # Mask Decoding
        low_res_masks, _ = self.sam_backbone.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam_backbone.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False
        )

        return F.interpolate(low_res_masks, size=(512, 512), mode='bilinear', align_corners=False)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        is_refinement_phase = (self.current_epoch >= self.refinement_warmup_epoch)
        
        if not is_refinement_phase:
            # Semantic Phase: Supervise Coarse Output
            pred = outputs['coarse']
            target = batch['label'].squeeze(1) # Standard 512 label
            loss_key = 'loss_semantic'
        else:
            # Refinement Phase: Supervise Refined Output
            pred = outputs['refined']
            target = batch.get('label_orig', batch['label']).squeeze(1) # Original high-res label
            loss_key = 'loss_refined'
            
        loss = self.loss_fn(pred, target)
        self.log(f'train/{loss_key}', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        
        # Evaluate Semantic Quality (512x512)
        loss_semantic = self.loss_fn(outputs['coarse'], batch['label'].squeeze(1))
        
        # Evaluate Refined Quality (Original Size)
        target_orig = batch.get('label_orig', batch['label']).squeeze(1)
        loss_refined = self.loss_fn(outputs['refined'], target_orig)
        
        # Metrics for monitoring
        pred_mask = (self._apply_activation(outputs['refined']) > 0.5).float()
        intersection = (pred_mask * target_orig).sum()
        union = pred_mask.sum() + target_orig.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)

        self.log('val/loss_semantic', loss_semantic, on_epoch=True)
        self.log('val/loss_refined', loss_refined, on_epoch=True)
        self.log('val/iou', iou, on_epoch=True, prog_bar=True)
        
        # Report the loss relevant to the current training phase
        current_loss = loss_refined if (self.current_epoch >= self.refinement_warmup_epoch) else loss_semantic
        self.log('val_loss', current_loss)
        
        # Store for epoch end logging
        self.validation_step_outputs.append({
            'loss': current_loss.detach(),
            'iou': iou.detach(),
            'loss_semantic': loss_semantic.detach(),
            'loss_refined': loss_refined.detach()
        })
        
        return current_loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
            
        avg_iou = torch.stack([x['iou'] for x in self.validation_step_outputs]).mean()
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_sem_loss = torch.stack([x['loss_semantic'] for x in self.validation_step_outputs]).mean()
        avg_ref_loss = torch.stack([x['loss_refined'] for x in self.validation_step_outputs]).mean()
        
        phase = "Refinement" if (self.current_epoch >= self.refinement_warmup_epoch) else "Semantic"
        
        # Force print to bypass tqdm overwriting
        print(f"\n\n{'='*20} Epoch {self.current_epoch} Results ({phase}) {'='*20}")
        print(f"Validation Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")
        print(f"Semantic Loss: {avg_sem_loss:.4f} | Refined Loss: {avg_ref_loss:.4f}")
        print(f"{'='*60}\n")
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        param_groups = [
            {'params': self.sam_params, 'lr': self.learning_rate, 'name': 'backbone'},
            {'params': self.refiner_params, 'lr': self.learning_rate * self.refiner_lr_scale, 'name': 'refiner'}
        ]
        
        optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        
        def lr_schedule(epoch):
            # Warmup period
            if epoch < 2: return 0.1 + 0.9 * (epoch / 2)
            # Refinement phase: constant stable LR
            if epoch >= self.refinement_warmup_epoch: return 1.0 
            # Semantic phase: cosine decay
            progress = (epoch - 2) / max(1, self.refinement_warmup_epoch - 2)
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def save_lora_checkpoint(self, path):
        self.sam_backbone.save_lora_parameters(path)
        
    def load_lora_checkpoint(self, path):
        self.sam_backbone.load_lora_parameters(path)