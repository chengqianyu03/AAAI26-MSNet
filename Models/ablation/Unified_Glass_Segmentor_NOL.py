import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import math
from segment_anything import sam_model_registry
from Models.Base import BaseModel
from Models.loss import CombinedBCEDiceFocalLoss
from Models.modulezoo import get_prompt_module
from Models.activation import get_activation

class SamPromptLearner_WithLoRA_DirectPhase2(BaseModel):
    """Simplified SAM model without LoRA, using direct fine-tuning approach"""
    def __init__(
        self,
        in_channels,
        out_channels,
        lr=1e-6,
        prompt_lr_factor=1,
        weight_decay=1e-4,
        sam_model_name="vit_l",
        sam_checkpoint="checkpoint/sam_vit_l_0b3195.pth",
        ft_dec=False,
        num_points=20,
        bce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.5,
        focal_gamma=2.0,
        focal_alpha=0.5,
        module_name='sam1',
        activation_type='scaled_sigmoid',
        temp_scaling=0.5,
        lora_rank = 0,
        lora_layers = None,
        clip_module_name = 'CS-VIT-B/16'
    ):
        super().__init__(in_channels, out_channels, lr, weight_decay)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.num_points = num_points
        self.prompt_lr_factor = prompt_lr_factor
        self.clip_module_name = clip_module_name
        
        # 初始化激活函数
        self.activation_type = activation_type
        self.temp_scaling = temp_scaling
        self.apply_sharpening = False
        self.activation_fct, _ = get_activation(
            activation_type,
            temp_scaling=temp_scaling,
            sharpness=None,
            apply_sharpening=False
        )
        
        # 初始化损失函数
        self.loss_fn = CombinedBCEDiceFocalLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha
        )

        # 初始化提示模块
        self.module_name = module_name
        self.prompt_embedding_module = get_prompt_module(
            module_name=self.module_name,
            num_points=self.num_points,
            d_model=256,
            clip_module_name = self.clip_module_name,
        )
        
        # 初始化SAM模型（无LoRA）
        self.sam = sam_model_registry[sam_model_name](checkpoint=sam_checkpoint)
        
        # 冻结SAM参数，根据需要设置mask_decoder为可训练
        for name, param in self.sam.named_parameters():
            if 'mask_decoder' in name:
                param.requires_grad = ft_dec
            else:
                param.requires_grad = False
                
        # 确保提示编码器的positional encoding保持冻结状态
        for name, param in self.sam.prompt_encoder.named_parameters():
            param.requires_grad = False

    def _apply_activation(self, logits, is_training=True):
        """只应用激活函数，不进行边界增强"""
        probs = self.activation_fct(logits).float()
        return probs

    def forward(self, batched_input):
        # 处理输入，使用已有的image_embeddings或通过encoder生成
        try:
            image_embeddings = batched_input['image_embeddings'].squeeze(1)
            if image_embeddings.dim() > 3:  # 处理可能的额外维度
                image_embeddings = image_embeddings.squeeze(1)
        except (KeyError, ValueError):
            data = batched_input['data'].squeeze(1)
            _, _, h, w = data.shape
            if h != 1024 or w != 1024:
                data = F.interpolate(
                    data,
                    size=(1024, 1024),
                    mode='bilinear',
                    align_corners=False
                )

            # 使用SAM的图像编码器生成embeddings
            with torch.no_grad():  # 图像编码器是冻结的
                image_embeddings = self.sam.image_encoder(data)
        
        # 获取SAM位置编码
        image_positional = self.sam.prompt_encoder.get_dense_pe()

        # 初始化提示编码
        sparse_embeddings2, dense_embeddings2 = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        
        # 获取输入数据
        orimg = batched_input['data'].squeeze(1)
        # 这些仅被提示模块使用
        rm_f = batched_input['removal_reflection'].squeeze(1) if 'removal_reflection' in batched_input else None
        refl = batched_input['reflection_map'].squeeze(1) if 'reflection_map' in batched_input else None
        
        if 'heatmap_diff' in batched_input:
            if batched_input['heatmap_diff'].dim() > 3:
                heatdif = batched_input['heatmap_diff'].squeeze(-1).squeeze(1)
            else:
                heatdif = batched_input['heatmap_diff'].squeeze(1)
        else:
            # 如果没有可用的热图，创建空热图
            heatdif = torch.zeros_like(orimg[:, 0])

        # 生成最终提示 - 由于我们跳过了粗分割阶段，所以传递None作为粗掩码
        sparse_embeddings, dense_embeddings = self.prompt_embedding_module(
            image_embeddings, orimg, refl, sparse_embeddings2, dense_embeddings2
        )

        # 使用SAM的掩码解码器解码最终掩码
        low_res_pred_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_positional,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        # 重采样到标签大小
        pred_masks = F.interpolate(
            low_res_pred_masks,
            (batched_input['label'].shape[-2], batched_input['label'].shape[-1]),
            mode="bilinear",
            align_corners=False
        )
        
        return pred_masks

    def training_step(self, batch, batch_idx):
        # 前向传播 - 获取主要预测
        logits = self.forward(batch)
        
        # 应用激活函数
        #out_probs = self._apply_activation(logits, is_training=True)
        
        # 真实标签
        y = batch['label'].squeeze(1)  # shape (B, H, W)
        
        # 计算主要预测损失
        #loss = self.loss_fn(out_probs, y)
        loss = self.loss_fn(logits, y)
        
        # 记录损失
        self.log('train_loss', loss, on_step=True)
        
        self.training_step_outputs.append({'loss': loss})
        
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()

        # 获取当前学习率
        lrs = self.get_current_lr()
        prompt_lr = lrs[0] if lrs else None
        decoder_lr = lrs[1] if len(lrs) > 1 else None

        # 打印训练损失和学习率
        lr_info = f"Prompt LR: {prompt_lr:.8f}"
        if decoder_lr is not None:
            lr_info += f", Decoder LR: {decoder_lr:.8f}"

        print(f'\nEpoch {self.current_epoch} - Training Loss: {avg_loss:.4f} - {lr_info}')

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        # 前向传播 - 获取预测
        logits = self.forward(batch)
        
        # 真实标签
        y = batch['label'].squeeze(1)  # shape (B, H, W)
        
        # 计算损失
        loss = self.loss_fn(logits, y)
        
        # 记录val_loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # 定义要评估的阈值
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        # 计算每个阈值的指标
        batch_results = {'loss': loss.detach()}
        
        # 处理每个样本
        batch_size = y.shape[0]
        for threshold in thresholds:
            # 存储每个样本的IoU
            batch_ious = []
            
            # 对批次中的每个样本单独计算IoU
            for i in range(batch_size):
                # 获取单个样本的预测和标签
                sample_logits = logits[i]
                sample_y = y[i]
                
                # 二值化预测
                pred_mask = (sample_logits > threshold).float()
                
                # 计算交集和并集
                intersection = torch.logical_and(pred_mask, sample_y).float().sum()
                union = torch.logical_or(pred_mask, sample_y).float().sum()
                
                # 计算样本IoU
                sample_iou = (intersection + 1e-6) / (union + 1e-6)
                batch_ious.append(sample_iou)
            
            # 计算批次IoU均值并保存
            batch_iou = torch.stack(batch_ious).mean()
            batch_results[f'iou_{threshold}'] = batch_iou.detach()

        # 存储所有指标
        self.validation_step_outputs.append(batch_results)
        return loss

    def on_validation_epoch_end(self):
        # 计算平均验证损失
        avg_val_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()

        # 定义阈值
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        # 计算每个阈值的IoU（平均IoU方式）
        ious = {}

        for threshold in thresholds:
            # 计算所有批次IoU的平均值
            avg_iou = torch.stack([x[f'iou_{threshold}'] for x in self.validation_step_outputs]).mean()
            ious[threshold] = avg_iou

            # 记录指标
            self.log(f'val_iou_{threshold}', avg_iou, prog_bar=(threshold == 0.5))

        # 打印结果
        print(f'\nEpoch {self.current_epoch} - Validation Loss: {avg_val_loss:.4f}')
        
        print("IoU: ", end='')
        for threshold in thresholds:
            print(f'@{threshold:.2f}: {ious[threshold]:.4f}', end='  ')
        print()

        # 清除存储的输出
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        
        # 应用激活函数
        out_probs = self._apply_activation(logits, is_training=False)
        
        # 获取预测掩码，阈值为0.5
        pred_masks = (out_probs > 0.5).float()  # (B, 1, H, W)

        # 计算主要预测指标
        main_metrics = self._compute_seg_metrics(pred_masks.squeeze(1), batch['label'].squeeze(1))
        for metric_name, metric_value in main_metrics.items():
            self.log(f'test/{metric_name}', metric_value)
            
        return main_metrics

    def get_current_lr(self):
        """返回每个参数组的当前学习率列表。"""
        if not hasattr(self.trainer, 'optimizers') or not self.trainer.optimizers:
            return [0.0]
        optimizer = self.trainer.optimizers[0]
        return [float(pg.get('lr', 0.0)) for pg in optimizer.param_groups]

    def _setup_optimizers(self):
        """构造优化器和调度器"""
        params_groups = []

        # 提示模块参数
        prompt_params = [p for p in self.prompt_embedding_module.parameters() if p.requires_grad]
        if prompt_params:
            params_groups.append({
                'name': 'prompt_module',
                'params': prompt_params,
                'lr': self.learning_rate * self.prompt_lr_factor
            })

        # 掩码解码器参数
        decoder_params = [p for n, p in self.sam.named_parameters()
                         if 'mask_decoder' in n and p.requires_grad]
        if decoder_params:
            params_groups.append({
                'name': 'decoder',
                'params': decoder_params,
                'lr': self.learning_rate
            })

        optimizer = optim.AdamW(params_groups, weight_decay=self.weight_decay)

        # 学习率调度：热身 + 余弦衰减，不低于5%
        warmup_epochs = 3
        max_epochs = getattr(self.trainer, 'max_epochs', 20)

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return 0.1 + 0.9 * (epoch / warmup_epochs)
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return max(0.05, cosine)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print("\n已配置优化器和调度器:")
        for g in optimizer.param_groups:
            print(f"- {g.get('name', 'group')}: lr={g['lr']:.8f}, params={len(g['params'])}")

        # 分配给trainer
        if hasattr(self, 'trainer') and self.trainer is not None:
            self.trainer.optimizers = [optimizer]
            self.trainer.lr_schedulers = [{"scheduler": scheduler, "interval": "epoch"}]

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def configure_optimizers(self):
        # 委托给_setup_optimizers以保持一致性
        return self._setup_optimizers()
    
