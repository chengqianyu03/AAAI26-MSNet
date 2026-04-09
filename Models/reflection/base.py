"""
Abstract base class for all reflection estimators.
Every new reflection model must subclass this and implement `estimate()`.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple


@dataclass
class ReflectionOutput:
    """
    Standardized output container for all reflection estimators.
    
    Attributes:
        reflection:   (B, 3, H, W) estimated reflection layer, [0, 1].
        transmission: (B, 3, H, W) estimated transmission layer, [0, 1].
        extras:       Optional dict for model-specific auxiliary outputs
                      (e.g., location maps, confidence maps, intermediate features).
    """
    reflection: torch.Tensor
    transmission: torch.Tensor
    extras: Dict[str, Any] = field(default_factory=dict)


class BaseReflectionEstimator(nn.Module, ABC):
    """
    Abstract interface that every reflection estimator must implement.

    Contract:
        Input:  images (B, 3, H, W) in [0, 1]
        Output: ReflectionOutput with reflection & transmission both (B, 3, H, W) in [0, 1]
    
    Design principles:
        1. Uniform I/O — callers never need to know which model is inside.
        2. Resolution handling — automatic resize to proc_size, then resize back.
        3. Freeze-by-default — weights frozen unless finetune=True.
        4. Checkpoint flexibility — handles DDP/DP prefix cleanup automatically.
    """

    def __init__(self, proc_size: int = 256, finetune: bool = False):
        super().__init__()
        self.proc_size = proc_size
        self._finetune = finetune

    # ------------------------------------------------------------------ #
    #  Subclass MUST implement this                                       #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def estimate(self, x: torch.Tensor) -> ReflectionOutput:
        """
        Core estimation logic operating on preprocessed input.

        Args:
            x: (B, 3, H_proc, W_proc) normalized to [0, 1], already resized.

        Returns:
            ReflectionOutput with tensors at (B, 3, H_proc, W_proc).
        """
        ...

    # ------------------------------------------------------------------ #
    #  Subclass MAY override these for finer control                      #
    # ------------------------------------------------------------------ #
    def freeze(self):
        """Freeze all parameters. Called after __init__ by default."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_for_finetune(self):
        """Selectively unfreeze layers for downstream fine-tuning.
        Override this in subclasses to unfreeze specific decoder/head layers.
        """
        for p in self.parameters():
            p.requires_grad = True

    @staticmethod
    def clean_state_dict(state_dict: dict) -> OrderedDict:
        """Remove common DDP / DataParallel prefixes from keys."""
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            new_k = k
            for prefix in ('module.', 'netG_T.', 'model.'):
                new_k = new_k.replace(prefix, '')
            cleaned[new_k] = v
        return cleaned

    def load_checkpoint(self, path: str, model: nn.Module = None, strict: bool = False):
        """Safely load a checkpoint with prefix cleanup."""
        if path is None or not os.path.exists(path):
            print(f"[{self.__class__.__name__}] Checkpoint not found: {path}")
            return
        state_dict = torch.load(path, map_location='cpu')
        # Unwrap nested dicts
        for key in ('state_dict', 'model', 'net'):
            if key in state_dict:
                state_dict = state_dict[key]
                break
        cleaned = self.clean_state_dict(state_dict)
        target = model if model is not None else self
        missing, unexpected = target.load_state_dict(cleaned, strict=strict)
        if missing:
            print(f"[{self.__class__.__name__}] Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"[{self.__class__.__name__}] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # ------------------------------------------------------------------ #
    #  Public forward (DO NOT override)                                   #
    # ------------------------------------------------------------------ #
    def forward(self, images: torch.Tensor) -> ReflectionOutput:
        """
        Unified forward pass with automatic normalization & resizing.

        Args:
            images: (B, 3, H, W) in [0, 1] or [0, 255].

        Returns:
            ReflectionOutput at original (H, W) resolution.
        """
        B, C, H_orig, W_orig = images.shape

        # 1. Normalize to [0, 1]
        x = images.float()
        if x.max() > 1.5:
            x = x / 255.0

        # 2. Resize to processing resolution
        need_resize = (H_orig != self.proc_size or W_orig != self.proc_size)
        if need_resize:
            x = F.interpolate(x, size=(self.proc_size, self.proc_size),
                              mode='bilinear', align_corners=False)

        # 3. Core estimation
        output = self.estimate(x)

        # 4. Resize back to original resolution
        if need_resize:
            output.reflection = F.interpolate(
                output.reflection, size=(H_orig, W_orig),
                mode='bilinear', align_corners=False
            )
            output.transmission = F.interpolate(
                output.transmission, size=(H_orig, W_orig),
                mode='bilinear', align_corners=False
            )
            # Resize any spatial extras
            for k, v in output.extras.items():
                if isinstance(v, torch.Tensor) and v.dim() == 4:
                    output.extras[k] = F.interpolate(
                        v, size=(H_orig, W_orig),
                        mode='bilinear', align_corners=False
                    )

        return output

    # ------------------------------------------------------------------ #
    #  Convenience: backward-compatible tuple output                      #
    # ------------------------------------------------------------------ #
    def estimate_tuple(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible interface returning (reflection, transmission) tuple."""
        out = self.forward(images)
        return out.reflection, out.transmission

    def __repr__(self):
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (f"{self.__class__.__name__}("
                f"proc_size={self.proc_size}, "
                f"params={n_total:,}, trainable={n_train:,})")