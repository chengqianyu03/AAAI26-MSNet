"""
LRM (Location-aware Recurrent Model) Reflection Estimator.
Original LSTM + Laplacian Pyramid based iterative approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import BaseReflectionEstimator, ReflectionOutput
from .registry import register_estimator


# ── LRM Building Blocks ──────────────────────────────────────────────── #

class _Conv2DLayer(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, stride=1, pad=None, dilation=1, norm=None, act=None, bias=False):
        super().__init__()
        pad = pad if pad is not None else dilation * (k - 1) // 2
        self.add_module('conv2d', nn.Conv2d(in_ch, out_ch, k, stride, pad, dilation=dilation, bias=bias))
        if norm is not None: self.add_module('norm', norm(out_ch))
        if act  is not None: self.add_module('act', act)


class _SElayer(nn.Module):
    def __init__(self, ch, red=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(ch, ch // red), nn.ReLU(True), nn.Linear(ch // red, ch), nn.Sigmoid())
    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.se(self.pool(x).view(b, c)).view(b, c, 1, 1)


class _ResBlock(nn.Module):
    def __init__(self, ch, norm=None, dilation=1, bias=False, se_red=None, res_scale=1, act=nn.ReLU(True)):
        super().__init__()
        self.c1 = _Conv2DLayer(ch, ch, 3, 1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.c2 = _Conv2DLayer(ch, ch, 3, 1, dilation=dilation, norm=norm, act=None, bias=None)
        self.se = _SElayer(ch, se_red) if se_red else None
        self.s  = res_scale
    def forward(self, x):
        r = self.c2(self.c1(x))
        if self.se: r = self.se(r)
        return x + r * self.s


class _ChannelAttn(nn.Module):
    def __init__(self, ch, red=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1); self.mx = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(ch, ch//red, 1, bias=False); self.relu = nn.ReLU(True)
        self.f2 = nn.Conv2d(ch//red, ch, 1, bias=False); self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.f2(self.relu(self.f1(self.avg(x)))) + self.f2(self.relu(self.f1(self.mx(x)))))


class _SpatialAttn(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=ks//2, bias=False); self.sig = nn.Sigmoid()
    def forward(self, x):
        return self.sig(self.conv(torch.cat([x.mean(1, True), x.max(1, True)[0]], 1)))


class _CBAMBlock(nn.Module):
    def __init__(self, ch, red=16):
        super().__init__()
        self.ca = _ChannelAttn(ch, red); self.sa = _SpatialAttn()
    def forward(self, x):
        return self.sa(self.ca(x) * x) * (self.ca(x) * x)


class _ResCbamBlock(nn.Module):
    def __init__(self, ch, norm=None, dilation=1, bias=False, cbam_red=None, act=nn.ReLU(True)):
        super().__init__()
        self.c1 = _Conv2DLayer(ch, ch, 3, 1, dilation=dilation, norm=norm, act=act, bias=bias)
        self.c2 = _Conv2DLayer(ch, ch, 3, 1, dilation=dilation, norm=norm, act=None, bias=None)
        self.cbam = _CBAMBlock(ch, cbam_red) if cbam_red else None
    def forward(self, x):
        r = self.c2(self.c1(x))
        if self.cbam: r = self.cbam(r)
        return x + r


class _LaplacianPyramid(nn.Module):
    def __init__(self, dim=3):
        super().__init__()
        self.dim = dim
        k = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
        k = np.repeat(k[None, None, :, :], dim, 0)
        self.kernel = nn.Parameter(torch.FloatTensor(k), requires_grad=False)
    def forward(self, x):
        laps = []
        for s in [0.125, 0.25, 0.5, 1.0]:
            xs = F.interpolate(x, scale_factor=s, mode='bilinear') if s < 1.0 else x
            lap = F.conv2d(xs, self.kernel, groups=self.dim, padding=1)
            if s < 1.0:
                lap = F.interpolate(lap, size=x.shape[2:], mode='bilinear')
            laps.append(lap)
        return torch.cat(laps, 1)


class _LRM(nn.Module):
    """Full LRM network (LSTM + Laplacian + U-Net decoder)."""
    def __init__(self):
        super().__init__()
        self.lap_pyramid = _LaplacianPyramid(dim=6)
        self.det_conv0 = nn.Sequential(nn.Conv2d(6, 32, 3, 1, 1), nn.ReLU())
        self.det_conv1 = _ResBlock(32, se_red=2, res_scale=0.1)
        self.det_conv2 = _ResBlock(32, se_red=2, res_scale=0.1)
        self.det_conv3 = _ResBlock(32, se_red=2, res_scale=0.1)
        self.det_conv4 = _ResBlock(32, se_red=2, res_scale=0.1)
        self.det_conv4_1 = _ResBlock(32, se_red=2, res_scale=0.1)
        self.det_conv4_2 = _ResBlock(32, se_red=2, res_scale=0.1)
        self.det_conv5 = nn.Sequential(nn.Conv2d(24, 32, 3, 1, 1), nn.PReLU())
        self.det_conv6  = _ResBlock(32, se_red=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv7  = _ResBlock(32, se_red=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv8  = _ResBlock(32, se_red=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv9  = _ResBlock(32, se_red=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv10 = _ResBlock(32, se_red=2, res_scale=0.1, act=nn.PReLU())
        self.det_conv11 = _ResBlock(32, se_red=2, res_scale=0.1, act=nn.PReLU())
        self.p_relu = nn.PReLU(); self.relu = nn.ReLU()
        self.det_conv_mask0 = nn.Sequential(nn.Conv2d(32,32,3,1,1),nn.ReLU(),nn.Conv2d(32,1,3,1,1),nn.Sigmoid())
        self.conv_i = nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(128,64,3,1,1),nn.Sigmoid())
        self.det_conv_mask1 = nn.Sequential(nn.Conv2d(64,32,3,1,1),nn.ReLU(),nn.Conv2d(32,3,3,1,1),nn.ReLU())
        self.conv1  = nn.Sequential(nn.Conv2d(10,64,5,1,2),nn.ReLU())
        self.conv2  = nn.Sequential(nn.Conv2d(64,128,3,2,1),nn.ReLU())
        self.conv3  = nn.Sequential(nn.Conv2d(128,128,3,1,1),nn.ReLU())
        self.conv4  = nn.Sequential(nn.Conv2d(128,256,3,2,1),nn.ReLU())
        self.conv5  = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.ReLU())
        self.conv6  = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.ReLU())
        self.diconv1 = nn.Sequential(nn.Conv2d(256,256,3,1,2,dilation=2),nn.ReLU())
        self.diconv2 = nn.Sequential(nn.Conv2d(256,256,3,1,4,dilation=4),nn.ReLU())
        self.diconv3 = nn.Sequential(nn.Conv2d(256,256,3,1,8,dilation=8),nn.ReLU())
        self.diconv4 = nn.Sequential(nn.Conv2d(256,256,3,1,16,dilation=16),nn.ReLU())
        self.conv7  = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.ReLU())
        self.conv8  = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.ReLU())
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1),nn.ReflectionPad2d((1,0,1,0)),nn.AvgPool2d(2,1),nn.ReLU())
        self.conv9   = nn.Sequential(nn.Conv2d(128,128,3,1,1),nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128,64,4,2,1),nn.ReflectionPad2d((1,0,1,0)),nn.AvgPool2d(2,1),nn.ReLU())
        self.conv10  = nn.Sequential(nn.Conv2d(64,32,3,1,1),nn.ReLU())
        self.outframe1 = nn.Sequential(nn.Conv2d(256,3,3,1,1),nn.ReLU())
        self.outframe2 = nn.Sequential(nn.Conv2d(128,3,3,1,1),nn.ReLU())
        self.output    = nn.Sequential(nn.Conv2d(32,3,3,1,1),nn.ReLU())
        self.cbam_block0 = _ResCbamBlock(64,  cbam_red=2)
        self.cbam_block1 = _ResCbamBlock(128, cbam_red=4)
        self.cbam_block2 = _ResCbamBlock(128, cbam_red=4)
        self.cbam_block3 = _ResCbamBlock(256, cbam_red=8)
        self.cbam_block4 = _ResCbamBlock(256, cbam_red=8)
        self.cbam_block5 = _ResCbamBlock(256, cbam_red=8)

    def forward(self, I, T, h, c):
        x = torch.cat([I, T], 1); lap = self.lap_pyramid(x)
        x = self.det_conv0(x)
        for blk in [self.det_conv1, self.det_conv2, self.det_conv3,
                     self.det_conv4, self.det_conv4_1, self.det_conv4_2]:
            x = F.relu(blk(x))
        lap = self.det_conv5(lap)
        for blk in [self.det_conv6, self.det_conv7, self.det_conv8]:
            lap = self.p_relu(blk(lap))
        c_map = self.det_conv_mask0(lap)
        for blk in [self.det_conv9, self.det_conv10, self.det_conv11]:
            lap = self.p_relu(blk(lap))
        lap = (1 - c_map) * lap
        x = torch.cat([x, lap, h], 1)
        i = self.conv_i(x); f = self.conv_f(x); g = self.conv_g(x); o = self.conv_o(x)
        c = f * c + i * g; h = o * torch.tanh(c)
        reflect = self.det_conv_mask1(h)
        x = torch.cat([I, T, reflect, c_map], 1)
        x = self.conv1(x); x = self.cbam_block0(x); res1 = x
        x = self.conv2(x); x = self.conv3(x); x = self.cbam_block1(x); x = self.cbam_block2(x); res2 = x
        x = self.conv4(x); x = self.conv5(x); x = self.conv6(x)
        x = self.cbam_block3(x); x = self.cbam_block4(x); x = self.cbam_block5(x)
        x = self.diconv1(x); x = self.diconv2(x); x = self.diconv3(x); x = self.diconv4(x)
        x = self.conv7(x); x = self.conv8(x)
        frame1 = self.outframe1(x); x = self.deconv1(x); x = x + res2
        x = self.conv9(x); frame2 = self.outframe2(x); x = self.deconv2(x); x = x + res1
        x = self.conv10(x); x = self.output(x)
        return h, c, c_map, reflect, frame1, frame2, x


# ── Registered Estimator ──────────────────────────────────────────────── #

@register_estimator('lrm')
class LRMEstimator(BaseReflectionEstimator):
    """LRM: Location-aware Recurrent Model with LSTM + Laplacian Pyramid (iterative)."""

    def __init__(
        self,
        checkpoint_path: str = None,
        proc_size: int = 256,
        n_iters: int = 3,
        finetune: bool = False,
        **kwargs,  # absorb unknown kwargs gracefully
    ):
        super().__init__(proc_size=proc_size, finetune=finetune)
        self.n_iters = n_iters
        self.lrm = _LRM()

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path, model=self.lrm)

        self.freeze()
        if finetune:
            self.unfreeze_for_finetune()

    def unfreeze_for_finetune(self):
        """Unfreeze Stage-2 decoder tail for lightweight fine-tuning."""
        for mod in [self.lrm.deconv1, self.lrm.deconv2, self.lrm.conv9,
                    self.lrm.conv10, self.lrm.outframe1, self.lrm.outframe2, self.lrm.output]:
            for p in mod.parameters():
                p.requires_grad = True

    def estimate(self, x: torch.Tensor) -> ReflectionOutput:
        B, _, H, W = x.shape
        h = torch.zeros(B, 64, H, W, device=x.device)
        c = torch.zeros(B, 64, H, W, device=x.device)
        fake_T = x.clone()

        for _ in range(self.n_iters):
            h, c, c_map, fake_R, frame1, frame2, fake_T = self.lrm(x, fake_T, h, c)

        return ReflectionOutput(
            reflection=torch.clamp(fake_R, 0, 1),
            transmission=torch.clamp(fake_T, 0, 1),
            extras={'confidence_map': c_map, 'frame1': frame1, 'frame2': frame2},
        )