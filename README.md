# MSNet: Multi-Semantic Modelling for Glass Surface Detection in the Wild

[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)]()
[![License](https://img.shields.io/badge/License-Academic-lightgrey)]()

Official implementation of **MSNet: Multi-Semantic Modelling for Glass Surface Detection in the Wild**.

> **AAAI 2026**  
> Qianyu Cheng, Huankang Guan, Rynson W.H. Lau  
> Department of Computer Science, City University of Hong Kong

**Links:** [Paper](https://www.cs.cityu.edu.hk/~rynson/papers/aaai26c.pdf) | [Project Page](https://www.cs.cityu.edu.hk/~rynson/projects/mirror_glass/MirrorGlassDetection.html) | [Code](https://github.com/chengqianyu03/AAAI26-MSNet)

---

## Overview

Glass images contain three semantic components:

| Component | Description | Origin |
|---|---|---|
| **Transmission** | Content visible *through* the glass | Far side of the glass |
| **Reflection** | Content *reflected on* the glass surface | Viewer's side |
| **Surrounding** | Non-glass regions adjacent to the glass | Viewer's side |

MSNet is built on the observation that:

> **Reflection ≈ Surrounding**, while both are different from **Transmission**.

This asymmetric multi-semantic relationship provides a discriminative cue for glass surface detection in the wild.

---

## Architecture

```text
Input Image I
    │
    ├──→ SDM: Semantic Decomposition Module
    │     ├── DSEB: LRM extracts reflection R
    │     │         CLIP(LoRA) encodes I→F_I, R→F_r
    │     └── SEB:  F_t = F_I·Attn(F_I,F_r) − F_r
    │               F_s = F_I − Attn(F_I,F_t)·F_t
    │
    ├──→ GSSAM: Glass-Specific SAM
    │     └── SAM ViT-H + LoRA → F_g
    │
    └──→ ASFM: Adaptive Semantic Fusion Module
          ├── Fuse {F_I, F_r, F_t, F_s, F_g}
          └── Generate sparse/dense prompts
                │
                └──→ SAM Mask Decoder → Glass Mask
```

---

## Repository Structure

```text
AAAI26-MSNet/
├── Models/
│   ├── MSNet.py                  # Main MSNet model
│   ├── Base.py                   # PyTorch Lightning base class
│   ├── loss.py                   # Loss functions
│   ├── activation.py
│   ├── ablation/                 # Ablation variants
│   └── reflection/
│       ├── base.py
│       ├── registry.py
│       └── location_estimator.py # LRM reflection estimator
│
├── Data/
│   ├── SAMDataLoader.py
│   ├── SAMDataLoader_ab.py
│   ├── PLdataModule.py
│   └── Tonpz_new.py              # Raw image/mask to NPZ
│
├── Utils/
│   ├── metric_utils.py
│   ├── dataset_utils.py
│   ├── regularization_utils.py
│   └── utils.py
│
├── clip/                         # Bundled CLIP Surgery code
├── TestImg/                      # Example images
├── train.py
├── test.py
├── inference.py
├── requirements.txt
└── README.md
```

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8
- Recommended GPU: NVIDIA RTX 4090 24GB or equivalent

### Setup

```bash
git clone https://github.com/chengqianyu03/AAAI26-MSNet.git
cd AAAI26-MSNet

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

---

## Checkpoints

Please download the following checkpoints and place them under `checkpoints/`.

| File | Description | Link |
|---|---|---|
| `sam_vit_h_4b8939.pth` | SAM ViT-H backbone | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `model.pth` | LRM reflection estimator | [LRM Repository](https://github.com/zdlarr/Location-aware-SIRR) |
| `best.ckpt` | Trained MSNet checkpoint | [Google Drive](https://drive.google.com/file/d/1iARIvO8pEOR-0UH23qRVw9_71h5sV_F5/view?usp=drive_link) |

Expected layout:

```text
checkpoints/
├── sam_vit_h_4b8939.pth
├── model.pth
└── best.ckpt
```

---

## Data Preparation

MSNet estimates reflection cues on-the-fly using the built-in **LRM** reflection estimator.  
For dataset preparation, only RGB images and binary glass masks are required.

### NPZ Format

Each `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `data` | `[1, 3, H, W]` | RGB image |
| `label` | `[1, 1, H, W]` | Binary glass mask |

### Convert Images and Masks to NPZ

Edit the paths in `Data/Tonpz_new.py`, then run:

```bash
python Data/Tonpz_new.py
```

Expected dataset layout:

```text
/path/to/dataset/
├── train/
│   ├── 0.npz
│   ├── 1.npz
│   └── ...
└── test/
    ├── 0.npz
    └── ...
```

---

## Training

```bash
python train.py \
    --data_dir /path/to/dataset \
    --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth \
    --sam_model_type vit_h \
    --lora_rank 512 \
    --ft_dec \
    --clip_lora_rank 128 \
    --clip_lora_alpha 256 \
    --reflection_estimator lrm \
    --reflection_checkpoint checkpoints/model.pth \
    --reflection_proc_size 256 \
    --reflection_n_iters 3 \
    --lr 2e-5 \
    --weight_decay 5e-5 \
    --max_epochs 50 \
    --gpu 0 \
    --ckpt_dir checkpoints/train
```

### Main Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--lora_rank` | 512 | SAM LoRA rank |
| `--clip_lora_rank` | 128 | CLIP LoRA rank |
| `--clip_lora_alpha` | 256 | CLIP LoRA scaling alpha |
| `--clip_layer_idx` | 10 | CLIP feature layer |
| `--lr` | `2e-5` | Base learning rate |
| `--weight_decay` | `5e-5` | Weight decay |
| `--reflection_estimator` | `lrm` | Reflection estimator |
| `--reflection_proc_size` | 256 | Reflection estimator input size |
| `--reflection_n_iters` | 3 | LRM iteration number |
| `--max_epochs` | 50 | Training epochs |
| `--patience` | 10 | Early stopping patience |

---

## Evaluation

Run evaluation on the test split:

```bash
python test.py \
    --data_dir /path/to/dataset \
    --checkpoint_path checkpoints/best.ckpt \
    --output_dir evaluation_results/
```

`test.py` performs multi-threshold evaluation by default.

**Metrics:** IoU, MAE, F_β (β=0.3), BER

---

## Inference

Run inference on a single image:

```bash
python inference.py \
    --input_image /path/to/photo.png \
    --checkpoint_path checkpoints/best.ckpt \
    --output_dir results/
```

Outputs:

```text
results/
├── photo_prob.png     # Probability map
├── photo_mask.png     # Binary mask
└── photo_overlay.png  # Overlay visualization
```

---

## Results

### GSD-S

| Method | IoU | MAE | F_β | BER |
|---|---:|---:|---:|---:|
| Mask2Former | 0.732 | 0.043 | 0.838 | 8.93 |
| GlassSemNet | 0.754 | 0.041 | 0.861 | 9.77 |
| SEEN-FT | 0.751 | 0.039 | 0.856 | 8.98 |
| **MSNet** | **0.817** | **0.027** | **0.892** | **6.09** |

### GDD

| Method | IoU | MAE | F_β | BER |
|---|---:|---:|---:|---:|
| GlassSemNet | 0.902 | 0.059 | 0.942 | 4.67 |
| GhostingNet | 0.893 | 0.054 | 0.943 | 5.13 |
| **MSNet** | **0.915** | **0.043** | **0.955** | **4.17** |

### GSD

| Method | IoU | MAE | F_β | BER |
|---|---:|---:|---:|---:|
| GlassSemNet | 0.854 | 0.068 | 0.903 | 5.69 |
| GhostingNet | 0.838 | 0.055 | 0.904 | 6.06 |
| **MSNet** | **0.878** | **0.042** | **0.916** | **4.69** |

---

## Citation

```bibtex
@inproceedings{cheng2026msnet,
  title     = {Multi-Semantic Modelling for Glass Surface Detection in the Wild},
  author    = {Cheng, Qianyu and Guan, Huankang and Lau, Rynson W.H.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

---

## Acknowledgements

This project builds upon the following excellent works:

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery)
- [LRM](https://github.com/zdlarr/Location-aware-SIRR)

---

## License

This project is released for academic research purposes only.
