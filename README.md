# MSNet: Multi-Semantic Modelling for Glass Surface Detection in the Wild

> **AAAI 2026** — Qianyu Cheng, Huankang Guan, Rynson W.H. Lau
>
> Department of Computer Science, City University of Hong Kong

## Core Idea

Glass images inherently contain three semantic components:

| Component | Description | Origin |
|---|---|---|
| **Transmission** | What is visible *through* the glass | Far side of the glass |
| **Reflection** | What is *reflected on* the glass surface | Viewer's side |
| **Surrounding** | Non-glass regions adjacent to the glass | Viewer's side |

**Key observation**: Reflection ≈ Surrounding (same side), both ≠ Transmission. This asymmetric similarity forms a **multi-semantic signature** unique to glass surfaces.

## Architecture

```
Input Image I
    │
    ├──→ SDM (Semantic Decomposition Module)
    │     ├── DSEB: LRM extracts reflection R;
    │     │         CLIP(LoRA) encodes I→F_I, R→F_r
    │     └── SEB:  F_t = F_I·Attn(F_I,F_r) − F_r    (transmission)
    │               F_s = F_I − Attn(F_I,F_t)·F_t     (surrounding)
    │
    ├──→ GSSAM (Glass-Specific SAM)
    │     └── SAM ViT-H + LoRA → F_g
    │
    └──→ ASFM (Adaptive Semantic Fusion Module)
          ├── Fuses {F_I, F_r, F_t, F_s, F_g} via softmax-weighted sum
          └── Generates P_sparse, P_dense prompts
                │
                └──→ SAM Mask Decoder → Glass Mask
```

## Repository Structure

```
AAAI26-MSNet/
├── Models/
│   ├── MSNet.py                 # Main model: BaselineSAMModel (SAM ViT-H + LoRA + SDM + ASFM)
│   ├── Base.py                  # PyTorch Lightning base class
│   ├── loss.py                  # Combined BCE + Dice + Focal Loss
│   └── reflection/              # Pluggable reflection estimator zoo
│       ├── __init__.py
│       ├── base.py              # BaseReflectionEstimator (abstract)
│       ├── registry.py          # Factory + @register_estimator decorator
│       ├── location_estimator.py    # 'lrm' — LRM (Dong et al., ICCV 2021)
│       └── reflection_estimator2024.py  # 'cvpr2024' — Zhu et al. (CVPR 2024)
│
├── Models/ablation/             # Ablation study variants
│
├── Data/
│   ├── SAMDataLoader.py         # NPZ dataset with MixUp + Albumentations
│   ├── PLdataModule.py          # PyTorch Lightning DataModule
│   └── Tonpz_new.py             # Raw image → NPZ conversion tool
│
├── clip/                        # CLIP Surgery ViT-B/16 (bundled)
│
├── Utils/
│   └── metric_utils.py          # Dice / IoU / Hausdorff metrics
│
├── train.py                     # Training script
├── test.py                      # Evaluation script (full dataset, 4 metrics)
├── inference.py                 # Single-image inference (outputs mask + overlay)
└── README.md
```

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA ≥ 11.8
- GPU: NVIDIA RTX 4090 (24 GB) or equivalent

### Setup

```bash
git clone https://github.com/chengqianyu03/AAAI26-MSNet.git
cd AAAI26-MSNet

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning albumentations opencv-python-headless monai
pip install pandas tqdm efficientnet_pytorch

# SAM ViT-H
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Checkpoints

Download and place in `checkpoints/`:

| File | Description | Link |
|---|---|---|
| `sam_vit_h_4b8939.pth` | SAM ViT-H backbone | [Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `model.pth` | LRM reflection estimator | [LRM repo](https://github.com/zdlarr/Location-aware-SIRR) |

## Data Preparation

MSNet estimates reflection cues **on-the-fly** inside the model via the built-in reflection estimator (e.g., LRM/CVPR2024).  
So for dataset preparation, you only need:

- input RGB images
- ground-truth binary masks

`removal_reflection` is a legacy field from earlier experiments and is **not used** in the current MSNet training/inference pipeline.

### NPZ Format

Each `.npz` file contains:

| Key | Shape | Description |
|---|---|---|
| `data` | `[1, 3, H, W]` | RGB image |
| `label` | `[1, 1, H, W]` | Binary glass mask |


### Convert Raw Images to NPZ

```bash
python Data/Tonpz_new.py
```

Edit the paths at the bottom of the script to point to your `image_dir`, `mask_dir`, and `output_dir`.

### On-the-fly Reflection Estimation

Reflection maps are generated during forward pass (no precomputed reflection map files are required). Configure the estimator with:

- `--reflection_estimator` (e.g., `lrm`, `cvpr2024`)
- `--reflection_checkpoint` (checkpoint path for the selected estimator)

### Directory Layout

```
/path/to/dataset/
├── train/       # Training NPZ files
│   ├── 0.npz
│   ├── 1.npz
│   └── ...
└── test/        # Validation / Test NPZ files
    ├── 0.npz
    └── ...
```

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
    --lr 1e-5 \
    --max_epochs 50 \
    --gpu 0 \
    --ckpt_dir /path/to/save/checkpoints
```

### Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `--lora_rank` | 512 | SAM LoRA rank |
| `--clip_lora_rank` | 128 | CLIP LoRA rank |
| `--clip_lora_alpha` | 256 | CLIP LoRA scaling alpha |
| `--clip_layer_idx` | 10 | CLIP feature extraction layer |
| `--ft_dec` / `--no_ft_dec` | True | Finetune SAM mask decoder |
| `--lr` | 1e-5 | Base learning rate |
| `--reflection_estimator` | lrm | Reflection estimator (`lrm`, `cvpr2024`) |
| `--gradient_clip_val` | 1.0 | Gradient clipping |
| `--accumulate_grad_batches` | 1 | Gradient accumulation steps |
| `--patience` | 10 | Early stopping patience |
| `--resume_from` | None | Resume from checkpoint path |

### Differential Learning Rates

| Module | LR multiplier |
|---|---|
| SAM LoRA | 1× |
| SAM Mask Decoder | 0.5× |
| SDM + ASFM (Prompt Gen) | 1× |
| Reflection Estimator | 0.1× (if unfrozen) |

## Evaluation

```bash
python test.py \
    --data_dir /path/to/dataset \
    --checkpoint_path /path/to/best.ckpt \
    --output_dir evaluation_results/ \
    --threshold 0.5
```

Multi-threshold sweep:

```bash
python test.py \
    --data_dir /path/to/dataset \
    --checkpoint_path /path/to/best.ckpt \
    --multi_threshold
```

**Metrics**: IoU, MAE, F_β (β=0.3), BER

## Inference

Run on a single image:

```bash
python inference.py \
    --input_image /path/to/photo.png \
    --checkpoint_path /path/to/best.ckpt \
    --output_dir results/
```

**Outputs**:
- `photo_prob.png` — probability heatmap (grayscale)
- `photo_mask.png` — binary mask
- `photo_overlay.png` — red overlay + green contour on original image

## Results

### GSD-S (NeurIPS 2022)

| Method | IoU | MAE | F_β | BER |
|---|---|---|---|---|
| Mask2Former (CVPR'22) | 0.732 | 0.043 | 0.838 | 8.93 |
| GlassSemNet (NeurIPS'22) | 0.754 | 0.041 | 0.861 | 9.77 |
| SEEN-FT (NeurIPS'23) | 0.751 | 0.039 | 0.856 | 8.98 |
| **MSNet (Ours)** | **0.817** | **0.027** | **0.892** | **6.09** |

### GDD (CVPR 2020)

| Method | IoU | MAE | F_β | BER |
|---|---|---|---|---|
| GlassSemNet (NeurIPS'22) | 0.902 | 0.059 | 0.942 | 4.67 |
| GhostingNet (TPAMI'25) | 0.893 | 0.054 | 0.943 | 5.13 |
| **MSNet (Ours)** | **0.915** | **0.043** | **0.955** | **4.17** |

### GSD (CVPR 2021)

| Method | IoU | MAE | F_β | BER |
|---|---|---|---|---|
| GlassSemNet (NeurIPS'22) | 0.854 | 0.068 | 0.903 | 5.69 |
| GhostingNet (TPAMI'25) | 0.838 | 0.055 | 0.904 | 6.06 |
| **MSNet (Ours)** | **0.878** | **0.042** | **0.916** | **4.69** |

## Citation

```bibtex
@inproceedings{cheng2026msnet,
  title     = {Multi-Semantic Modelling for Glass Surface Detection in the Wild},
  author    = {Cheng, Qianyu and Guan, Huankang and Lau, Rynson W.H.},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

## Acknowledgements

- [SAM](https://github.com/facebookresearch/segment-anything) — Meta AI
- [CLIP Surgery](https://github.com/xmed-lab/CLIP_Surgery) — Li et al.
- [LRM](https://github.com/zdlarr/Location-aware-SIRR) — Dong et al.

## License

This project is for academic research purposes.
