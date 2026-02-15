# Offroad Scene Segmentation

Semantic segmentation of offroad environments into **11 classes** using DeepLabV3+ with MiT-B3 encoder.

## Classes

| ID | Class | ID | Class |
|----|-------|----|-------|
| 0 | Background | 6 | Flowers |
| 1 | Trees | 7 | Logs |
| 2 | Lush Bushes | 8 | Rocks |
| 3 | Dry Grass | 9 | Landscape |
| 4 | Dry Bushes | 10 | Sky |
| 5 | Ground Clutter | | |

## Setup

### Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA (tested on RTX 4060, 8GB VRAM)
- Anaconda / Miniconda

### Install

```bash
conda create --name EDU python=3.10 -y
conda activate EDU

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install segmentation-models-pytorch albumentations opencv-contrib-python tqdm matplotlib pillow numpy
```

### Verify

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import segmentation_models_pytorch; print('smp OK')"
python -c "import albumentations; print('albumentations OK')"
```

## Data Structure

```
Parent_Directory/
├── Offroad_Segmentation_Scripts/     ← this repo
│   ├── train_segmentation.py
│   ├── test_segmentation.py
│   └── visualize.py
├── Offroad_Segmentation_Training_Dataset/
│   ├── train/
│   │   ├── Color_Images/
│   │   └── Segmentation/
│   └── val/
│       ├── Color_Images/
│       └── Segmentation/
└── Offroad_Segmentation_testImages/
    ├── Color_Images/
    └── Segmentation/
```

## Training

```bash
conda activate EDU
cd Offroad_Segmentation_Scripts
python train_segmentation.py
```

**Configuration** (edit at top of `main()` in `train_segmentation.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `img_size` | 512 | Input resolution |
| `batch_size` | 4 | GPU batch size |
| `accumulate_steps` | 4 | Gradient accumulation (effective BS = 16) |
| `lr` | 1e-3 | Peak learning rate |
| `n_epochs` | 150 | Max training epochs |
| `encoder_name` | mit_b3 | Encoder backbone |
| `patience` | 30 | Early stopping patience |

**Outputs** saved to `train_stats/`:
- `checkpoints/best_model.pth` — best validation mIoU weights
- `checkpoints/latest_checkpoint.pth` — full state (resumable)
- `training_history.json` — per-epoch metrics (updated live)
- `training_curves.png` — loss & accuracy plots
- `iou_curves.png` — IoU progression
- `all_metrics_curves.png` — combined 4-panel plot

## Evaluation / Inference

```bash
python test_segmentation.py
```

With custom paths:
```bash
python test_segmentation.py \
    --model_path ./train_stats/checkpoints/best_model.pth \
    --data_dir ../Offroad_Segmentation_testImages \
    --encoder_name mit_b3
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./train_stats/checkpoints/best_model.pth` | Path to model weights |
| `--data_dir` | `../Offroad_Segmentation_testImages` | Test dataset path |
| `--output_dir` | `./predictions` | Output directory |
| `--encoder_name` | `mit_b3` | Must match training encoder |
| `--img_size` | 512 | Must match training resolution |
| `--batch_size` | 4 | Inference batch size |
| `--num_samples` | 10 | Number of comparison visualizations |
| `--tta` | True | Multi-scale TTA (3 scales × hflip = 6 predictions) |

**Outputs** saved to `predictions/`:
- `masks/` — raw prediction masks (class IDs 0-10)
- `masks_color/` — colored RGB prediction masks
- `comparisons/` — side-by-side input / ground truth / prediction
- `evaluation_metrics.txt` — mIoU, Dice, pixel accuracy
- `per_class_metrics.png` — per-class IoU bar chart

## Visualization

```bash
python visualize.py
```

Colorizes raw segmentation masks for visual inspection.

## Architecture

- **Model**: DeepLabV3+ (ASPP decoder for multi-scale context)
- **Encoder**: MiT-B3 (Mix Transformer, hierarchical self-attention)
- **Loss**: Focal Loss + Dice Loss
- **Optimizer**: AdamW with differential LR (encoder at 0.3× base LR)
- **Scheduler**: OneCycleLR (10% warmup + cosine annealing)
- **Training**: AMP (mixed precision), gradient accumulation (effective BS=16)
- **Inference**: Multi-scale TTA (0.75×, 1.0×, 1.25× + horizontal flip)

## Metric

**Mean Intersection over Union (mIoU)** — averaged across all 11 classes.

```
IoU_c = TP / (TP + FP + FN)    per class
mIoU  = mean(IoU_0 ... IoU_10)
```

---

*KrackHack Hackathon — AI/ML Track*
