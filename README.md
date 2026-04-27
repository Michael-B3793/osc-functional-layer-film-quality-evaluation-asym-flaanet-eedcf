## AFM Surface Defect Detection & Multi-dimensional Quality Evaluation

This project is a Python-based pipeline for **AFM (Atomic Force Microscopy) surface image defect segmentation** (white protrusions / black pits) and **multi-dimensional surface quality scoring**.  
It includes:

- A **Gradio UI** (`app.py`) for batch inference + visualization + report export
- Multiple **training scripts** (baseline / structure / attention / dual-expert / ablations / final model)
- **Evaluation & plotting** utilities for curves, PR, heatmaps, radar, ranking, distributions, etc.

## License / citation

This repository permits academic/research use. Any commercial use requires prior written permission from the copyright holder. See `LICENSE`.

If you use this repository in academic work, please cite:

Michael B. Paper A: Shape-Aware Dual-Expert Network for AFM Image Detection. TBD, 2026. DOI: TBD

Michael B. Paper B: TBD. TBD, 2026. DOI: TBD

Citation metadata is also available in `CITATION.cff`.

## Project structure

```
image_detection/                       # repo root
├── app.py                             # Gradio app (inference + scoring + reports)
├── requirements.txt                   # Python dependencies
├── update_val_metrics.py              # helper script (CSV post-process, local path inside)
└── image_detection/                   # main project directory
    ├── dataset_400/                   # dataset (images + label jsons)
    │   ├── train_images/              # training images
    │   ├── train_jsons/               # training annotations (LabelMe-like JSON)
    │   ├── val_images/                # validation images
    │   ├── val_jsons/                 # validation annotations
    │   ├── total_images/              # full set (optional)
    │   ├── total_jsons/               # full set annotations (optional)
    │   ├── text_images/ text_jsons/   # text/demo subset (optional)
    ├── train_code/                    # individual training scripts
    │   ├── train_baseline_unet.py
    │   ├── train_unetpp.py
    │   ├── train_attention_transunet.py
    │   ├── train_dual_expert_strategy.py
    │   ├── train_ablation_no_aspp.py
    │   ├── train_ablation_no_ag.py
    │   └── train_full_final.py        # AFMBWNetV2_Full training (saves best.pth)
    ├── All_Models_Training_Logs/      # unified runner (train 8 models + metrics)
    │   └── train.py
    ├── Comparison_And_Ablation_Runs/ # comparison/ablation/final outputs (train_code outputs)
    │   └── <experiment_name>/best.pth
    ├── Final_Model_Training_shape/    # final model output (used by app.py)
    │   └── best.pth
    ├── Figure/                        # plotting scripts (train/val curves, PR, heatmap...)
    ├── feature_images/                # feature visualization scripts
    ├── formula/                       # metric formula evaluation scripts
    └── visualize/                     # training result visualization scripts
```

## Environment setup

Recommended: Python 3.9+ (GPU is optional but recommended).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start (run the Gradio app)

The UI loads the pretrained weight by default:

- `./image_detection/Final_Model_Training_shape/best.pth`

Run:

```bash
python app.py
```

In the UI you can upload multiple AFM images (`.tif/.tiff/.png/.jpg`). The app will:

- preprocess image → build 3-channel input (`[clahe+detrend, local_rms, grad_mag]`)
- segment **white** and **black** defect regions (2-channel output)
- compute quality metrics: **EEDCF (0–100, higher is better)**, **ASDI (lower is better)**, **MMCSO (dB, higher is better)**
- draw contours and output a sorted table (ranked by EEDCF)
- save a run record under `System_Records/`

## Training

### Option A: train the final AFMBWNetV2_Full model

From repo root:

```bash
python image_detection/train_code/train_full_final.py
```

Default dataset paths inside the script (relative to `image_detection/train_code/`):

- `../dataset_400/train_images`, `../dataset_400/train_jsons`
- `../dataset_400/val_images`, `../dataset_400/val_jsons`

Outputs:

- `image_detection/Comparison_And_Ablation_Runs/08_Final_AFMBWNetV2_Full/best.pth`

### Option B: train & record all models (8 experiments)

```bash
python image_detection/All_Models_Training_Logs/train.py
```

This runs baseline / UNet++ / TransUNet / dual experts / ablations / final model and writes logs into:

- `image_detection/All_Models_Training_Logs/<experiment_name>/`

## Dataset format (annotations)

The training scripts expect **LabelMe-like JSON** per image:

- JSON filename: `same_stem_as_image.json`
- `shapes[*].label` must be one of:
  - `white` (white protrusions)
  - `black` (black pits)
- `shapes[*].points`: polygon points in original image coordinates

Masks are rasterized after resizing to `TARGET_SIZE=576`.

## Common issues

- **Path issues when running scripts**: some scripts use relative paths like `../dataset_400/...` (assuming you run the script from its folder). If you get “Training set is empty”, run from repo root exactly as shown above, or adjust the constants at the top of the script.
- **No GPU**: training will be much slower on CPU. Inference via `app.py` can still run on CPU.

