# EviDA: Evidential Domain Adaptation for Multimodal Fake News Detection

Cross-domain fake news detection that combines multimodal (text + image) modeling, domain adversarial training, and evidential deep learning — with a core innovation: **uncertainty-weighted domain adaptation**, where the model's own evidential uncertainty guides how strongly domain alignment is applied to each sample.

## Key Idea

Standard domain adversarial training treats every sample equally:

```
loss = cls_loss + λ · domain_loss
```

EviDA instead weights the domain loss by per-sample evidential uncertainty:

```
loss = cls_loss + λ · (1 + α · uncertainty) · domain_loss
```

where `α` can be fixed (**static**), thresholded (**threshold**), or learned during training (**adaptive** — our main contribution). Samples the model is uncertain about (typically cross-domain samples) receive stronger domain alignment pressure.

## Architecture

| Component | Implementation |
|---|---|
| Text encoder | XLM-RoBERTa (multilingual, HuggingFace) |
| Vision encoder | Swin Transformer (timm) |
| Fusion | Bidirectional multi-head cross-modal attention |
| Classifier | Evidential head (Dirichlet parameters → prediction + uncertainty) |
| Domain adaptation | Gradient reversal + domain discriminator, per-domain LayerNorm, MMD |
| Optional | MAML-style meta-learning phase, explanation generator |

The evidential classifier outputs Dirichlet parameters (Sensoy et al., NeurIPS 2018), giving calibrated per-sample uncertainty (vacuity) at no extra inference cost — no MC Dropout or ensembles required.

## Project Structure

```
├── config/                  # Experiment configuration
├── scripts/
│   ├── prepare_data.py      # Data validation & split creation
│   ├── create_french_dataset.py  # Build French dataset via translation
│   ├── train.py             # Training entry point
│   ├── evaluate.py          # Comprehensive domain-shift evaluation
│   ├── run_experiments.py   # Full experimental suite
│   ├── inference.py         # Single-sample prediction
│   └── analyze_results.py
├── src/
│   ├── data/                # Datasets, flexible image loading, balancing, samplers
│   ├── models/              # Encoders, fusion, classifiers, domain adaptation
│   ├── losses/              # Combined classification + domain + KL losses
│   ├── training/            # Trainers (standard, uncertainty-weighted, meta), evaluators
│   └── utils/               # Metrics, logging
├── train.sh                 # SLURM job script (train → evaluate → experiments)
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/hormone03/EviDA.git
cd EviDA

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Requirements: Python 3.8+, CUDA-capable GPU recommended (8 GB+ VRAM).

## Data Preparation
Data sets are publicly available. Please refer to the documentation in our paper. Place each source (domain) under `data/`, e.g.:

```
data/
├── twitter/
│   ├── images/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── weibo/
└── fakeddit/
```

CSVs need a text column (`text` / `post_text` / `content`), an image column (`image_path` / `image_id` / `img_path`), and a `label` column (`real`/`fake` or 0/1). Image folders may be flat, split-nested, or folder-nested — the loader auto-detects the structure per source.

Validate everything before training:

```bash
python scripts/prepare_data.py --data_dir data
```

Domain adaptation works best with 3+ distinct sources.

## Training

**Baseline** (no domain adaptation, no evidential head):

```bash
python scripts/train.py --data_dir data --epochs 30 --lr 2e-5 \
    --no_domain_adversarial --no_evidential --exp_name baseline
```

**Full model with uncertainty-weighted domain adaptation** (recommended):

```bash
python scripts/train.py --data_dir data --batch_size 16 --epochs 30 --lr 2e-5 \
    --text_encoder xlm-roberta-base \
    --vision_encoder swin_tiny_patch4_window7_224 \
    --use_domain_adversarial --use_evidential \
    --uncertainty_weighting adaptive --uncertainty_alpha 0.5 \
    --class_balance focal_loss --source_balance balanced_batch \
    --exp_name full_model
```

Only the best checkpoint (by validation F1) is kept, at `checkpoints/<exp_name>/best_model.pt`.

Monitor with TensorBoard:

```bash
tensorboard --logdir logs
```

### Uncertainty weighting strategies

| `--uncertainty_weighting` | Behavior |
|---|---|
| `none` | Standard domain adversarial training (baseline) |
| `static` | Fixed weight `1 + α·u` |
| `adaptive` | `α` is a learned parameter (main contribution) |
| `threshold` | Extra weight only when `u` exceeds a threshold |

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/full_model/best_model.pt \
    --data_dir data \
    --output_dir evaluation_results/full_model
```

Produces overall and per-domain metrics, uncertainty analysis (correct vs. incorrect predictions, per-domain distributions), prediction/error CSVs, and calibration plots.

## Experiments

Reproduce the full experimental suite:

```bash
python scripts/run_experiments.py --experiments all --data_dir data \
    --output_dir experiments \
    --existing_model checkpoints/full_model/best_model.pt
```

Individual experiments:

1. Uncertainty ↔ domain-shift correlation
2. Uncertainty method comparison (evidential vs. MC Dropout, ensembles, temperature scaling)
3. Uncertainty-weighted training ablation (none / static / adaptive / threshold)
4. Component ablation
5. SOTA comparison

## Inference

```bash
python scripts/inference.py \
    --checkpoint checkpoints/full_model/best_model.pt \
    --text "Breaking: Major announcement expected tomorrow!" \
    --image path/to/image.jpg
```

Returns the prediction, class probabilities, and an evidential uncertainty score.

## HPC / SLURM

`train.sh` is a ready-to-adapt SLURM script that runs training → evaluation → experiments in sequence, redirects HuggingFace/Torch caches to scratch space, and emails on completion. Adjust paths, account details, and resource requests for your cluster.

## Multilingual Support

`scripts/create_french_dataset.py` builds a French version of existing sources by machine translation, enabling cross-lingual experiments on top of the multilingual XLM-RoBERTa backbone:

```bash
python scripts/create_french_dataset.py --data_dir data \
    --sources twitter fakeddit --device cuda
```