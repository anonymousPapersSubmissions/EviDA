#!/bin/bash
#SBATCH --job-name=fakenews_full
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:nvidia_a100_7g.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=akinloluoluwabusayo.ojo@concordia.ca

echo "================================================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================================================================"

# ── Cache directories on scratch (avoid home quota) ──────────────────────────
export HF_HOME=/speed-scratch/ak_oj/.cache/huggingface
export TRANSFORMERS_CACHE=/speed-scratch/ak_oj/.cache/huggingface/hub
export HF_DATASETS_CACHE=/speed-scratch/ak_oj/.cache/huggingface/datasets
export TORCH_HOME=/speed-scratch/ak_oj/.cache/torch
export MPLCONFIGDIR=/speed-scratch/ak_oj/.cache/matplotlib

mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$MPLCONFIGDIR"

echo "Cache directories:"
echo "  HF_HOME:            $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  TORCH_HOME:         $TORCH_HOME"

# ── Activate virtualenv ───────────────────────────────────────────────────────
source /speed-scratch/ak_oj/my_project/IJCAI2026/venv311/bin/activate

echo ""
echo "Python:  $(which python)"
echo "Version: $(python --version)"
echo ""
nvidia-smi

# ── Create required directories ───────────────────────────────────────────────
mkdir -p logs checkpoints evaluation_results experiments

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo "STEP 1: TRAINING MODEL"
echo "================================================================================"

python scripts/train.py \
    --data_dir data \
    --batch_size 8 \
    --epochs 30 \
    --lr 2e-5 \
    --text_encoder xlm-roberta-base \
    --vision_encoder swin_tiny_patch4_window7_224 \
    --use_domain_adversarial \
    --use_evidential \
    --meta_epochs 3 \
    --k_shot 3 \
    --uncertainty_weighting adaptive \
    --uncertainty_alpha 0.5 \
    --class_balance focal_loss \
    --source_balance balanced_batch \
    --exp_name full_model

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "✓ Training completed successfully!"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo "STEP 2: EVALUATING TRAINED MODEL"
echo "================================================================================"

python scripts/evaluate.py \
    --checkpoint checkpoints/full_model/best_model.pt \
    --data_dir data \
    --output_dir evaluation_results/full_model

EVAL_EXIT_CODE=$?

if [ $EVAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    exit $EVAL_EXIT_CODE
fi

echo "✓ Evaluation completed successfully!"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo "STEP 3: RUNNING EXPERIMENTAL SUITE"
echo "================================================================================"

python scripts/run_experiments.py \
    --experiments all \
    --data_dir data \
    --output_dir experiments \
    --existing_model checkpoints/full_model/best_model.pt

EXP_EXIT_CODE=$?

if [ $EXP_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Experiments failed with exit code $EXP_EXIT_CODE"
    exit $EXP_EXIT_CODE
fi

echo "✓ Experiments completed successfully!"

# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================================"
echo "ALL TASKS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo "End Time: $(date)"
echo ""
echo "Results saved to:"
echo "  checkpoints/full_model/best_model.pt"
echo "  evaluation_results/full_model/"
echo "  experiments/"
echo ""
echo "View results:"
echo "  cat evaluation_results/full_model/evaluation_summary.txt"
echo "  cat experiments/experiment_*_results.json"
echo "================================================================================"

exit 0