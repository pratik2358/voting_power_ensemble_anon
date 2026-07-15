#!/bin/bash
#SBATCH --job-name=cinic-train
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=1-3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# If your cluster needs an account or modules, add e.g.:
# #SBATCH --account=<account>
# module load cuda

# Trains the three CINIC-10 model-ensemble runs of the paper by
# executing the original notebooks verbatim (run 1: 12 models, run 2:
# 16, run 3: 8; label-flipping and class-imbalance regimes each).
#
# Submit FROM THE REPOSITORY ROOT (Slurm runs a spooled copy of this
# script, so paths are resolved from the submission directory):
#   sbatch slurm/cinic_train.sbatch
#
# Prerequisites, from the repository root:
#   pip install -r requirements.txt && pip install -e . jupyter nbconvert
#   python3 experiments/cinic_prepare_data.py --root notebooks/cinic
# (the notebooks expect cinic_10_data/ relative to their own directory)

set -eu
# Environment setup: batch jobs do not inherit your login
# environment; put module loads / conda or venv activation in
# slurm/env.sh (see slurm/env.sh.example)
[ -f "${SLURM_SUBMIT_DIR}/slurm/env.sh" ] && . "${SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "${SLURM_SUBMIT_DIR}/notebooks/cinic"

# the notebooks save checkpoints into these directories but do not
# create them
mkdir -p cinic_various_models cinic_various_models_2 cinic_various_models_3

case "${SLURM_ARRAY_TASK_ID}" in
  1) NB=train_cinic_noisy_models.ipynb ;;
  2) NB=train_cinic_noisy_models_2.ipynb ;;
  3) NB=train_cinic_noisy_models_3.ipynb ;;
esac

jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 "$NB"
