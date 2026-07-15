#!/bin/bash
#SBATCH --job-name=cinic-post
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# If your cluster needs an account or modules, add e.g.:
# #SBATCH --account=<account>
# module load cuda

# After the training array job (cinic_train.sbatch):
# 1. materializes the prediction tensors for all runs/regimes/splits
#    (GPU inference over the CINIC validation and test splits);
# 2. computes the voting powers with the corrected code and evaluates
#    the 20-repetition main experiments (CPU);
# 3. runs the differential-privacy experiments (Section 5) on run 1
#    (CPU post-processing of the prediction tensors).
#
# Submit FROM THE REPOSITORY ROOT:
#   sbatch slurm/cinic_postprocess.sbatch

set -eu
# Environment setup: batch jobs do not inherit your login
# environment; put module loads / conda or venv activation in
# slurm/env.sh (see slurm/env.sh.example)
[ -f "${SLURM_SUBMIT_DIR}/slurm/env.sh" ] && . "${SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "${SLURM_SUBMIT_DIR}/notebooks/cinic"

python3 ../../experiments/cinic_predictions.py --data cinic_10_data
python3 ../../experiments/cinic_eval.py
python3 ../../experiments/cinic_dp.py \
  --wf-val weight_finders_run1/wf_fl_val.pkl \
  --wf-test weight_finders_run1/wf_fl_test.pkl

# Results to bring back:
#   results/cinic_main/*.json      (main experiments, corrected code)
#   results/dp/cinic/*.json        (DP experiments)
#   notebooks/cinic/weight_finders_run*/  (prediction pickles, optional)
