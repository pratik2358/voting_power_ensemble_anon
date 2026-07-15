#!/bin/bash
#SBATCH --job-name=mnist-timing
#SBATCH --output=slurm-%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# CPU-only job: pick your cluster's CPU partition if the default does
# not fit, e.g.:
# #SBATCH --partition=cpu
# #SBATCH --account=<account>

# Times the voting-power computations for the MNIST ensembles
# (experiments/mnist_timing.py, Figure "time comparison" of the paper).
# The measurements are single-threaded; the extra CPUs only serve the
# initial teacher training.
#
# MNIST is fetched from OpenML into data/ on first use: if the compute
# nodes have no internet access, run once on a login node first:
#   python3 -c "import sys; sys.path.insert(0, 'experiments'); \
#               import mnist_main; mnist_main.load_mnist()"
#
# Submit FROM THE REPOSITORY ROOT:
#   sbatch slurm/mnist_timing.sbatch

set -e
# Environment setup: batch jobs do not inherit your login
# environment; put module loads / conda or venv activation in
# slurm/env.sh (see slurm/env.sh.example). Sourced without `set -u`:
# system init scripts (/etc/bashrc, module system) read unset variables.
[ -f "${SLURM_SUBMIT_DIR}/slurm/env.sh" ] && . "${SLURM_SUBMIT_DIR}/slurm/env.sh"
set -u
cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=1

python3 experiments/mnist_timing.py

# Result to bring back: results/timing/mnist_times.json
