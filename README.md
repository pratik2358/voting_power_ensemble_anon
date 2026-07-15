# Voting Power for Ensemble Methods: Discovery and Differential Privacy

Code and experimental data for the paper *"Voting Power for Ensemble
Methods: Discovery and Differential Privacy"*. The repository contains
the implementation of all voting-power (weight) assignment methods
compared in the paper (accuracy, regression, Shapley values,
leave-one-out, CRH truth discovery, inverse entropy), the weighted
voting rules (plurality and Borda), the experiments on the four
benchmark datasets (MNIST, CINIC-10, DMOZ, webpage phishing), and the
differential-privacy experiments of Section 5.

## Repository structure

```
vpe/                       Python package
  voting_utils.py          voting-power computation (WeightFinding for
                           PyTorch models, WeightFinding_sklearn for
                           scikit-learn models) and weighted voting
  dp_mechanisms.py         differentially private aggregation mechanisms,
                           DP release of voting powers, budget allocation
                           across teachers, margin/entropy diagnostics
  models_used.py           CINIC-10 model architectures (CNN, logistic)
  noise_added.py           dataset partitioning and label-noise utilities

experiments/               scripts for the DP experiments (Section 5)
  mnist_dp.py              end-to-end on MNIST: trains logistic-regression
                           teachers, computes voting powers, runs the
                           mechanism comparison (D1), the weighting-scheme
                           comparison under DP (D2), the budget-allocation
                           experiment (D3), and the Proposition 1
                           empirical study (D4)
  dmoz_dp.py               same design on DMOZ (needs the Kaggle CSV)
  cinic_dp.py              same design on CINIC-10, as post-processing of
                           the pickled prediction tensors produced by the
                           CINIC notebooks
  synthetic_dp.py          synthetic mechanism comparison (sanity check)

notebooks/                 main (non-private) experiments, Sections 3-4
  cinic/                   CINIC-10: train_cinic_noisy_models*.ipynb train
                           the noisy model ensembles (3 independent runs);
                           cinic_models_weights*.ipynb compute and save the
                           voting powers; voting_*_repeat.ipynb evaluate
                           the weighted ensembles over 20 repetitions
  dmoz/dmoz.ipynb          DMOZ URL classification, end to end
  phishing/webpage_class.ipynb  phishing single-model baseline

analysis/
  significance_tests.py    pairwise Wilcoxon signed-rank tests with Holm
                           correction on the archived accuracy results;
                           produces the significance table of the paper
  plot_new.ipynb           generates the accuracy/time figures of the
                           paper from the archived results

results/
  paper/                   archived outputs of the main experiments
                           (cinic_noise_results/, dmoz_exps/, phising/)
  dp/                      outputs of the DP experiments
  significance/            outputs of analysis/significance_tests.py
```

## Setup

```
pip install -r requirements.txt
pip install -e .        # makes the vpe package importable everywhere
```

The notebooks also work without installation: they add the repository
root to `sys.path` in their first cell.

## Datasets

- **MNIST**: downloaded automatically (OpenML) by
  `experiments/mnist_dp.py` into `data/`.
- **CINIC-10**: follow
  [BayesWatch/cinic-10](https://github.com/BayesWatch/cinic-10), then
  split each of `train`, `valid`, `test` into a CIFAR half and an
  ImageNet half (images prefixed `c` vs. `n`), e.g. with the snippet
  below; the notebooks expect
  `cinic_10_data/{train2,valid2,test2}/{cifar,imagenet}/<class>/`.
- **DMOZ**: [Kaggle: URL classification dataset
  (DMOZ)](https://www.kaggle.com/datasets/shawon10/url-classification-dataset-dmoz).
- **Phishing**: [Kaggle: web page phishing detection
  dataset](https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset).

<details>
<summary>CINIC-10 reorganization snippet</summary>

```python
import os, shutil

for i in ['train', 'valid', 'test']:
    source_dir = '/path/to/cinic10/' + i
    dest_dir = '/path/to/cinic_10_data/' + i + '2'
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    for cls in classes:
        os.makedirs(os.path.join(dest_dir, 'cifar', cls), exist_ok=True)
        os.makedirs(os.path.join(dest_dir, 'imagenet', cls), exist_ok=True)
    for cls in classes:
        class_dir = os.path.join(source_dir, cls)
        for img in os.listdir(class_dir):
            half = 'cifar' if img.startswith('c') else 'imagenet'
            shutil.move(os.path.join(class_dir, img),
                        os.path.join(dest_dir, half, cls, img))
```
</details>

## Reproducing the paper's results

- **Main comparison (Sections 3-4).** MNIST:
  `python3 experiments/mnist_main.py` (a few CPU-hours). DMOZ:
  `python3 experiments/dmoz_main.py` (script version of
  `notebooks/dmoz/dmoz.ipynb`; a few CPU-hours). Phishing:
  `python3 experiments/phishing_main.py` (minutes). CINIC-10: see the
  GPU-cluster section below. `analysis/make_pgf_figures.py` builds all
  paper figures as pgfplots fragments from the results (preferring
  reruns under `results/` over the archived outputs in
  `results/paper/`).
- **Timing and entropy/CRH anomaly studies (Sections 4 and 6).**
  `python3 experiments/mnist_timing.py` times the voting-power
  computations for MNIST ensembles of 3-16 models (single-threaded;
  `slurm/mnist_timing.sbatch` runs it on a cluster);
  `python3 experiments/mnist_anomaly.py` runs the label-interchange
  study behind the entropy/CRH anomaly and correlation figures.
- **Significance tests.** `python3 analysis/significance_tests.py`
  writes per-setting p-value matrices and the LaTeX summary table to
  `results/significance/`.
- **Differential privacy (Section 5).**
  `python3 experiments/mnist_dp.py` runs the complete MNIST pipeline on
  CPU in a few minutes and writes `results/dp/mnist/dp_results.json`;
  `experiments/dmoz_dp.py --csv <dmoz.csv>` and
  `experiments/cinic_dp.py --wf-val <pkl> --wf-test <pkl>` run the same
  design on the other datasets.

## Running the CINIC-10 experiments on a GPU cluster

The CINIC-10 experiments are the only GPU-bound part of the paper
(training 72 CNN/VGG16 models over three independent runs). The
workflow, with SLURM templates under `slurm/`:

```
pip install -r requirements.txt && pip install -e . jupyter nbconvert
python3 experiments/cinic_prepare_data.py --root notebooks/cinic
sbatch slurm/cinic_train.sbatch        # array job, one GPU per run (1-3)
sbatch slurm/cinic_postprocess.sbatch  # after training completes
```

Submit from the repository root (the scripts locate the repository
through `SLURM_SUBMIT_DIR`). They request `--partition=gpu`; adjust the
partition, account, and module lines to your cluster if needed.

- `cinic_prepare_data.py` downloads the official CINIC-10 archive
  (~700 MB) and reorganizes it into the CIFAR/ImageNet halves the
  notebooks expect (`cinic_10_data/{train2,valid2,test2}/...`).
- `cinic_train.sbatch` executes the three original training notebooks
  verbatim with nbconvert (checkpoints under
  `notebooks/cinic/cinic_various_models*/`); roughly a few GPU-hours
  per run, dominated by the VGG16 models.
- `cinic_postprocess.sbatch` runs `cinic_predictions.py` (one GPU pass
  over the validation and test splits, pickling the prediction
  tensors), then `cinic_eval.py` (the 20-repetition main experiments,
  CPU) and `cinic_dp.py` (the Section 5 DP experiments, CPU).
- Files to copy back: `results/cinic_main/*.json`,
  `results/dp/cinic/*.json`, and optionally the
  `weight_finders_run*/` pickles. `analysis/make_pgf_figures.py` and
  `analysis/significance_tests.py` pick the rerun results up
  automatically once they are placed under `results/`.

Edit the `#SBATCH` partition/account/module lines to match your
cluster; the scripts have no other site-specific assumptions.

Make sure the installed PyTorch wheel matches the NVIDIA driver of the
GPU nodes (`torch.cuda.is_available()` silently returns False
otherwise, and everything falls back to CPU): CUDA 13 wheels need
driver >= 580, while CUDA 12.x wheels
(`pip install torch --index-url https://download.pytorch.org/whl/cu126`)
work on any driver >= 525. On clusters with heterogeneous driver
versions across nodes, the CUDA 12.x wheels are the safe choice.
