"""Materialize the CINIC-10 prediction tensors from trained checkpoints.

For each training run (cinic_various_models, _2, _3, produced by the
notebooks in notebooks/cinic/) and each noise regime (fl = label
flipping, di = class imbalance), loads all model checkpoints
(CIFAR10_Net and VGG16 with a 10-class head, as in the notebooks),
runs them over the combined cifar+imagenet validation and test splits,
and pickles the resulting WeightFinding objects:
    weight_finders_run{1,2,3}/wf_{fl,di}_{val,test}.pkl
These pickles are all that the evaluation (cinic_eval.py) and the DP
experiments (cinic_dp.py) need; no further GPU inference is required.

Run from the directory containing the checkpoint folders and the data
(notebooks/cinic/ when the notebooks were executed in place):
    python3 ../../experiments/cinic_predictions.py [--data cinic_10_data] [--runs 1 2 3]
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vpe.models_used import CIFAR10_Net
from vpe.voting_utils import WeightFinding

CINIC_MEAN = [0.47889522, 0.47227842, 0.43047404]
CINIC_STD = [0.24205776, 0.23828046, 0.25874835]
RUN_DIR = {1: "cinic_various_models", 2: "cinic_various_models_2",
           3: "cinic_various_models_3"}


def loaders(data_root, split):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CINIC_MEAN, std=CINIC_STD)])
    out = []
    for half in ["cifar", "imagenet"]:
        ds = torchvision.datasets.ImageFolder(
            os.path.join(data_root, split, half), transform=tf)
        out.append(torch.utils.data.DataLoader(
            ds, batch_size=512, shuffle=False, num_workers=4,
            pin_memory=True))
    return out


def load_models(directory, noise_type):
    models = []
    for file in sorted(os.listdir(directory)):
        if noise_type not in file or not file.endswith(".pt"):
            continue
        if "cifar10_net" in file:
            model = CIFAR10_Net()
        else:
            model = torchvision.models.vgg16()
            model.classifier[6] = nn.Linear(
                model.classifier[6].in_features, 10)
        model.load_state_dict(torch.load(
            os.path.join(directory, file), map_location="cpu"))
        models.append(model)
        print("loaded", file)
    return models


def main(data_root, runs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    for run in runs:
        model_dir = RUN_DIR[run]
        if not os.path.isdir(model_dir):
            print(f"skipping run {run}: no directory {model_dir}")
            continue
        out_dir = f"weight_finders_run{run}"
        os.makedirs(out_dir, exist_ok=True)
        for regime in ["fl", "di"]:
            models = load_models(model_dir, regime)
            if not models:
                print(f"run {run} {regime}: no checkpoints found")
                continue
            for split, tag in [("valid2", "val"), ("test2", "test")]:
                path = os.path.join(out_dir, f"wf_{regime}_{tag}.pkl")
                if os.path.exists(path):
                    print("exists, skipping:", path)
                    continue
                wf = WeightFinding(models, loaders(data_root, split),
                                   device)
                wf.save(path)
                print("saved", path, "y_preds", wf.y_preds.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cinic_10_data")
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()
    main(args.data, args.runs)
