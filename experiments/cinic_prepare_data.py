"""Download and lay out the CINIC-10 data for the CINIC experiments.

Downloads the official CINIC-10 archive (~700 MB, from the Edinburgh
DataShare) unless --tarball points to an existing copy, extracts it,
and reorganizes each split into a CIFAR half and an ImageNet half
(images prefixed 'c' vs. 'n'), producing the layout expected by the
notebooks and scripts:
    cinic_10_data/{train2,valid2,test2}/{cifar,imagenet}/<class>/

Run from the repository root (or pass --root):
    python3 experiments/cinic_prepare_data.py [--root .] [--tarball path]
"""

import argparse
import os
import shutil
import tarfile
import urllib.request

URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def main(root, tarball):
    raw = os.path.join(root, "cinic_10_raw")
    out = os.path.join(root, "cinic_10_data")
    if tarball is None:
        tarball = os.path.join(root, "CINIC-10.tar.gz")
        if not os.path.exists(tarball):
            print("downloading", URL)
            urllib.request.urlretrieve(URL, tarball)
    if not os.path.exists(raw):
        print("extracting", tarball)
        os.makedirs(raw)
        with tarfile.open(tarball) as tf:
            tf.extractall(raw)
    for split in ["train", "valid", "test"]:
        dest = os.path.join(out, split + "2")
        for half in ["cifar", "imagenet"]:
            for cls in CLASSES:
                os.makedirs(os.path.join(dest, half, cls), exist_ok=True)
        for cls in CLASSES:
            src_dir = os.path.join(raw, split, cls)
            n_c = n_n = 0
            for img in os.listdir(src_dir):
                half = "cifar" if img.startswith("c") else "imagenet"
                shutil.move(os.path.join(src_dir, img),
                            os.path.join(dest, half, cls, img))
                n_c += half == "cifar"
                n_n += half == "imagenet"
            print(f"{split}/{cls}: {n_c} cifar + {n_n} imagenet")
    print("done; data in", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tarball", default=None)
    args = ap.parse_args()
    main(args.root, args.tarball)
