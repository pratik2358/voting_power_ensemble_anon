from collections import defaultdict
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import random
import copy

def simple_partition(dataset, divs):
    # Divides the dataset into divs non-overlapping subsets
    num_samples = len(dataset)
    num_subsamples = num_samples // divs
    subsets = []
    all_indices = list(range(num_samples))
    for i in range(divs):
        indices = random.sample(all_indices, num_subsamples)
        all_indices = [idx for idx in all_indices if idx not in indices]
        subset = Subset(dataset, indices)
        subsets.append(subset)

    batch_size = 512
    dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
    return dataloaders

def partition_imbalance(dataset, divs):
    class_to_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(dataset):
        class_to_indices[class_idx].append(idx)
    subsets = []
    num_classes = len(class_to_indices)
    for i in range(divs):
        selected_classes = range(num_classes - i)
        selected_indices = []
        for class_idx in selected_classes:
            selected_indices.extend(class_to_indices[class_idx])
        subsets.append(Subset(dataset, selected_indices))

    batch_size = 512
    dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
    return dataloaders

class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.data = [dataset[i][0] for i in indices]
        self.targets = [dataset.targets[i] for i in indices]
        self.indices = range(len(indices))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        return image, label

def flip_labels(dataset, flip_percentage):
    num_samples = len(dataset.targets)
    num_flips = int(num_samples * flip_percentage)
    indices = np.random.choice(num_samples, num_flips, replace=False)
    for i in indices:
        dataset.targets[i] = random.choice([j for j in range(10) if j != dataset.targets[i]])
    return dataset

def flip_labels_imbalance(dataset, flip_percentages, divs):
    num_samples = len(dataset)
    num_subsamples = num_samples // divs
    subsets = []
    for i in range(divs):
        indices = random.sample(range(num_samples), num_subsamples)
        subset = SubsetDataset(dataset, indices)
        subset = flip_labels(subset, flip_percentages[i])
        subsets.append(subset)

    batch_size = 512
    dataloaders = [DataLoader(subset, batch_size=batch_size, shuffle=True) for subset in subsets]
    return dataloaders