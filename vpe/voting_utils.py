"""Voting-power computation for ensembles of classifiers.

Two entry points:
- WeightFinding: for PyTorch models. Runs every model over the given
  DataLoaders once, stores the softmax of the outputs as probability
  vectors (y_preds, shape n_models x n_samples x n_classes) together with
  the labels, and computes voting powers from them: Shapley values,
  leave-one-out, CRH truth discovery, regression, accuracy, inverse
  mean entropy.
- WeightFinding_sklearn: same interface for scikit-learn models, using
  predict_proba on (X, y) arrays.

voting() evaluates a weighted ensemble (Borda or plurality) given the
probability tensor, labels, and a weight vector.
"""

import numpy as np
import itertools
from tqdm import tqdm
from math import comb
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from scipy.special import softmax


class WeightFinding:
    def __init__(self, models, loaders, device):
        self.models = models
        self.loaders = loaders
        self.device = device
        self.y_preds, self.labels = self.compute_predictions()
        self.models = []

    @classmethod
    def from_predictions(cls, y_preds, labels, device='cpu'):
        """Build a WeightFinding object directly from an
        (n_models, n_samples, n_classes) array of probability vectors and
        the corresponding labels, without re-running any model."""
        obj = cls.__new__(cls)
        obj.models = []
        obj.loaders = None
        obj.device = device
        obj.y_preds = np.asarray(y_preds)
        obj.labels = np.asarray(labels)
        return obj

    def compute_predictions(self):
        y_preds = []

        for model in self.models:
            model.to(self.device)
            model.eval()
            labels = []
            ys = []
            for loader in self.loaders:
                for data in tqdm(loader):
                    X, y = data
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_pred = model(X)
                    ys.extend(y_pred.detach().cpu().numpy())
                    labels.extend(y.detach().cpu().numpy())
            y_preds.append(np.array(ys))
        
        y_preds = np.array(y_preds)
        labels = np.array(labels)

        return softmax(y_preds, axis = 2), labels

    def shapley_pytorch(self, method='borda'):
        print("Calculating Shapley values for", len(self.y_preds), "models")
        shapley_values = np.zeros(len(self.y_preds))
        indices = list(range(len(self.y_preds)))

        y_pred_acc = {}
        for r in tqdm(range(len(self.y_preds) + 1)):
            for subset in itertools.combinations(indices, r):
                coalition = subset
                if method == 'borda':
                    y_pred = np.argmax(np.mean(self.y_preds[list(coalition)], axis=0), axis=1)
                elif method == 'plurality':
                    y_pred = mode(np.argmax(self.y_preds[list(coalition)], axis=2), axis=0)[0]
                if r == 0:
                    y_pred_acc[coalition] = 0.1
                else:
                    y_pred_acc[coalition] = accuracy_score(self.labels, y_pred)
        
        for i in tqdm(range(len(self.y_preds))):
            rest = indices.copy()
            rest.remove(i)
            for r in range(len(rest) + 1):
                c = (len(self.y_preds) * comb(len(self.y_preds) - 1, len(self.y_preds) - r - 1))
                for subset in itertools.combinations(rest, r):
                    coalition = subset
                    coalition_i = tuple(sorted(list(coalition) + [i]))
                    if r == 0:
                        shapley_values[i] += (y_pred_acc[coalition_i] - 0.1) / c
                    else:
                        shapley_values[i] += (y_pred_acc[coalition_i] - y_pred_acc[coalition]) / c
        return shapley_values

    def loo_pytorch(self, method='borda'):
        print("Calculating LOO values for", len(self.y_preds), "models")
        loo_values = np.zeros(len(self.y_preds))
        indices = list(range(len(self.y_preds)))

        if method == 'borda':
            total_accuracy = accuracy_score(self.labels, np.argmax(np.mean(self.y_preds, axis=0), axis=1))
        elif method == 'plurality':
            total_accuracy = accuracy_score(self.labels, mode(np.argmax(self.y_preds, axis=2), axis=0)[0])
        
        for i in indices:
            coalition = [j for j in indices if j != i]
            if method == 'borda':
                y_pred = np.argmax(np.mean(self.y_preds[list(coalition)], axis=0), axis=1)
            elif method == 'plurality':
                y_pred = mode(np.argmax(self.y_preds[list(coalition)], axis=2), axis=0)[0]
            
            y_pred_acc = accuracy_score(self.labels, y_pred)
            loo_values[i] = total_accuracy - y_pred_acc
        return loo_values

    def crh_pytorch(self, iterations=1):
        trust = np.ones(len(self.y_preds)) / len(self.y_preds)
        for i in range(iterations):
            y_preds_new = []
            for j in range(len(self.y_preds)):
                y_preds_new.append(self.y_preds[j] * trust[j])
            y_preds_new = np.array(y_preds_new)
            y_preds_new = np.argmax(y_preds_new, axis=2).T
            nq, nv = y_preds_new.shape
            x = mode(y_preds_new, axis=1)[0]
            d = 1 - 1 * (y_preds_new == np.tile(x.reshape(nq, 1), (1, nv)))
            c = np.nansum(d)
            trust = -np.log(np.nansum(d, 0) / c)
            trust = trust / np.sum(trust)
        return trust

    def regression_pytorch(self, num_iterations=1000, lr = 0.001):
        print("Calculating Regression weights for", len(self.y_preds), "models")

        w = np.ones(len(self.y_preds))
        w = w / np.sum(w)

        labels_vec = np.eye(self.y_preds.shape[2])[self.labels.astype(int)]
        inputs = torch.tensor(self.y_preds).to(self.device)
        targets = torch.tensor(labels_vec).to(self.device)

        w = nn.Parameter(torch.tensor(w).to(self.device))
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam([w], lr=lr, weight_decay=0.0001)

        for i in range(num_iterations):
            optimizer.zero_grad()
            outputs = torch.sum(w[:, None, None] * inputs, 0)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        return (w.cpu()).detach().numpy()
    
    def accuracy_weights(self):
        weights = np.ones(len(self.y_preds))
        for w in range(len(weights)):
            weights[w] = accuracy_score(self.labels, np.argmax(self.y_preds[w], axis=1))
        return weights
    
    def entropy_weights(self):
        """Inverse mean-entropy weights. self.y_preds already holds
        probability vectors (softmax is applied in compute_predictions),
        so the entropy is computed on them directly."""
        weights = np.ones(len(self.y_preds))
        for w in range(len(weights)):
            probs = self.y_preds[w]
            weights[w] = -(len(self.y_preds[w]))/np.sum(probs * np.log(probs + 1e-10))
        return weights

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

class WeightFinding_sklearn:
    def __init__(self, models, dataset, device, apply_softmax=False):
        """predict_proba already returns probability vectors, which are used
        directly by default. apply_softmax=True applies an extra softmax to
        them (flattening the distributions); this reproduces the behaviour
        of the original-submission experiments, where it was applied
        inadvertently."""
        self.models = models
        self.dataset = dataset
        self.device = device
        self.apply_softmax = apply_softmax
        self.y_preds, self.labels = self.compute_predictions()
        self.models = []

    def compute_predictions(self):
        y_preds = []
        labels = self.dataset[1]
        for model in tqdm(self.models):
            y_preds.append(model.predict_proba(self.dataset[0]))
        y_preds = np.array(y_preds)

        if self.apply_softmax:
            y_preds = softmax(y_preds, axis = 2)
        return y_preds, labels

    def shapley(self, method='borda'):
        print("Calculating Shapley values for", len(self.y_preds), "models")
        shapley_values = np.zeros(len(self.y_preds))
        indices = list(range(len(self.y_preds)))

        y_pred_acc = {}
        for r in tqdm(range(len(self.y_preds) + 1)):
            for subset in itertools.combinations(indices, r):
                coalition = subset
                if method == 'borda':
                    y_pred = np.argmax(np.mean(self.y_preds[list(coalition)], axis=0), axis=1)
                elif method == 'plurality':
                    y_pred = mode(np.argmax(self.y_preds[list(coalition)], axis=2), axis=0)[0]
                if r == 0:
                    y_pred_acc[coalition] = 0.1
                else:
                    y_pred_acc[coalition] = accuracy_score(self.labels, y_pred)
        
        for i in tqdm(range(len(self.y_preds))):
            rest = indices.copy()
            rest.remove(i)
            for r in range(len(rest) + 1):
                c = (len(self.y_preds) * comb(len(self.y_preds) - 1, len(self.y_preds) - r - 1))
                for subset in itertools.combinations(rest, r):
                    coalition = subset
                    coalition_i = tuple(sorted(list(coalition) + [i]))
                    if r == 0:
                        shapley_values[i] += (y_pred_acc[coalition_i] - 0.1) / c
                    else:
                        shapley_values[i] += (y_pred_acc[coalition_i] - y_pred_acc[coalition]) / c
        return shapley_values

    def loo(self, method='borda'):
        print("Calculating LOO values for", len(self.y_preds), "models")
        loo_values = np.zeros(len(self.y_preds))
        indices = list(range(len(self.y_preds)))

        if method == 'borda':
            total_accuracy = accuracy_score(self.labels, np.argmax(np.mean(self.y_preds, axis=0), axis=1))
        elif method == 'plurality':
            total_accuracy = accuracy_score(self.labels, mode(np.argmax(self.y_preds, axis=2), axis=0)[0])
        
        for i in indices:
            coalition = [j for j in indices if j != i]
            if method == 'borda':
                y_pred = np.argmax(np.mean(self.y_preds[list(coalition)], axis=0), axis=1)
            elif method == 'plurality':
                y_pred = mode(np.argmax(self.y_preds[list(coalition)], axis=2), axis=0)[0]
            
            y_pred_acc = accuracy_score(self.labels, y_pred)
            loo_values[i] = total_accuracy - y_pred_acc
        return loo_values

    def crh(self, iterations=1):
        print("Calculating CRH values for", len(self.y_preds), "models")
        trust = np.ones(len(self.y_preds)) / len(self.y_preds)
        for i in range(iterations):
            y_preds_new = []
            for j in range(len(self.y_preds)):
                y_preds_new.append(self.y_preds[j] * trust[j])
            y_preds_new = np.array(y_preds_new)
            y_preds_new = np.argmax(y_preds_new, axis=2).T
            nq, nv = y_preds_new.shape
            x = mode(y_preds_new, axis=1)[0]
            d = 1 - 1 * (y_preds_new == np.tile(x.reshape(nq, 1), (1, nv)))
            c = np.nansum(d)
            trust = -np.log(np.nansum(d, 0) / c)
            trust = trust / np.sum(trust)
        return trust

    def regression(self, num_iterations=1000, lr = 0.01):
        print("Calculating Regression weights for", len(self.y_preds), "models")

        w = np.ones(len(self.y_preds))
        w = w / np.sum(w)

        labels_vec = np.eye(self.y_preds.shape[2])[self.labels.astype(int)]
        inputs = torch.tensor(self.y_preds).to(self.device)
        targets = torch.tensor(labels_vec).to(self.device)

        w = nn.Parameter(torch.tensor(w).to(self.device))
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD([w], lr=lr, weight_decay=0.0001)

        for i in tqdm(range(num_iterations)):
            optimizer.zero_grad()
            outputs = torch.sum(w[:, None, None] * inputs, 0)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        return (w.cpu()).detach().numpy()
    
    def accuracy_weights(self):
        print("Calculating Accuracy weights for", len(self.y_preds), "models")
        weights = np.ones(len(self.y_preds))
        for w in range(len(weights)):
            weights[w] = accuracy_score(self.labels, np.argmax(self.y_preds[w], axis=1))
        return weights
    
    def entropy_weights(self):
        """Inverse mean-entropy weights; see WeightFinding.entropy_weights."""
        print("Calculating Entropy weights for", len(self.y_preds), "models")
        weights = np.ones(len(self.y_preds))
        for w in range(len(weights)):
            probs = self.y_preds[w]
            weights[w] = -(len(self.y_preds[w]))/np.sum(probs * np.log(probs + 1e-10))
        return weights

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)

def voting(y_preds, labels, weights = None, method='borda'):
    m, n, k = y_preds.shape  # Adjust to handle 3D shape

    if weights is None:
        weights = np.ones(m) / m
    if method == 'borda':
        # Weighted average of predictions
        weighted_preds = np.tensordot(weights, y_preds, axes=(0, 0))  # shape: (n, k)
        y_pred_labels = np.argmax(weighted_preds, axis=1)  # shape: (n,)
    elif method == 'plurality':
        # Majority vote of argmax predictions using weights
        y_pred_labels = np.zeros(n, dtype=int)
        for i in range(n):
            count_array = np.zeros(k)
            for j in range(m):
                predicted_class = np.argmax(y_preds[j, i])
                count_array[predicted_class] += weights[j]
            y_pred_labels[i] = np.argmax(count_array)
    else:
        raise ValueError("Unsupported voting method. Choose 'borda' or 'plurality'.")

    # Calculate accuracy
    accuracy = 100*np.mean(y_pred_labels == labels)
    return accuracy