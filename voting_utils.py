import numpy as np
import itertools
from tqdm import tqdm
from math import comb
from scipy.stats import mode
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from scipy.special import softmax, entr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def shapley_pytorch(models, loaders, method = 'borda'):
    print("Calculating Shapley values for", len(models), "models")
    shapley_values = np.zeros(len(models))
    y_preds = []
    indices = list(range(len(models)))
    for model in models:
        model.to(device)
        model.eval()
        ys = []
        labels = []
        for loader in loaders:
            for data in tqdm(loader):
                X,y = data
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                ys.extend(y_pred.detach().cpu().numpy())
                labels.extend(y.detach().cpu().numpy())
        ys = np.array(ys)
        y_preds.append(ys)
        labels = np.array(labels)
    y_preds = np.array(y_preds)
    y_pred_acc = {}
    for r in tqdm(range(len(models)+1)):
        for subset in itertools.combinations(indices, r):
            coalition = subset
            if method == 'borda':
                y_pred = np.argmax(np.mean(y_preds[list(coalition)], axis = 0), axis = 1)
            elif method == 'plurality':
                y_pred = mode(np.argmax(y_preds[list(coalition)], axis = 2), axis=0)[0]
            if r == 0:
                y_pred_acc[coalition] = 0.1
            else:
                y_pred_acc[coalition] = accuracy_score(labels, y_pred)
    for i in tqdm(range(len(models))):
        rest = indices.copy()
        rest.remove(i)
        for r in range(len(rest)+1):
            c = (len(models)*comb(len(models)-1, len(models)-r-1))
            for subset in itertools.combinations(rest, r):
                coalition = subset
                coalition_i = tuple(sorted(list(coalition) + [i]))
                if r == 0:
                    shapley_values[i] += (y_pred_acc[coalition_i] - 0.1)/c
                else:
                    shapley_values[i] += (y_pred_acc[coalition_i] - y_pred_acc[coalition])/c
    return shapley_values

def loo_pytorch(models, loaders, method = 'borda'):
    print("Calculating LOO values for", len(models), "models")
    loo_values = np.zeros(len(models))
    y_preds = []
    indices = list(range(len(models)))
    for model in models:
        model.to(device)
        model.eval()
        ys = []
        labels = []
        for loader in loaders:
            for data in tqdm(loader):
                X,y = data
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                ys.extend(y_pred.detach().cpu().numpy())
                labels.extend(y.detach().cpu().numpy())
        ys = np.array(ys)
        y_preds.append(ys)
        labels = np.array(labels)
    y_preds = np.array(y_preds)
    y_pred_acc = {}
    if method == 'borda':
        total_accuracy = accuracy_score(labels, np.argmax(np.mean(y_preds, axis = 0), axis = 1))
    elif method == 'plurality':
        total_accuracy = accuracy_score(labels, mode(np.argmax(y_preds, axis = 2), axis=0)[0])
    for i in indices:
        coalition = [j for j in indices if j != i]
        if method == 'borda':
            y_pred = np.argmax(np.mean(y_preds[list(coalition)], axis = 0), axis = 1)
        elif method == 'plurality':
            y_pred = mode(np.argmax(y_preds[list(coalition)], axis = 2), axis=0)[0]
            
        y_pred_acc[tuple(coalition)] = accuracy_score(labels, y_pred)
        loo_values[i] = total_accuracy - y_pred_acc[tuple(coalition)]
    return loo_values

def crh_pytorch(models, loaders, iterations = 1):
    y_preds = []
    for model in models:
        model.to(device)
        model.eval()
        ys = []
        for loader in loaders:
            for data in tqdm(loader):
                X,y = data
                X = X.to(device)
                y_pred = model(X)
                ys.extend(y_pred.detach().cpu().numpy())
        ys = np.array(ys)
        y_preds.append(ys)
    y_preds = np.array(y_preds)
    trust = np.ones(len(models))/len(models)
    for i in range(iterations):
        y_preds_new = []
        for j in range(len(models)):
            y_preds_new.append(y_preds[j]*trust[j])
        y_preds_new = np.array(y_preds_new)
        y_preds_new = np.argmax(y_preds_new, axis = 2).T
        nq, nv = y_preds_new.shape
        x = mode(y_preds_new, axis=1)[0]
        d = 1-1*(y_preds_new == np.tile(x.reshape(nq,1), (1,nv)))
        c = np.nansum(d)
        trust = -np.log(np.nansum(d,0)/c)
        trust = trust/np.sum(trust)
    return trust

def regression_pytorch(models, loaders, num_iterations = 100, method = 'borda'):

    def weighted_average(weights, probabilities):
        # Calculate the weighted average of the probabilities for each class
        weighted_probs = np.zeros_like(probabilities[0])
        for i in range(len(weights)):
            weighted_probs += weights[i] * probabilities[i]
        return weighted_probs
    
    # def loss_function(weights, probabilities, labels):
    #     # Calculate the L2 distance between the one-hot encoded labels and the weighted averages
    #     weighted_probs = weighted_average(weights, probabilities)
    #     labels_one_hot = np.zeros_like(weighted_probs)
    #     labels_one_hot[np.arange(len(labels)), labels] = 1  # Convert labels to one-hot encoding
    #     loss = np.sum((labels_one_hot - weighted_probs) ** 2)
    #     return loss
    
    print("Calculating Regression weights for", len(models), "models")
    y_preds = []
    for model in models:
        model.to(device)
        model.eval()
        ys = []
        labels = []
        for loader in loaders:
            for data in tqdm(loader):
                X,y = data
                X = X.to(device)
                y = y.to(device)
                y_pred = model(X)
                ys.extend(y_pred.detach().cpu().numpy())
                labels.extend(y.detach().cpu().numpy())
        ys = np.array(ys)
        y_preds.append(ys)
        labels = np.array(labels)
    y_preds = np.array(y_preds)

    w = np.ones(len(y_preds))
    w = w/np.sum(w)

    labels_vec = np.eye(10)[labels]
    inputs = torch.tensor(y_preds).cuda()
    targets = torch.tensor(labels_vec).cuda()

    w = nn.Parameter(torch.tensor(w).cuda())
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam([w], lr=0.1)

    num_iterations = 100
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        outputs = torch.sum(w[:,None,None]*inputs,0)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    return (w.cpu()).detach().numpy()

def validate_ensemble(models, valid_loaders, weights):
    if len(models) == 0:
        return 0.1
    else:
        correct = 0
        total = 0
        with torch.no_grad():
            for loader in valid_loaders:
                for data in tqdm(loader):
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = torch.zeros(images.size(0), 10).to(device)
                    for i in range(len(models)):
                        outputs += torch.tensor(weights[i], dtype = torch.float).to(device)*models[i](images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
    return correct / total

def find_acc(models, valid_loaders):
    acc = []
    for i in range(len(models)):
        acc.append(validate_ensemble([models[i]], valid_loaders, [1]))
    return acc

def find_mean_entropy(model, valid_loaders):
    entropy = 0
    total = 0
    with torch.no_grad():
        for loader in valid_loaders:
            for data in tqdm(loader):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)
                outputs = torch.zeros(images.size(0), 10).to(device)
                outputs += model(images)
                outputs = F.softmax(outputs, dim=1)
                entropy += -torch.sum(outputs * torch.log(outputs + 1e-10), dim=1).sum().item()
                total += labels.size(0)
    return total/entropy

def find_entropy(models, valid_loaders):
    entropy = []
    for i in range(len(models)):
        entropy.append(find_mean_entropy(models[i], valid_loaders))
    return entropy


class WeightFinding:
    def __init__(self, models, loaders, device):
        self.models = models
        self.loaders = loaders
        self.device = device
        self.y_preds, self.labels = self.compute_predictions()
        self.models = []

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
        weights = np.ones(len(self.y_preds))
        for w in range(len(weights)):
            probs = softmax(self.y_preds[w], axis=1)
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
    def __init__(self, models, dataset, device):
        self.models = models
        self.dataset = dataset
        self.device = device
        self.y_preds, self.labels = self.compute_predictions()
        self.models = []

    def compute_predictions(self):
        y_preds = []
        labels = self.dataset[1]
        for model in tqdm(self.models):
            y_preds.append(model.predict_proba(self.dataset[0]))
        y_preds = np.array(y_preds)
        
        return softmax(y_preds, axis = 2), labels

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
        print("Calculating Entropy weights for", len(self.y_preds), "models")
        weights = np.ones(len(self.y_preds))
        for w in range(len(weights)):
            probs = softmax(self.y_preds[w], axis=1)
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