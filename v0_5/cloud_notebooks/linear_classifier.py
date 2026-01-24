import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
from dataclasses import dataclass


@dataclass
class LinearClassifierConfig:
    input_dim: int = 8
    num_classes: int = 10


class LinearClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.input_dim, config.num_classes)

    def forward(self, x):
        return self.linear(x)


def train_linear_classifier(X_train, y_train, X_val, y_val, num_classes, device, epochs=100, lr=1e-3):
    '''Train a linear classifier on frozen embeddings.'''
    input_dim = X_train.shape[1]
    config = LinearClassifierConfig(input_dim=input_dim, num_classes=num_classes)
    classifier = LinearClassifier(config).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(X_val_t)
                val_preds = val_logits.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    if best_state:
        classifier.load_state_dict(best_state)

    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(X_val_t)
        val_preds = val_logits.argmax(dim=1).cpu().numpy()

    return classifier, val_preds, best_val_acc
