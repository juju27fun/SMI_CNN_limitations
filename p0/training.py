"""Core training and evaluation loops shared by P0 scripts."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch. Returns ``(avg_loss, accuracy)``."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    """Evaluate model. Returns ``(loss, accuracy, preds, labels, probas)``."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probas = [], [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * signals.size(0)
            probas = F.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probas.append(probas)

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    all_probas = np.concatenate(all_probas, axis=0)
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), all_probas
