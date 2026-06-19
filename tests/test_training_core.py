"""Smoke tests for the package training/evaluation loops."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from p0.training import evaluate, train_one_epoch


def test_train_and_evaluate_tiny_classifier():
    torch.manual_seed(0)
    x = torch.randn(8, 1, 16)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 2))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    train_loss, train_acc = train_one_epoch(model, loader, criterion, optimizer, device)
    val_loss, val_acc, preds, labels, probas = evaluate(model, loader, criterion, device)

    assert train_loss > 0
    assert 0.0 <= train_acc <= 1.0
    assert val_loss > 0
    assert 0.0 <= val_acc <= 1.0
    assert preds.shape == labels.shape == (8,)
    assert probas.shape == (8, 2)
