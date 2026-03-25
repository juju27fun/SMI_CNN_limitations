"""Training script for particle classification using Conv1D model."""

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchsummary import summary

RAW_SIGNAL_LENGTH = 2500


class ParticleDataset(Dataset):
    """Dataset for loading particle signals from .npy files (no filtering or decimation)."""

    def __init__(self, root_dir: Path, class_names: list, transforms=None):
        self.samples = []
        self.labels = []
        self.root_dir = root_dir
        self.class_names = class_names
        self.transforms = transforms

        for class_idx, class_name in enumerate(class_names):
            class_dir = root_dir / class_name
            if not class_dir.exists():
                continue
            for npy_file in class_dir.glob("*.npy"):
                self.samples.append(npy_file)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal = np.load(self.samples[idx]).astype(np.float32)
        signal = signal[np.newaxis, :]  # Shape: (1, seq_len) for Conv1d
        signal_tensor = torch.from_numpy(signal)

        # Apply transforms sequentially
        if self.transforms is not None:
            for transform in self.transforms:
                signal_tensor = transform(signal_tensor)

        label = self.labels[idx]
        return signal_tensor, label
    
class BandpassFilter:
    """FFT bandpass filter (5–100 kHz). Best filter per filter_study_report.md."""
    def __init__(self, low_cutoff_khz: float = 5.0, high_cutoff_khz: float = 100.0, sample_rate_mhz: float = 2.0):
        self.low_cutoff = low_cutoff_khz * 1000  # Convert kHz to Hz
        self.high_cutoff = high_cutoff_khz * 1000  # Convert kHz to Hz
        self.sample_rate = sample_rate_mhz * 1_000_000  # Convert MHz to Hz

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        fft_signal = torch.fft.fft(signal)
        freqs = torch.fft.fftfreq(signal.size(-1), 1 / self.sample_rate)
        mask = (torch.abs(freqs) >= self.low_cutoff) & (torch.abs(freqs) <= self.high_cutoff)
        filtered_fft = fft_signal * mask
        return torch.fft.ifft(filtered_fft).real

class Decimate:
    """Decimation by slicing every Nth sample."""
    def __init__(self, decimate: int):
        self.decimate = decimate

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if self.decimate > 1:
            return signal[..., ::self.decimate]
        return signal


class Conv1DClassifier(nn.Module):
    """1D Convolutional classifier for particle signals."""
    
    def __init__(self, input_length: int = 250, num_classes: int = 3, dropout: float = 0.2):
        super(Conv1DClassifier, self).__init__()
        
        # Conv1D layers with increasing channels and max pooling to reduce sequence length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.drop2 = nn.Dropout(dropout)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.drop3 = nn.Dropout(dropout)
        
        # Flatten size: 256 channels * (seq_len / 8) width
        flatten_size = 256 * (input_length // 2 // 2 // 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flatten_size, 256)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Input shape: (batch, seq_len) - 1D signal format
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(self.pool1(x))
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(self.pool2(x))
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(self.pool3(x))
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.drop_fc(x)
        x = self.fc2(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train model for one epoch. Returns (avg_loss, accuracy)."""
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
    """Full model evaluation. Returns (loss, accuracy, preds, labels, probas)."""
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


def run_training(args, device, class_names, train_loader, val_loader, test_loader):
    """Execute full training pipeline with validation and testing."""
    input_length = RAW_SIGNAL_LENGTH // args.decimate

    model = Conv1DClassifier(input_length=input_length, num_classes=len(class_names)).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params} trainable parameters.")
    summary(model, input_size=(1, input_length))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    
    best_val_acc = 0.0
    train_loss_plots, val_loss_plots = [], []
    val_acc_plots = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_loss_plots.append(train_loss)
        val_loss_plots.append(val_loss)
        val_acc_plots.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    print("-" * 60)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    model.load_state_dict(torch.load(output_dir / "best_model.pth", weights_only=True))
    test_loss, test_acc, y_pred, y_true, _ = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Save the confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()

    # Figure for train loss, val loss and val acc
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_loss_plots, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(val_loss_plots, label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(val_acc_plots, label="Validation Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_plots.png")
    plt.close()
    
    torch.save(model, output_dir / "model.pth.tar")
    print(f"\nModel saved to {output_dir / 'model.pth.tar'}")


# ----------------- Main function -----------------
def main():
    parser = argparse.ArgumentParser(description="Train Conv1D classifier for particle classification")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset root directory")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save model and logs")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--decimate", type=int, default=4, help="Decimation factor for input signals")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of training data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    class_names = ["2um", "4um", "10um"]

    bandpass = BandpassFilter(low_cutoff_khz=5.0, high_cutoff_khz=100.0, sample_rate_mhz=2.0)
    decimate = Decimate(decimate=args.decimate)

    train_dataset = ParticleDataset(data_dir / "train", class_names, transforms=[bandpass, decimate])
    test_dataset = ParticleDataset(data_dir / "test", class_names, transforms=[bandpass, decimate])
    
    val_size = int(len(train_dataset) * args.val_split)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    run_training(args, device, class_names, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()

