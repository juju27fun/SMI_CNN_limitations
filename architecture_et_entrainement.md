# Model Architecture and Training Loop

## 1. Conv1DClassifier Architecture

> Input: 1D particle signal (1 channel, 250 samples after 4x decimation)

```mermaid
graph TD
    INPUT["Input<br/>(batch, 1, 250)"]

    subgraph BLOC1["Conv Block 1"]
        CONV1["Conv1d<br/>in=1, out=64, kernel=5, pad=2<br/>Params: 384"]
        BN1["BatchNorm1d(64)<br/>Params: 128"]
        RELU1["ReLU"]
        POOL1["MaxPool1d(2)<br/>(batch, 64, 125)"]
        DROP1["Dropout(0.2)"]
    end

    subgraph BLOC2["Conv Block 2"]
        CONV2["Conv1d<br/>in=64, out=128, kernel=5, pad=2<br/>Params: 41 088"]
        BN2["BatchNorm1d(128)<br/>Params: 256"]
        RELU2["ReLU"]
        POOL2["MaxPool1d(2)<br/>(batch, 128, 62)"]
        DROP2["Dropout(0.2)"]
    end

    subgraph BLOC3["Conv Block 3"]
        CONV3["Conv1d<br/>in=128, out=256, kernel=5, pad=2<br/>Params: 164 096"]
        BN3["BatchNorm1d(256)<br/>Params: 512"]
        RELU3["ReLU"]
        POOL3["MaxPool1d(2)<br/>(batch, 256, 31)"]
        DROP3["Dropout(0.2)"]
    end

    subgraph MLP["Classification Head (MLP)"]
        FLAT["Flatten<br/>(batch, 7936)"]
        FC1["Linear(7936, 256)<br/>Params: 2 031 872"]
        RELU_FC["ReLU"]
        DROP_FC["Dropout(0.5)"]
        FC2["Linear(256, 3)<br/>Params: 771"]
    end

    OUTPUT["Output logits<br/>(batch, 3)<br/>Classes: 2um, 4um, 10um"]

    INPUT --> CONV1 --> BN1 --> RELU1 --> POOL1 --> DROP1
    DROP1 --> CONV2 --> BN2 --> RELU2 --> POOL2 --> DROP2
    DROP2 --> CONV3 --> BN3 --> RELU3 --> POOL3 --> DROP3
    DROP3 --> FLAT --> FC1 --> RELU_FC --> DROP_FC --> FC2
    FC2 --> OUTPUT

    style INPUT fill:#e1f5fe
    style OUTPUT fill:#e8f5e9
    style BLOC1 fill:#fff3e0
    style BLOC2 fill:#fff3e0
    style BLOC3 fill:#fff3e0
    style MLP fill:#f3e5f5
```

**Total trainable parameters: ~2 239 107**

| Layer | Parameters |
|-------|-----------|
| Conv1 + BN1 | 512 |
| Conv2 + BN2 | 41 344 |
| Conv3 + BN3 | 164 608 |
| FC1 | 2 031 872 |
| FC2 | 771 |
| **Total** | **~2 239 107** |

> 90% of parameters are in the FC1 layer (7936 to 256 projection).

---

## 2. Training Loop — One Iteration (One Batch)

```mermaid
graph TD
    START["Start of iteration<br/>Load a batch of signals and labels"]
    TO_DEVICE["Transfer to device<br/>signals.to device - labels.to device"]
    ZERO_GRAD["optimizer.zero_grad<br/>Reset gradients to zero"]

    subgraph FORWARD["Forward pass"]
        FWD1["Pass signals through the model<br/>outputs = model signals"]
        FWD2["Conv1 - BN1 - ReLU - Pool - Dropout<br/>Conv2 - BN2 - ReLU - Pool - Dropout<br/>Conv3 - BN3 - ReLU - Pool - Dropout<br/>Flatten - FC1 - ReLU - Dropout - FC2"]
        FWD3["Output: logits of shape batch x 3"]
    end

    subgraph LOSS_CALC["Loss computation"]
        LOSS["loss = CrossEntropyLoss outputs labels<br/>1. Softmax on logits<br/>2. Negative log-likelihood<br/>3. Average over batch"]
    end

    subgraph BACKWARD["Backpropagation"]
        BACK1["loss.backward<br/>Compute gradients dL/dw<br/>for each parameter w"]
        BACK2["Traverse the computation graph<br/>in reverse order:<br/>FC2 - FC1 - Conv3 - Conv2 - Conv1"]
        BACK3["Each parameter accumulates<br/>its gradient in .grad"]
    end

    subgraph UPDATE["Weight update"]
        OPT["optimizer.step<br/>Adam: w = w - lr * m / sqrt v + eps<br/>with weight_decay=0.0001"]
    end

    ACCUM["Accumulate loss for reporting<br/>total_loss += loss.item * batch_size"]
    NEXT["Next batch or end of epoch"]

    START --> TO_DEVICE --> ZERO_GRAD
    ZERO_GRAD --> FWD1 --> FWD2 --> FWD3
    FWD3 --> LOSS
    LOSS --> BACK1 --> BACK2 --> BACK3
    BACK3 --> OPT
    OPT --> ACCUM --> NEXT

    style START fill:#e1f5fe
    style FORWARD fill:#e8f5e9
    style LOSS_CALC fill:#fff9c4
    style BACKWARD fill:#ffcdd2
    style UPDATE fill:#f3e5f5
    style NEXT fill:#e1f5fe
```

### Complete Epoch Cycle

```mermaid
graph LR
    subgraph EPOCH["For each epoch from 1 to 150"]
        TRAIN["model.train<br/>Loop over all batches<br/>returns average train_loss"]
        EVAL_STEP["model.eval + torch.no_grad<br/>Validation on val_loader<br/>returns val_loss and val_accuracy"]
        SCHED["LR scheduler step<br/>CosineAnnealingLR or<br/>ReduceLROnPlateau"]
        CHECK["val_acc better than best_val_acc?<br/>Yes: save best_model.pth"]
        LOG["Display every 10 epochs"]
    end

    TRAIN --> EVAL_STEP --> SCHED --> CHECK --> LOG

    FINAL["Load best_model.pth<br/>Test on test_loader<br/>Classification report + Confusion matrix"]

    LOG --> FINAL

    style TRAIN fill:#e8f5e9
    style EVAL_STEP fill:#fff9c4
    style SCHED fill:#e1f5fe
    style CHECK fill:#f3e5f5
    style FINAL fill:#ffcdd2
```
