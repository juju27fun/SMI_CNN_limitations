# Conv1D vs Conv1DGAP

Comparison on Benchmark 2, Tier 1 standard laser dataset.

Context: input trace `1 x 625`, `3` output classes, base `M` variants with
`width_mult = 1.0`, averaged over 3 seeds.

---

## Architecture 1v1

```mermaid
flowchart LR
    X["Input trace<br/>1 x 625"]:::input
    B["Shared conv backbone<br/>k=5, filters 64 -> 128 -> 256<br/>3x Conv + BN + ReLU + MaxPool /2<br/>206,464 params"]:::backbone
    L["Latent map<br/>256 x 78"]:::latent

    F["Conv1D<br/>Flatten<br/>19,968 features"]:::flatten
    FH["Dense head<br/>19,968 -> 256 -> 3<br/>5,112,835 params"]:::heavy
    FT["Total<br/>5,319,299 params"]:::heavy

    G["Conv1DGAP<br/>Global average pooling<br/>256 features"]:::gap
    GH["Dense head<br/>256 -> 256 -> 3<br/>66,563 params"]:::small
    GT["Total<br/>273,027 params"]:::small

    X --> B --> L
    L --> F --> FH --> FT
    L --> G --> GH --> GT

    classDef input fill:#f7f7f7,stroke:#666,color:#111;
    classDef backbone fill:#dbeafe,stroke:#2563eb,color:#111;
    classDef latent fill:#eef2ff,stroke:#4f46e5,color:#111;
    classDef flatten fill:#ffedd5,stroke:#ea580c,color:#111;
    classDef heavy fill:#fed7aa,stroke:#c2410c,color:#111;
    classDef gap fill:#dcfce7,stroke:#16a34a,color:#111;
    classDef small fill:#bbf7d0,stroke:#15803d,color:#111;
```

Reading the diagram:

- Blue is shared by both models.
- Orange is the expensive `Flatten` path.
- Green is the compact `Global Average Pooling` path.

**Main difference.** `Conv1DGAP` keeps the same convolutional feature extractor
but replaces `Flatten(256 x 78)` with global average pooling. The dense head
shrinks from `5.11M` parameters to `66.6K`, reducing the full model by about
`19.5x`.

### Parameter Footprint

```mermaid
flowchart LR
    C1["Conv1D<br/>5.32M params<br/>100%"]:::heavy
    C2["Conv1DGAP<br/>273K params<br/>5.1%"]:::small
    S["Same task<br/>about 19.5x smaller"]:::note

    C1 --> S
    C2 --> S

    classDef heavy fill:#fed7aa,stroke:#c2410c,color:#111;
    classDef small fill:#bbf7d0,stroke:#15803d,color:#111;
    classDef note fill:#f7f7f7,stroke:#666,color:#111;
```

---

## Accuracy 1v1

```mermaid
flowchart LR
    A["Conv1D<br/>96.68% test accuracy"]:::heavy
    B["Conv1DGAP<br/>97.12% test accuracy"]:::small
    T["Accuracy stays close<br/>model size changes a lot"]:::note

    A --> T
    B --> T

    classDef heavy fill:#fed7aa,stroke:#c2410c,color:#111;
    classDef small fill:#bbf7d0,stroke:#15803d,color:#111;
    classDef note fill:#f7f7f7,stroke:#666,color:#111;
```

### Test Accuracy

| Model | Test accuracy | Std | Seeds |
|---|---:|---:|---:|
| Conv1D | 96.68% | 0.33% | 3 |
| Conv1DGAP | 97.12% | 1.02% | 3 |

### Validation Accuracy

| Model | Best val accuracy | Seeds |
|---|---:|---:|
| Conv1D | 98.75% | 3 |
| Conv1DGAP | 98.46% | 3 |

**Takeaway.** On the base laser classification task, `Conv1DGAP` matches or
slightly improves mean test accuracy while using only about `5.1%` of the
parameters of `Conv1D`.

---

## Sources

- Architectures: `models/conv1d.py`, `models/conv1d_gap.py`
- Test accuracy and parameter counts: `outputs/benchmarks/results/benchmark2/summary.csv`
- Validation accuracy: `outputs/benchmarks/results/benchmark2/runs/*-dataset-tier1-seed*.json`
