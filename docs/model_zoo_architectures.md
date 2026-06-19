# Model Zoo — Architecture Diagrams

All 10 models take a 1D signal `(batch, 1, L)` and output class logits `(batch, C)`.

**Legend.** Edge labels show the tensor shape `channels×length` at each step. Signature blocks are highlighted in the family colour.

---

## 1. Conv1D

Plain CNN — 3 conv blocks, flatten, FC head.

```mermaid
graph LR
    A["Input"] -->|"1×L"| B["Conv k=5\n64 ch"]
    B -->|"64×L/2"| C["Conv k=5\n128 ch"]
    C -->|"128×L/4"| D["Conv k=5\n256 ch"]
    D -->|"256×L/8"| E["Flatten"]
    E -->|"256·L/8"| F["Latent 256"]
    F --> G["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style G fill:#f5f5f5,stroke:#333
    style E fill:#444,stroke:#333,color:#fff
    linkStyle default stroke:#888
```

> Block = Conv1d + BN + ReLU + MaxPool(/2) + Dropout. **Flatten** couples head size to L.

---

## 2. Conv1DGAP

Same backbone as Conv1D but **Global Average Pooling** replaces flatten → input-length agnostic, much smaller head.

```mermaid
graph LR
    A["Input"] -->|"1×L"| B["Conv k=5\n64 ch"]
    B -->|"64×L/2"| C["Conv k=5\n128 ch"]
    C -->|"128×L/4"| D["Conv k=5\n256 ch"]
    D -->|"256×L/8"| E["GAP"]
    E -->|"256"| F["Latent 256"]
    F --> G["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style G fill:#f5f5f5,stroke:#333
    style E fill:#E6AB02,stroke:#333,color:#000
    linkStyle default stroke:#888
```

> **GAP** = AdaptiveAvgPool1d(1) — collapses the temporal axis to 1, regardless of L.

---

## 3. LeNet1D

Classic LeNet-5 adapted to 1D — 2 conv stages, heavy FC head.

```mermaid
graph LR
    A["Input"] -->|"1×L"| B["Conv k=5\n32 ch"]
    B -->|"32×L/2"| C["Conv k=5\n64 ch"]
    C -->|"64×L/4"| D["Flatten"]
    D -->|"64·L/4"| E["Dense 512"]
    E --> F["Latent 256"]
    F --> G["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style G fill:#f5f5f5,stroke:#333
    style D fill:#444,stroke:#333,color:#fff
    linkStyle default stroke:#888
```

> Minimal backbone, heavy FC head — parameter count scales with L.

---

## 4. VGG1D

Stacked small (k=3) convolutions, doubling channels — 4 blocks.

```mermaid
graph LR
    A["Input"] -->|"1×L"| B["2× Conv k=3\n64 ch"]
    B -->|"64×L/2"| C["2× Conv k=3\n128 ch"]
    C -->|"128×L/4"| D["3× Conv k=3\n256 ch"]
    D -->|"256×L/8"| E["3× Conv k=3\n256 ch"]
    E -->|"256×L/16"| F["Flatten"]
    F -->|"256·L/16"| G["Dense 400"]
    G --> H["Latent 256"]
    H --> I["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style I fill:#f5f5f5,stroke:#333
    style F fill:#444,stroke:#333,color:#fff
    linkStyle default stroke:#888
```

> Block = N × (Conv + BN + ReLU) + MaxPool(/2). Heavy flatten → large FC.

---

## 5. ResNet1D

Residual blocks with **skip connections** — ResNet-18 style. Width `w` = `base_width` (default 74).

```mermaid
graph LR
    A["Input"] -->|"1×L"| S["Stem\nConv k=7"]
    S -->|"w×L/2"| L1["Stage 1\n2× BasicBlock"]
    L1 -->|"w×L/2"| L2["Stage 2\n2× BasicBlock"]
    L2 -->|"2w×L/4"| L3["Stage 3\n2× BasicBlock"]
    L3 -->|"4w×L/8"| L4["Stage 4\n2× BasicBlock"]
    L4 -->|"8w×L/16"| G["GAP"]
    G -->|"8w"| FC["Latent 256"]
    FC --> O["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style O fill:#f5f5f5,stroke:#333
    linkStyle default stroke:#888
```

```mermaid
graph TB
    subgraph BasicBlock
        X["x"] --> C1["Conv k=3 + BN + ReLU"]
        C1 --> C2["Conv k=3 + BN"]
        X --> |"skip"| ADD(("+"))
        C2 --> ADD
        ADD --> R["ReLU"]
    end

    style X fill:#f5f5f5,stroke:#333
    style R fill:#f5f5f5,stroke:#333
    style ADD fill:#D55E00,stroke:#333,color:#fff
```

> Stage 2+ doubles channels and halves length via strided first block. Identity skip when shapes match, else 1×1 projection.

---

## 6. InceptionTime1D

**Multi-scale parallel convolutions** — 6 modules, residual shortcut every 3.

```mermaid
graph LR
    A["Input"] -->|"1×L"| M1["6× Inception\nModule"]
    M1 -->|"4f×L"| G["GAP"]
    G -->|"4f"| FC["Latent 256"]
    FC --> O["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style O fill:#f5f5f5,stroke:#333
    style M1 fill:#CC79A7,stroke:#333,color:#000
    linkStyle default stroke:#888
```

```mermaid
graph TB
    subgraph InceptionModule
        X["x"] --> BN["Bottleneck\n1×1 conv"]
        BN --> K11["Conv k=11"]
        BN --> K21["Conv k=21"]
        BN --> K41["Conv k=41"]
        X --> MP["MaxPool +\n1×1 conv"]
        K11 --> CAT["Concat\n+ BN + ReLU"]
        K21 --> CAT
        K41 --> CAT
        MP --> CAT
    end

    style X fill:#f5f5f5,stroke:#333
    style CAT fill:#CC79A7,stroke:#333,color:#000
```

> Length preserved inside the module; output = concat of 4 parallel branches (→ 4f channels). `f` = `num_filters` (default 148).

---

## 7. MobileNet1D

**Depthwise separable convolutions** with inverted residual bottleneck (MobileNetV2).

```mermaid
graph LR
    A["Input"] -->|"1×L"| S["Stem\nConv k=3, s=2"]
    S -->|"16×L/2"| B["7 stages\nInvResidual"]
    B -->|"320×L/32"| H["Head\n1×1 conv"]
    H -->|"1280×L/32"| G["GAP"]
    G -->|"1280"| FC["Latent 256"]
    FC --> O["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style O fill:#f5f5f5,stroke:#333
    style B fill:#56B4E9,stroke:#333,color:#000
    linkStyle default stroke:#888
```

```mermaid
graph TB
    subgraph InvertedResidual
        X["x (narrow)"] --> PW1["1×1 Expand 6×"]
        PW1 --> DW["Depthwise k=3"]
        DW --> PW2["1×1 Project"]
        X --> |"skip (same shape)"| ADD(("+"))
        PW2 --> ADD
        ADD --> Y["y (narrow)"]
    end

    style X fill:#f5f5f5,stroke:#333
    style Y fill:#f5f5f5,stroke:#333
    style ADD fill:#56B4E9,stroke:#333,color:#000
```

> ReLU6 activations. Linear bottleneck (no activation after project). Skip only when input/output shapes match.

---

## 8. EfficientNet1D

Like MobileNet but adds **Squeeze-and-Excitation** + **stochastic depth** (EfficientNet-B0 style).

```mermaid
graph LR
    A["Input"] -->|"1×L"| S["Stem\nConv k=3, s=2"]
    S -->|"16×L/2"| B["7 stages\nMBConv"]
    B -->|"320×L/32"| H["Head\n1×1 conv"]
    H -->|"1280×L/32"| G["GAP"]
    G -->|"1280"| FC["Latent 256"]
    FC --> O["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style O fill:#f5f5f5,stroke:#333
    style B fill:#009E73,stroke:#333,color:#fff
    linkStyle default stroke:#888
```

```mermaid
graph TB
    subgraph MBConv
        X["x"] --> PW1["1×1 Expand"]
        PW1 --> DW["Depthwise"]
        DW --> SE["Squeeze &\nExcite"]
        SE --> PW2["1×1 Project"]
        X --> |"skip + stoch. depth"| ADD(("+"))
        PW2 --> ADD
        ADD --> Y["y"]
    end

    style X fill:#f5f5f5,stroke:#333
    style Y fill:#f5f5f5,stroke:#333
    style SE fill:#009E73,stroke:#333,color:#fff
    style ADD fill:#009E73,stroke:#333,color:#fff
```

> SiLU activations. SE recalibrates channel importance. Stochastic depth drops entire blocks during training.

---

## 9. DenseNet1D

**Dense connectivity** — each layer receives all preceding feature maps via concatenation.

```mermaid
graph LR
    A["Input"] -->|"1×L"| S["Stem\nConv k=7 + MaxPool"]
    S -->|"c0×L/4"| D1["Dense Block\n6 layers"]
    D1 -->|"+6g"| T1["Transition /2"]
    T1 --> D2["Dense Block\n12 layers"]
    D2 -->|"+12g"| T2["Transition /2"]
    T2 --> D3["Dense Block\n24 layers"]
    D3 -->|"c×L/16"| G["GAP"]
    G --> FC["Latent 256"]
    FC --> O["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style O fill:#f5f5f5,stroke:#333
    style D1 fill:#999,stroke:#333,color:#fff
    style D2 fill:#999,stroke:#333,color:#fff
    style D3 fill:#999,stroke:#333,color:#fff
    linkStyle default stroke:#888
```

```mermaid
graph TB
    subgraph DenseLayer
        X["x"] --> BN1["BN + ReLU\n1×1 conv"]
        BN1 --> BN2["BN + ReLU\nConv k=3"]
        X --> CAT["Concat"]
        BN2 --> CAT
        CAT --> Y["x ⊕ new"]
    end

    style X fill:#f5f5f5,stroke:#333
    style Y fill:#f5f5f5,stroke:#333
    style CAT fill:#999,stroke:#333,color:#fff
```

> Growth rate `g` = new channels per layer. Transition = BN + 1×1 conv + AvgPool (halves channels + length).

---

## 10. ConvNeXt1D

**Modernised CNN** inspired by Transformers — patchify stem, large-kernel DWConv, inverted bottleneck MLP, LayerNorm + GELU, layer scale.

```mermaid
graph LR
    A["Input"] -->|"1×L"| S["Patchify\nConv k=4, s=4"]
    S -->|"d×L/4"| B1["Stage 1\n3× ConvNeXt"]
    B1 -->|"d×L/4"| D1["Down /2"]
    D1 -->|"2d×L/8"| B2["Stage 2\n3× ConvNeXt"]
    B2 --> D2["Down /2"]
    D2 -->|"4d×L/16"| B3["Stage 3\n9× ConvNeXt"]
    B3 --> D3["Down /2"]
    D3 -->|"8d×L/32"| B4["Stage 4\n3× ConvNeXt"]
    B4 --> N["LayerNorm"]
    N --> G["GAP"]
    G -->|"8d"| FC["Latent 256"]
    FC --> O["Logits C"]

    style A fill:#f5f5f5,stroke:#333
    style O fill:#f5f5f5,stroke:#333
    style S fill:#0072B2,stroke:#333,color:#fff
    linkStyle default stroke:#888
```

```mermaid
graph TB
    subgraph ConvNeXtBlock
        X["x"] --> DW["Depthwise k=7"]
        DW --> LN["LayerNorm"]
        LN --> PW1["1×1 Expand 4×"]
        PW1 --> GE["GELU"]
        GE --> PW2["1×1 Project"]
        PW2 --> LS["Layer Scale γ"]
        X --> |"skip + stoch. depth"| ADD(("+"))
        LS --> ADD
        ADD --> Y["y"]
    end

    style X fill:#f5f5f5,stroke:#333
    style Y fill:#f5f5f5,stroke:#333
    style LS fill:#0072B2,stroke:#333,color:#fff
    style ADD fill:#0072B2,stroke:#333,color:#fff
```

> Depths [3, 3, 9, 3] (ConvNeXt-Tiny). Channels double per stage: [d, 2d, 4d, 8d], with d = `base_dim` (default 42). Per-channel layer scale γ (init 1e-6) stabilises training.

---

## Quick Comparison

| Model | Key Idea | Conv backbone | Classifier head | Head params (→ Latent 256) |
|---|---|---|---|---|
| **Conv1D** | Plain CNN | 3 × Conv k=5, 64→256 ch | Flatten(256·L/8) → 256 → C | 8192·L |
| **Conv1DGAP** | Plain CNN + GAP | 3 × Conv k=5, 64→256 ch | GAP(256) → 256 → C | 66k |
| **LeNet1D** | Minimal (LeNet-5) | 2 × Conv k=5, 32→64 ch | Flatten(64·L/4) → 512 → 256 → C | 8192·L + 132k |
| **VGG1D** | Stacked k=3 | 10 Conv k=3 (4 blocks), 64→256 ch | Flatten(256·L/16) → 400 → 256 → C | 6400·L + 103k |
| **ResNet1D** | Skip connections | 8 blocks × k=3, w→8w ch | GAP(8w) → 256 → C | 2048·w |
| **InceptionTime1D** | Multi-scale parallel | 6 modules × (k=11,21,41), 4f ch | GAP(4f) → 256 → C | 1024·f |
| **MobileNet1D** | Depthwise separable | 17 InvRes blocks, 16→320 ch | GAP(1280) → 256 → C | 328k |
| **EfficientNet1D** | MBConv + SE | 16 MBConv blocks, 16→320 ch | GAP(1280) → 256 → C | 328k |
| **DenseNet1D** | Dense concat | 42 layers (3 blocks), +g ch/layer | GAP(ch) → 256 → C | 256·ch_final |
| **ConvNeXt1D** | Modernised CNN (DWConv k=7 + MLP) | 18 blocks (4 stages [3,3,9,3]), d→8d ch | GAP(8d) → 256 → C | 2048·d |

> **Flatten-based heads** (Conv1D, LeNet, VGG) have parameter counts that scale with input length L, while **GAP-based heads** are L-agnostic.
