#!/usr/bin/env python3
"""Regenerate all grid and Pareto figures from cached JSON results.

Does NOT need torch / the training pipeline — mocks torch so we can
import benchmark_zoo for its plotting functions.
"""
import sys, os, types

# Build a torch mock complete enough that scipy / seaborn don't crash
# when they probe for torch.Tensor during import.
_torch = types.ModuleType("torch")

class _FakeTensor:
    pass

_torch.Tensor = _FakeTensor
_torch.nn = types.ModuleType("torch.nn")
_torch.nn.Module = type("Module", (), {})
_torch.nn.functional = types.ModuleType("torch.nn.functional")
_torch.utils = types.ModuleType("torch.utils")
_torch.utils.data = types.ModuleType("torch.utils.data")
_torch.utils.data.DataLoader = type("DataLoader", (), {})
_torch.utils.data.Dataset = type("Dataset", (), {})
_torch.utils.data.Subset = type("Subset", (), {})
_torch.optim = types.ModuleType("torch.optim")
_torch.optim.lr_scheduler = types.ModuleType("torch.optim.lr_scheduler")
_torch.device = lambda x: x
_torch.no_grad = lambda: type("ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()
_torch.float32 = "float32"
_torch.cuda = types.ModuleType("torch.cuda")
_torch.cuda.is_available = lambda: False

for name, mod in [
    ("torch", _torch),
    ("torch.nn", _torch.nn),
    ("torch.nn.functional", _torch.nn.functional),
    ("torch.utils", _torch.utils),
    ("torch.utils.data", _torch.utils.data),
    ("torch.optim", _torch.optim),
    ("torch.optim.lr_scheduler", _torch.optim.lr_scheduler),
    ("torch.cuda", _torch.cuda),
]:
    sys.modules[name] = mod

# Also mock wandb and torchinfo
for mod_name in ["wandb", "torchinfo", "torchsummary"]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        m.summary = lambda *a, **kw: None
        sys.modules[mod_name] = m

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_zoo import (
    generate_scaling_grid,
    generate_pareto_publication,
    generate_pareto_latency_focus,
    generate_tier_grid,
)

OUTPUT_DIR = "results/benchmark2"

generate_scaling_grid(OUTPUT_DIR, x_col="macs", x_label="MACs",
                      fname="scaling_grid.pdf")
generate_scaling_grid(OUTPUT_DIR, x_col="params", x_label="Parameters",
                      fname="scaling_grid_size.pdf")
generate_tier_grid(OUTPUT_DIR)
generate_pareto_publication(OUTPUT_DIR, x_col="macs", x_label="MACs",
                            fname="pareto.pdf")
generate_pareto_publication(OUTPUT_DIR, x_col="size_mb",
                            x_label="Model size (MB)", x_log=True,
                            fname="pareto_size.pdf")
generate_pareto_latency_focus(OUTPUT_DIR)

print("Done — all 6 figures regenerated.")
