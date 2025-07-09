import torch
import torch.nn as nn

def add_nan_hook(model):
    """
    Thêm hook để kiểm tra NaN sau mỗi layer forward.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LayerNorm, nn.MultiheadAttention, nn.Embedding)):
            module.register_forward_hook(make_hook(name))


def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"❌ NaN xuất hiện trong layer: {name}")
                print(f"→ Layer: {module}")
                print(f"→ Input NaN: {torch.isnan(input[0]).any().item()}")
                print(f"→ Output shape: {output.shape}")
        elif isinstance(output, (tuple, list)):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor) and torch.isnan(out).any():
                    print(f"❌ NaN trong output[{i}] của layer {name}")
    return hook
