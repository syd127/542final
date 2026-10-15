"""Shape and masking checks for the velocity field, before any training runs."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import RhythmFlow

dev = "cuda" if torch.cuda.is_available() else "cpu"
m = RhythmFlow().to(dev)
print(f"params: {sum(p.numel() for p in m.parameters())/1e6:.1f} M")

B, T = 4, 150
valid = torch.ones(B, T, device=dev)
valid[2, 80:] = 0  # a short clip, padded
kf = torch.zeros(B, T, device=dev)
kf[:, 0] = 1
kf[0, T - 1] = kf[1, T - 1] = kf[3, T - 1] = 1
kf[2, 79] = 1

out = m(torch.randn(B, T, 151, device=dev), torch.rand(B, device=dev),
        torch.randn(B, T, 151, device=dev), kf,
        torch.randn(B, T, 35, device=dev), torch.rand(B, device=dev), valid,
        drop_audio=torch.tensor([0, 1, 0, 1], dtype=torch.bool, device=dev),
        drop_style=torch.tensor([1, 0, 0, 1], dtype=torch.bool, device=dev))

print("out", tuple(out.shape), "| finite", torch.isfinite(out).all().item())
print("padded frames zeroed:", out[2, 80:].abs().max().item())

out.sum().backward()
nog = [n for n, p in m.named_parameters() if p.grad is None]
print("params with no gradient:", nog if nog else "none")
print("OK")
