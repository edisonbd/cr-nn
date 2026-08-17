"""Diagnose which SDPA backend torch actually uses, and at what memory cost.

Tests each backend (forced) for a single-head bf16 attention at N=12167 and
reports peak memory, to pin down whether 'flash' is O(N) or silently falls
back to math (O(N^2)) in this torch build.
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

dev = "cuda"
B, N, H, D = 4, 12167, 8, 64
q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)

for name, bk in [("MATH", SDPBackend.MATH),
                 ("EFFICIENT", SDPBackend.EFFICIENT_ATTENTION),
                 ("FLASH", SDPBackend.FLASH_ATTENTION),
                 ("CUDNN", SDPBackend.CUDNN_ATTENTION)]:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with sdpa_kernel(bk):
            out = F.scaled_dot_product_attention(q, k, v)
        out.sum().backward()
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"{name:>10}: OK   peak={mem:.3f}GB", flush=True)
    except Exception as e:
        print(f"{name:>10}: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)

# default dispatch (no forcing)
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
out = F.scaled_dot_product_attention(q, k, v)
out.sum().backward()
print(f"{'default':>10}: OK   peak={torch.cuda.max_memory_allocated()/1e9:.3f}GB")
