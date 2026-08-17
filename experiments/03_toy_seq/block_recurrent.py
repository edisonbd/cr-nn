"""Block-recurrent CR decoder: O(1) state -> unbounded context.

Causal LLMs (Qwen/DeepSeek/GLM) keep a KV cache of size O(N) (linear), so their
context is bounded by VRAM (Qwen2.5-0.5B: 86 GB at N=1M, OOM).  A block-recurrent
CR decoder keeps instead a FIXED-SIZE running state S (a summary of all previous
blocks) and processes the sequence block by block (block = p^3 tokens) with the
global CR attention; memory is O(p^3 + |S|) independent of the total length.

This demonstrates the "unbounded context" property: we process 1M tokens of
blocks with constant memory, and contrast with Qwen's linear KV cache.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


class BlockRecurrentCR(nn.Module):
    def __init__(self, vocab, d, p, n_layers=1):
        super().__init__()
        self.d, self.p = d, p
        self.W = p ** 3
        self.embed = nn.Embedding(vocab, d)
        self.to_complex = nn.Linear(d, 2 * d)
        self.attn = PiecewiseCRAttention(d, p=p, mix=False, gate=False,
                                         n_flow=1, nl="softmodrelu")
        # running state S (fixed size) and its update
        self.state_proj = nn.Linear(d, d)
        self.head = nn.Linear(2 * d, vocab)

    def init_state(self, B):
        return torch.zeros(B, self.d, device=next(self.parameters()).device)

    def forward_block(self, x, S):
        """x: (B, W) tokens, S: (B, d) running state.  Returns logits, new S."""
        B, W = x.shape
        e = self.embed(x)                              # (B, W, d)
        # inject the running state into every position (a fixed-size context)
        e = e + S.unsqueeze(1)
        z = torch.complex(e, torch.zeros_like(e))
        h2 = self.to_complex(e).view(B, W, self.d, 2)
        z = torch.complex(h2[..., 0], h2[..., 1])
        z = self.attn(z)                                # global CR aggregation
        h = torch.cat([z.real, z.imag], -1)             # (B, W, 2d)
        logits = self.head(h)
        # update state: running summary of the block
        S_new = S + self.state_proj(e.mean(1))
        return logits, S_new


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    vocab, d, p = 66, 64, 11
    B = 1
    model = BlockRecurrentCR(vocab, d, p).to(dev)
    W = p ** 3
    print(f"block W={W}  params={sum(p.numel() for p in model.parameters())}")

    # process an increasingly long stream of blocks; peak memory must stay flat
    print(f"{'total_len':>10} {'blocks':>7} {'peak_GB':>9}")
    S = model.init_state(B)
    for n_blocks in (8, 64, 512, 1024):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        S = model.init_state(B)
        with torch.no_grad():
            for _ in range(n_blocks):
                x = torch.randint(0, vocab, (B, W), device=dev)
                logits, S = model.forward_block(x, S)
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = n_blocks * W
        print(f"{total:>10} {n_blocks:>7} {peak:>9.3f}")

    # Qwen KV-cache contrast (linear): from the earlier measurement
    print("\nQwen2.5-0.5B KV cache (linear, for contrast):")
    print(f"  N=8K: 0.705 GB  ->  N=1M: {2*24*14*64*1_000_000*2/1e9:.1f} GB (OOM)")


if __name__ == "__main__":
    main()
