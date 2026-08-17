"""Profile WHERE the full-model memory actually goes.

The user's question: attention saves memory, FFN saves memory, why doesn't the
full chain?  Profile each component's peak allocation to find the hidden hog.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention
from crnn.layers.complex_nn import GeoFFN


def peak_gb():
    return torch.cuda.max_memory_allocated() / 1e9


def reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    vocab, d, p, L, B = 8192, 128, 11, 2, 4
    N = p ** 3

    embed = nn.Embedding(vocab, d).to(dev)
    to_complex = nn.Linear(d, 2 * d).to(dev)
    attn = PiecewiseCRAttention(d, p=p, mix=False, gate=False, n_flow=1,
                                nl="softmodrelu", szego=True).to(dev)
    ffn = GeoFFN(d, rounds=2, nl="softmodrelu").to(dev)
    head = nn.Linear(2 * d, vocab).to(dev)

    x = torch.randint(0, vocab, (B, N), device=dev)

    print(f"config: vocab={vocab} d={d} p={p} N={N} B={B} layers={L}")
    print("-" * 70)
    print(f"{'stage':<34} {'peak GB':>8} {'delta GB':>8}")
    print("-" * 70)

    # 1. embedding
    reset()
    e = embed(x)
    gb = peak_gb()
    print(f"{'embedding (vocab x d)':<34} {gb:>8.3f} {gb:>8.3f}")
    base = gb

    # 2. + to_complex -> complex field
    reset()
    e = embed(x)
    z = torch.complex(*to_complex(e).view(B, N, d, 2).chunk(2, -1)).squeeze(-1)
    gb = peak_gb()
    print(f"{'+ to_complex -> complex field (B,N,d)':<34} {gb:>8.3f} {gb-base:>8.3f}")
    base = gb

    # 3. + attention (FFT buffers)
    reset()
    e = embed(x)
    z = torch.complex(*to_complex(e).view(B, N, d, 2).chunk(2, -1)).squeeze(-1)
    z = z + attn(z)
    gb = peak_gb()
    print(f"{'+ attention (FFT over p^3)':<34} {gb:>8.3f} {gb-base:>8.3f}")
    base = gb

    # 4. + FFN
    reset()
    e = embed(x)
    z = torch.complex(*to_complex(e).view(B, N, d, 2).chunk(2, -1)).squeeze(-1)
    z = z + attn(z)
    z = z + ffn(z)
    gb = peak_gb()
    print(f"{'+ FFN (GeoFFN)':<34} {gb:>8.3f} {gb-base:>8.3f}")
    base = gb

    # 5. + head logits (B, N, vocab)
    reset()
    e = embed(x)
    z = torch.complex(*to_complex(e).view(B, N, d, 2).chunk(2, -1)).squeeze(-1)
    z = z + attn(z)
    z = z + ffn(z)
    logits = head(torch.cat([z.real, z.imag], -1))
    gb = peak_gb()
    print(f"{'+ head logits (B,N,vocab)':<34} {gb:>8.3f} {gb-base:>8.3f}")
    base = gb

    # 6. + CE backward (autograd graph)
    reset()
    e = embed(x)
    z = torch.complex(*to_complex(e).view(B, N, d, 2).chunk(2, -1)).squeeze(-1)
    z = z + attn(z)
    z = z + ffn(z)
    logits = head(torch.cat([z.real, z.imag], -1))
    loss = F.cross_entropy(logits.reshape(-1, vocab), x.reshape(-1))
    loss.backward()
    gb = peak_gb()
    print(f"{'+ CE loss + backward':<34} {gb:>8.3f} {gb-base:>8.3f}")

    print("-" * 70)
    # theoretical sizes
    def mb(n): return n * 4 / 1e6
    print(f"logits tensor = B*N*vocab = {B*N*vocab/1e6:.1f}M floats "
          f"= {mb(B*N*vocab):.0f} MB (fp32)")
    print(f"head weight  = 2d*vocab  = {2*d*vocab/1e6:.1f}M params "
          f"= {mb(2*d*vocab):.0f} MB")
    print(f"field+FFT    = B*N*d     = {B*N*d/1e6:.1f}M complex "
          f"= {mb(2*B*N*d):.0f} MB (c64)")


if __name__ == "__main__":
    main()
