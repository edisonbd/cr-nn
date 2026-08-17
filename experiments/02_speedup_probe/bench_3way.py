"""Three-way speed/memory: naive softmax vs flash (SDPA) vs matrix-free CR.

Full-model fwd+bwd at attention-relevant scale (embedding + N layers + head).
All three are *the same LM* differing only in the attention operator:
  - naive softmax : Q K^T / sqrt(d) softmax V,  O(N^2) compute AND memory
  - flash (SDPA)  : F.scaled_dot_product_attention, O(N) memory, O(N^2) compute
  - CR            : piecewise Szego spectral flow, O(N log N) compute, O(N) memory

This is the deployment-relevant 3-way the paper needs: flash already fixes the
O(N^2) *memory*, so CR's durable win over flash is *compute* (O(N log N)) at
long context, while matching naive/flash on memory (all O(N)).

Run: python experiments/02_speedup_probe/bench_3way.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


def attention_naive(q, k, v):
    att = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    return att.softmax(-1) @ v


def attention_flash(q, k, v):
    # (B,N,d) -> (B,1,N,d): the fused flash/mem-efficient kernels require a
    # 4-D (B,H,N,D) layout; single head H=1 keeps it parameter-matched to CR.
    q4 = q.unsqueeze(1)
    k4 = k.unsqueeze(1)
    v4 = v.unsqueeze(1)
    return F.scaled_dot_product_attention(q4, k4, v4).squeeze(1)


class Block(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.kind = kind
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, d))
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, h):
        r = self.norm1(h)
        q, k, v = self.wq(r), self.wk(r), self.wv(r)
        if self.kind == "naive":
            a = attention_naive(q, k, v)
        else:
            a = attention_flash(q, k, v)
        h = h + self.wo(a)
        return h + self.ffn(self.norm2(h))


class CRBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.attn = PiecewiseCRAttention(d, p=p, mix=False, gate=False,
                                         n_flow=1, nl="softmodrelu")
        self.n1 = nn.LayerNorm(2 * d)
        self.n2 = nn.LayerNorm(2 * d)
        self.ffn = nn.Sequential(nn.Linear(2 * d, 8 * d), nn.GELU(),
                                 nn.Linear(8 * d, 2 * d))

    def forward(self, z):
        z = z + self.attn(z)
        r = torch.cat([z.real, z.imag], -1)
        r = self.n2(r + self.ffn(self.n1(r)))
        return torch.complex(r[..., :r.shape[-1] // 2], r[..., r.shape[-1] // 2:])


class TransLM(nn.Module):
    def __init__(self, vocab, d, p, n_layers, kind):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(p ** 3, d)
        self.blocks = nn.ModuleList([Block(d, kind) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, N = x.shape
        h = self.embed(x) + self.pos(torch.arange(N, device=x.device))
        for b in self.blocks:
            h = b(h)
        return self.head(h)


class CRLM(nn.Module):
    def __init__(self, vocab, d, p, n_layers):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.to_complex = nn.Linear(d, 2 * d)
        self.blocks = nn.ModuleList([CRBlock(d, p) for _ in range(n_layers)])
        self.head = nn.Linear(2 * d, vocab)

    def forward(self, x):
        B, N = x.shape
        e = self.embed(x)
        h2 = self.to_complex(e).view(B, N, self.d, 2)
        z = torch.complex(h2[..., 0], h2[..., 1])
        for b in self.blocks:
            z = b(z)
        return self.head(torch.cat([z.real, z.imag], -1))


def bench(model, x, dev, repeat=5):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        out = model(x)
        out.float().mean().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = model(x)
        out.float().mean().backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return ((time.perf_counter() - t0) / repeat * 1000,
            torch.cuda.max_memory_allocated() / 1e9)


def main():
    dev = torch.device("cuda")
    vocab, d, n_layers, B = 8192, 128, 4, 4
    print(f"full-model fwd+bwd  (vocab={vocab} d={d} layers={n_layers} B={B})")
    print("naive=fp32 O(N^2)  flash=bf16 SDPA  CR=fp32 complex64 O(N log N)")
    print(f"{'p':>4} {'N':>6} | {'naive-fp32':>11} | {'flash-bf16':>11} | {'CR':>10}")
    print("      " + "      |  ms    GB    |  ms    GB    |  ms   GB")
    for p in (11, 13, 17, 19, 23):
        N = p ** 3
        x = torch.randint(0, vocab, (B, N), device=dev)
        row = f"{p:>4} {N:>6} |"
        # naive softmax, fp32 (explicit O(N^2) materialization)
        m = TransLM(vocab, d, p, n_layers, "naive").to(dev)
        t, mem = bench(m, x, dev)
        row += f" {t:>5.1f} {mem:>5.2f} |"
        del m
        # flash attention, bf16 (triggers the flash kernel; O(N) memory)
        mb = TransLM(vocab, d, p, n_layers, "flash").to(dev, dtype=torch.bfloat16)
        t, mem = bench(mb, x, dev)
        row += f" {t:>5.1f} {mem:>5.2f} |"
        del mb
        # CR, fp32 complex64
        cr = CRLM(vocab, d, p, n_layers).to(dev)
        t, mem = bench(cr, x, dev)
        row += f" {t:>5.1f} {mem:>5.2f} |"
        del cr
        print(row, flush=True)


if __name__ == "__main__":
    main()
