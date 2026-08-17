"""Segmented (piecewise) spectral attention: break the single smooth fluid.

The user's idea: the fluid (convolution kernel) is *too smooth* — one global
translation-invariant kernel.  Cut it into K position segments, each with its
own kernel, breaking translation-equivariance while keeping local smoothness
within each segment.  This is 'K connected fluids', FFT applied per segment
(O(N log(N/K)) still sub-quadratic).

  x -> split into K segments -> per-segment FFT * per-segment kernel -> IFFT -> concat
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast


def fourier_pe(W, d):
    pe = torch.zeros(W, d)
    pos = torch.arange(W).unsqueeze(1).float()
    i = torch.arange(d // 2).unsqueeze(0).float()
    div = torch.exp(-math.log(10000.0) * 2 * i / d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class SegmentedSpectralAttention(nn.Module):
    def __init__(self, d, n_heads, n_segments):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.hd = d // n_heads
        self.n_segments = n_segments
        self.w_v = nn.Linear(d, d)
        self.w_o = nn.Linear(d, d)
        # per-head per-segment kernel: learn L = N/(K) after rfft
        # stored lazily: shaped (H, K, L+1, hd) with L set at forward time

    def forward(self, x):
        B, N, d = x.shape
        H, hd = self.n_heads, self.hd
        K = self.n_segments
        L = (N + K - 1) // K                    # ceil(N/K)
        pad = K * L - N
        if not hasattr(self, "W") or self.W.shape[-2] != L // 2 + 1:
            self.register_parameter(
                "W", nn.Parameter(torch.zeros(H, K, L // 2 + 1, hd,
                                              device=x.device)))
        v = self.w_v(x).view(B, N, H, hd).permute(0, 2, 1, 3)  # (B,H,N,hd)
        v = F.pad(v, (0, 0, 0, pad))                            # (B,H,N+pad,hd)
        v = v.view(B, H, K, L, hd)                               # (B,H,K,L,hd)
        X = torch.fft.rfft(v, dim=3)                            # (B,H,K,L/2+1,hd)
        X = X * self.W                                          # per-segment kernel
        out = torch.fft.irfft(X, n=L, dim=3)                    # (B,H,K,L,hd)
        out = out.reshape(B, H, K * L, hd).permute(0, 2, 1, 3)
        out = out[:, :N].reshape(B, N, d)
        return self.w_o(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        self.gate = nn.Linear(d, expansion * d, bias=False)
        self.up = nn.Linear(d, expansion * d, bias=False)
        self.down = nn.Linear(expansion * d, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, n_heads, n_segments):
        super().__init__()
        self.attn = SegmentedSpectralAttention(d, n_heads, n_segments)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ffn = SwiGLUFFN(d)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Decoder(nn.Module):
    def __init__(self, vocab, d, W, n_layers, n_heads, n_segments):
        super().__init__()
        self.d = d
        self.W = W
        self.embed = nn.Embedding(vocab, d)
        self.register_buffer("pe", fourier_pe(W, d))
        self.blocks = nn.ModuleList([Block(d, n_heads, n_segments)
                                     for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, W = x.shape
        e = self.embed(x) + self.pe[:W]
        for blk in self.blocks:
            e = blk(e)
        return self.head(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-segments", type=int, default=16)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=80_000_000)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    W = args.p ** 3
    tok = PreTrainedTokenizerFast.from_pretrained("data/bpe8k-tok")
    vocab = tok.vocab_size
    with open("data/formal_full.txt", encoding="utf-8") as f:
        text = f.read(args.maxlen * 4)
    ids = tok(text, return_tensors="pt", add_special_tokens=False,
              truncation=True, max_length=args.maxlen)["input_ids"][0]
    blocks = [ids[i:i + W] for i in range(0, len(ids) - W, W)]
    blocks = [b for b in blocks if len(b) == W]
    n_tr = int(len(blocks) * 0.9)
    tr, ev = blocks[:n_tr], blocks[n_tr:]

    model = Decoder(vocab, args.d, W, args.layers, args.n_heads,
                    args.n_segments).to(dev)
    print(f"SEGMENTED segs={args.n_segments} heads={args.n_heads} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    pairs = [(tr[i], tr[i + 1]) for i in range(len(tr) - 1)]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    B = args.batch
    while step < args.steps:
        for i in range(0, len(pairs) - B + 1, B):
            if step >= args.steps:
                break
            x = torch.stack([p[0] for p in pairs[i:i + B]]).to(dev)
            y = torch.stack([p[1] for p in pairs[i:i + B]]).to(dev)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            if step % 2000 == 0:
                ppl = float(torch.exp(torch.tensor(min(loss.item(), 20))))
                print(f"  step {step} ce={loss.item():.3f} ppl={ppl:.2f}",
                      flush=True)
    tok_s = args.steps * B * W / (time.time() - t0)
    mem = torch.cuda.max_memory_allocated() / 1e9
    model.eval(); tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ev) - B, B):
            x = torch.stack(ev[i:i + B]).to(dev)
            y = torch.stack(ev[i + 1:i + B + 1]).to(dev)
            logits = model(x)
            tot += F.cross_entropy(logits.reshape(-1, vocab),
                                   y.reshape(-1)).item() * B
            n += B
    ppl = float(torch.exp(torch.tensor(min(tot / n, 20))))
    print(f"RESULT SEGMENTED: eval_ppl={ppl:.2f} tokens/s={tok_s:.0f} "
          f"peak={mem:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
