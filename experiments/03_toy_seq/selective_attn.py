"""Selective (gated) spectral attention: break the fixed-convolution wall.

The real limitation of CR/FNet is NOT translation-equivariance but
*content-independence*: a fixed convolution kernel treats every input the same.
Mamba (input-dependent SSM) and Hyena (gated FFT convolution) break this by an
input-dependent gate, while keeping O(N log N).  This implements the Hyena-style
mechanism: gate = sigmoid(W_g x) modulates the value, then a spectral (FFT)
convolution mixes, then an output projection.  All sub-quadratic.

  x -> g = sigmoid(W_g x), v = W_v x
     -> h = SpectralConv(g * v)   (FFT + learned kernel, O(N log N))
     -> out = W_o h
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


class SelectiveSpectralAttention(nn.Module):
    def __init__(self, d, p, n_heads, n_flow=1):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.hd = d // n_heads
        self.n_flow = n_flow
        self.L = p // 2 + 1
        self.W = nn.Parameter(torch.zeros(n_heads, n_flow, p, p, self.L))
        self.w_g = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.w_o = nn.Linear(d, d)

    def forward(self, x):
        B, N, d = x.shape
        p = round(N ** (1 / 3))
        g = torch.sigmoid(self.w_g(x))       # (B,N,d) input-dependent gate
        v = self.w_v(x)                       # (B,N,d) value
        vg = (g * v).permute(0, 2, 1).reshape(B, d, p, p, p)
        vg = vg.reshape(B, self.n_heads, self.hd, p, p, p)
        for k in range(self.n_flow):
            vh = torch.fft.rfftn(vg, dim=(-3, -2, -1))
            w = self.W[:, k].reshape(1, self.n_heads, 1, p, p, self.L)
            vh = vh * w
            vg = torch.fft.irfftn(vh, s=(p, p, p), dim=(-3, -2, -1))
        vg = vg.reshape(B, d, p, p, p).reshape(B, d, N).permute(0, 2, 1)
        return self.w_o(vg)


class SwiGLUFFN(nn.Module):
    def __init__(self, d, expansion=4):
        super().__init__()
        self.gate = nn.Linear(d, expansion * d, bias=False)
        self.up = nn.Linear(d, expansion * d, bias=False)
        self.down = nn.Linear(expansion * d, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, p, n_heads, n_flow):
        super().__init__()
        self.attn = SelectiveSpectralAttention(d, p, n_heads, n_flow)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ffn = SwiGLUFFN(d)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))


class Decoder(nn.Module):
    def __init__(self, vocab, d, p, n_layers, n_heads, n_flow):
        super().__init__()
        self.d = d
        self.W = p ** 3
        self.embed = nn.Embedding(vocab, d)
        self.register_buffer("pe", fourier_pe(self.W, d))
        self.blocks = nn.ModuleList([Block(d, p, n_heads, n_flow)
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
    ap.add_argument("--n-flow", type=int, default=2)
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

    model = Decoder(vocab, args.d, args.p, args.layers, args.n_heads,
                    args.n_flow).to(dev)
    print(f"SELECTIVE spectral n_heads={args.n_heads} n_flow={args.n_flow} "
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
    print(f"RESULT SELECTIVE: eval_ppl={ppl:.2f} tokens/s={tok_s:.0f} "
          f"peak={mem:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
