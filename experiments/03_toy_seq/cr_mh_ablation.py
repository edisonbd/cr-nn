"""Multi-head CR attention + 4x FFN: close the expressiveness gap vs Qwen2.

The 2.2x ppl gap is dominated by single-head global attention (CR has 1 head,
Qwen2 has 8) and the no-expansion FFN.  This adds:
  - Multi-head CR: split d channels into H groups, each its own piecewise
    Szego spectral flow (H independent "CR heads"), still matrix-free O(N log N)
  - 4x FFN: ComplexFFN with 4x expansion (replacing no-expansion GeoFFN)
  - Fourier positional encoding (already shown to give -13.6%)
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from crnn.layers import PiecewiseCRAttention
from crnn.layers.complex_nn import ComplexFFN, ComplexRMSNorm


def fourier_pe(W, d):
    pe = torch.zeros(W, d)
    pos = torch.arange(W).unsqueeze(1).float()
    i = torch.arange(d // 2).unsqueeze(0).float()
    div = torch.exp(-math.log(10000.0) * 2 * i / d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class MultiHeadCRAttention(nn.Module):
    """H independent CR heads: split channels, each head its own Szego flow."""

    def __init__(self, d, p, n_heads, n_flow=1, nl="softmodrelu", mix=True):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.hd = d // n_heads
        self.heads = nn.ModuleList([
            PiecewiseCRAttention(self.hd, p=p, mix=mix, gate=False,
                                 n_flow=n_flow, nl=nl)
            for _ in range(n_heads)])

    def forward(self, z):
        parts = z.chunk(self.n_heads, dim=-1)
        outs = [h(p) for h, p in zip(self.heads, parts)]
        return torch.cat(outs, dim=-1)


class CRBlock(nn.Module):
    def __init__(self, d, p, n_heads, n_flow, nl, norm, mix):
        super().__init__()
        self.attn = MultiHeadCRAttention(d, p, n_heads, n_flow, nl, mix)
        self.ffn = ComplexFFN(d, expansion=4)
        self.norm = norm
        if norm:
            self.n1 = ComplexRMSNorm(d)
            self.n2 = ComplexRMSNorm(d)

    def forward(self, z):
        if self.norm:
            z = z + self.attn(self.n1(z))
            return z + self.ffn(self.n2(z))
        z = z + self.attn(z)
        return z + self.ffn(z)


class CRDecoder(nn.Module):
    def __init__(self, vocab, d, p, n_layers, n_heads, n_flow, nl, norm, mix):
        super().__init__()
        self.d = d
        self.W = p ** 3
        self.embed = nn.Embedding(vocab, d)
        self.register_buffer("pe", fourier_pe(self.W, d))
        self.to_complex = nn.Linear(d, 2 * d)
        self.blocks = nn.ModuleList([CRBlock(d, p, n_heads, n_flow, nl, norm,
                                             mix)
                                     for _ in range(n_layers)])
        self.head = nn.Linear(2 * d, vocab)

    def forward(self, x):
        B, W = x.shape
        e = self.embed(x) + self.pe[:W]
        h2 = self.to_complex(e).view(B, W, self.d, 2)
        z = torch.complex(h2[..., 0], h2[..., 1])
        for blk in self.blocks:
            z = blk(z)
        return self.head(torch.cat([z.real, z.imag], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--n-flow", type=int, default=1)
    ap.add_argument("--nl", default="softmodrelu")
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--no-mix", action="store_true")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=20_000_000)
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

    mix = not args.no_mix
    model = CRDecoder(vocab, args.d, args.p, args.layers, args.n_heads,
                      args.n_flow, args.nl, args.norm, mix).to(dev)
    print(f"CR multihead heads={args.n_heads} n_flow={args.n_flow} nl={args.nl} "
          f"norm={args.norm} mix={mix} "
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
    print(f"RESULT heads={args.n_heads}: eval_ppl={ppl:.2f} "
          f"tokens/s={tok_s:.0f} peak={mem:.2f}GB", flush=True)


if __name__ == "__main__":
    main()
