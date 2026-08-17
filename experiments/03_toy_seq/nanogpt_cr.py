"""Open-source model conversion: a nanoGPT-style character model with the
self-attention swapped for the matrix-free CR attention.

This is the "existing open-source model -> our structure" demonstration.  The
nanoGPT architecture is the standard decoder block

    x = x + attention(ln1(x));   x = x + mlp(ln2(x))

with a causal self-attention.  The CR (Szegő) attention is a *global,
non-causal* aggregation, so it replaces the self-attention in the *encoder*
configuration (bidirectional / masked LM), exactly as BERT uses bidirectional
self-attention.  We keep everything else (token+position embedding, MLP, head)
identical and only swap the attention operator, then compare training speed
and MLM accuracy:

    --attn cr   : PiecewiseCRAttention (matrix-free, O(N log N))
    --attn softmax : nn.MultiheadAttention (the nanoGPT/open-source default)

Runs on tiny-shakespeare MLM, so the conversion is verified on real text.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


class Block(nn.Module):
    def __init__(self, d, p, attn="cr", nhead=4, n_flow=1, nl="softmodrelu"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = attn
        if attn == "cr":
            self.cr_attn = PiecewiseCRAttention(d, p=p, mix=False, gate=False,
                                                n_flow=n_flow, nl=nl)
        else:
            self.mha = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        if self.attn == "cr":
            # CR attention acts on the complex field on H_p
            x = x + self.cr_attn(
                torch.complex(self.ln1(x), torch.zeros_like(self.ln1(x)))).real
        else:
            x = x + self.mha(self.ln1(x), self.ln1(x), self.ln1(x),
                             need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab, d, n_layers, p, attn="cr", nhead=4, n_flow=1,
                 nl="softmodrelu"):
        super().__init__()
        self.p, self.d = p, d
        self.W = p ** 3
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(self.W, d)
        self.blocks = nn.ModuleList(
            [Block(d, p, attn, nhead, n_flow, nl) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, W = x.shape
        pos = torch.arange(W, device=x.device)
        h = self.tok(x) + self.pos(pos)
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))


def load_text(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text)) + ["\x00M"]
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, len(chars), stoi, chars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["cr", "softmax"], default="cr")
    ap.add_argument("--p", type=int, default=7)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-flow", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, vocab, stoi, chars = load_text("data/tinyshakespeare.txt")
    W = args.p ** 3
    wins = [data[i * W:(i + 1) * W] for i in range(len(data) // W)]
    n_tr = int(len(wins) * 0.9)
    tr, va = wins[:n_tr], wins[n_tr:]
    mask_id = vocab - 1

    def batcher(ws):
        def g():
            for i in range(0, len(ws) - args.batch + 1, args.batch):
                yield torch.stack(ws[i:i + args.batch]).to(dev)
        return g

    model = NanoGPT(vocab, args.d, args.layers, args.p, args.attn,
                    n_flow=args.n_flow).to(dev)
    print(f"[{args.attn}] params={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    while step < args.steps:
        for w in batcher(tr)():
            if step >= args.steps:
                break
            B, L = w.shape
            masked = torch.rand(B, L, device=dev) < 0.15
            tgt = w.clone()
            xin = w.clone()
            r = torch.rand(B, L, device=dev)
            xin[masked & (r < 0.8)] = mask_id
            xin[masked & (r >= 0.9)] = torch.randint(0, vocab - 1, (B, L),
                                                     device=dev)[masked & (r >= 0.9)]
            logits = model(xin)
            loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1),
                                   reduction="none")
            loss = (loss * masked.reshape(-1).float()).sum() / masked.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 300 == 0:
                model.eval()
                with torch.no_grad():
                    ces = []
                    for vw in batcher(va)():
                        B, L = vw.shape
                        masked = torch.rand(B, L, device=dev) < 0.15
                        tgt = vw.clone(); xin = vw.clone()
                        xin[masked] = mask_id
                        lg = model(xin)
                        ce = F.cross_entropy(lg.reshape(-1, vocab),
                                             tgt.reshape(-1), reduction="none")
                        ces.append((ce * masked.reshape(-1).float()).sum().item()
                                   / masked.sum().item())
                ppl = math.exp(min(sum(ces) / len(ces), 20))
                print(f"[{args.attn}] step {step} val_ppl={ppl:.2f}", flush=True)
                model.train()
    print(f"[{args.attn}] wall={time.time()-t0:.1f}s "
          f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.3f}GB")


if __name__ == "__main__":
    main()
