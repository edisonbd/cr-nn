"""LRA ListOps: hierarchical list arithmetic (long-range, hard for attention).

Plan A benchmark.  ListOps nests MAX/MIN/MED/SUM_MOD operators over digits; the
model must parse a ~1000-2000 token expression and output the result digit
(10-class).  This is exactly where the CR attention's *global* group
aggregation should help (long-range hierarchical dependency), so we compare
the matrix-free CR attention vs softmax (transformer) at matched capacity.

Data is generated locally (the standard LRA ListOps generation), no download.
Sequence length is padded to N = p^3.

Run: python experiments/03_toy_seq/lra_listops.py --attn cr --p 11
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


OPS = ["MAX", "MIN", "MED", "SUM_MOD"]


def gen_expr(depth, rng):
    if depth == 0:
        return [str(int(rng.integers(0, 10)))]
    op = OPS[int(rng.integers(0, 4))]
    left = gen_expr(depth - 1, rng)
    right = gen_expr(depth - 1, rng)
    return ["[", op] + left + right + ["]"]


def eval_expr(expr):
    def parse(i):
        if expr[i] != "[":
            return int(expr[i]), i + 1
        i += 1
        op = expr[i]; i += 1
        a, i = parse(i)
        b, i = parse(i)
        i += 1  # ']'
        if op == "MAX":
            r = max(a, b)
        elif op == "MIN":
            r = min(a, b)
        elif op == "MED":
            r = (a + b) // 2
        else:
            r = (a + b) % 10
        return r, i
    v, _ = parse(0)
    return v


def make_dataset(n, depth, vocab, stoi, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n):
        e = gen_expr(depth, rng)
        xs.append([stoi[t] for t in e])
        ys.append(eval_expr(e))
    return xs, ys


class ListOpsModel(nn.Module):
    def __init__(self, vocab, d, n_layers, p, attn="cr", nhead=4):
        super().__init__()
        self.p, self.attn = p, attn
        self.W = p ** 3
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(self.W, d)          # positional embedding
        if attn == "cr":
            self.to_complex = nn.Linear(d, 2 * d)
            self.blocks = nn.ModuleList([
                CRBlock(d, p) for _ in range(n_layers)])
        else:
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4 * d,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
                for _ in range(n_layers)])
        self.head = nn.Linear(2 * d if attn == "cr" else d, 10)

    def forward(self, x):
        B, N = x.shape
        pos = torch.arange(N, device=x.device).clamp(max=self.W - 1)
        h = self.embed(x) + self.pos(pos)
        if self.attn == "cr":
            h2 = self.to_complex(h).view(B, N, -1, 2)
            z = torch.complex(h2[..., 0], h2[..., 1])
            for blk in self.blocks:
                z = blk(z)
            h = torch.cat([z.real, z.imag], -1)
        else:
            for blk in self.blocks:
                h = blk(h)
        return self.head(h[:, 0])          # [CLS] token for classification


class CRBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.attn = PiecewiseCRAttention(d, p=p, mix=False, gate=False,
                                         n_flow=1, nl="softmodrelu")
        self.ffn = nn.Sequential(nn.Linear(2 * d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, 2 * d))
        self.norm1 = nn.LayerNorm(2 * d)
        self.norm2 = nn.LayerNorm(2 * d)

    def forward(self, z):
        B, N, d = z.shape
        r = torch.cat([z.real, z.imag], -1)
        z = z + self.attn(z)
        r = torch.cat([z.real, z.imag], -1)
        z2 = self.ffn(self.norm1(r))
        z = z + torch.complex(z2[..., :d], z2[..., d:])
        return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", choices=["cr", "softmax"], default="cr")
    ap.add_argument("--p", type=int, default=11)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # vocabulary: [CLS], digits 0-9, '[', ']', 4 ops, PAD
    toks = ["[CLS]"] + [str(i) for i in range(10)] + ["[", "]"] + OPS
    stoi = {t: i for i, t in enumerate(toks)}
    vocab = len(toks) + 1
    PAD = vocab - 1
    W = args.p ** 3

    xs_tr, ys_tr = make_dataset(args.n_train, args.depth, vocab, stoi, 0)
    xs_va, ys_va = make_dataset(args.n_val, args.depth, vocab, stoi, 1)

    def to_tensor(xs, ys):
        x = torch.zeros(len(xs), W, dtype=torch.long) + PAD
        for i, e in enumerate(xs):
            x[i, 0] = stoi["[CLS]"]             # [CLS] at position 0
            x[i, 1:len(e) + 1] = torch.tensor(e)
        return x, torch.tensor(ys)

    Xtr, Ytr = to_tensor(xs_tr, ys_tr)
    Xva, Yva = to_tensor(xs_va, ys_va)
    print(f"vocab={vocab} W={W} train={len(Xtr)} val={len(Xva)} "
          f"maxlen={max(len(e) for e in xs_tr)}")

    model = ListOpsModel(vocab, args.d, args.layers, args.p, args.attn).to(dev)
    print(f"[{args.attn}] params={sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    while step < args.steps:
        idx = torch.randint(0, len(Xtr), (args.batch,))
        x, y = Xtr[idx].to(dev), Ytr[idx].to(dev)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                tracc = (model(Xtr[:500].to(dev)).argmax(-1)
                         == Ytr[:500].to(dev)).float().mean().item()
                accs = []
                for i in range(0, len(Xva), args.batch):
                    x = Xva[i:i + args.batch].to(dev)
                    y = Yva[i:i + args.batch].to(dev)
                    accs.append((model(x).argmax(-1) == y).float().mean().item())
            acc = sum(accs) / len(accs)
            print(f"[{args.attn}] step {step} train_acc={tracc:.3f} "
                  f"val_acc={acc:.3f} loss={loss.item():.3f}", flush=True)
            model.train()
    print(f"[{args.attn}] wall={time.time()-t0:.1f}s "
          f"steps/s={args.steps/(time.time()-t0):.1f} "
          f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.3f}GB")


if __name__ == "__main__":
    main()
