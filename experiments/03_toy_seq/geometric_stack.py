"""Quality check: fully-geometric stack (szego attn + GeoFFN) vs baseline.

MLM on the 8K BPE corpus, d=128, p=11, 2 layers, short budget.  Compares
  baseline : diffusion attention + Euclidean FFN (Linear 2d->8d)
  geometric: szego (lambda=0 collapse) attention + GeoFFN (hypersurface collapse)
reports held-out MLM ppl + peak VRAM.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention
from crnn.layers.complex_nn import GeoFFN


class Block(nn.Module):
    def __init__(self, d, p, attn_szego, ffn_geo):
        super().__init__()
        self.attn = PiecewiseCRAttention(d, p=p, mix=False, gate=False,
                                         n_flow=1, nl="softmodrelu",
                                         szego=attn_szego)
        self.n1 = nn.LayerNorm(2 * d)
        self.n2 = nn.LayerNorm(2 * d)
        if ffn_geo:
            self.ffn = GeoFFN(d, rounds=2, nl="softmodrelu")
        else:
            self.ffn = nn.Sequential(nn.Linear(2 * d, 8 * d), nn.GELU(),
                                     nn.Linear(8 * d, 2 * d))

    def forward(self, z):
        z = z + self.attn(z)
        if isinstance(self.ffn, GeoFFN):
            r = torch.cat([z.real, z.imag], -1)
            zc = self.ffn(z)
            z = z + zc
            r = torch.cat([z.real, z.imag], -1)
        else:
            r = torch.cat([z.real, z.imag], -1)
            r = self.n2(r + self.ffn(self.n1(r)))
            z = torch.complex(r[..., :r.shape[-1] // 2],
                              r[..., r.shape[-1] // 2:])
        return z


class Encoder(nn.Module):
    def __init__(self, vocab, d, p, n_layers, attn_szego, ffn_geo):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.to_complex = nn.Linear(d, 2 * d)
        self.blocks = nn.ModuleList(
            [Block(d, p, attn_szego, ffn_geo) for _ in range(n_layers)])
        self.head = nn.Linear(2 * d, vocab)

    def forward(self, x):
        B, W = x.shape
        e = self.embed(x)
        h2 = self.to_complex(e).view(B, W, self.d, 2)
        z = torch.complex(h2[..., 0], h2[..., 1])
        for blk in self.blocks:
            z = blk(z)
        return self.head(torch.cat([z.real, z.imag], -1))


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained("data/bpe8k-tok")
    vocab = tok.vocab_size
    mask_id = tok.convert_tokens_to_ids("[MASK]")
    text = open("data/formal_corpus.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    p, d, L, W = 11, 128, 2, 11 ** 3
    blocks = [ids[i:i + W] for i in range(0, len(ids) - W, W)]
    blocks = [b for b in blocks if len(b) == W]
    n_tr = int(len(blocks) * 0.9)
    tr, ev = blocks[:n_tr], blocks[n_tr:]

    def mask_batch(x):
        m = torch.rand(*x.shape, device=dev) < 0.15
        tgt = x.clone(); xin = x.clone(); xin[m] = mask_id
        return xin, tgt, m

    def train(model, steps):
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        step = 0
        while step < steps:
            for i in range(0, len(tr) - 4 + 1, 4):
                if step >= steps:
                    break
                x = torch.stack(tr[i:i + 4]).to(dev)
                xin, tgt, m = mask_batch(x)
                lg = model(xin)
                ce = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1),
                                     reduction="none")
                loss = (ce * m.reshape(-1).float()).sum() / m.sum()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); step += 1
        mem = torch.cuda.max_memory_allocated() / 1e9
        # eval
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ev) - 4 + 1, 4):
                x = torch.stack(ev[i:i + 4]).to(dev)
                xin, tgt, m = mask_batch(x)
                lg = model(xin)
                ce = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1),
                                     reduction="none")
                tot += (ce * m.reshape(-1).float()).sum().item(); n += m.sum().item()
        ppl = float(torch.exp(torch.tensor(min(tot / n, 20))))
        return ppl, mem, steps / (time.time() - t0)

    for name, szego, geo in [
        ("baseline(diff+euclidFFN)", False, False),
        ("geometric(szego+GeoFFN)", True, True),
    ]:
        model = Encoder(vocab, d, p, L, szego, geo).to(dev)
        npar = sum(p.numel() for p in model.parameters())
        ppl, mem, sp = train(model, 1200)
        print(f"{name}: params={npar/1e6:.2f}M eval_ppl={ppl:.2f} "
              f"peak={mem:.3f}GB steps/s={sp:.1f}", flush=True)


if __name__ == "__main__":
    main()
