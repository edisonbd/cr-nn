"""Formal-scale subword MLM: CR (global) vs bidirectional transformer.

MLM is the correct benchmark for the global CR attention (paper §5.4); the
"next-block" task is a documented negative result (saturating for both models).
This compares the CR global aggregation against a bidirectional transformer
(BERT-style, no causal mask) on the same subword corpus and budget, reporting
meaningful held-out MLM ppl.

15% masking, loss on masked positions only, [MASK] token from the BPE vocab.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention


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


class CREncoder(nn.Module):
    def __init__(self, vocab, d, p, n_layers):
        super().__init__()
        self.d = d
        self.W = p ** 3
        self.embed = nn.Embedding(vocab, d)
        self.to_complex = nn.Linear(d, 2 * d)
        self.blocks = nn.ModuleList([CRBlock(d, p) for _ in range(n_layers)])
        self.head = nn.Linear(2 * d, vocab)

    def forward(self, x):
        B, W = x.shape
        e = self.embed(x)
        h2 = self.to_complex(e).view(B, W, self.d, 2)
        z = torch.complex(h2[..., 0], h2[..., 1])
        for blk in self.blocks:
            z = blk(z)
        return self.head(torch.cat([z.real, z.imag], -1))


class BiTransformer(nn.Module):
    """Bidirectional (BERT-style) transformer — no causal mask."""

    def __init__(self, vocab, d, p, n_layers):
        super().__init__()
        self.W = p ** 3
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(self.W, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, 8, dim_feedforward=4 * d,
                                       batch_first=True, norm_first=True,
                                       activation="gelu")
            for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        B, W = x.shape
        pos = torch.arange(W, device=x.device)
        h = self.embed(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)  # no src_mask => bidirectional
        return self.head(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", default="data/bpe8k-tok")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--p", type=int, default=11)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(args.tok)
    vocab = tok.vocab_size
    mask_id = tok.convert_tokens_to_ids("[MASK]")
    text = open("data/formal_corpus.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    W = args.p ** 3
    blocks = [ids[i:i + W] for i in range(0, len(ids) - W, W)]
    blocks = [b for b in blocks if len(b) == W]
    n_tr = int(len(blocks) * 0.9)
    tr, ev = blocks[:n_tr], blocks[n_tr:]
    print(f"vocab={vocab} mask_id={mask_id} W={W} blocks={len(blocks)} "
          f"(tr={len(tr)} ev={len(ev)})", flush=True)

    def mask_batch(x):
        m = torch.rand(*x.shape, device=x.device) < 0.15
        tgt = x.clone()
        xin = x.clone()
        xin[m] = mask_id
        return xin, tgt, m

    @torch.no_grad()
    def evaluate(model):
        tot, n = 0.0, 0
        for i in range(0, len(ev) - 4 + 1, 4):
            x = torch.stack(ev[i:i + 4]).to(dev)
            xin, tgt, m = mask_batch(x)
            lg = model(xin)
            ce = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1),
                                 reduction="none")
            tot += (ce * m.reshape(-1).float()).sum().item()
            n += m.sum().item()
        return float(torch.exp(torch.tensor(min(tot / n, 20))))

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
                opt.step()
                step += 1
                if step % 600 == 0:
                    ppl = float(torch.exp(torch.tensor(min(loss.item(), 20))))
                    print(f"  step {step} ce={loss.item():.3f} ppl={ppl:.2f}",
                          flush=True)
        sp = steps / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9
        return sp, mem, evaluate(model)

    for name, model in [
        ("CR-MLM", CREncoder(vocab, args.d, args.p, args.layers).to(dev)),
        ("bi-transformer-MLM", BiTransformer(vocab, args.d, args.p,
                                             args.layers).to(dev)),
    ]:
        n = sum(p.numel() for p in model.parameters())
        sp, mem, ev_ppl = train(model, args.steps)
        print(f"{name}: params={n/1e6:.1f}M steps/s={sp:.1f} "
              f"peak={mem:.3f}GB eval_ppl={ev_ppl:.2f}", flush=True)


if __name__ == "__main__":
    main()
