"""Blockwise-autoregressive CR decoder: the decoder adaptation.

A standard decoder is causal at *token* granularity (a triangular mask), which
breaks the group-convolution structure of the CR attention.  The adaptation is
**blockwise autoregression** — the standard block-causal paradigm (parallel
block decoding, cf. Medusa / blockwise parallel decoding):

  * the sequence is processed in blocks of W = p^3 tokens;
  * the CR attention aggregates *globally inside* a block (its natural fit);
  * causality lives *between* blocks: block i is predicted from blocks < i;
  * generation appends one predicted block at a time (W tokens in parallel).

This is a genuine decoder (it generates text), with the CR attention's O(N log N)
compute / O(N) memory instead of the O(N^2) causal softmax.  The tradeoff is
block-granularity (not token-granularity) causality.

Run:
    python experiments/03_toy_seq/blockwise_decoder.py --steps 3000 --p 7
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from crnn.models import ToyLM


def load_text(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    return [stoi[c] for c in text], len(chars), itos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=7)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n-flow", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids, vocab, itos = load_text("data/tinyshakespeare.txt")
    W = args.p ** 3
    blocks = [torch.tensor(ids[i:i + W], dtype=torch.long)
              for i in range(0, len(ids) - W, W)]
    blocks = [b for b in blocks if len(b) == W]
    n_tr = int(len(blocks) * 0.9)
    tr, va = blocks[:n_tr], blocks[n_tr:]
    # next-block pairs (causal at block granularity)
    tr_pairs = [(tr[i], tr[i + 1]) for i in range(len(tr) - 1)]
    va_pairs = [(va[i], va[i + 1]) for i in range(len(va) - 1)]
    print(f"vocab={vocab} W={W} train_blocks={len(tr_pairs)} "
          f"val_blocks={len(va_pairs)}")

    model = ToyLM(vocab=vocab, d_model=64, n_layers=args.layers, p=args.p,
                  block_type="cr-vec", attn_type="piecewise", n_flow=args.n_flow,
                  nl="softmodrelu", gate=False, use_pos=False).to(dev)
    print(f"params={model.param_count()}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    def batch(pairs):
        for i in range(0, len(pairs) - args.batch + 1, args.batch):
            x = torch.stack([p[0] for p in pairs[i:i + args.batch]]).to(dev)
            y = torch.stack([p[1] for p in pairs[i:i + args.batch]]).to(dev)
            yield x, y

    step = 0
    while step < args.steps:
        for x, y in batch(tr_pairs):
            if step >= args.steps:
                break
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 500 == 0:
                model.eval()
                with torch.no_grad():
                    ce = sum(F.cross_entropy(model(x).reshape(-1, vocab),
                                             y.reshape(-1)).item()
                             for x, y in batch(va_pairs)) / max(1, len(va_pairs))
                print(f"step {step} val_ppl={float(torch.exp(torch.tensor(min(ce,20)))):.2f}",
                      flush=True)
                model.train()
    print(f"wall={time.time()-t0:.1f}s "
          f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.3f}GB")

    # generation: seed = first val block, autoregressively predict blocks
    model.eval()
    with torch.no_grad():
        seed = va[0].to(dev)
        out = [seed]
        for _ in range(3):                      # generate 3 more blocks
            nxt = model(seed.unsqueeze(0))[0].argmax(-1)
            out.append(nxt)
            seed = nxt
    print("=== GENERATION (blockwise autoregressive) ===")
    for blk in out:
        print("".join(itos[i] for i in blk.tolist())[:200])


if __name__ == "__main__":
    main()
