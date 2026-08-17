"""Plan B: long-context crossover benchmark (quality + speed + memory vs N).

For p = 7,11,13,17 (window N=p^3 = 343..4913) we train masked-LM for a fixed
step budget with (a) the matrix-free CR attention and (b) a same-parameter
transformer (flash/SDPA attention), and record val ppl, throughput (steps/s),
and peak VRAM.  This produces the crossover table: as N grows, the CR's
O(N log N) compute / O(N) memory pulls ahead of softmax's O(N^2), while quality
stays comparable.

Run: python experiments/03_toy_seq/bench_longctx.py
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from crnn.models import ToyLM


def load_text(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text)) + ["\x00M"]
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, len(chars)


def run(p, model_kind, steps, vocab, dev):
    W = p ** 3
    data, vocab = load_text("data/tinyshakespeare.txt")
    wins = [data[i:i + W] for i in range(0, len(data) - W, W)]
    wins = [w for w in wins if len(w) == W]
    n_tr = int(len(wins) * 0.9)
    tr, va = wins[:n_tr], wins[n_tr:]
    mask_id = vocab - 1

    if model_kind == "cr":
        model = ToyLM(vocab=vocab, d_model=64, n_layers=3, p=p,
                      block_type="cr-vec", attn_type="piecewise",
                      n_flow=1, nl="softmodrelu", gate=False,
                      use_pos=False).to(dev)
    else:
        model = ToyLM(vocab=vocab, d_model=64, n_layers=10, p=p,
                      block_type="transformer").to(dev)
    n_params = model.param_count()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    def batch(ws, B=8):
        for i in range(0, len(ws) - B + 1, B):
            yield torch.stack(ws[i:i + B]).to(dev)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    while step < steps:
        for w in batch(tr):
            if step >= steps:
                break
            masked = torch.rand(*w.shape, device=dev) < 0.15
            tgt = w.clone(); xin = w.clone(); xin[masked] = mask_id
            logits = model(xin)
            loss = F.cross_entropy(logits.reshape(-1, vocab), tgt.reshape(-1),
                                   reduction="none")
            loss = (loss * masked.reshape(-1).float()).sum() / masked.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
    wall = time.time() - t0
    # eval ppl
    model.eval()
    with torch.no_grad():
        ces = []
        for w in batch(va):
            masked = torch.rand(*w.shape, device=dev) < 0.15
            tgt = w.clone(); xin = w.clone(); xin[masked] = mask_id
            lg = model(xin)
            ce = F.cross_entropy(lg.reshape(-1, vocab), tgt.reshape(-1),
                                 reduction="none")
            ces.append((ce * masked.reshape(-1).float()).sum().item()
                       / masked.sum().item())
    ppl = float(torch.exp(torch.tensor(min(sum(ces) / len(ces), 20))))
    mem = torch.cuda.max_memory_allocated() / 1e9
    return ppl, steps / wall, mem, n_params


def main():
    dev = torch.device("cuda")
    steps = 800
    print(f"{'p':>3} {'N':>6} | {'model':>12} {'ppl':>7} {'steps/s':>8} "
          f"{'VRAM(GB)':>8} {'params':>7}")
    print("-" * 60)
    for p in (7, 11, 13, 17):
        N = p ** 3
        ppl_cr, sp_cr, mem_cr, np_cr = run(p, "cr", steps, 0, dev)
        ppl_tr, sp_tr, mem_tr, np_tr = run(p, "transformer", steps, 0, dev)
        print(f"{p:>3} {N:>6} | {'CR':>12} {ppl_cr:>7.2f} {sp_cr:>8.1f} "
              f"{mem_cr:>8.3f} {np_cr:>7d}")
        print(f"{p:>3} {N:>6} | {'transformer':>12} {ppl_tr:>7.2f} {sp_tr:>8.1f} "
              f"{mem_tr:>8.3f} {np_tr:>7d}")
        print(f"      {'':>6} | {'speedup CR':>12} {sp_cr/sp_tr:>7.2f}x   "
              f"mem_ratio {mem_cr/mem_tr:.2f}x")


if __name__ == "__main__":
    main()
