"""Char-level language-model terminal test on real text (tiny-shakespeare).

End-to-end usability check: load real text, build a character vocabulary,
windowed next-window prediction (window = p^3, matching the CR grid), train
the piecewise / szego / transformer models, report eval ppl + acc + throughput
+ peak VRAM, and generate a sample continuation.

Usage (run from repo root):
    PYTHONPATH=. python experiments/03_toy_seq/train_text.py \
        --data data/tinyshakespeare.txt --attn piecewise --p 7 --steps 600

The CR models (piecewise/szego) operate on the p^3 grid and use a complex
field; the transformer baseline uses the same windowed task with positional
encoding via the standard encoder layer.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from crnn.models import ToyLM


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab, stoi, itos, chars


def windowed_pairs(data, window, train_frac=0.9):
    """(context=w_i, target=w_{i+1}) pairs; causal at window granularity."""
    n_full = (len(data) - 1) // window - 1   # number of complete (ctx,tgt) pairs
    starts = torch.arange(0, n_full * window, window)
    ctx = [data[s:s + window] for s in starts]
    tgt = [data[s + window:s + 2 * window] for s in starts]
    n = len(ctx)
    n_train = int(n * train_frac)
    return ctx[:n_train], tgt[:n_train], ctx[n_train:], tgt[n_train:]


def make_batcher(ctx, tgt, batch, device):
    n = len(ctx)

    def gen():
        for i in range(0, n - batch + 1, batch):
            x = torch.stack(ctx[i:i + batch]).to(device)
            y = torch.stack(tgt[i:i + batch]).to(device)
            yield x, y
    return gen


@torch.no_grad()
def evaluate(model, eval_batcher, n_batches=8):
    model.eval()
    ces, accs = [], []
    for k, (x, y) in enumerate(eval_batcher()):
        if k >= n_batches:
            break
        logits = model(x)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        acc = (logits.argmax(-1) == y).float().mean().item()
        ces.append(ce.item())
        accs.append(acc)
    model.train()
    return float(sum(ces) / max(1, len(ces))), float(sum(accs) / max(1, len(accs)))


@torch.no_grad()
def generate(model, seed, itos, n_windows, window, device):
    """Continue a seed window by repeatedly predicting the next window."""
    model.eval()
    ctx = seed.to(device)
    out_chars = []
    for _ in range(n_windows):
        logits = model(ctx.unsqueeze(0))
        nxt = logits[0].argmax(-1)          # (window,) greedy next-window
        out_chars.append("".join(itos[i] for i in nxt.tolist()))
        ctx = nxt                            # slide: predicted window -> context
    model.train()
    return out_chars


def main():
    ap = argparse.ArgumentParser(description="char-level CR LM terminal test")
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--model", choices=["cr", "cr-vec", "transformer"],
                    default="cr-vec")
    ap.add_argument("--attn", choices=["szego", "fluid", "piecewise"],
                    default="piecewise")
    ap.add_argument("--p", type=int, default=7, help="grid prime; window=p^3")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--spectrum", choices=["full", "mlp", "diffusion"],
                    default="full")
    ap.add_argument("--n-flow", type=int, default=3)
    ap.add_argument("--nl",
                    choices=["modrelu", "softmodrelu", "radial", "gelu", "none"],
                    default="gelu")
    ap.add_argument("--no-twist", action="store_true")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, vocab, stoi, itos, chars = load_text(args.data)
    window = args.p ** 3
    print(f"corpus={len(data)} chars vocab={vocab} window={window} p={args.p}")

    ctx, tgt, vctx, vtgt = windowed_pairs(data, window)
    print(f"train_windows={len(ctx)} val_windows={len(vctx)}")
    train_batcher = make_batcher(ctx, tgt, args.batch, device)
    eval_batcher = make_batcher(vctx, vtgt, args.batch, device)

    model = ToyLM(vocab=vocab, d_model=args.d_model, n_layers=args.layers,
                  p=args.p, block_type=args.model, attn_type=args.attn,
                  n_flow=args.n_flow, nl=args.nl, twist=not args.no_twist,
                  spectrum=args.spectrum, use_pos=False).to(device)
    print(f"params={model.param_count()}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    while step < args.steps:
        for x, y in train_batcher():
            if step >= args.steps:
                break
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                ce, acc = evaluate(model, eval_batcher)
                ppl = float(torch.exp(torch.tensor(min(ce, 20.0))))
                print(f"step {step}/{args.steps} train_ce={loss.item():.3f} "
                      f"val_ce={ce:.3f} ppl={ppl:.2f} acc={acc:.3f}")

    elapsed = max(1e-9, time.time() - t0)
    print(f"throughput={args.steps / elapsed:.2f} steps/s "
          f"({args.steps * args.batch * window / elapsed:.0f} chars/s) "
          f"wall={elapsed:.1f}s")
    if torch.cuda.is_available():
        print(f"peak_vram={torch.cuda.max_memory_allocated() / 1e9:.3f} GB")

    # final eval + generation sample
    ce, acc = evaluate(model, eval_batcher, n_batches=16)
    print(f"FINAL val_ppl={float(torch.exp(torch.tensor(min(ce, 20.0)))):.2f} "
          f"acc={acc:.3f}")
    seed = vctx[0].clone()
    print("=== generation (seed + continuation) ===")
    seed_str = "".join(itos[i] for i in seed.tolist())
    print("SEED: " + seed_str[:120].replace("\n", " "))
    cont = generate(model, seed, itos, n_windows=2, window=window, device=device)
    print("CONT: " + "".join(cont)[:200].replace("\n", " "))


if __name__ == "__main__":
    main()
