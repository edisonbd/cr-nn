"""Masked language modeling (MLM) on real text — the standard global-aggregation
task for the CR attention.

The CR (Szegő) attention is a *global, non-causal* aggregation over a window of
length N = p^3.  The natural standard benchmark for it is BERT-style masked
language modeling: mask 15% of a window and predict the masked characters from
the whole window context.  This replaces the earlier "next-window prediction"
formulation (which was non-standard and saturating for *both* CR and
transformer) with a task whose inductive bias matches the architecture.

Models compared (same vocab, window, optimizer):
    --attn piecewise (matrix-free CR)   vs   --model transformer

Metrics: masked-position perplexity, accuracy, throughput, peak VRAM.
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
    chars = sorted(list(set(text))) + ["\x00MASK"]      # add [MASK] token
    vocab = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab, stoi, itos, chars


def make_windows(data, window, train_frac=0.9):
    n = len(data) // window
    wins = [data[i * window:(i + 1) * window] for i in range(n)]
    k = int(len(wins) * train_frac)
    return wins[:k], wins[k:]


def make_batcher(wins, batch, device):
    n = len(wins)

    def gen():
        for i in range(0, n - batch + 1, batch):
            yield torch.stack(wins[i:i + batch]).to(device)
    return gen


def mask_batch(x, mask_id, vocab, mask_frac=0.15):
    """BERT-style masking: of masked tokens, 80% [MASK], 10% random, 10% keep."""
    B, W = x.shape
    dev = x.device
    masked = torch.rand(B, W, device=dev) < mask_frac
    targets = x.clone()
    r = torch.rand(B, W, device=dev)
    x_in = x.clone()
    x_in[masked & (r < 0.8)] = mask_id
    rand_tok = torch.randint(0, vocab - 1, (B, W), device=dev)
    x_in[masked & (r >= 0.9)] = rand_tok[masked & (r >= 0.9)]
    return x_in, targets, masked


@torch.no_grad()
def evaluate(model, eval_batcher, mask_id, vocab, n_batches=16):
    model.eval()
    ces, accs, ns = [], [], []
    for k, w in enumerate(eval_batcher()):
        if k >= n_batches:
            break
        x_in, tgt, m = mask_batch(w, mask_id, vocab)
        logits = model(x_in)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             tgt.reshape(-1), reduction="none")
        ce = (ce * m.reshape(-1).float()).sum() / m.sum().clamp(min=1)
        pred = logits.argmax(-1)
        acc = ((pred == tgt) & m).float().sum() / m.sum().clamp(min=1)
        ces.append(ce.item())
        accs.append(acc.item())
    model.train()
    return (sum(ces) / max(1, len(ces)),
            sum(accs) / max(1, len(accs)))


def main():
    ap = argparse.ArgumentParser(description="MLM on real text: CR vs transformer")
    ap.add_argument("--data", default="data/tinyshakespeare.txt")
    ap.add_argument("--model", choices=["cr-vec", "transformer"], default="cr-vec")
    ap.add_argument("--attn", choices=["szego", "piecewise", "fluid"],
                    default="piecewise")
    ap.add_argument("--p", type=int, default=7, help="window = p^3")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--n-flow", type=int, default=3)
    ap.add_argument("--nl", choices=["modrelu", "softmodrelu", "radial", "gelu",
                                     "none"], default="softmodrelu")
    ap.add_argument("--no-twist", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--prune-rate", type=float, default=0.0)
    ap.add_argument("--spectrum", choices=["full", "mlp", "diffusion"],
                    default="full")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, vocab, stoi, itos, chars = load_text(args.data)
    window = args.p ** 3
    train_wins, eval_wins = make_windows(data, window)
    print(f"corpus={len(data)} vocab={vocab} window={window} p={args.p} "
          f"train_wins={len(train_wins)} eval_wins={len(eval_wins)}")
    mask_id = vocab - 1
    train_batcher = make_batcher(train_wins, args.batch, device)
    eval_batcher = make_batcher(eval_wins, args.batch, device)

    model = ToyLM(vocab=vocab, d_model=args.d_model, n_layers=args.layers,
                  p=args.p, block_type=args.model, attn_type=args.attn,
                  n_flow=args.n_flow, nl=args.nl, twist=not args.no_twist,
                  gate=not args.no_gate, prune_rate=args.prune_rate,
                  spectrum=args.spectrum, use_pos=False).to(device)
    print(f"params={model.param_count()}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    while step < args.steps:
        for w in train_batcher():
            if step >= args.steps:
                break
            x_in, tgt, m = mask_batch(w, mask_id, vocab)
            opt.zero_grad()
            logits = model(x_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   tgt.reshape(-1), reduction="none")
            loss = (loss * m.reshape(-1).float()).sum() / m.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                ce, acc = evaluate(model, eval_batcher, mask_id, vocab)
                ppl = float(torch.exp(torch.tensor(min(ce, 20.0))))
                print(f"step {step}/{args.steps} train_ce={loss.item():.3f} "
                      f"val_ce={ce:.3f} ppl={ppl:.2f} acc={acc:.3f}")

    elapsed = max(1e-9, time.time() - t0)
    print(f"throughput={args.steps / elapsed:.2f} steps/s "
          f"({args.steps * args.batch * window / elapsed:.0f} chars/s) "
          f"wall={elapsed:.1f}s")
    if torch.cuda.is_available():
        print(f"peak_vram={torch.cuda.max_memory_allocated() / 1e9:.3f} GB")
    ce, acc = evaluate(model, eval_batcher, mask_id, vocab, n_batches=32)
    print(f"FINAL val_ppl={float(torch.exp(torch.tensor(min(ce, 20.0)))):.2f} "
          f"acc={acc:.3f}")


if __name__ == "__main__":
    main()
