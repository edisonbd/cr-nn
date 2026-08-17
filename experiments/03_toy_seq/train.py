"""M4 toy sequence quality comparison: CR-NN vs Transformer.

Usage (run from repo root; requires the crnn package installed -e):

    python -m pytest experiments/01_unit_tests -q          # sanity (20 tests)
    python experiments/03_toy_seq/train.py --model cr --steps 300 --p 7
    python experiments/03_toy_seq/train.py --model transformer --steps 300 --p 7
    python experiments/03_toy_seq/train.py --model cr --cr-weight 0.01 --steps 300

Outputs metrics.csv in the output dir and prints a comparison summary.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from crnn.losses import CollapseLoss, CombinedLoss, SobolevEmbeddingLoss
from crnn.models import ToyLM

try:
    from .dataset import SyntheticSequenceDataset
except ImportError:  # run as a plain script (python experiments/03_toy_seq/train.py)
    from dataset import SyntheticSequenceDataset


def build_windows(ds: SyntheticSequenceDataset, window: int, batch_size: int,
                  seed: int, device, frac_eval: float = 0.1):
    pairs = ds.windowed_pairs(window)
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * frac_eval))
    eval_pairs, train_pairs = pairs[:n_eval], pairs[n_eval:]

    def make_batcher(plist):
        """Return a *factory* that yields fresh generators, so evaluate()
        can re-iterate the eval split on every call (generators are one-shot)."""
        def gen():
            n = len(plist)
            for i in range(0, n - batch_size + 1, batch_size):
                sel = [plist[j] for j in range(i, i + batch_size)]
                ctx = torch.tensor(np.stack([p[0] for p in sel]), dtype=torch.long,
                                   device=device)
                tgt = torch.tensor(np.stack([p[1] for p in sel]), dtype=torch.long,
                                   device=device)
                yield ctx, tgt
        return gen

    return (make_batcher(train_pairs), make_batcher(eval_pairs),
            len(train_pairs), len(eval_pairs))


@torch.no_grad()
def evaluate(model, eval_batcher, loss_fn, p, n_batches: int = 8):
    model.eval()
    is_vec = getattr(model, "block_type", "") in ("cr-vec", "cr-geo")
    ces = []
    accs = []
    t0 = time.perf_counter()
    n_ev = 0
    for ctx, tgt in eval_batcher():
        if n_ev >= n_batches:
            break
        logits, hidden = model(ctx, return_hidden=True)
        hidden_grid = model.hidden_to_grid(hidden)
        if is_vec:
            target_grid = model.embed_target_grid(tgt)
            _, stats = loss_fn(logits, tgt, hidden_grid, target_grid, p=p)
        else:
            _, stats = loss_fn(logits, tgt, hidden_grid, p=p)
        ce = stats["ce"]
        pred = logits.argmax(-1)
        acc = (pred == tgt).float().mean().item()
        ces.append(ce)
        accs.append(acc)
        n_ev += 1
    model.train()
    if not ces:
        return 0.0, 0.0, 0.0
    mean_ce = float(np.mean(ces))
    mean_acc = float(np.mean(accs))
    latency_ms = (time.perf_counter() - t0) / max(1, n_ev) * 1000.0
    return mean_ce, mean_acc, latency_ms


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = SyntheticSequenceDataset(args.vocab, args.n_seq, args.seq_len,
                                  seed=args.seed)
    window = args.p ** 3
    train_batcher, eval_batcher, n_train, n_eval = build_windows(
        ds, window, args.batch, seed=args.seed, device=device)
    if n_train == 0:
        raise SystemExit("no training pairs; increase --seq-len")

    model = ToyLM(vocab=args.vocab, d_model=args.d_model, n_layers=args.layers,
                  p=args.p, block_type=args.model, ff_expansion=args.ff_expansion,
                  nhead=args.nhead, dropout=args.dropout,
                  use_pos=not args.no_pos, use_mix=not args.no_mix,
                  curvature_m=args.curvature_m,
                  log_correction=args.log_correction,
                  eps_init=args.eps_init, attn_type=args.attn,
                  spectrum=args.spectrum, spectral_mix=args.spectral_mix,
                  n_flow=args.n_flow, nl=args.nl,
                  twist=not args.no_twist, gate=not args.no_gate,
                  prune_rate=args.prune_rate, checkpoint=args.checkpoint)
    model = model.to(device)
    if args.model in ("cr-vec", "cr-geo"):
        if args.attn in ("fluid", "piecewise"):
            loss_fn = CollapseLoss(ce_weight=1.0,
                                   col_weight=args.col_weight,
                                   mu=args.cr_mu)
        else:
            loss_fn = SobolevEmbeddingLoss(ce_weight=args.ce_weight,
                                           so_weight=args.so_weight,
                                           mu=args.cr_mu)
    else:
        loss_fn = CombinedLoss(ce_weight=1.0, cr_weight=args.cr_weight,
                               mu=args.cr_mu, sobolev_target="detach")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "step", "train_ce", "train_cr",
                         "eval_ce", "eval_ppl", "eval_acc", "latency_ms"])

    print(f"[{args.model}] params={model.param_count()} window={window} "
          f"train_pairs={n_train} eval_pairs={n_eval} p={args.p}")
    is_vec = args.model in ("cr-vec", "cr-geo")
    t_step0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step = 0
    while step < args.steps:
        for ctx, tgt in train_batcher():
            if step >= args.steps:
                break
            if args.debug:
                print(f"[dbg] step {step}: zero_grad", flush=True)
            opt.zero_grad()
            if args.debug:
                print(f"[dbg] step {step}: forward", flush=True)
            logits, hidden = model(ctx, return_hidden=True)
            if args.debug:
                print(f"[dbg] step {step}: forward ok", flush=True)
            hidden_grid = model.hidden_to_grid(hidden)
            if args.debug:
                print(f"[dbg] step {step}: loss", flush=True)
            if is_vec:
                target_grid = model.embed_target_grid(tgt)
                loss, stats = loss_fn(logits, tgt, hidden_grid, target_grid,
                                      p=args.p)
            else:
                loss, stats = loss_fn(logits, tgt, hidden_grid, p=args.p)
            if args.debug:
                print(f"[dbg] step {step}: backward", flush=True)
            loss.backward()
            if args.debug:
                print(f"[dbg] step {step}: clip", flush=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if args.debug:
                print(f"[dbg] step {step}: opt", flush=True)
            opt.step()
            if args.debug:
                print(f"[dbg] step {step}: done", flush=True)
            if args.debug:
                print(f"[dbg] step {step}: elapsed {time.time() - t_step0:.2f}s", flush=True)
            step += 1
            if step % args.log_every == 0 or step == args.steps:
                eval_ce, eval_acc, lat = evaluate(
                    model, eval_batcher, loss_fn, args.p,
                    n_batches=args.eval_batches)
                ppl = float(np.exp(min(eval_ce, 20.0)))
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        args.model, step, round(stats["ce"], 4),
                        round(stats.get("cr", 0.0), 6),
                        round(eval_ce, 4), round(ppl, 4),
                        round(eval_acc, 4), round(lat, 3)])
                print(f"[{args.model}] step {step}/{args.steps} "
                      f"train_ce={stats['ce']:.3f} eval_ce={eval_ce:.3f} "
                      f"ppl={ppl:.3f} acc={eval_acc:.3f} lat={lat:.1f}ms")
    if args.curvature_m > 0:
        attn = getattr(model.blocks[0], "attn", None)
        if attn is not None and hasattr(attn, "eps"):
            print(f"[{args.model}] final eps (block 0): "
                  f"{attn.eps.detach().cpu().tolist()}", flush=True)

    # throughput + peak memory summary (training-memory / speed evidence)
    elapsed = max(1e-9, time.time() - t_step0)
    steps_per_sec = args.steps / elapsed
    tokens_per_sec = steps_per_sec * args.batch * window
    print(f"[{args.model}] throughput={steps_per_sec:.2f} steps/s "
          f"({tokens_per_sec:.0f} tok/s) wall={elapsed:.1f}s", flush=True)
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[{args.model}] peak_vram={peak_gb:.3f} GB", flush=True)
    return csv_path


def main():
    ap = argparse.ArgumentParser(description="M4 toy LM comparison")
    ap.add_argument("--model", choices=["cr", "cr-vec", "cr-geo", "transformer"],
                    default="cr")
    ap.add_argument("--p", type=int, default=7, help="grid resolution (prime); window=p^3")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--ff-expansion", type=int, default=4)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--cr-weight", type=float, default=0.0)
    ap.add_argument("--cr-mu", type=float, default=1e-3)
    ap.add_argument("--so-weight", type=float, default=1.0,
                    help="CR-Sobolev main-loss weight (cr-vec only)")
    ap.add_argument("--ce-weight", type=float, default=0.1,
                    help="auxiliary CE weight (cr-vec only)")
    ap.add_argument("--no-pos", action="store_true",
                    help="cr-vec: disable the learnable complex position field")
    ap.add_argument("--no-mix", action="store_true",
                    help="cr-vec: disable the complex channel mixer")
    ap.add_argument("--curvature-m", type=int, default=0,
                    help="curvature perturbation order M (M5; 0 = flat)")
    ap.add_argument("--log-correction", action="store_true",
                    help="enable R1 log-correction placeholder term")
    ap.add_argument("--eps-init", type=float, default=0.0,
                    help="initial curvature amplitude eps (M5; 0 = start flat)")
    ap.add_argument("--attn", choices=["szego", "fluid", "piecewise"],
                    default="szego",
                    help="cr-vec attention type (fluid = orthogonal spectral "
                         "flow; piecewise = activation-segmented manifold)")
    ap.add_argument("--spectrum", choices=["full", "mlp", "diffusion"],
                    default="full",
                    help="fluid/piecewise spectral filter expressiveness")
    ap.add_argument("--spectral-mix", action="store_true",
                    help="fluid/piecewise: cross-channel complex mixing in freq domain")
    ap.add_argument("--n-flow", type=int, default=3,
                    help="piecewise attention: number of spectral-flow stages")
    ap.add_argument("--nl",
                    choices=["modrelu", "softmodrelu", "radial", "gelu", "none"],
                    default="gelu",
                    help="piecewise attention: segmenting nonlinearity "
                         "(radial/softmodrelu = CR-friendly smooth, no sharp "
                         "breakpoint)")
    ap.add_argument("--no-twist", action="store_true",
                    help="piecewise attention: disable the symplectic chirp twist")
    ap.add_argument("--no-gate", action="store_true",
                    help="piecewise attention: disable the dbar_b gate (4 fewer "
                         "FFTs per stage)")
    ap.add_argument("--prune-rate", type=float, default=0.0,
                    help="CR pruning rate (structured dropout on channels)")
    ap.add_argument("--checkpoint", action="store_true",
                    help="gradient checkpointing on attention FFT + FFN")
    ap.add_argument("--col-weight", type=float, default=1.0,
                    help="collapse-loss weight (fluid attention)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/03_toy_seq/runs")
    ap.add_argument("--debug", action="store_true", help="print per-phase timing")
    args = ap.parse_args()

    csv_path = train(args)
    print(f"metrics saved: {csv_path}")


if __name__ == "__main__":
    main()
