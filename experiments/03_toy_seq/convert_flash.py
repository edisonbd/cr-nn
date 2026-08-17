"""Complete conversion: BERT-tiny (SDPA flash attention) -> CR-Bert (CR attention).

Fair comparison that isolates the ATTENTION replacement:
  * load BERT-tiny (real open-source model, flash attention);
  * report the transferable weight fraction (non-attention = ~97%);
  * build CR-Bert with the SAME frozen non-attention weights (transferred as
    the real part of complex weights) and a fresh CR attention;
  * freeze the non-attention weights in BOTH models, train ONLY the attention;
  * compare masked-LM ppl, throughput, peak VRAM on real book text.

Run: python experiments/03_toy_seq/convert_flash.py --steps 800
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_to_cr import CRBert, load_bert, weight_breakdown
from transformers import BertForMaskedLM, BertTokenizerFast


def freeze_non_attention(model, attn_param_names):
    for n, p in model.named_parameters():
        if not any(a in n for a in attn_param_names):
            p.requires_grad = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=7)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-flow", type=int, default=1)
    args = ap.parse_args()

    dev = torch.device("cuda")
    bert, tok = load_bert()
    tot, attn, rest = weight_breakdown(bert)
    print(f"BERT-tiny params={tot} attention={attn} ({attn/tot:.1%}) "
          f"transferable={rest} ({rest/tot:.1%})")

    # real book corpus (Moby Dick + Frankenstein), wordpiece
    text = open("data/corpus.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False,
              truncation=True, max_length=400000)["input_ids"][0]
    W = args.p ** 3
    wins = [ids[i:i + W] for i in range(0, len(ids) - W, W)]
    wins = [w for w in wins if len(w) == W]
    n_tr = int(len(wins) * 0.9)
    tr, va = wins[:n_tr], wins[n_tr:]
    print(f"corpus windows: train={len(tr)} val={len(va)} vocab={tok.vocab_size}")
    mask_id = tok.mask_token_id

    def batch(ws):
        for i in range(0, len(ws) - args.batch + 1, args.batch):
            yield torch.stack(ws[i:i + args.batch]).to(dev)

    # CR-Bert with transferred (frozen) non-attention weights
    cr = CRBert(bert, p=args.p, n_flow=args.n_flow, nl="softmodrelu").to(dev)
    freeze_non_attention(cr, ["attn", "norm1", "norm2"])  # train attn + norms
    n_trainable = sum(p.numel() for p in cr.parameters() if p.requires_grad)
    print(f"CR-Bert total={sum(p.numel() for p in cr.parameters())} "
          f"trainable={n_trainable}")

    def train_one(model, name, steps):
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        step = 0
        while step < steps:
            for w in batch(tr):
                if step >= steps:
                    break
                masked = torch.rand(*w.shape, device=dev) < 0.15
                tgt = w.clone(); xin = w.clone(); xin[masked] = mask_id
                out = model(xin)
                logits = out.logits if hasattr(out, "logits") else out
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       tgt.reshape(-1), reduction="none")
                loss = (loss * masked.reshape(-1).float()).sum() / masked.sum()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                step += 1
                if step % 200 == 0:
                    model.eval()
                    with torch.no_grad():
                        ces = []
                        for vw in batch(va):
                            masked = torch.rand(*vw.shape, device=dev) < 0.15
                            tgt = vw.clone(); xin = vw.clone(); xin[masked] = mask_id
                            out = model(xin)
                            lg = out.logits if hasattr(out, "logits") else out
                            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                                 tgt.reshape(-1), reduction="none")
                            ces.append((ce * masked.reshape(-1).float()).sum().item()
                                       / masked.sum().item())
                    ppl = min(1e5, float(torch.exp(torch.tensor(min(sum(ces) / len(ces), 20)))))
                    print(f"[{name}] step {step} val_ppl={ppl:.2f}", flush=True)
                    model.train()
        wall = time.time() - t0
        print(f"[{name}] wall={wall:.1f}s steps/s={steps/wall:.1f} "
              f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.3f}GB")

    # baseline: BERT-tiny, freeze non-attention, train only softmax attention
    bert = bert.to(dev)
    freeze_non_attention(bert, ["attention", "LayerNorm"])
    n_trainable_b = sum(p.numel() for p in bert.parameters() if p.requires_grad)
    print(f"BERT-tiny trainable={n_trainable_b}")
    train_one(cr, "CR-Bert", args.steps)
    train_one(bert, "BERT-tiny(flash)", args.steps)


if __name__ == "__main__":
    main()
