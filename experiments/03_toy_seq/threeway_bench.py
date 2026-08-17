"""Formal 3-way benchmark: Qwen2 (GQA+RoPE+flash) vs GPT-2 (flash) vs CR.

All three use the *open-source* architecture (transformers Qwen2 / GPT-2,
which dispatch to fused SDPA/flash on A800), scaled to the same size, trained
from scratch on the SAME corpus with the SAME steps, at a 40GB memory budget.
CR is the blockwise decoder (block-causal, global CR attention within a block).

Metrics are all measured, not theoretical:
  - held-out ppl (causal next-token for Qwen2/GPT-2; blockwise next-block for CR)
  - training throughput tokens/s
  - peak VRAM (GB)

Scale: vocab 8K, d=512, 4 layers, W=1331 (p=11).  Use --steps to raise the
budget; 40GB allows large batch/sequence.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (GPT2Config, GPT2LMHeadModel, Qwen2Config,
                          Qwen2ForCausalLM, PreTrainedTokenizerFast)

from crnn.layers import PiecewiseCRAttention
from crnn.layers.complex_nn import GeoFFN


# ---- CR blockwise decoder (block-causal, global CR within block) ----
class CRBlock(nn.Module):
    def __init__(self, d, p, ffn_geo=True):
        super().__init__()
        self.attn = PiecewiseCRAttention(d, p=p, mix=False, gate=False,
                                         n_flow=1, nl="softmodrelu")
        self.ffn_geo = ffn_geo
        if ffn_geo:
            self.ffn = GeoFFN(d, rounds=2, nl="softmodrelu")
        else:
            self.n1 = nn.LayerNorm(2 * d)
            self.n2 = nn.LayerNorm(2 * d)
            self.ffn = nn.Sequential(nn.Linear(2 * d, 8 * d), nn.GELU(),
                                     nn.Linear(8 * d, 2 * d))

    def forward(self, z):
        z = z + self.attn(z)
        if self.ffn_geo:
            z = z + self.ffn(z)
        else:
            r = torch.cat([z.real, z.imag], -1)
            r = self.n2(r + self.ffn(self.n1(r)))
            z = torch.complex(r[..., :r.shape[-1] // 2],
                              r[..., r.shape[-1] // 2:])
        return z


class CRDecoder(nn.Module):
    def __init__(self, vocab, d, p, n_layers, ffn_geo=True):
        super().__init__()
        self.d = d
        self.W = p ** 3
        self.embed = nn.Embedding(vocab, d)
        self.to_complex = nn.Linear(d, 2 * d)
        self.blocks = nn.ModuleList(
            [CRBlock(d, p, ffn_geo) for _ in range(n_layers)])
        self.head = nn.Linear(2 * d, vocab)

    def forward(self, x):
        B, W = x.shape
        e = self.embed(x)
        h2 = self.to_complex(e).view(B, W, self.d, 2)
        z = torch.complex(h2[..., 0], h2[..., 1])
        for blk in self.blocks:
            z = blk(z)
        return self.head(torch.cat([z.real, z.imag], -1))


def build_models(vocab, d, n_layers, W):
    qwen = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=vocab, hidden_size=d, num_hidden_layers=n_layers,
        num_attention_heads=8, num_key_value_heads=2, intermediate_size=4 * d,
        max_position_embeddings=W))
    gpt2 = GPT2LMHeadModel(GPT2Config(
        vocab_size=vocab, n_embd=d, n_layer=n_layers, n_head=8,
        n_positions=W, n_inner=4 * d))
    cr = CRDecoder(vocab, d, 11, n_layers, ffn_geo=True)
    return qwen, gpt2, cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--maxlen", type=int, default=20_000_000)
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    tok = PreTrainedTokenizerFast.from_pretrained("data/bpe8k-tok")
    vocab = tok.vocab_size
    # read only a byte slice ~4 bytes/token, avoiding a full 423MB tokenize
    with open("data/formal_full.txt", encoding="utf-8") as f:
        text = f.read(args.maxlen * 4)
    ids = tok(text, return_tensors="pt", add_special_tokens=False,
              truncation=True, max_length=args.maxlen)["input_ids"][0]
    W = 11 ** 3
    blocks = [ids[i:i + W] for i in range(0, len(ids) - W, W)]
    blocks = [b for b in blocks if len(b) == W]
    n_tr = int(len(blocks) * 0.9)
    tr, ev = blocks[:n_tr], blocks[n_tr:]
    print(f"vocab={vocab} W={W} blocks={len(blocks)} (tr={len(tr)} ev={len(ev)})",
          flush=True)

    qwen, gpt2, cr = build_models(vocab, args.d, args.layers, W)
    for name, m in [("Qwen2-GQA-flash", qwen), ("GPT2-flash", gpt2),
                    ("CR-blockwise", cr)]:
        print(f"{name}: params={sum(p.numel() for p in m.parameters())/1e6:.1f}M",
              flush=True)

    # ---- causal next-token trainer (Qwen2 / GPT2) ----
    def train_causal(model, name):
        model = model.to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        step = 0
        B = args.batch
        while step < args.steps:
            for i in range(0, len(tr) - B + 1, B):
                if step >= args.steps:
                    break
                x = torch.stack(tr[i:i + B]).to(dev)       # (B, W)
                logits = model(x).logits                    # (B, W, vocab)
                loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab),
                                       x[:, 1:].reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); step += 1
                if step % 2000 == 0:
                    ppl = float(torch.exp(torch.tensor(min(loss.item(), 20))))
                    print(f"  [{name}] step {step} ce={loss.item():.3f} "
                          f"ppl={ppl:.2f}", flush=True)
        tok_s = args.steps * B * W / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9
        # eval held-out ppl
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ev) - B + 1, B):
                x = torch.stack(ev[i:i + B]).to(dev)
                logits = model(x).logits
                tot += F.cross_entropy(logits[:, :-1].reshape(-1, vocab),
                                       x[:, 1:].reshape(-1)).item() * B
                n += B
        ppl = float(torch.exp(torch.tensor(min(tot / n, 20))))
        return ppl, tok_s, mem

    # ---- blockwise next-block trainer (CR) ----
    def train_cr(model):
        model = model.to(dev)
        pairs = [(tr[i], tr[i + 1]) for i in range(len(tr) - 1)]
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        step = 0
        B = args.batch
        while step < args.steps:
            for i in range(0, len(pairs) - B + 1, B):
                if step >= args.steps:
                    break
                x = torch.stack([p[0] for p in pairs[i:i + B]]).to(dev)
                y = torch.stack([p[1] for p in pairs[i:i + B]]).to(dev)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); step += 1
                if step % 2000 == 0:
                    ppl = float(torch.exp(torch.tensor(min(loss.item(), 20))))
                    print(f"  [CR] step {step} ce={loss.item():.3f} "
                          f"ppl={ppl:.2f}", flush=True)
        tok_s = args.steps * B * W / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ev) - B, B):
                x = torch.stack(ev[i:i + B]).to(dev)
                y = torch.stack(ev[i + 1:i + B + 1]).to(dev)
                logits = model(x)
                tot += F.cross_entropy(logits.reshape(-1, vocab),
                                       y.reshape(-1)).item() * B
                n += B
        ppl = float(torch.exp(torch.tensor(min(tot / n, 20))))
        return ppl, tok_s, mem

    print("\n=== results (all measured) ===")
    print(f"{'model':<20} {'params':>8} {'eval ppl':>9} {'tokens/s':>9} "
          f"{'peak GB':>8}")
    for name, m, fn in [
        ("Qwen2-GQA-flash", qwen, train_causal),
        ("GPT2-flash", gpt2, train_causal),
        ("CR-blockwise", cr, train_cr),
    ]:
        if fn is train_causal:
            ppl, tok_s, mem = fn(m, name)
        else:
            ppl, tok_s, mem = fn(m)
        npar = sum(p.numel() for p in m.parameters())
        print(f"{name:<20} {npar/1e6:>7.1f}M {ppl:>9.2f} {tok_s:>9.0f} "
              f"{mem:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
