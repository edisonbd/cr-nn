"""Clean 3-way benchmark: SAME task, open-source structures, long block.

All three do BLOCKWISE next-block prediction (input block t -> predict block t+1,
full bidirectional context within the block) — one single task, so ppl is
comparable.  Structures are the open-source families (scaled to equal size),
all bidirectional within the block:

  Qwen2-style : GQA (2 KV heads) + RoPE + RMSNorm + SwiGLU, flash (SDPA)
  GPT2-style  : MHA + LayerNorm + GELU, flash (SDPA)
  CR          : global piecewise Szego attention + GeoFFN (no positional)

Long block W = p^3 = 4913 (p=17) so CR's O(N log N) speed shows.  Metrics all
measured: eval ppl, tokens/s, peak VRAM.
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from crnn.layers import PiecewiseCRAttention
from crnn.layers.complex_nn import GeoFFN


# ================= Qwen2-style (GQA + RoPE + RMSNorm + SwiGLU, bidirectional) ===
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x * self.w / (rms + self.eps)


def precompute_rope(head_dim, seq_len, base=10000.0, device="cuda"):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float()
                                / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape_as(x)
    return x * cos + x2 * sin


class Qwen2BiBlock(nn.Module):
    def __init__(self, d, n_q_heads, n_kv_heads, head_dim):
        super().__init__()
        self.n_q = n_q_heads
        self.n_kv = n_kv_heads
        self.hd = head_dim
        self.wq = nn.Linear(d, n_q_heads * head_dim, bias=False)
        self.wk = nn.Linear(d, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(d, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_q_heads * head_dim, d, bias=False)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.gate = nn.Linear(d, 4 * d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, h, cos, sin):
        B, N, d = h.shape
        r = self.norm1(h)
        q = self.wq(r).view(B, N, self.n_q, self.hd).transpose(1, 2)
        k = self.wk(r).view(B, N, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(r).view(B, N, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # GQA: repeat KV heads to match Q heads
        rep = self.n_q // self.n_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        # bidirectional (no causal mask)
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, N, self.n_q * self.hd)
        h = h + self.wo(a)
        r2 = self.norm2(h)
        # SwiGLU
        ff = self.down(F.silu(self.gate(r2)) * self.up(r2))
        return h + ff


# ================= GPT2-style (MHA + LayerNorm + GELU, bidirectional) ==========
def gpt2_bi_layer(d, n_head):
    return nn.TransformerEncoderLayer(d, n_head, dim_feedforward=4 * d,
                                      batch_first=True, norm_first=True,
                                      activation="gelu")


# ================= CR blockwise decoder ========================================
class CRBlock(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.attn = PiecewiseCRAttention(d, p=p, mix=False, gate=False,
                                         n_flow=1, nl="softmodrelu")
        self.ffn = GeoFFN(d, rounds=2, nl="softmodrelu")

    def forward(self, z):
        z = z + self.attn(z)
        return z + self.ffn(z)


class CRDecoder(nn.Module):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--p", type=int, default=17)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--maxlen", type=int, default=20_000_000)
    ap.add_argument("--models", default="qwen,gpt2,cr")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda")
    W = args.p ** 3
    tok = PreTrainedTokenizerFast.from_pretrained("data/bpe8k-tok")
    vocab = tok.vocab_size
    with open("data/formal_full.txt", encoding="utf-8") as f:
        text = f.read(args.maxlen * 4)
    ids = tok(text, return_tensors="pt", add_special_tokens=False,
              truncation=True, max_length=args.maxlen)["input_ids"][0]
    blocks = [ids[i:i + W] for i in range(0, len(ids) - W, W)]
    blocks = [b for b in blocks if len(b) == W]
    n_tr = int(len(blocks) * 0.9)
    tr, ev = blocks[:n_tr], blocks[n_tr:]
    print(f"vocab={vocab} W={W} blocks={len(blocks)} "
          f"(tr={len(tr)} ev={len(ev)})", flush=True)

    d, L = args.d, args.layers
    qwen = nn.Sequential(
        nn.Embedding(vocab, d),
        *[Qwen2BiBlock(d, 8, 2, d // 8) for _ in range(L)],
    )
    # wrap with head for logits
    class QwenHead(nn.Module):
        def __init__(self, body, d, vocab):
            super().__init__()
            self.body = body
            self.head = nn.Linear(d, vocab, bias=False)

        def forward(self, x):
            B, N = x.shape
            h = self.body[0](x)
            cos, sin = precompute_rope(d // 8, N, device=x.device)
            for blk in self.body[1:]:
                h = blk(h, cos, sin)
            return self.head(h)

    qwen = QwenHead(qwen, d, vocab)

    gpt2 = nn.Sequential(
        nn.Embedding(vocab, d),
        nn.Embedding(W, d),
        *[gpt2_bi_layer(d, 8) for _ in range(L)],
        nn.LayerNorm(d),
    )

    class GPT2Head(nn.Module):
        def __init__(self, body, d, vocab, W):
            super().__init__()
            self.body = body
            self.head = nn.Linear(d, vocab, bias=False)
            self.W = W

        def forward(self, x):
            B, N = x.shape
            h = self.body[0](x) + self.body[1](torch.arange(N, device=x.device))
            for blk in self.body[2:-1]:
                h = blk(h)
            h = self.body[-1](h)
            return self.head(h)

    gpt2 = GPT2Head(gpt2, d, vocab, W)

    cr = CRDecoder(vocab, d, args.p, L)

    # open-source Qwen2/GPT2 deploy in bf16 (triggers the fused flash kernel);
    # CR stays fp32 (cuFFT fp16 is power-of-2 only, incompatible with prime p).
    qwen = qwen.to(torch.bfloat16)
    gpt2 = gpt2.to(torch.bfloat16)

    for name, m in [("Qwen2-bi", qwen), ("GPT2-bi", gpt2), ("CR-blockwise", cr)]:
        print(f"{name}: params={sum(p.numel() for p in m.parameters())/1e6:.1f}M",
              flush=True)

    def train_blockwise(model, name):
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
                loss = F.cross_entropy(logits.float().reshape(-1, vocab),
                                       y.reshape(-1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); step += 1
                if step % 2000 == 0:
                    ppl = float(torch.exp(torch.tensor(min(loss.item(), 20))))
                    print(f"  [{name}] step {step} ce={loss.item():.3f} "
                          f"ppl={ppl:.2f}", flush=True)
        tok_s = args.steps * B * W / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ev) - B, B):
                x = torch.stack(ev[i:i + B]).to(dev)
                y = torch.stack(ev[i + 1:i + B + 1]).to(dev)
                logits = model(x)
                tot += F.cross_entropy(logits.float().reshape(-1, vocab),
                                       y.reshape(-1)).item() * B
                n += B
        ppl = float(torch.exp(torch.tensor(min(tot / n, 20))))
        return ppl, tok_s, mem

    print("\n=== clean results (same task, blockwise, W=%d) ===" % W)
    print(f"{'model':<16} {'params':>8} {'eval ppl':>9} {'tokens/s':>9} "
          f"{'peak GB':>8}")
    want = set(args.models.split(","))
    for name, m in [("Qwen2-bi", qwen), ("GPT2-bi", gpt2),
                    ("CR-blockwise", cr)]:
        if not any(name.lower().startswith(k) for k in want):
            continue
        ppl, tok_s, mem = train_blockwise(m, name)
        npar = sum(p.numel() for p in m.parameters())
        print(f"{name:<16} {npar/1e6:>7.1f}M {ppl:>9.2f} {tok_s:>9.0f} "
              f"{mem:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
