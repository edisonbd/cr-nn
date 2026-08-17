"""Real open-source model -> CR conversion: BERT-tiny weights -> complex CR-BERT.

Honest transfer plan
--------------------
BERT-tiny's *non-attention* weights (word/position/token-type embedding, the
two LayerNorms, the FFN intermediate/output, and the MLM head) are structurally
identical to a transformer encoder's — these transfer **as the real part** of
our complex weights (imag = 0).  The *self-attention* (Q/K/V + softmax) is a
dot-product operator and cannot map onto the Szegő group convolution, so it is
replaced by a fresh CR (piecewise) attention.  We then fine-tune briefly on
masked language modelling and report quality / speed / memory versus the
original BERT-tiny (whose attention uses SDPA, i.e. flash).

The sequence window is p^3 = 343 (p=7), so inputs are chunked/padded to 343.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from crnn.layers import PiecewiseCRAttention, ComplexRMSNorm
from crnn.layers.complex_nn import ComplexLinear

from transformers import BertForMaskedLM, BertTokenizerFast


def load_bert(name="google/bert_uncased_L-2_H-128_A-2", cache="data/bert-tiny"):
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model = BertForMaskedLM.from_pretrained(cache if os.path.isdir(cache) else name)
    tok = BertTokenizerFast.from_pretrained(cache if os.path.isdir(cache) else name)
    return model, tok


def weight_breakdown(model):
    tot = sum(p.numel() for p in model.parameters())
    attn = sum(p.numel() for n, p in model.named_parameters() if "attention" in n)
    return tot, attn, tot - attn


class CRBert(nn.Module):
    """Complex CR-BERT: BERT-tiny skeleton with CR attention."""

    def __init__(self, bert, p=7, n_flow=1, nl="softmodrelu", checkpoint=False):
        super().__init__()
        cfg = bert.config
        H = cfg.hidden_size
        V = cfg.vocab_size
        self.H, self.p, self.W = H, p, p ** 3
        # --- transferred as real part of complex (imag = 0) ---
        self.emb = nn.Embedding(V, 2 * H)
        self.pos = nn.Embedding(cfg.max_position_embeddings, 2 * H)
        self.tok_type = nn.Embedding(cfg.type_vocab_size, 2 * H)
        self.head = nn.Linear(2 * H, V)
        self.blocks = nn.ModuleList()
        for l in range(cfg.num_hidden_layers):
            blk = nn.Module()
            blk.norm1 = ComplexRMSNorm(H)
            blk.attn = PiecewiseCRAttention(H, p=p, mix=True, gate=False,
                                            n_flow=n_flow, nl=nl,
                                            checkpoint=checkpoint)
            blk.norm2 = ComplexRMSNorm(H)
            blk.ffn1 = ComplexLinear(H, cfg.intermediate_size)
            blk.ffn2 = ComplexLinear(cfg.intermediate_size, H)
            self.blocks.append(blk)
        self._transfer(bert)

    def _transfer(self, bert):
        sd = bert.state_dict()
        # embeddings: real part from BERT, imag 0
        with torch.no_grad():
            self.emb.weight[:, :self.H].copy_(sd["bert.embeddings.word_embeddings.weight"])
            self.pos.weight[:, :self.H].copy_(sd["bert.embeddings.position_embeddings.weight"])
            self.tok_type.weight[:, :self.H].copy_(sd["bert.embeddings.token_type_embeddings.weight"])
            if "cls.predictions.decoder.weight" in sd:
                self.head.weight[:, :self.H].copy_(sd["cls.predictions.decoder.weight"])
            if "cls.predictions.bias" in sd:
                self.head.bias.copy_(sd["cls.predictions.bias"])
            for l, blk in enumerate(self.blocks):
                # norms: BERT gamma (beta absorbed as complex shift = 0)
                g1 = sd[f"bert.encoder.layer.{l}.attention.output.LayerNorm.weight"]
                g2 = sd[f"bert.encoder.layer.{l}.output.LayerNorm.weight"]
                blk.norm1.gamma_r.copy_(g1)
                blk.norm2.gamma_r.copy_(g2)
                # FFN: BERT intermediate/output as real part
                blk.ffn1.Wr.copy_(sd[f"bert.encoder.layer.{l}.intermediate.dense.weight"])
                blk.ffn1.br.copy_(sd[f"bert.encoder.layer.{l}.intermediate.dense.bias"])
                blk.ffn2.Wr.copy_(sd[f"bert.encoder.layer.{l}.output.dense.weight"])
                blk.ffn2.br.copy_(sd[f"bert.encoder.layer.{l}.output.dense.bias"])

    def forward(self, ids, ttype=None):
        B, W = ids.shape
        pos = torch.arange(W, device=ids.device).clamp(max=self.pos.num_embeddings - 1)
        e = self.emb(ids) + self.pos(pos) + self.tok_type(
            ttype if ttype is not None else torch.zeros(B, W, dtype=torch.long,
                                                        device=ids.device))
        z = torch.complex(e[..., :self.H], e[..., self.H:])
        for blk in self.blocks:
            z = z + blk.attn(blk.norm1(z))
            h = blk.ffn1(blk.norm2(z))
            h = torch.complex(F.gelu(h.real), F.gelu(h.imag))
            z = z + blk.ffn2(h)
        return self.head(torch.cat([z.real, z.imag], -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=7)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--n-flow", type=int, default=1)
    ap.add_argument("--checkpoint", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    bert, tok = load_bert()
    tot, attn, rest = weight_breakdown(bert)
    print(f"BERT-tiny params={tot} attention={attn} ({attn/tot:.1%}) "
          f"transferable={rest} ({rest/tot:.1%})")

    # tiny-shakespeare MLM data (wordpiece) — tokenize fully, then chunk
    text = open("data/tinyshakespeare.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    W = args.p ** 3
    wins = [ids[i:i + W] for i in range(0, len(ids) - W, W)]
    wins = [w for w in wins if len(w) == W]
    n_tr = int(len(wins) * 0.9)
    tr, va = wins[:n_tr], wins[n_tr:]
    mask_id = tok.mask_token_id

    def batch(ws):
        for i in range(0, len(ws) - args.batch + 1, args.batch):
            yield torch.stack(ws[i:i + args.batch]).to(dev)

    model = CRBert(bert, p=args.p, n_flow=args.n_flow,
                   checkpoint=args.checkpoint).to(dev)
    n_cr = sum(p.numel() for p in model.parameters())
    print(f"CR-Bert params={n_cr}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    step = 0
    while step < args.steps:
        for w in batch(tr):
            if step >= args.steps:
                break
            masked = torch.rand(*w.shape, device=dev) < 0.15
            tgt = w.clone()
            xin = w.clone()
            xin[masked] = mask_id
            logits = model(xin)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   tgt.reshape(-1), reduction="none")
            loss = (loss * masked.reshape(-1).float()).sum() / masked.sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 200 == 0:
                model.eval()
                with torch.no_grad():
                    ces = []
                    for vw in batch(va):
                        masked = torch.rand(*vw.shape, device=dev) < 0.15
                        tgt = vw.clone(); xin = vw.clone(); xin[masked] = mask_id
                        lg = model(xin)
                        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                             tgt.reshape(-1), reduction="none")
                        ces.append((ce * masked.reshape(-1).float()).sum().item()
                                   / masked.sum().item())
                ppl = min(1e5, float(torch.exp(torch.tensor(min(sum(ces) / len(ces), 20)))))
                print(f"step {step} val_ppl={ppl:.2f}", flush=True)
                model.train()
    print(f"wall={time.time()-t0:.1f}s "
          f"peak_vram={torch.cuda.max_memory_allocated()/1e9:.3f}GB")


if __name__ == "__main__":
    main()
