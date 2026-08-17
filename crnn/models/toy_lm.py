"""Toy language model for the M4 sequence-quality comparison.

Design (see experiments/03_toy_seq/README.md):
  * Windowed next-window prediction.  Each sequence is split into windows of
    length W = p^3 (prime p).  The model consumes window i and predicts all
    tokens of window i+1 in parallel.  This is causal at window granularity
    and keeps the CR machinery intact: CR-Attention's global group
    convolution operates inside the window without needing a causal mask
    (a causal restriction of the Szegő kernel would break the group-convolution
    FFT structure, risk R4).
  * ``block_type="cr"`` uses CRBlock (Szegő-projection attention); 
    ``block_type="transformer"`` uses nn.TransformerEncoderLayer with pre-norm
    (norm_first=True) and zero dropout so the baselines match the CRBlock
    architecture (CRBlock has no dropout).

Shapes:
    input  : (B, W) int64 token ids, W = p^3
    output : (B, W, vocab) logits
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..layers import CRBlock, VecCRBlock, GeoCRBlock


class ToyLM(nn.Module):
    """Small LM: embedding -> N blocks -> LM head.

    Parameters
    ----------
    vocab : int
        Vocabulary size.
    d_model : int
        Hidden dimension (also the CR channel count).
    n_layers : int
        Number of blocks.
    p : int
        Heisenberg grid resolution per axis; window length W = p^3. Must be
        prime (assumptions.md R7).
    block_type : str
        "cr" (CRBlock), "cr-vec" (VecCRBlock, fully complex CR stack), or
        "transformer" (nn.TransformerEncoderLayer).
    ff_expansion : int
        FFN hidden multiplier (default 4).
    nhead : int
        Transformer baseline attention heads (ignored for "cr").
    dropout : float
        Baseline dropout (kept 0.0 by default to match CRBlock).
    use_pos : bool
        cr-vec: add the learnable complex position field on H_p.  Default
        False: the M4/CR-completion ablation showed the field slightly
        hurts quality at toy scale (ppl 17.29 vs 17.57).
    use_mix : bool
        cr-vec: enable the complex channel mixer inside VecCRAttention.
    curvature_m : int
        Curvature perturbation truncation order M (M5 ablation; 0 = flat).
    log_correction : bool
        Enable the R1 log-correction placeholder term (M5).
    attn_type : str
        cr-vec attention: "szego" (VecCRAttention), "fluid"
        (FluidCRAttention, orthogonal spectral flow), or "piecewise"
        (PiecewiseCRAttention, activation-segmented manifold).
    spectrum : str
        fluid/piecewise spectral filter: "full" | "mlp" | "diffusion".
    spectral_mix : bool
        fluid/piecewise: cross-channel complex mixing in the frequency
        domain (O(p^3 d^2), no p x p matrices).
    n_flow : int
        piecewise attention: number of spectral-flow stages (breakpoints).
    nl : str
        piecewise attention: segmenting nonlinearity "modrelu" | "gelu" |
        "none".
    """

    def __init__(self, vocab: int, d_model: int, n_layers: int, p: int,
                 block_type: str = "cr", ff_expansion: int = 4,
                 nhead: int = 4, dropout: float = 0.0,
                 use_pos: bool = False, use_mix: bool = True,
                 curvature_m: int = 0, log_correction: bool = False,
                 eps_init: float = 0.0, attn_type: str = "szego",
                 spectrum: str = "full", spectral_mix: bool = False,
                 n_flow: int = 3, nl: str = "gelu", twist: bool = True,
                 gate: bool = True, prune_rate: float = 0.0,
                 checkpoint: bool = False):
        super().__init__()
        if block_type not in ("cr", "cr-vec", "cr-geo", "transformer"):
            raise ValueError(f"block_type={block_type!r} "
                             "(expected 'cr'|'cr-vec'|'cr-geo'|'transformer')")
        self.vocab = vocab
        self.d_model = d_model
        self.p = p
        self.W = p ** 3
        self.block_type = block_type
        self.use_pos = use_pos
        self.use_mix = use_mix
        self.curvature_m = curvature_m
        self.log_correction = log_correction
        self.eps_init = eps_init
        self.attn_type = attn_type
        self.spectrum = spectrum
        self.spectral_mix = spectral_mix
        self.n_flow = n_flow
        self.nl = nl
        self.twist = twist
        self.gate = gate
        self.prune_rate = prune_rate
        self.checkpoint = checkpoint

        if block_type == "cr-geo":
            # matrix-free embedding: direct complex lookup (no Linear).
            self.embed = nn.Embedding(vocab, 2 * d_model)
        else:
            self.embed = nn.Embedding(vocab, d_model)
        if block_type == "cr-vec":
            # real -> complex field: Linear(d, 2d) -> (..., re, im)
            self.to_complex = nn.Linear(d_model, 2 * d_model)
            # explicit CR position field: learnable complex offset per grid
            # point (positions are group elements of H_p), added after embed
            if use_pos:
                self.pos_re = nn.Parameter(torch.randn(self.W, d_model) * 0.02)
                self.pos_im = nn.Parameter(torch.randn(self.W, d_model) * 0.02)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            if block_type == "cr":
                self.blocks.append(CRBlock(d_model, p=p, M=curvature_m,
                                           gate=True,
                                           ff_expansion=ff_expansion,
                                           log_correction=log_correction,
                                           eps_init=eps_init))
            elif block_type == "cr-vec":
                self.blocks.append(VecCRBlock(d_model, p=p, gate=gate,
                                              mix=use_mix,
                                              ff_expansion=ff_expansion,
                                              M=curvature_m,
                                              log_correction=log_correction,
                                              eps_init=eps_init,
                                              attn_type=attn_type,
                                              spectrum=spectrum,
                                              spectral_mix=spectral_mix,
                                              n_flow=n_flow, nl=nl,
                                              twist=twist,
                                              prune_rate=prune_rate,
                                              checkpoint=checkpoint))
            elif block_type == "cr-geo":
                self.blocks.append(GeoCRBlock(d_model, p=p, gate=True,
                                              n_flow=n_flow, spectrum=spectrum,
                                              nl=nl, twist=twist))
            else:
                self.blocks.append(nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward=d_model * ff_expansion,
                    dropout=dropout, batch_first=True, norm_first=True,
                    activation="gelu"))
        self.head = nn.Linear(2 * d_model if block_type in ("cr-vec", "cr-geo")
                              else d_model, vocab)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """x: (B, W) token ids. Returns logits (B, W, vocab)."""
        if x.shape[-1] != self.W:
            raise ValueError(f"input length {x.shape[-1]} != window {self.W} (p={self.p})")
        h = self.embed(x)
        if self.block_type == "cr-vec":
            h2 = self.to_complex(h).view(*h.shape[:-1], self.d_model, 2)
            z = torch.complex(h2[..., 0], h2[..., 1])       # (B, N, d) complex
            if self.use_pos:
                pos = torch.complex(self.pos_re, self.pos_im)   # (N, d)
                z = z + pos
            h = z
        elif self.block_type == "cr-geo":
            h2 = h.view(*h.shape[:-1], self.d_model, 2)     # direct complex lookup
            h = torch.complex(h2[..., 0], h2[..., 1])       # (B, N, d) complex
        for blk in self.blocks:
            h = blk(h)
        if self.block_type in ("cr-vec", "cr-geo"):
            logits = self.head(torch.cat([h.real, h.imag], dim=-1))
        else:
            logits = self.head(h)
        if return_hidden:
            return logits, h
        return logits

    def embed_target_grid(self, tgt: torch.Tensor) -> torch.Tensor:
        """Complex embedding field of target tokens (detached), packed to the
        grid: (B*d, p, p, p) complex. Used as the Sobolev regression target.
        The position field is intentionally NOT added (target = token content).
        """
        with torch.no_grad():
            e = self.embed(tgt)
            if self.block_type == "cr-geo":
                h2 = e.view(*e.shape[:-1], self.d_model, 2)
            else:
                h2 = self.to_complex(e).view(*e.shape[:-1], self.d_model, 2)
            z = torch.complex(h2[..., 0], h2[..., 1])       # (B, N, d)
        return self.hidden_to_grid(z)

    def hidden_to_grid(self, h: torch.Tensor) -> torch.Tensor:
        """Pack (B, W, d) hidden states into (B*d, p, p, p) complex fields.

        Uses the same packing as CRAttention.forward so the CR-Sobolev loss
        can treat each channel as an independent scalar field on H_p.
        """
        B, W, d = h.shape
        assert W == self.W
        f = h.permute(0, 2, 1).reshape(B * d, self.p, self.p, self.p).contiguous()
        return f if torch.is_complex(f) else f.to(torch.complex64)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
