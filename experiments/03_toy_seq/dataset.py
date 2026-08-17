"""Synthetic sequence dataset for the M4 toy LM comparison.

Three pattern families with controllable complexity (docs/assumptions.md A4:
the toy task must be non-trivial so that quality differences are visible):

  periodic : x_t = base + ((t // k) % period)              -- short-range
  nested   : stack-balanced symbol stream (bracket-like)   -- nested structure
  longrange: x_t = f(x_{t-L}) xor small noise              -- long-range dep

The windowed-pairs function splits each sequence into windows of length
W = p^3 and returns (context=window i, target=window i+1) pairs for
windowed next-window prediction.
"""

from __future__ import annotations

import numpy as np


def make_sequence(vocab: int, length: int, pattern: str,
                  seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if pattern == "periodic":
        period = int(rng.integers(4, 12))
        k = int(rng.integers(2, 6))
        base = int(rng.integers(0, max(1, vocab - 8)))
        t = np.arange(length)
        return ((base + (t // k)) % (vocab - 2)).astype(np.int64) + 1

    if pattern == "nested":
        # stack-balanced stream: push/pop symbols with a depth cap.
        tokens = np.zeros(length, dtype=np.int64)
        stack = []
        max_depth = int(rng.integers(2, 5))
        for i in range(length):
            if stack and rng.random() < 0.45:
                depth = len(stack)
                tokens[i] = 10 + depth
                stack.pop()
            else:
                if len(stack) >= max_depth:
                    tokens[i] = 10 + len(stack)
                    stack.pop()
                else:
                    tokens[i] = 1 + int(rng.integers(0, 5))
                    stack.append(tokens[i])
        return tokens

    if pattern == "longrange":
        lag = int(rng.integers(8, 32))
        base = int(rng.integers(1, vocab - 2))
        tokens = np.zeros(length, dtype=np.int64)
        tokens[:lag] = 1 + rng.integers(0, vocab - 2, size=lag)
        for i in range(lag, length):
            prev = tokens[i - lag]
            if rng.random() < 0.8:
                tokens[i] = ((prev * base + i) % (vocab - 2)) + 1
            else:
                tokens[i] = 1 + rng.integers(0, vocab - 2)
        return tokens

    raise ValueError(f"unknown pattern {pattern!r}")


class SyntheticSequenceDataset:
    """A fixed corpus of synthetic token sequences."""

    def __init__(self, vocab: int, n_seq: int, seq_len: int,
                 patterns=("periodic", "nested", "longrange"), seed: int = 0):
        self.vocab = vocab
        self.n_seq = n_seq
        self.seq_len = seq_len
        rng = np.random.default_rng(seed)
        self.sequences = np.zeros((n_seq, seq_len), dtype=np.int64)
        for i in range(n_seq):
            pattern = patterns[i % len(patterns)]
            self.sequences[i] = make_sequence(vocab, seq_len, pattern,
                                              seed=int(rng.integers(0, 2**31)))

    def windowed_pairs(self, window: int):
        """(context=w_i, target=w_{i+1}) pairs for every window boundary."""
        n_windows = self.seq_len // window
        pairs = []
        for seq in self.sequences:
            for i in range(n_windows - 1):
                ctx = seq[i * window:(i + 1) * window]
                tgt = seq[(i + 1) * window:(i + 2) * window]
                pairs.append((ctx.copy(), tgt.copy()))
        return pairs
