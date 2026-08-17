# M4: Toy sequence quality comparison

## Goal

Verify assumption A4: CR-NN matches a same-scale Transformer on a toy
sequence task within +/-10% ppl (quality, not speed -- speed is M3).

## Task design

* **Windowed next-window prediction.** Sequences are split into windows of
  length `W = p^3` (prime `p`). The model sees window `i` and predicts all
  tokens of window `i+1` in parallel.
  * Causal at window granularity.
  * Keeps CR-Attention's global group convolution intact (a causal mask on
    the Szego kernel would break the group-convolution FFT structure, R4).
* **Fair baseline.** TransformerLM uses `nn.TransformerEncoderLayer` with
  pre-norm (`norm_first=True`) and zero dropout, mirroring CRBlock (which has
  no dropout). Same dims/layers/optimizer/data.
* **Loss.** Cross-entropy; optionally combined with a CR-Sobolev
  representation regulariser (`CombinedLoss`, cr_weight > 0) that reduces to
  `mu * ||dbar_b h||^2` on the packed hidden field (assumption A3 test).

## Data

`dataset.py` generates three pattern families (periodic / nested / long-range)
with controllable complexity, mixed across the corpus.

## Run

```bash
python -m pytest experiments/01_unit_tests -q   # sanity
python experiments/03_toy_seq/train.py --model cr --steps 300
python experiments/03_toy_seq/train.py --model transformer --steps 300
python experiments/03_toy_seq/train.py --model cr --cr-weight 0.01 --steps 300
```

Metrics (train CE, eval CE/ppl/acc, latency) are written to
`experiments/03_toy_seq/runs/metrics.csv`.

## Notes / risks

* Short windows (p=5, N=125) are fast but the task may be too easy; prefer
  p=7 (N=343) or p=11 (N=1331). CR is slower than softmax below N~10K, so
  training is slower for the CR model at these sizes -- expected, and not
  the point of M4 (quality only).
* `CombinedLoss` default `sobolev_target="detach"` makes the CR term a pure
  `dbar_b` energy regulariser; see `crnn/losses/combined.py`.
