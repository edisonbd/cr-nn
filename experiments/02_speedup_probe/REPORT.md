# M3a speed probe (superseded)

This is a historical note; the current, GPU-measured results are in
`docs/paper.md` §5.2 and `HANDOFF.md` §5.

The original CPU probe (torch 2.13+cpu, d=64, B=1) found the crossover at
N ≈ 10,000 (p ≈ 23), after which the group-convolution CR path beats softmax
attention, with the advantage growing in N (softmax $O(N^2 d)$ vs CR
$O(N^{4/3})$). The matrix-free piecewise attention later improved this to
$O(N\log N)$, and the A800 GPU measurements give a **16× speedup over flash at
N = 50,653**.
