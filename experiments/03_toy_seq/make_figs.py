"""Generate the two key paper figures from the measured results.

fig_precision.png : precision evolution (complex CR -> real spectral -> Q*K -> LDR)
                    vs Qwen2/GPT2 baselines.
fig_speed.png     : long-context attention speed crossover (CR vs flash).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Figure 1: precision evolution ----
labels = ["complex\nSzegő", "real\nspectral", "Q⋆K\n(corr.)", "LDR\n(Q⋆K+lowr.)",
          "GPT-2\n(flash)", "Qwen2\n(GQA+RoPE)"]
ppl = [1833, 1460, 1381, 1201, 1301, 1068]
colors = ["#c0392b", "#e67e22", "#f1c40f", "#2ecc71", "#7f8c8d", "#3498db"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(range(len(labels)), ppl, color=colors)
for b, v in zip(bars, ppl):
    ax.text(b.get_x() + b.get_width()/2, v + 20, str(v),
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("held-out ppl (lower is better)", fontsize=10)
ax.set_title("Precision: matrix-free attention vs Transformer baselines\n"
             "(blockwise next-block, 80M tokens, d=512, 4 layers)", fontsize=10)
ax.set_ylim(0, 2050)
ax.grid(axis="y", alpha=0.3)
ax.axhline(1068, color="#3498db", ls="--", lw=1, alpha=0.5)
plt.tight_layout()
plt.savefig("docs/fig_precision.png", dpi=150)
plt.close()

# ---- Figure 2: long-context speed crossover ----
N = [12167, 24389, 29791, 50653]
speedup = [4.4, 8.8, 10.8, 16.2]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(N, speedup, "o-", color="#2ecc71", lw=2, ms=6,
        label="CR vs flash speedup")
for x, y in zip(N, speedup):
    ax.text(x, y + 0.6, f"{y:.1f}×", ha="center", fontsize=9, fontweight="bold")
ax.set_xlabel("sequence length N", fontsize=10)
ax.set_ylabel("CR speedup over flash (fwd+bwd)", fontsize=10)
ax.set_title("Long-context speed: O(N log N) vs O(N²)\n"
             "(attention operator, A800, d=128)", fontsize=10)
ax.set_xscale("log")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/fig_speed.png", dpi=150)
plt.close()

print("saved docs/fig_precision.png, docs/fig_speed.png")
