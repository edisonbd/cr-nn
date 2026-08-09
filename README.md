# CR-NN：基于 CR 几何的神经网络

用 Heisenberg 群 $\mathbb{H}^n$ 上的内蕴 CR 几何算子替代 Transformer 的注意力机制，以 CR-Sobolev 范数替代欧氏损失。目标是验证：在不依赖 $O(N^2)$ 矩阵乘法的前提下，在玩具序列任务上达到同规模 Transformer 的质量，同时降低显存、加快运算。

> **状态**：研究阶段（Phase 1–3）。先在 PyTorch 上验证数学正确性与"提速降存"假设，确认收益后再投入 MLX/Metal kernel 工程化。

---

## 核心思路

标准注意力的本质是 $O(N^2)$ 的核回归，核为 softmax 内积。本项目以 **CR 流形上的 Szegő 投影**作为信息聚合算子替代之：

- **平坦模型**：$\mathbb{H}^n$ 上 Szegő 投影核有闭式 $S(z,t)=c_n(|z|^2-it)^{-(n+1)}$，群卷积结构使其可经离散 Heisenberg 群 FFT 降到 $O(N\log N)$。
- **曲率扰动**：以截断微扰展开 $S_\text{curved}\approx S_\text{flat}+\sum_j\varepsilon^j L_j[S_\text{flat}]$ 引入表达力，$\varepsilon$ 为可学小参数。
- **信息汇总到复维**：训练损失加 $\bar\partial_b$ 能量正则，推动表示趋向 CR 函数（全纯子空间）。
- **CR 损失**：用 CR-Sobolev 范数 $\|(I+\Delta_b)^{s/2}\cdot\|^2$ 替代欧氏 $\|\cdot\|^2$，按子拉普拉斯谱频率加权。

---

## 数学叙事的诚实约束

经三路并行调研，本项目对原始设想做了修正（详见 `docs/assumptions.md`）。摘要：

| 表述 | 是否可用 |
|---|---|
| "softmax 注意力 = Bargmann 相干态核的离散截断" | ❌ 无文献支撑，仅作结构类比引言 |
| "曲率扰动 = 低秩修正" | ❌ 曲率扰动一般满秩 |
| "弯曲模型上精确 $O(N\log N)$ 快速变换" | ❌ 仅平坦模型精确，弯曲为近似 |
| "截断微扰展开，曲率项为有限阶 symbol 修正" | ✅ Barilari arXiv:1105.1285 等支撑 |
| "$\Delta_b$ 谱 $(2k+n)\|\lambda\|$ 可计算" | ✅ 经典稳固 |
| "CR Szegő 投影作为注意力的几何替代" | ✅ 替代关系，非重解释 |

---

## 仓库结构

```
cr-nn/
├── docs/
│   ├── math.md            # 数学定型（唯一定义参照）
│   ├── assumptions.md     # 假设清单与风险登记
│   └── references.bib     # 文献（标注稳固性）
├── crnn/
│   ├── backend.py         # 后端抽象（TorchBackend / MLXBackend）
│   ├── geometry/          # H^n 结构、算子、谱
│   ├── transforms/        # Heisenberg FFT、Szegő 投影
│   ├── layers/            # CR-Attention、CR-FFN、block
│   ├── losses/            # CR-Sobolev 损失
│   ├── models/            # 玩具 LM
│   └── curvature/         # 可学曲率扰动
├── experiments/
│   ├── 01_unit_tests/     # 算子数值正确性（解析解比对）
│   ├── 02_speedup_probe/  # CR-attention vs softmax-attention 速度/显存
│   ├── 03_toy_seq/        # 玩具序列建模质量对比
│   └── 04_curvature_ablation/  # 曲率扰动消融
└── tests/                 # 双后端数值一致性
```

---

## 里程碑与止损点

| 里程碑 | 产出 | 判定标准 |
|---|---|---|
| M1 | 数学定型文档 | 定义无歧义，假设可追溯 |
| M2 | Torch 算子 + 单元测试 | 解析解比对通过 |
| M3 | 速度探针 | 平坦 CR-attention 在大 N 下 FLOPs/显存优于 softmax-attention |
| M4 | 玩具序列质量 | CR-NN ppl 与同规模 Transformer 持平（±10%） |
| M5 | 曲率消融 | 曲率项的质量-速度 tradeoff 量化 |
| M6 | MLX 后端 | 双后端数值一致，Apple Silicon 延迟达标 |
| M7 | 论文初稿 | 措辞合规，caveat 显式 |

**早期止损点**：M3 若未拿到速度收益，暂停后续，回头审查"是否真化归 FFT / 常数因子 / 离散化是否破坏群结构"。

---

## 环境

- Python ≥ 3.10
- PyTorch（M1–M5 先行）
- MLX（M6，Apple Silicon）
- NumPy, SciPy

> 注：当前宿主机尚无 Python 环境，需在 M2 前安装。

---

## 参考文献

见 `docs/references.bib`，每条标注稳固性分级 [S]/[C]/[W]。
