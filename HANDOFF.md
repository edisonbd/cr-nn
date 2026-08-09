# HANDOFF — CR-NN 项目交接

> 最后更新：2026-08-10
> 当前进度：M1–M3 完成，M4 待开始
> 测试状态：20/20 通过

---

## 1. 这个项目在做什么

用 **CR 几何**（Heisenberg 群上的 Szegő 投影）替代 Transformer 的 softmax
注意力，用 **CR-Sobolev 范数**替代欧氏损失。目标：在玩具序列任务上达到
同规模 Transformer 的质量，同时在长序列下降低显存、加快运算。

数学骨架与所有假设见 `docs/math.md`（数学定型）和 `docs/assumptions.md`
（假设清单 + 风险登记）。**接手前必读这两个文件**——它们是实现的唯一参照，
所有断言都标注了稳固性分级。

## 2. 关键约束（踩过的坑，必读）

### p 必须是素数
有限 Heisenberg 群 $H_p$ 的矩阵值 Fourier 变换**只在 p 为素数时正确**。
合数 p 下 Schrödinger 表示族不完整（漏掉低维不可约表示），Plancherel 失效，
约 5% 能量泄漏。这是 M2 调试一整天的根因，记录在 `docs/assumptions.md` R7。

**工程含义**：格点分辨率取邻近素数（3, 5, 7, 11, 13, 17, 19, 23, 29, ...）。
序列长度 N = p³。若序列长度不是 p³，CR-Attention 会自动 pad/truncate。

### 群法则用单向形式
实现采用 $c = c_1 + c_2 + a_1 b_2$（系数 1，单向），**不是**连续 $\mathbb{H}^n$
的对称形式 $2(x_1 y_2 - y_1 x_2)$（系数 2）。二者同构但参数化不同。连续核
（Korányi/Szegő）在格点上求值时按此群法则解释。见 `docs/math.md` §7.2。

### numpy FFT 用负号，Schrödinger 表示用正号
`omega = e^{+2πi/p}`，但 numpy/torch `fft` 用 `e^{-2πi/p}`。所以前向变换
用 `conj(fft2(conj(x)))` 实现正指数 2D 变换。这个符号陷阱在
`crnn/transforms/heisenberg_fft.py` 模块文档里有详细说明。

### 不可宣称 softmax = Bargmann 核
"softmax 注意力等价于 Bargmann 相干态核的离散截断"——**无文献支撑，仅结构类比**。
论文中只能作类比引言。见 `docs/assumptions.md` 叙事合规检查表。

### 曲率扰动是"截断微扰展开"，不是"低秩修正"
混合方案（平坦 + 可学曲率）的正确定位是截断微扰展开（Barilari arXiv:1105.1285）。
曲率扰动一般满秩，不可称低秩。见 `docs/math.md` §5。

## 3. 仓库结构

```
cr-nn/
├── docs/
│   ├── math.md            # ★数学定型（唯一定义参照）
│   ├── assumptions.md     # ★假设清单 A1-A4 + 风险 R1-R7
│   └── references.bib     # 文献，标注 [S]/[C]/[W] 稳固性
├── crnn/
│   ├── backend.py         # 后端抽象（Backend Protocol）
│   ├── backends/
│   │   └── torch_backend.py    # TorchBackend（CPU/CUDA，complex64）
│   ├── geometry/
│   │   ├── heisenberg.py       # 群结构、Korányi 范数、离散格点
│   │   ├── operators.py        # X_j,Y_j,∂̄_b,Δ_b,Korányi核,Szegő核
│   │   └── spectrum.py         # Hermite-Laguerre基, 谱值 (2k+n)|λ|
│   ├── transforms/
│   │   ├── heisenberg_fft.py   # ★矩阵值群FFT + 群卷积（素数p）
│   │   └── szego.py            # Szegő投影（平坦+曲率微扰占位）
│   ├── layers/
│   │   ├── cr_attention.py         # ★CR-Attention（注意力替代）
│   │   ├── cr_attention_backend.py # 向量化群卷积（生产版）
│   │   ├── cr_feedforward.py       # 复值FFN
│   │   └── cr_block.py             # CR-Block（drop-in替代Transformer层）
│   ├── losses/
│   │   └── cr_sobolev.py       # CR-Sobolev损失 + ∂̄_b正则
│   └── models/                 # （M4 待写）玩具LM
├── experiments/
│   ├── 01_unit_tests/          # ★20个测试，全过
│   ├── 02_speedup_probe/
│   │   ├── probe.py            # 速度探针脚本
│   │   └── REPORT.md           # ★M3速度报告
│   ├── 03_toy_seq/             # （M4 待写）
│   └── 04_curvature_ablation/  # （M5 待写）
├── pyproject.toml
├── README.md
└── HANDOFF.md                  # 本文件
```

★ = 接手时优先看的文件。

## 4. 环境配置

机器上原本没有 Python 和 git，都已装好：

```
Python 3.12.7    → C:\Users\admin\AppData\Local\Python312\python.exe
torch 2.13.0+cpu → pip（清华镜像）
numpy 2.5.2, scipy 1.18.0, pytest 9.1.1
git 2.46.0       → C:\Program Files\Git\bin\git.exe
```

**重要**：当前 shell 的 PATH 是安装前的快照，不包含 python/git。新开终端
才会自动加载 PATH。在当前会话里要用完整路径调用，或：

```bat
set PATH=%LOCALAPPDATA%\Python312;C:\Program Files\Git\cmd;%PATH%
```

pip 已配清华镜像（`C:\Users\admin\AppData\Roaming\pip\pip.ini`）。

```bat
:: 安装 crnn 为可编辑包（已装过，重装环境时需要）
cd D:\code\cr-nn
python -m pip install -e .
```

## 5. 常用命令

```bat
:: 跑全部测试（20个，~5秒）
python -m pytest experiments/01_unit_tests -q

:: 跑速度探针（~3分钟，扫 p=3..29）
python experiments/02_speedup_probe/probe.py

:: git 操作（新终端里 git 已在 PATH）
git status
git log --oneline
```

## 6. 已完成的里程碑

### M1 数学定型 ✅
- `docs/math.md`：CR 流形、算子、谱、核、微扰展开、损失、离散化
- `docs/assumptions.md`：假设 A1-A4 + 风险 R1-R7
- `docs/references.bib`：文献含稳固性标注
- 三路并行调研修正了原始设想的 4 处过度声明（见 README 叙事约束表）

### M1 项目骨架 + 环境 ✅
- Python 3.12 + torch + numpy/scipy/pytest 装好
- `crnn/` 包可编辑安装，导入正常

### M2 核心算子 ✅（20/20 测试通过）
- 后端抽象层 `backend.py` + `TorchBackend`
- 几何算子：群结构、水平向量场、∂̄_b、Δ_b、Korányi/Szegő 核（闭式）
- 谱：Hermite-Laguerre 基、子拉普拉斯谱值 (2k+n)|λ|
- **矩阵值群 Fourier 变换 + 群卷积**（素数 p，往返/Parseval/卷积 1e-14 精度）
- 测试：群结构(5) + 核(2) + 谱(2) + FFT/卷积(4)

### M3 CR-Attention + 速度验证 ✅
- **速度探针**：crossover 在 N≈10K，p=29(N=24389) 时 CR 快 2.18x。
  早期止损点**未触发**，"速度加快"假设在长序列下成立。
  报告：`experiments/02_speedup_probe/REPORT.md`
- CR-Attention 层：Szegő投影 → ∂̄_b门控 → 复值混合，前向/反向通过
- CR-FFN + CR-Block：drop-in 替代 Transformer 层
- CR-Sobolev 损失：可微，∂̄_b 能量对 CR 函数（常数）为零
- 测试：层(3) + 损失(4)

## 7. 速度探针关键数据（M3 核心结论）

| p | N=p³ | softmax (ms) | CR-vec (ms) | speedup |
|---|------|-------------|-------------|---------|
| 11 | 1,331 | 12 | 37 | 0.33x |
| 19 | 6,859 | 392 | 537 | 0.73x |
| **23** | **12,167** | **1,765** | **1,453** | **1.22x** ✅ |
| **29** | **24,389** | **8,551** | **3,922** | **2.18x** ✅ |

- 短序列（N<10K）CR 没优势——常数因子大，softmax 的 N² 还小
- 长序列（N>10K）CR 胜出，优势随 N 增长
- 这是**下界**：CR-vec 未完全优化，且未用 Diaconis–Rockmore 快速路径（O(p³log p)）

## 8. 下一步：M4 玩具序列质量对比

M4 是质量验证——除速度外另一个核心假设的检验。

**目标**：CR-NN 在玩具序列任务上 ppl 与同规模 Transformer 持平（±10%）。

**计划**：
1. 搭玩具自回归 LM（字符级，~1M 参数），放 `crnn/models/toy_lm.py`
2. 合成序列数据集（可控复杂度：周期/嵌套/长程依赖），放 `experiments/03_toy_seq/`
3. 三组对比（曲率扰动是 M5 占位，M4 先比前两组）：
   - 标准 Transformer（`nn.TransformerEncoderLayer`）
   - CR-NN（平坦，M=0）
   - （M5）CR-NN（平坦+曲率，M=2）
4. 指标：ppl、训练曲线、推理延迟

**注意点**：
- CR-Attention 在短序列慢（常数因子），M4 要么用长序列（p≥11, N≥1331），
  要么接受训练慢但看质量趋势
- CR-Block 已设计为 Transformer block 的 drop-in 替代，模型结构可直接套用
- CR-Sobolev 损失当前是回归形式；LM 任务需要它和交叉熵配合（见下方 TODO）

## 9. 已知 TODO / 待办

### M4 即将处理
- [ ] `crnn/models/toy_lm.py`：玩具 LM 模型（embedding + N×CRBlock + LM head）
- [ ] CR-Sobolev 损失与交叉熵的配合：当前 `CRSobolevLoss` 是纯回归形式，
      LM 任务需要先用 CE 算 logits 损失，CR-Sobolev 作为表示空间正则。
      可能需要 `CombinedLoss(ce_weight, cr_weight, mu)`
- [ ] 合成序列数据集生成器
- [ ] 训练循环 + 评估脚本

### M5（曲率消融，待 M4 通过后）
- [ ] `crnn/curvature/perturbation.py`：实现截断微扰 L_j = Δ_b^j
- [ ] `crnn/layers/cr_attention.py` 的 `_apply_perturbation` 当前是
      `NotImplementedError`，需填实
- [ ] 对数修正项（R1）单独标记 + 开关
- [ ] 扫 M=0,1,2,3 与 ε 幅度

### M6（MLX 后端）
- [ ] `crnn/backends/mlx_backend.py`：复用 `mlx.core.fft` + complex64
- [ ] 仅 Korányi 核复幂/奇点处用 `mx.fast.metal_kernel` + 手写 vjp
- [ ] 双后端数值一致性测试（`tests/`）

### 工程改进（非阻塞）
- [ ] `heisenberg_fft.py` 的 inverse 仍用 numpy 循环（参考路径）；
      生产路径在 `cr_attention_backend.py` 已向量化
- [ ] `_solution_fft.py` 是子代理写的独立参考实现，保留作对照，不纳入包
- [ ] CR-Attention 的 Szegő 核缓存 `_kernel_cache` 是类变量，多进程训练
      可能有问题（改实例变量或用 lru_cache）

## 10. 调试经验（避免重复踩坑）

1. **cmd 下单行 Python 复杂脚本会静默失败**：多行 `-c` 脚本在 Windows cmd
   下经常没输出。写成 `.py` 文件再跑，可靠得多。

2. **变量名 `b` 冲突**：backend 变量常叫 `b`，但循环里也常用 `b` 作 b 轴
   索引。`heisenberg_ifft` 里因此出过 bug（`b` 被循环覆盖成 int）。当前
   代码用 `bb` 避开，但接手写新代码时注意。

3. **einsum 输出下标不能重复**：`"blau,lbcu->babc"` 会报错（b 既是 batch
   又是 b 轴）。用 `z` 表示 batch：`"zlau,lbcu->zabc"`。

4. **numpy gather 维度要对齐**：`take_along_axis` 的 index 必须和数组同
   维数。复杂 gather 用 meshgrid 构建索引更清晰。

5. **Szegő 核奇异性正则**：`w + eta` 不够（当 |w|<<eta）。正确做法是
   `conj(w)^{n+1} / (|w|² + eta)^{n+1}`，保持相位、软化模长。见
   `cr_attention.py` 的 `_get_szego_kernel`。

## 11. 文献锚点

最关键的几篇（完整列表在 `docs/references.bib`）：

- **Thangavelu 1998**：$\mathbb{H}^n$ 调和分析教材，谱 $(2k+n)|\lambda|$ 的来源
- **Folland 1975**：Korányi 核闭式 $c_n$，$\Delta_b$ 基本解
- **Diaconis & Rockmore 1990**：有限群 FFT，$O(N\log N)$ 的依据
- **Barilari arXiv:1105.1285**：子 Riemannian 热核微扰展开（混合方案依据，已验证）
- **Tao "finite uncertainty principle"**：Schrödinger 表示的清晰讲解

---

## 接手检查清单

接手时按顺序确认：
1. `python -m pytest experiments/01_unit_tests -q` → 20 passed
2. 读 `docs/math.md` §7（离散化 + 群法则）和 `docs/assumptions.md` R7（素数 p）
3. 读 `experiments/02_speedup_probe/REPORT.md`（M3 速度结论）
4. 看 `crnn/layers/cr_attention.py` 的 forward 流程
5. 从 M4 TODO 开始
