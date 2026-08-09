# CR-NN 数学定型

本文档钉死 CR-NN 项目使用的全部数学定义，作为实现的唯一参照。任何实现中的歧义以本文档为准；本文档的修正须经显式版本化。所有断言的稳固性分级见 `assumptions.md`。

> **符号约定**：除非另注，$n$ 为 CR 复维度，$Q=2n+2$ 为齐次维（homogeneous dimension），$\mathbb{H}^n$ 为 $(2n+1)$ 维 Heisenberg 群。

---

## 1. CR 流形：Heisenberg 群 $\mathbb{H}^n$

### 1.1 群结构与坐标

Heisenberg 群 $\mathbb{H}^n = \mathbb{C}^n \times \mathbb{R}$，元素 $g=(z,t)$，$z=x+iy\in\mathbb{C}^n$，$t\in\mathbb{R}$。群乘法

$$
(z,t)\cdot(z',t') = \left(z+z',\ t+t' + 2\,\mathrm{Im}(z\cdot \bar{z}')\right)
$$

其中 $z\cdot\bar z'=\sum_{j=1}^n z_j\overline{z'_j}$。在实坐标 $(x,y,t)$ 下即

$$
(x,y,t)\cdot(x',y',t')=(x+x',\ y+y',\ t+t' + 2(x\cdot y' - y\cdot x'))。
$$

**为什么用它**：$\mathbb{H}^n$ 是最简单的非阿贝尔幂零李群，其边界 CR 结构正是 $\mathbb{C}^{n+1}$ 中 Siegel 上半空间边界。它是"平坦"CR 流形的原型——Webster 曲率恒为零，子拉普拉斯有闭式谱。

### 1.2 水平分布与复结构

切空间的水平子分布由以下左不变向量场张成：

$$
X_j = \partial_{x_j} + 2y_j\,\partial_t,\qquad
Y_j = \partial_{y_j} - 2x_j\,\partial_t,\qquad j=1,\dots,n.
$$

中心方向 $T=\partial_t$ 不属于水平分布（这恰是子椭圆性之源）。非交换括号

$$
[X_j,Y_j] = -4\,\partial_t = -4T
$$

满足 Hörmander 横截条件，故由 $X_j,Y_j$ 生成的子拉普拉斯是亚椭圆（hypoelliptic）而非椭圆。

**复结构** $J$ 定义在水平分布上：

$$
J X_j = Y_j,\qquad J Y_j = -X_j.
$$

记 $Z_j = \tfrac{1}{2}(X_j - iY_j)$（$(1,0)$ 型），$\bar Z_j=\tfrac{1}{2}(X_j+iY_j)$（$(0,1)$ 型）。

### 1.3 齐次维

伸缩 $\delta_r(z,t)=(rz, r^2 t)$。在此下 $\mathbb{H}^n$ 的齐次维

$$
Q = 2n + 2,
$$

而非拓扑维 $2n+1$。$Q$ 出现在所有核的齐次性中（如 Korányi 核的 $-Q$ 次齐性）。

---

## 2. CR 算子

### 2.1 tangential Cauchy–Riemann 算子 $\bar\partial_b$

$$
\bar\partial_b f = \sum_{j=1}^n \bar Z_j f\ d\bar z_j
   = \tfrac{1}{2}\sum_{j=1}^n (X_j + iY_j)f\ d\bar z_j.
$$

**CR 函数**：$\bar\partial_b f = 0$。这是 $\mathbb{H}^n$ 上"全纯"的内蕴概念。形式伴随记 $\bar\partial_b^*$。复合给出 Kohn 子拉普拉斯（见 2.3）。

> **项目语义**：训练中推动表示趋向 CR 函数等价于把信息压缩到全纯（复）子空间——这是"信息汇总到复维度"的几何载体（假设 A3）。

### 2.2 水平向量场的离散实现

连续向量场 $X_j,Y_j$ 在实现时取两种等价表示之一（实现时择一，记录在算子实现注释中）：

- **有限差分**：在 $(x,y,t)$ 离散格点上，$X_j f \approx [f(\cdot+2y_j\Delta t)-f(\cdot-2y_j\Delta t)]/(2\Delta x)$ 类的中心差分。简单但破坏精确谱结构。
- **谱表示（首选）**：通过中心方向 Fourier 变换后，$X_j,Y_j$ 在每个 $\lambda$ 层化为乘以 $i\xi_j$、$\partial_{z_j}$ 等，可精确作用于 Hermite 基。**实现默认走谱表示**，以保证单元测试中 $\Delta_b$ 作用于特征函数回吐谱值。

### 2.3 子拉普拉斯 $\Delta_b$（Kohn sub-Laplacian）

$$
\Delta_b = -\sum_{j=1}^n (X_j^2 + Y_j^2) = -2\sum_{j=1}^n (Z_j\bar Z_j + \bar Z_j Z_j).
$$

（等价地 $\Delta_b = 2\,\bar\partial_b^*\bar\partial_b$ 在适当规范下；具体常数因子在 `operators.py` 中固定并测试。）

**亚椭圆非椭圆**：$\Delta_b$ 不含 $\partial_t^2$，但因 Hörmander 条件仍亚椭圆，正则性提升 $1/2$ 阶而非 $1$ 阶。

---

## 3. 谱结构（平坦模型，最稳固的部分）

### 3.1 中心 Fourier 分解

对 $t$ 做 Fourier 变换 $f\mapsto\hat f(\cdot,\lambda)$，参数 $\lambda\in\mathbb{R}\setminus\{0\}$。子拉普拉斯约化为一族标度谐振子：

$$
\widehat{\Delta_b f}(\,\cdot\,,\lambda) = \mathcal L_\lambda\,\hat f(\,\cdot\,,\lambda),\qquad
\mathcal L_\lambda = -\Delta_{\mathbb{R}^{2n}} + \lambda^2|x|^2 + i\lambda\,(x\cdot\nabla_y - y\cdot\nabla_x).
$$

即 $|\lambda|$ 尺度的 Hermite 算子加复相位项。

### 3.2 谱值

$\mathcal L_\lambda$ 的特征值（**本项目核心谱公式**）

$$
\boxed{\;\sigma_{k,\lambda} = (2k+n)\,|\lambda|,\qquad k=0,1,2,\dots\;}
$$

特征函数为 rescaled Hermite 函数（按 $\lambda$ 缩放、带特殊 Hermite–Laguerre 角向结构）。$n$ 重简并（在 $j=1,\dots,n$ 上）。

> 这是"平坦模型可快速计算"的全部数学依据。所有谱运算回到对角化这一步。

### 3.3 截断

非紧 $\mathbb{H}^n$ 上谱在 $\lambda$ 连续、$k$ 离散。实现中：

- $\lambda$：离散化为 $L$ 个频率点（中心方向 FFT 自动给出）。
- $k$：截断到前 $K$ 个模（$k=0,\dots,K-1$），低频能量近似。
- 截断误差 $O(e^{-cK})$（Hermite 函数速降），记录在 `spectrum.py`。

> 紧替代：若发现非紧截断引发问题，可切换到球面 $S^{2n+1}$ 上的离散谱。记录为备选方案。

---

## 4. 核函数（闭式）

### 4.1 Korányi 核

$$
\Gamma(g) = c_n\,\rho(g)^{-Q},\qquad
\rho(z,t) = \bigl(|z|^4 + t^2\bigr)^{1/4},
$$

$c_n$ 由 Folland (1975) 给出使 $\Delta_b\Gamma=\delta_0$。$\rho$ 为 Korányi–Cygan 距离的范数。这是 $\Delta_b$ 的基本解，$-Q=-(2n+2)$ 次齐性。

更一般 Riesz 核 $I_\alpha(g)=C_{n,\alpha}\,\rho(g)^{\alpha-Q}$。

### 4.2 Szegő 投影核（平坦闭式）

在 Siegel 上半空间边界（即 $\mathbb{H}^n$ 的 CR 结构）上，Szegő 投影 $\Pi$ 的核

$$
S(g) = c_n'\,\bigl(|z|^2 - i t\bigr)^{-(n+1)}.
$$

（注意：此处 $|z|^2-it$ 为复值，幂次取主支；与 Korányi 核 $|z|^4+t^2$ 的关系是 $|z|^2-it$ 的模长平方 $=|z|^4+t^2$。）

**用途**：Szegő 投影是到 CR 全纯函数的正交投影——即"去掉非全纯部分"。这正是 CR-Attention 层中信息聚合的内蕴算子。

### 4.3 数值注意事项（实现必读）

- **奇异性**：$g\to 0$ 时 $\Gamma\sim\rho^{-Q}$、$S\sim\rho^{-2(n+1)}$ 发散。实现中加正则 $\rho^2\to\rho^2+\eta$，$\eta$ 为可调小正数（默认 $10^{-6}$）。
- **复幂 branch cut**：$(|z|^2-it)^{-(n+1)}$ 取主支对数。两端后端 `log`/`pow` 的支点约定须显式对齐，否则梯度爆炸。
- **卷积加速**：$S*f$ 走 FFT 卷积（群结构使之为群卷积）降至 $O(N\log N)$。**不要**做朴素 $O(N^2)$。

---

## 5. 微扰展开：混合方案

### 5.1 定位（重要）

本项目采用"平坦快速变换 + 可学曲率扰动"的混合方案。经调研核实，其正确数学定位为：

> **截断微扰展开**，非"低秩修正"。曲率扰动一般是满秩的局部 symbol 扰动；不存在"曲率 = 低秩"的文献依据。离开平坦模型后，精确 $O(N\log N)$ 谱变换不再存在；本方案的"快速"来自截断近似 + 学习项补偿，误差可控但非零。

理论依据：子 Riemannian 热核 / Szegő 核的小曲率微扰展开（Barilari arXiv:1105.1285；Boutet de Monvel–Sjöstrand FIO 渐近）。

### 5.2 形式

设 $S_\text{flat}$ 为 4.2 的平坦 Szegő 投影。弯曲伪厄米流形上的 Szegő 投影近似为

$$
S_\text{curved} \;\approx\; S_\text{flat} \;+\; \sum_{j=1}^{M} \varepsilon^j\, L_j[S_\text{flat}],
$$

其中
- $M$ 为截断阶（超参，默认 $M=2$）；
- $\varepsilon\in\mathbb{R}$ 为可学小参数（曲率幅度，初始化 $0$，软约束 $|\varepsilon|<\varepsilon_\text{max}=0.1$）；
- $L_j$ 为第 $j$ 阶 transport 算子，其 symbol 由 Webster 曲率不变量决定。在实现中 $L_j$ 取参数化形式（一组可学系数作用在 $\Delta_b$ 的低阶 symbol 上），并非精确几何 transport——这是"学习项吸收截断余项"的工程化。

### 5.3 误差与风险

- 截断误差形式上 $O(\varepsilon^{M+1})$，由 $L_j$ 的可学参数吸收。
- **已知未解决风险**（记录在 `assumptions.md`）：
  - 子 Riemannian 特有的**对数修正项** $\log\rho$ 可能落在截断阶上，不能简单当幂次项；实现中单独标记并可选关闭。
  - **特征值重数跳变**：小曲率下谱连续但简并可分裂；若快速变换依赖 Hermite 重数结构，须在训练中容许。
  - **全局 vs 局部频段**：微扰展开局部（小时间/高频）成立；低频大尺度行为偏离更大，需评估网络工作频段。

---

## 6. CR-Sobolev 损失

### 6.1 定义

以子拉普拉斯谱定义的 Sobolev 范数（最稳固的损失组件）：

$$
\|f\|^2_{S_b^s} = \sum_{k,\lambda} (1 + \sigma_{k,\lambda})^{s}\, |\hat f_{k,\lambda}|^2,
$$

其中 $\hat f_{k,\lambda}$ 为 $f$ 在 $\Delta_b$ 特征基下的系数，$\sigma_{k,\lambda}=(2k+n)|\lambda|$。等价地 $\|f\|_{S_b^s}=\|(I+\Delta_b)^{s/2}f\|_{L^2}$。

### 6.2 训练损失

$$
\boxed{\;\mathcal L_\text{CR} = \|\mathrm{out}-y\|^2_{S_b^s} + \mu\,\|\bar\partial_b\,\mathrm{out}\|^2\;}
$$

- 第一项：CR-Sobolev 回归/匹配（替代欧氏 $\|\cdot\|^2$），按频率加权，低频（信息核）权重高。
- 第二项：$\bar\partial_b$ 能量正则，推动输出趋向 CR 函数 = 信息压缩到全holomorphic子空间。
- $\mu>0$ 为正则强度（超参），$s$ 为 Sobolev 阶（默认 $s=1$）。

### 6.3 与欧氏损失的关系

欧氏损失是 $s=0,\mu=0$ 的退化情形。CR-Sobolev 在谱上加权，等价于对表示施加"频域先验"——与 Transformer 中的权重衰减作用域不同（后者在参数空间，前者在表示空间）。

---

## 7. 离散化与序列嵌入

### 7.1 序列 → $\mathbb{H}^n$ 格点

序列长度 $N$ 映射到有限 Heisenberg 群格点。默认 $n=2$，取 $p\times p\times p$ 格点 $p\approx N^{1/3}$（向上下取整，不足补零）。

格点坐标 $(x_a,y_b,t_c)$，$a,b,c\in\mathbb{Z}/p\mathbb{Z}$，对应有限 Heisenberg 群 $H_p$（$3\times 3$ 上三角矩阵群，阶 $p^3$）。其上存在 $O(p^3\log p)=O(N\log N)$ 的 FFT（Diaconis–Rockmore 1990）——这是"速度加快"的工程依据。

> **关键约束（v0.2 修正）**：$p$ **必须是素数**。对合数 $p$，Schrödinger 表示 $\rho_\lambda$ 在 $\gcd(\lambda,p)>1$ 时可约，且整个 $\{\rho_\lambda:\lambda=1..p{-}1\}$ 族**不完整**——缺失低维不可约表示，$\sum d_\rho^2 < |G|$，导致 Schur 正交性、Plancherel、反演公式全部失效（实测约 5% 能量泄漏，往返误差不为零）。
>
> 素数 $p$ 下，$\{\rho_\lambda:\lambda=1..p{-}1\}$（p 维）$\cup$ $\{\chi_{m,n}:m,n\in\mathbb{Z}_p\}$（p² 个 1 维字符，$\lambda=0$ 扇区）构成完整不可约表示集，$\sum d_\rho^2=(p{-}1)p^2+p^2=p^3=|G|$，一切成立。
>
> 工程含义：格点分辨率取邻近素数（3, 5, 7, 11, 13, ...）。几何/核/谱理论不变，仅格点间距不同。若必须用合数 $p$（如 $p=2^k$），需手工构造缺失的射影不可约表示，无统一简洁公式，本项目不支持。

### 7.2 群法则与表示（实现定型）

实现采用的群法则（使 Schrödinger 表示成为同态的版本，单向交换子，系数 1）：

$$(a_1,b_1,c_1)\cdot(a_2,b_2,c_2)=(a_1{+}a_2,\ b_1{+}b_2,\ c_1{+}c_2{+}a_1 b_2)\ \bmod p$$

逆元：$(a,b,c)^{-1}=(-a,\,-b,\,-c{+}ab)\bmod p$。

> **注意**：这与连续 $\mathbb{H}^n$ 的对称形式 $t{+}t'{+}2\,\mathrm{Im}(z\bar z')$（系数 2，双向）是**同构但不同的参数化**。实现统一采用上述单向形式；连续核（Korányi/Szegő）在格点上求值时按此群法则解释坐标。文档其他处若写连续形式，仅为几何直觉，实现以此处为准。

Schrödinger 表示（$\omega=e^{+2\pi i/p}$）：

$$[\rho_\lambda(a,b,c)]_{u,v}=\omega^{\lambda c}\,\omega^{\lambda b u}\,\delta_{v,(u+a)\bmod p}$$

群 Fourier 变换：标量 $f\mapsto$ 矩阵值 $\hat f$，
- $\lambda=0$ 扇区：$\hat f(0;m,n)=\sum_{a,b,c}f(a,b,c)\,\omega^{ma+nb}$（p² 个标量，c 不可见）
- $\lambda=1..p{-}1$ 扇区：$\hat f(\lambda)_{u,v}=\sum_{a,b,c}f(a,b,c)\,[\rho_\lambda(a,b,c)]_{u,v}$（p×p 矩阵）

反演（Plancherel，$|G|=p^3$，$d_\rho=p$ 对矩阵扇区，$d=1$ 对字符扇区）：

$$f(a,b,c)=\frac{1}{p^3}\Big[\sum_{m,n}\hat f(0;m,n)\,\omega^{-ma-nb}+\sum_{\lambda=1}^{p-1}p\sum_{u,v}\hat f(\lambda)_{u,v}\,\overline{[\rho_\lambda(a,b,c)]_{u,v}}\Big]$$

群卷积定理：$(f*g)\widehat{}(\lambda)=\hat f(\lambda)\,\hat g(\lambda)$（矩阵乘法，$\hat f$ 在左），$(f*g)\widehat{}(0;m,n)=\hat f(0;m,n)\cdot\hat g(0;m,n)$（逐点）。朴素 $O(p^6)$ 降为 $O(p^4)$（p−1 个 p×p 矩阵乘 + 2D FFT）。

### 7.2 token 嵌入

特征维 $d$ 映射到 $\mathbb{H}^n$ 格点上的 $d$ 通道复值场 $f:G_p\to\mathbb{C}^d$。嵌入层为可学线性映射 $\mathbb{R}^{d_\text{tok}}\to\mathbb{C}^{d\times p^3}$，按位置展开。

---

## 8. softmax 注意力的几何类比（仅作叙事，不作数学等价）

经调研核实：

- softmax 核 $e^{q\cdot k}$ 在结构上类比 Bargmann 再生核 $e^{z\bar w}=\sum(z\bar w)^m/m!$，均为指数型核。
- 但**"softmax = Bargmann 相干态核的离散截断"不是已证等式**：严格对应要求 query/key 视为复共轭对，标准 Transformer 不满足。
- **本项目叙事**：以 CR Szegő 投影作为注意力的几何替代（而非"softmax 的几何解释"）。两者都是"核回归式信息聚合"，但 CR 聚合的内蕴核来自流形谱结构，softmax 来自内积。这是"替代"而非"重解释"。

> 论文与文档中**禁止**宣称 softmax 与 Bargmann 核等价；仅可作结构类比引言。

---

## 版本

- v0.1 (2026-08-09)：初版数学定型，基于三路调研修正。
