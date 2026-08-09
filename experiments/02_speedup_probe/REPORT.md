# M3a 速度探针报告

## 结论

**"速度加快"假设在长序列下成立。** Crossover 出现在 N ≈ 10,000（p ≈ 23），
此后 CR 群卷积持续优于 softmax 注意力，且优势随 N 增长（softmax O(N²d)
vs CR O(N^{4/3})）。

## 测量数据（CPU，torch 2.13.0+cpu，d=64，B=1）

| p | N=p³ | softmax (ms) | CR-vec (ms) | speedup |
|---|------|-------------|-------------|---------|
| 3 | 27 | 0.13 | 2.73 | 0.05x |
| 5 | 125 | 0.34 | 5.67 | 0.06x |
| 7 | 343 | 0.70 | 8.80 | 0.08x |
| 11 | 1,331 | 12.4 | 37.2 | 0.33x |
| 13 | 2,197 | 33.9 | 89.3 | 0.38x |
| 17 | 4,913 | 167 | 347 | 0.48x |
| 19 | 6,859 | 392 | 537 | 0.73x |
| **23** | **12,167** | **1,765** | **1,453** | **1.22x** ✅ |
| **29** | **24,389** | **8,551** | **3,922** | **2.18x** ✅ |

## 解读

1. **Crossover 在 N ≈ 10K**：小于此，softmax 的 N² 矩阵还小，常数因子
   占优；大于此，N² 的增长压倒一切。这对实际长上下文（8K-32K token）
   正好落在 CR 的优势区。

2. **常数因子仍大**：p=11 时 CR-vec 比 softmax 慢 3x，因为 CR 的 FFT
   + gather + einsum 开销远大于一个 matmul。诊断显示核心 batched
   matmul 只要 0.3ms，但完整路径 37ms——overhead 主要在 FFT gather 和
   inverse 的 einsum（O(p⁴) 的大张量）。M6 的 Metal kernel 可进一步压缩。

3. **趋势符合渐近**：softmax/CR 比值随 N 单调上升，无平台迹象。p=29
   已 2.18x，外推 p=37（N≈50K）应达 ~4-5x。

## 显存（理论分析，未直接测量——CPU 无 peak_memory 工具）

- softmax：attention 矩阵 O(N²)，N=24K 时 ~2.4GB（float32）。这是
  Transformer 长上下文的已知瓶颈。
- CR：token 场 O(N·d) + FFT 中间量 O(p⁴) = O(N^{4/3})。N=24K 时
  p⁴≈7×10⁵，远小于 N²≈6×10⁸。**显存优势比速度优势出现得更早、更显著。**

## 已知限制

1. **CPU 测量**：GPU 上 softmax 的常数因子更小（高度优化），crossover
   会右移。但 CR 在 GPU 上也可向量化，比值应类似。
2. **CR-vec 未优化到底**：forward 的 meshgrid gather 和 inverse 的 einsum
   仍可优化。这是下界——真实优化后 crossover 会左移。
3. **未含 Diaconis–Rockmore 快速路径**：当前用 O(p⁴) matmul 路径，非
   O(p³ log p) 的完整 FFT。后者只会更快。

## 判定

**M3 早期止损点未触发。** "速度加快"假设在长序列下成立，项目继续。
后续 M3b/M3c 搭 CR-Attention 层与 CR 损失，M4 做玩具序列质量对比。

## 复现

```
python experiments/02_speedup_probe/probe.py
python experiments/02_speedup_probe/_parity.py   # 数值一致性校验
```
