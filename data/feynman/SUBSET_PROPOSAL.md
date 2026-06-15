# Feynman 子集候选清单（T-20 的待拍板决策）

**状态**：candidates —— 需要用户拍板后才生成数据与 loader。

**相关**：`ROADMAP.md` T-20 Notes —— "先出候选清单附理由再拍板"。

---

## 1. 数据源选择

**拟用**：**AI Feynman FSRD v1**（Udrescu & Tegmark 2020，[space.mit.edu/home/tegmark/aifeynman.html](https://space.mit.edu/home/tegmark/aifeynman.html)）—— 100 条 Feynman 讲义公式 + 每变量采样区间的 `bosons.csv`。这是 SRBench 2021（Cava et al. NeurIPS datasets）官方使用的数据源，也是本领域现代 benchmark 的事实标准。

**未选方案及原因**：

| 候选 | 不选的原因 |
|------|-----------|
| TPSR `upstream/pse/baselines/algorithms/TPSR/.../feynman.csv` | 是 TPSR 论文自造的小子集（形如 `sin(x_3) + sin(x_3/(x_1-x_2))`），不是 Feynman 物理公式，仅用于 TPSR 比较。**跟 Feynman 名号无关** |
| DGSR `upstream/pse/baselines/algorithms/DGSR/datasets/feynman_benchmark.csv` | 格式 `fn-0, 9.0, x3*x4*x5/…` 看起来像 DGSR 自己的简化版。缺官方 ID（`I.X.Y`），没对齐 SRBench，不方便跟文献里的结果对齐 |
| SRBench `pmlb/feynman_*` 包 | 已经是 FSRD 的衍生包；我们直接用 FSRD 源数据等价，且少一层依赖 |

---

## 2. 选择规则（需拍板）

下面三条规则一起过滤，从 100 条 → 预期 15–25 条。

### R1 —— 数据变量数 ≤ 4

为什么：HPEC 首篇的 benchmark 需要 compute 成本可控，同时保留异构 K 分布。FSRD 里 n=5+ 的公式常是 9-变量引力、6-变量内积等，对我们 bench 性能差异没有额外贡献。

### R2 —— 去除 op 集合不支持的公式

HPEC MVP 只承诺 ~16 个光滑 op（`ADD/SUB/MUL/DIV/POW/SIN/COS/TAN/SINH/COSH/TANH/LOG/EXP/SQRT/NEG/INV`，见 `gpu_lm_direction_zh.md` §5.2）。FSRD 里出现的**会被排除**的 op：

- `arcsin` / `arccos` / `arctan`（反三角）—— 排除 I.26.2、I.30.5 等
- 分段 / `|x|` / `floor` —— FSRD 里极少，几乎不触发
- 特殊函数（Γ、Bessel）—— FSRD 里没有

这条是**硬规则**。

### R3 —— 采样区间避开奇点

原始 FSRD 每变量有区间（例如 `q1: [1,5]`）——我们直接沿用。拒绝：

- 分母变量跨 0 —— FSRD 已避开（`r: [1,5]` 而不是 `[-5,5]`），不用再筛
- `log(x)` 的 `x` 跨 0 —— FSRD 的 log 对应变量都 `[1,5]`，安全
- `sqrt(x)` 的 `x` 负值 —— FSRD 安全

预期 R3 0 条额外排除。

### R4（可选，让用户拍板）—— 参数化后的**常数数量 ≤ 4**

FSRD 公式本身**没有自由常数**（全是符号变量）。我们要做的 NLS-for-constants bench 需要把公式改写成骨架 `skeleton + c`。**参数化方案单独决策**（见 §4），但不管选哪种方案，常数数量都会受限。R4 预留一位——如果某公式参数化后超过 4 个 c，从候选里 drop。

---

## 3. 候选清单（预拍板）

以下按**变量数升序** + **简单度优先**，给出 ≈20 条候选。每条都满足 R1–R3。

| FSRD ID | 公式（原始） | #vars | 备注 / 采样区间要点 |
|---------|------------|-------|---------------------|
| **1 var** | | | |
| I.6.2a | `exp(-θ²/2)` | 1 | θ∈[1,3]；干净 exp |
| **2 vars** | | | |
| I.6.2 | `exp(-(θ/σ)²/2) / (√(2π)·σ)` | 2 | σ∈[1,3], θ∈[1,3]；Gauss |
| I.12.1 | `μ·Nn` | 2 | 纯乘；但无可拟合常数 —— **用来 sanity check 参数化 scheme**，本身意义有限 |
| I.12.5 | `q2·Ef` | 2 | 同上 |
| I.14.4 | `0.5·k·x²` | 2 | 有 literal `0.5` 可做 c，经典 PE |
| I.25.13 | `q / C` | 2 | 纯除 |
| I.29.4 | `ω / c` | 2 | 纯除 |
| I.34.27 | `ℏ·ω` | 2 | 纯乘（光子能量）|
| II.8.31 | `ε·Ef² / 2` | 2 | literal `2`, `0.5` 可做 c |
| II.11.28 | `1 + n·α/(1 - n·α/3)` | 2 | **有 literal `1`, `3`**；有理式 |
| II.27.18 | `ε·Ef²` | 2 | 纯乘|
| **3 vars** | | | |
| I.12.2 | `q1·q2 / (4π·ε·r²)` *(4 vars 原始，此处视 4π·ε 为 1 个 c)* | 3（降维参数化后）| literal `4π` → c |
| I.14.3 | `m·g·z` | 3 | 纯乘 |
| I.18.4 | `m1·r1 / (m1+m2)` | 3 | 分母不跨 0 |
| I.18.12 | `r·F·sin(θ)` | 3 | 含 sin |
| I.26.2 | ~~`arcsin(n·sin(θ2))`~~ | — | **R2 排除（arcsin）**，仅列此作透明 |
| I.27.6 | `d1·d2 / (d1 + d2·n)` | 3 | 有 literal 无 |
| I.34.8 | `q·v·B / p` | — | **4 vars，归到 4 vars 组** |
| I.39.22 | `p_F / (γ-1)` | 2 | literal `1` |
| I.43.31 | `μ·kB·T` | 3 | 纯乘 |
| II.2.42 | `κ·(T2-T1)·A / d` | 4 | 归到 4 vars 组 |
| II.8.7 | `0.6·q² / (4π·ε·d)` | 3 | literals `0.6, 4π` |
| II.15.4 | `-μ·B·cos(θ)` | 3 | 含 cos + 负号 |
| II.15.5 | `-p_d·Ef·cos(θ)` | 3 | 含 cos |
| II.24.17 | `√(ω²/c² - π²/d²)` | 3 | 含 sqrt；需确认 ω/c > π/d（FSRD 区间已保证）|
| II.27.16 | `ε·c·Ef²` | 3 | 纯乘 |
| II.34.2 | `q·v / (2π·r)` | 3 | literal `2π` |
| II.38.3 | `Y·x / d` | 3 | 纯乘除 |
| III.4.32 | `1 / (exp(ℏω/(kB·T)) - 1)` | 3 | Bose-Einstein 占据；有 exp + 分母 - 1，需要严格 ℏω/kBT > 0 |
| III.7.38 | `2·μ·B / ℏ` | 3 | literal `2` |
| III.12.43 | `n·ℏ` | 2 | 纯乘 |
| III.15.14 | `ℏ² / (2·E·d²)` | 3 | literal `2` |
| **4 vars** | | | |
| I.8.14 | `√((x2-x1)² + (y2-y1)²)` | 4 | 距离公式；含 sqrt + pow；**无 literal**（除 `2`）|
| I.13.4 | `0.5·m·(v²+u²+w²)` | 4 | literal `0.5`；KE |
| I.15.10 | `m0·v / √(1-v²/c²)` | — | **3 vars**，归到 3 组 |
| I.16.6 | `(u+v) / (1 + u·v/c²)` | — | 3 vars（速度合成）|
| I.18.14 | `m·r·v·sin(θ)` | 4 | sin |
| I.34.8 | `q·v·B / p` | 4 | 纯乘除 |
| I.48.2 | `m·c² / √(1-v²/c²)` | — | 2 vars（相对论能量）|
| II.2.42 | `κ·(T2-T1)·A / d` | 4 | 差 + 纯乘除 |
| II.11.3 | `q·Ef / (m·(ω0²-ω²))` | 4 | 有理 |
| III.13.18 | `2·E_n·d²·k / ℏ` | 4 | 纯乘除 + literal |

**总计约 25 条**（含重复归类的，去重后约 22 条）。

---

## 4. 参数化方案（单独决策点）

候选 A：**literal-to-c**
- 把公式里所有数值常量（`0.5`, `2`, `π`, `4π`, `ℏ`, `kB` 等）替换成 `c_i`
- GT = 原数值
- 优点：机械可推广，不需要物理直觉；bench 里 FSRD 都自带这些常量
- 缺点：像 `I.12.1 μ·Nn` 这种没 literal 的公式就没 c 可拟合 → 要么排除，要么手动加一个 leading `c0`

候选 B：**leading-c**
- 不管原公式有无 literal，统一在 **最外层乘一个 `c0`**
- 例：`μ·Nn` → `c0·μ·Nn`，GT c0 = 1.0
- 例：`0.5·k·x²` → `c0·k·x²`，GT c0 = 0.5
- 优点：所有公式至少 1 个 c，bench 一致
- 缺点：把原公式里多个 literal 合成一个 —— 可能让问题变简单

候选 C：**literal-to-c + leading-c 保底**
- 有 literal 的按 A 做；无 literal 的 fallback 到 B 加一个 leading `c0`
- 优点：结合 A 的保真度 + B 的一致性
- 缺点：规则稍复杂

我的推荐：**C**（literal-to-c + leading-c 保底）。跟 Nguyen 的 "leading constant per term + GT=1" 风格保持一致；同时保留 FSRD 原公式里所有显式 literal 作 c。

---

## 5. 物理单位 / 采样区间来源

FSRD `bosons.csv` 每变量自带区间（例：`theta: [1,5]`, `sigma: [1,3]`）。我们**完全沿用**这些区间，每个公式生成 `n_samples = 1000`（SRBench 默认），clean + 1% + 5% 三版。

唯一例外：如果 FSRD 区间让**参数化后的 fit target 退化**（例如某条 `exp(...)` 的指数永远 ≈0 导致 y 永远接近 1），再单独调整。

---

## 6. 交付结构（拍板后生成）

```
experiments/003_nls_bench/data/feynman/
├── I.6.2.parquet            # feature columns: x0=sigma, x1=theta; y
├── I.14.4.parquet           # x0=k, x1=x; y
├── ...
└── ground_truth.yaml        # 每条公式的 skeleton_expr + GT constants + 变量名映射
```

Registry 里每条公式 3 个条目：`feynman/I.6.2` / `feynman/I.6.2-noisy-01pct` / `feynman/I.6.2-noisy-05pct`（复用 `bench/transform.py` 的 `gaussian_noise`）。

---

## 7. 需要用户拍板的事项

1. **R1 ≤4 变量**这条定死可以吗？有没有哪几条 5-var 是你觉得 HPEC track 一定要留（比如 Planck 公式 I.41.16 是 5 vars）？
2. **参数化方案 A/B/C**：默认推荐 C，能同意吗？
3. **采样数 n_samples=1000 per dataset** 行不行？Nguyen 1-8 是 20、9-12 是 100，Feynman 要拉到 1000 是因为 SRBench 标准；但也可以压到 100-200 省磁盘。
4. **噪声变体 1% / 5%** 跟 Nguyen 一致，还是换成 SRBench 的 "clean + 1e-3 relative + 1e-2 relative"？目前默认前者。
5. **具体哪 ~20 条**：§3 清单里有几条边界（比如 II.8.7 里的 `0.6` 是 `3/5` 的物理常数，要不要展开；I.12.1/I.12.5 无 literal 需靠方案 B fallback）。拍板前你要不要过一遍。

给个 go 我就开始动代码 + 下载 FSRD 原文件。

---

## 8. 预估工作量（拍板后）

- FSRD `bosons.csv` 下载 + 对齐 ID ⇒ ~1h
- 参数化 + 生成 20 条 parquet ⇒ ~2h（机械，参照 `bench/sources/synthetic.py` 模式）
- Feynman loader 写进 `bench/sources/`（类似 `synthetic.py` 的 `FeynmanSource`）⇒ ~2h
- 单测（形状、GT 一致性、noisy 变体）⇒ ~1h
- 更新 `data/README.md` 和 `registry.yaml` 索引 ⇒ ~30min

合计 **~6-7 h** wall-clock。
