# 009 — SR 符号发现 benchmark

构建一套**符号发现率(symbolic recovery)** benchmark,用来验证并展示
"能力解锁型"故事:GPU 批量二阶常数拟合(CO)让激进 memetic GP 跑得起,
配方(自适应阻尼 + 选择改造)把算力兑现成更高的发现率。

**为什么 / 完整理由**:见 [`../../discussion/memetic_sr_benchmark_direction_zh.md`](../../discussion/memetic_sr_benchmark_direction_zh.md)
(status: current, 2026-06-07)。本 README 只讲这套 benchmark 怎么构成、怎么用。

---

## 这是什么 / 不是什么

- **是**:"整条 GP 能不能**找回真公式的结构**"——symbolic recovery,对齐 SRBench。
- **不是**:`bench/` / `experiments/003_nls_bench` / `experiments/006_nls_benchmark`
  那一类——那些测的是"给定骨架,优化器拟合常数的质量/速度"(NLS optimizer quality),
  跟这里**不是一回事**,别混。
  - 注意(2026-06-07 勘查):`bench/` 主体虽是 optimizer-quality,但它**已经长出了
    符号恢复判定器**(`bench/sources/evogp.py` 的 `check_recovery*`)+ Feynman/Nguyen
    题集 + 造数 + 噪声。009 **复用**这些(import),不重写。

---

## 问题集由哪些组成

| 类别 | 是什么 | 作用 |
|---|---|---|
| **SRSD-Feynman**(主力) | ~120 条费曼物理公式,**物理真实采样区间**,易/中/难分档(Matsubara 2022,`omron-sinicx/srsd-benchmark`) | 可信、贴"科学发现"、是我们 CO 优势该显的地方(真实区间→常数更难拟) |
| **Korns**(机制放大镜) | 人造题,常数**藏在 sin/cos 频率、exp 衰减率内部**(Korns 2011) | 专门放大 CO 价值(内部常数,别人随机生成撞墙)。偏合成,小份额。已知 f11/f12 谁都解不出,当"公认难"参照不当战利品 |
| **Strogatz**(补覆盖) | 14 个非线性动力学 ODE(`lacava/ode-strogatz`) | SRBench 自带的另一类 ground-truth |
| **Nguyen / Koza**(阴性对照) | 多项式/三角,系数全整数、**无自由常数** | CO 在这儿不该帮上忙;若我们这儿也"赢"= 假信号。反证优势真来自 CO |

- **AI-Feynman(原版区间)不当第二套主力**(同 120 方程、只是区间不同,结构重复)。
  只在需要蹭 SRBench 公开 14-方法排行榜时**薄跑一层**(那张榜在原版区间上跑的)。
- 用 manifest 里的 `range_variant`(easy / realistic)字段承载"区间难度"——
  以后想做"我们的 CO 优势随区间变真实而变大"的消融,随时能加,现在不强制。

---

## 每道题记录什么(manifest schema,初稿)

不管来自哪一类,统一记:

```
id              问题唯一名(如 srsd_feynman_I.18.12 / korns_11 / nguyen_5)
source          srsd-feynman | korns | strogatz | nguyen | koza | feynman-srbench
true_expr       真表达式(sympy 语法)
n_vars          变量数
var_ranges      每个变量的采样区间
n_consts        常数个数 K
n_inner_consts  ★嵌在非线性函数内部的常数个数(我们最关心的分层轴)
difficulty      easy | medium | hard(来源若有就沿用)
range_variant   easy | realistic(同一方程的区间变体)
is_control      是否阴性对照(Nguyen/Koza = true)
known_ceiling   是否"公认谁都解不出"(如 Korns f11/f12)
notes           备注
```

数据生成器按 manifest 吐 `(X, y)`,跑完用 SRBench 的 SymPy 判定比对 `true_expr`。

---

## 指标与协议(对齐 SRBench)

- **主指标**:符号恢复率(SRBench 的 SymPy 判定:差或比化简成常数即算找回,
  容忍外层加/乘常数、`r2_test>0.5` 门控)。
- **连续副指标**:NED(SRSD 的归一化树编辑距离,对"部分接近 / 早熟锁死"灵敏)。
- **headline(贡献新轴)**:固定总算力下的**恢复率-算力曲线**。
- **诊断(机制)**:逐代 适应度区分度塌缩 / 结构多样性 / 种群 K 分布。
- **噪声**:γ ∈ {0, 1e-3, 1e-2, 1e-1}(按目标 RMS 缩放,SRBench 协议)。
- **种子**:25–30(exp007 证明 10 太吵)。
- **统计**:成对 Wilcoxon signed-rank + Bonferroni(SRBench 协议)。
- **复杂度**:化简后 运算符+特征+常数 个数,model-size vs 精度 Pareto。

构建方法学:**先全放进去跑总结果,再按事先声明的规则裁**(裁是为去掉
饱和/无区分度的题,不是挑对自己有利的)。

---

## 现状 / 下一步

**已有可复用件(2026-06-07 勘查)**:原计划第一块砖要写的判定器/造数/部分题目,
`bench/` 里已有且带 TDD,直接复用、不重写:
- 判定器 `cusr/bench/sources/evogp.py` 的 `check_recovery` / `check_recovery_numeric` /
  `check_recovery_composite`(SRBench 风格,容忍外层常数),测试在 `tests/bench/test_recovery.py`。
- 题集+造数:Feynman 98 题(`data/feynman/formulas.yaml`,
  AI-Feynman 原始区间)、Nguyen 12 题(`cusr/bench/sources/synthetic.py`);噪声 `cusr/bench/transform.py`。

**第一块砖(已完成,green)**:`seed_bench.py` + `test_seed_bench.py` —— 用 009 自己的
manifest 跑通 `entry → 造 (X,y) → 复用 composite 判定`。种子集 = Nguyen-1(阴性对照,
无内部常数)+ Korns-12(双内部频率)。14 个测试过,含关键判别:**内部频率差 ~7%
(9.8→10.5)判 not recovered,外层加/乘常数容忍**。

**判定器标定(已完成 2026-06-08)**:见 [`judge_calibration.md`](judge_calibration.md)。
落在 `judge.py`(`test_judge.py` 18 测试)——自定义统一相对容忍(`sig=3`,跨量级一致,堵住
SRBench 小常数过松+抹零的病理)+ 严格/宽松两列 + SRBench 可比列。

**题集扩展(已完成 2026-06-08)**:`problems.py`(`test_problems.py` 10 测试)+
`taxonomy.py`(`test_taxonomy.py` 16 测试)。共 **113 题**:
- **Feynman 主干 98 题**(复用 bench,AI-Feynman 原始区间,`range_variant=ai_feynman_original`)。
- **Nguyen 阴性对照 12 题**(复用 bench,无自由常数原版)。
- **Korns 放大镜 3 题**:korns_7(衰减率)、korns_11(频率+立方,known_ceiling)、korns_12(双频率);
  常数核自 arXiv:2412.02126。korns_8 **故意剔除**(它的 sqrt 内常数可被外层吸收,非真内部常数)。
- **`n_inner_consts` 自动标注**(`taxonomy.count_inner_consts`,位置法上界):全集分布
  `{0:76, 1:12, 2:20, 3:5}`,即 37 题带内部常数;对照全为 0(已测)。
- **守门**:113 题全部生成有限 y(已测,挡 log/tan/sqrt 域炸)。

**LM 接入 EvoGP in-loop(已完成 2026-06-08)**:
- **可插拔 CO backend**(`co_backend.py`,`test_co_backend.py` 18 测试):`ConstantOptimizer.fit_batch`
  接口,货币=`Skeleton`;`ScipyLM`(基线)/`TorchLM`(GPU 原生 LM,autograd Jacobian)/`CudaKernelLM`(空壳)。
  详见 [[project_009_co_backend_interface]]。torch 残差对齐 bench 的 1e10 哨兵(等算力公平对比),
  并在**真实 EvoGP 骨架**上交叉验过(torch 有限且不劣于 scipy)。
- **memetic pipeline**(`pipeline.py`,`test_pipeline.py`):009 自己的 `MemeticPipeline`(不碰 bench 的),
  每代 extract→`fit_batch`→writeback→re-eval→rollback。Nguyen-1 用 TorchLM **和** ScipyLM 都在环里恢复。
- **端到端 runner**(`run_bench.py`):pilot(nguyen_1/nguyen_5/korns_12,TorchLM)跑通——
  nguyen_1 6 代恢复;hard 题在此小配置下不恢复(预期)。出表 + jsonl。

**待做**:
1. **全量 sweep(建议开 exp010)**:113 题 × 多 seed × **对照臂**(无CO / 朴素CO / 配方;top-K vs 全种群);
   `r2_test>0.5` 门控放 runner 层;出 headline 恢复率-算力曲线。dtype 接入时已选(见 pipeline.py 注)。
2. **补更多题**:Strogatz 覆盖、SRSD **真实区间** overlay(`range_variant=realistic`)。
3. judge 侧:`simplify` 加超时(扩到全量重算时)。

---

## 放置说明

**自包含在 `experiments/009_sr_benchmark/` 下**(用户 2026-06-07 决定):009 自己的件
(manifest、Korns/Strogatz 新题、生成器、恢复率-算力 harness)都建在本文件夹;只**复用**
(import)`bench/` 里已测的判定器,**不挪它**(exp007 还在 import 它)。代码用脚本式导入
(测试/模块里把仓库根加进 `sys.path`),因为目录名以数字开头、不能当常规 Python 包。
日后若 ≥2 个实验依赖,再议是否提升为共享基础设施。
