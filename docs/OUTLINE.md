标题: 面向 GPU 符号回归的批量常数优化 CUDA kernel (英文正式标题待定)
更新 2026-06-21 · HPEC 2026 投稿大纲 + 进度. 标记 (已有) / (待做 xxx).

> **范围 (2026-06-21).** 本文 = **HPEC 系统篇**,只承载**贡献①(异构批量 CO CUDA kernel,系统/性能)**。
> 贡献②(选题判据)与③(实证发现)见 [`research/contributions.md`](research/contributions.md),**留作后续 benchmark 篇**,本文不承重。
> 诚实边界(prior-art 各归谁)以 `research/contributions.md` / `research/co_benchmark_strategy.md` 为准:
> **不喊 "first GPU CO"(Kozax=de Vries 2025 已有,一阶)/ 不喊 "新算法"(LM 是老的)**;按 HPEC 口径定位为 **"高质量原型 + 强定量结果"**。

> **进度更新 2026-06-21 (e6 — 修正 e5 测错 kernel 的重大返工).**
> **e5 的速度数字作废**: 之前的扫描误用了 in-process `libcusr_co_fd.so` = host-FD 慢核
> (每 FD 步 H↔D 往返, ~1.3–2k trees/s 平), 不是部署用的片上核。**"29×" 是真的, 但活在
> host-FD 这个错分母上**。已坐实 (confirm micro-bench, M=64k, A100, 时钟未锁草测):
>   · 片上 fusedfd vs in-proc host-FD = **21× (early-gen) / 12.5× (inner-const)**, 质量逐位相同;
>   · 片上 fusedfd vs **64 核 EPYC 7763×2** = **6.5× (early) / 2.1× (inner)** —— GPU 反超并行 CPU。
> **e6 修正扫描已建好 (harness + TDD 20 测试绿 + dry-run 端到端通过), 待全跑**: 两个真·片上核
> **fusedfd + ad** (子进程 PROFILE binary, 测纯 LM-loop `loop_ms`) × 3 preset × M∈{1k,4k,16k,64k,256k}
> × **N∈{100,1k,10k}(新轴)** × Operon **{1,16,64,128 核}**; fp64 质量门、显存/耗时跳过都记账、
> 可续跑。GPU/CPU **分两段计时不并发**(GPU 的 LM loop 是 host 编排的, 跑满核的 Operon 会抢 host/PCIe
> 撑大 GPU 计时)。反作假判据从"吞吐量地板"换成**结构性判据**(查 PROFILE 里 Jacobian 是否在 GPU 上算
> —— N-不变, 因为高 N 小 M 的真实吞吐量实测 ~517 t/s 会和 host-FD 带重叠, 绝对地板无法区分)。
> 校准实测 (M=2000, N=1000, 时钟未锁): fusedfd 10.6k/9.1k、**ad 19.3k/14.9k** trees/s (early/inner)。
> **质量天花板 (inner-const) 与变体无关** (fused==ad==fd 数值): kernel ~5.0 vs Operon ~3.6,
> 根因 = fp32 + 乘性 Marquardt 阻尼无法正则化秩亏 JtJ → 另开修复 (加性 λI/λ·diag, 改 kernel)。
> 硬件框定: EPYC 7763×2 (256 线程) ≈ 10× 一颗 9850HX → **A100-vs-本机 = 最对抗的 CPU 基线**,
> 即便如此片上核仍赢; 消费级 5070Ti-vs-9850HX 则 GPU 完胜。


故事

符号回归用进化算法养一大群候选公式、逐代择优。每代里公式结构定了, 但其中的常数还没调准;
把常数调到最优这一步叫常数优化(CO)。实测 CO 是整条流程的瓶颈: 进化一代仅几毫秒, 全种群
CO 一次要几百毫秒。现有 CO 要么在 CPU 串行(scipy / Operon / PySR, 慢), 要么 GPU 上缺一个
针对"大批量、且每棵树常数个数不一"的专用 kernel(EvoGP 有 GPU 进化但**无 CO kernel** —— 这正是我们补的缺口)。我们做了这个 kernel(一次并行调好上千棵
树的常数), 并搭一套公平的评测对比。两条主结论: (1) 用对片上 kernel 时, 在 CO 最难处(常数多)
比 CPU 快很多 —— 对**具名强基线 64 核 EPYC** 在 M=64k 上 **2–6.5×**(片上核 confirm 值; 旧 "29×"
是 e5 误测 host-FD 慢核的错分母, 见上方进度更新; 正式曲线待 e6 全跑 + A100 终值); (2) 单精度足够 —— CO 只为给候选排序、择优喂下一代,
排序对就行, 不需要双精度的位数(另带 fp64 诚实守卫:交付解不劣于初值)。

贡献定位: 提出的 kernel(目前为原型, 仍有优化空间) + 评测框架与口径 + 硬件极限分析, 均为
系统/性能层面 —— 按 HPEC 口径是**"高质量原型 + 强定量结果"**, **不声称新算法**(LM 是 1944/1963 的;
新意只在"异构(结构与 K 皆变)+ 二阶 LM + 显式 per-tree Jacobian + 单次 CUDA launch"这一窄实现片,
对比 Kozax 的一阶 JAX、Gpufit/JAXFit 的单一固定模型)。早期版本只是优化路径上的对照, 不算贡献。
端到端"能多发现公式"的演示不承重。


大纲 (HPEC 结构)

一、摘要: 问题(CO 是瓶颈、缺批量异构 kernel) + 贡献(kernel + 评测框架) + 数字(加速 ~xxx 倍、
   单精度排序≈双精度、硬件利用率 xxx%). (终稿数字待 A100)

二、引言 (已有骨架): 介绍 SR / 进化算法 / CO; CO 是瓶颈(几毫秒 vs 几百毫秒)= 全文动机;
   缺口与难点(每棵树常数个数不一, 直接并行会浪费、对不齐); 贡献列点.

三、相关工作 (待做 xxx): 带诚实边界各归各:
   · EvoGP(GPU 进化, **无 CO** —— 我们的缺口)/ Operon(CPU CO 天花板, fp64)/ PySR(CPU)
   · Kozax = **de Vries, Keemink & van Gerven 2025**(GPU CO 已存在,但**一阶 AD 梯度 + GA、JAX/vmap、无 LM**)→ 我们 = **二阶 LM + per-tree Jacobian + 自定义 CUDA**
   · Gpufit / JAXFit(GPU 批量 NLS,但**单一固定模型**)→ 我们 = **异构树**
   · Kronberger 2022(CO conditioning / 诚实轴的主人)/ MegBA / 批量拟合先例
   · 红线: 不喊 "first GPU CO" / "novel algorithm". 梳理待补.

四、设计 (已有): 公式编码与共享数据; 每棵树分一组线程并行跑 Levenberg-Marquardt(LM)拟合;
   变长 K 的异构批量法方程在一次 launch 内打包; 核内 forward-mode AD 逐树构造 Jacobian;
   失败返回当前最优、不丢进度; 全程单精度 + **fp64 诚实守卫(deliver≤init: fp32 迭代 / fp64 验收)**.
   (守卫框定: 引用 Kronberger 2022 为 conditioning/诚实轴主人, 只 claim "GPU SR 第一个把它作为运行时保证强制执行", 非新机制.)
   (待做 xxx: 结构图 / 伪码)

四之二、负载与设计空间 (workload, 已有数据; 待补图):
   一份 CO 负载 = 一代种群长啥样: 每棵树几个常数 K、几个节点、多深、什么算子, 而且随代数漂
   (早期树小, 后期 bloat 变大). 三个标准 preset 跑遍所有性能实验: early-gen / late-bloated /
   inner-const-heavy.
   · bloat 对 kernel 不是 bug, 是一种负载: 节点变多 → eval 变贵, K 变多 → CO 更贵 (正好是我们赢
     最多的地方, 见 五A 高 K 那段), 同时 bloated 高 K 树常秩亏 → 质量天花板 (五D 里 tier-B 65% 软肋
     的来源). late-bloated preset 就是专门测这个 regime.
   · 树深度进 kernel 设计三处: (1) 每线程栈固定 64 槽, 树太深 (栈需求 >64) kernel 收不了, 退回 CPU
     算; (2) 越深栈占越多 local 内存 → 同时能跑的 warp 越少 (occupancy 降); (3) 越深嵌套越多 →
     Jacobian 越病态 → CO 越难收敛 (质量). 记: 节点数决定计算量, 深度决定栈深, bloat 时俩一起涨.

五、评测 (核心):
   A 加速随难度↑ + 随种群规模 M / 数据量 N 变化的吞吐量曲线 (e6 修正扫描: fusedfd+ad vs Operon
     1/16/64/128 核; **旧 2.4–29× 是 e5 误测 host-FD 错分母 → 作废**, 见进度更新)。confirm 已坐实
     片上核 M=64k 反超 64 核 EPYC **2–6.5×**; **全曲线 + A100 终值 + scaling 待 e6 跑完 —— 关键路径**
     · **种群规模 M 是 GP 的设计选择, 不是固定值**: CPU 时代用几千的小种群, 正因 CPU 养不起大种群
       (CO ∝ M)。GPU 在 M≈64k 才饱和、优势随 M 增长, 而 Operon 封顶 ~10k t/s 且过 16 核几乎不再扩展
       (实测 M=256k N=100: nc 16→128 仅 8.4k→9.9k)。⟹ 头条不是"GPU 在小 M 被饿着", 而是"**GPU 解锁了
       CPU 养不起的大种群区间**, 自然 operating point ~64k = CPU 时代 4000 的 16×"(可经 大种群 / island /
       多重启 / 多问题批处理达到; 显存 N=1000 时单卡 M 可到 ~1M)。**两层诚实**: kernel 层 = 大 M 的 CO
       在 GPU 上便宜、优势随 M 增长 (已测); GP 层 = "用大 M 因此更快/更好做 SR" 是端到端命题, 归 F 演示, 不由吞吐量本身证。
   B 单精度排序≈双精度(难题排序吻合 Spearman 0.82–0.91, 前 25% 重合 76–87%)→ 单精度够用 (已有)
   C 优化阶梯(消融): 早期版本 → 最终版的提速(本机约 8–13 倍)(已有; A100 上预期收窄)
     · 收敛了就踢出去 (待做, 后面可能要做): 现在每代把所有树重算一遍, 拟好的树还在白算, 整批被
       最慢那棵拖着。改法: 树拟好了就踢出去, 后面只算没好的。不是学 Operon 那样 1-2 步就撒手 (常数
       没拟准、质量会塌), 是真拟好了才踢。值不值看情况 (能画成一张图放 E): 树多到 GPU 一次放不下、
       又有快有慢拖长尾时才省时间; 树少、一波就跑完就省不出来。
   D 质量与局限: 已知答案的合成题上, 结果离理论最优和双精度 CPU 一样近(差 ≤2.3 个百分点);
     难题上的差距源于题目本身, 非 kernel 缺陷 (已有).
     注: 旧稿"Operon 在 bloated 树上深 3×"已被自己的 report.json finding[2] 推翻, **删除**;
     bloat→秩亏 改作**质量天花板的解释**(SVD 数值秩, 引用 Kronberger 2022), 不当性能贬损.
     注 (e6 坐实): inner-const 质量天花板与 Jacobian 方法/变体**无关**(fused==ad==fd 数值相同),
     且**不是可修的 solve bug**: exp012 (docs/MIXED_PRECISION_PROBE.md) 已证伪/否决全部修法 —— fp64 解 +0;
     pivot-floor 炸 W0 parity gate (伤 123 棵健康树); QR/rank-reveal 也不行 (健康树同样普遍秩亏、谱重叠)。
     根因 = 高 K 树**结构秩亏**, 安全基线 (整步放弃 + 升 λ) 是对的; 加性 λI 是唯一未实测候选, 但文档预测同样失败。
     ⟹ 诚实写成 **intrinsic 质量上限** (SVD 数值秩, 引 Kronberger 2022), 不当待修 bug, **别再提 damping 当解法**。
   E 硬件极限分析(roofline): xxx(计算量/访存量 + Nsight 剖析, 看离硬件天花板多近, A100)(**待做 —— HPEC 最看重、当前最缺的一块, 关键路径**)
   F 端到端演示(可选, 不承重): 把 kernel 装进进化循环, 看能不能找回 stock 找不回的公式.
     实测 (构造 inner-const 题, 真 EvoGP 配置, 5 seed): stock 0/30 → 装上 CO 11/30 (stock 全
     找不回 = 富采样也撞不到精确常数, 不是我们砍常数池骗的). 但解锁强弱受结构搜索限制: 内部常数
     题偏弱 (3/20), 因为树一 bloat 结构就搜不对、CO 没正确结构可补; 外部常数/简单结构题强 (8/10).
     诚实补注: **解锁集中在 single-inner; multi-inner 目前所有方法都失败** —— 写成开放前沿,
     **绝不写"收益集中在多内部常数题"**(与 Study B 数据相反). 数字用修完 zoo bug 的干净重跑值.
     口径: 这是存在性演示不是排行榜 (不承重), 不声称普遍提升 SR.

六、讨论 (有素材): 难题差距的根因 = 这类公式里常数互相冗余、信息不足, 无干净解法(试过几种均
   不行), 是题目性质非缺陷; 单精度是有意选择(数据本就单精度, 下游只需排序); 局限(本机粗测
   vs A100、演示场景可能受限).

七、结论 (待补): 把 CO 当一个算子做快 + 公平对比; 贡献不依赖端到端结果. 后续: A100 扫描、更多
   CPU 对比、更优 CO 策略 (待做 xxx).

八、参考文献 (待做 xxx)


findings 池 (待筛进 五)
1. fp32 比 fp64 更好, 不需要那么高的精度, co 的意义是给候选公式排序 所以只要保证选择信号就ok
2. 真·片上核有两个变体: fusedfd(片上融合 FD)与 ad(片上前向 AD, 精确 Jacobian)。AD 在小 M / 高 K
   上常比 fusedfd 还快(校准 ad 19.3k vs fusedfd 10.6k @M=2000 early-gen)且质量略好 → e6 两个都测,
   ad 可能是主推变体。
3. N(每棵树数据点数)是新性能轴: N 越大 GPU 数据并行优势越大(每棵树 32 lane 跨 N); 但 trees/s
   随 N 上升而下降(每棵树算更多点)→ 吞吐量必须按 (M,N) 配对看, 不能跨 N 比绝对 trees/s。
4. e5→e6 教训: 出现与已知基线(tier0 fused 36–50k)差一个数量级的数, 先怀疑"测错变体", 别编物理
   解释。e6 harness 内置结构性反作假判据(查 PROFILE 的 Jacobian 是否在 GPU 上算, N-不变)。
