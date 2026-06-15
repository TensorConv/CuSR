# E1 协议 — CO 算子层 benchmark 口径

* version 2 (2026-06-12): 加**选择保真度**指标 (§3/§6) + **精度口径** (fp32 是既定目标,
  oracle 仍 fp64 当金标准) + **指标面板哲学** (定义现在钉死, 每次全算全报, 写作只挑"主推哪个"
  不挑"披露哪个"). v1 (2026-06-11, W1 定稿): pop.bin 货币 / 串行 oracle / tier-A 1.005×+tier-B 1.05×.
* 适用: 论文核心图 E1 (多后端同批常数优化对比); 改口径必须 bump version 并在此记录原因
* 状态: 口径已定; 全量执行等 A100; 冒烟用 008 `data/pop.bin`

## 0. 指标面板哲学 (避免 cherry-pick 翻车)

* **定义现在钉死** (本协议), 后端实测前不再动 → 避免"看到谁赢再改定义"的 gaming
* harness **每次全算全存** tier-A / tier-B / 选择保真度 (合成 preset 另加噪声地板), 不挑着算
* 写作时可以选**主推哪个图** (emphasis), 但 **reviewer 预期看到的指标 (尤其 tier-A) 不能藏** —
  不利数字照实摆 + 给机制解释 (如 tier-A 低 = fp32 数值天花板), 比回避更抗打
* 一句话: **cherry-pick 的是 lead, 不是 disclosure**

## 1. 比什么

同一份 workload (M 棵树, 每树 K_m 个常数初值, 共享 X[N,n_vars], 每树 y[N]),
各后端把常数拟到最好。比 **达到质量档位的吞吐**, 不比裸 trees/s, 不比裸 loss。

## 2. workload

* 货币 = `pop.bin` (008 格式: 16×int32 header, nt/nv/ci/metas/c_init/xs/ym, fp32)
* 来源 = W4 三 preset (early-gen / late-gen-bloated / inner-const-heavy) + 合成单轴 sweep
* 同一文件喂所有后端 — bit 级相同输入
* 数据本体 fp32 交付; 后端内部升精度允许, 结果表必须披露 dtype

## 3. 质量口径 (oracle + tier)

* oracle = scipy fp64 紧公差 run: `least_squares(method='lm', xtol=ftol=1e-10, max_nfev=200)`,
  residual 用本目录 `interp.py` fp64 解释器 (已对 008 `tree_interpreter.py` 交叉验证)
* **oracle 串行跑** (nproc=1): 实测多进程池下 ~1% 退化树 (平坦谷底, identifiability 退化)
  结果非确定 — 同 payload 串行/单 worker/全新进程/任意核都 bit 级稳定, 满池就抖,
  loss 相对差 ≤2.3e-3 (300 树 × 15 池次实测; 排除 chunk 顺序/进程复用/核类型/BLAS 线程,
  根因未追到底). 串行 oracle 可复现, 结果按文件 sha 缓存, 慢一点无所谓
* loss ≡ 0.5·Σ_N r² (同 verify.py / kernel 口径)
* **harness 用 fp64 参考解释器从后端返回的 c_final 统一重算 loss** — 不信后端自报 loss
* Tier A: L_b(m) ≤ max(1.005·L*_m, L*_m + 1e-10) — "跟 oracle 一样好";
  0.5% 容差 = 上述抖动上界 ×2 余量, 否则确定性后端对自己都到不了 100%
* Tier B: L_b(m) ≤ max(1.05·L*_m, L*_m + 1e-10) — ≡ 008 parity 的 fitness-equivalent 档
* `+1e-10` 绝对垫底防 L*≈0; K=0 树不进 tier 统计
* oracle 不是被测后端; 结果按 pop 文件 sha1 缓存 (`_cache/`), 换 oracle 设置必须清缓存 + bump version
* 已知妥协: oracle 的 max_nfev=200 含 FD evals, 高 K 树可能在收敛前触顶 → L* 偏松。
  与 verify.py 同口径, 沿用; 若收紧要连 parity 历史一起重算

### 3b. 精度口径 (oracle fp64 金标准, 后端按各自精度)

* oracle 用 fp64 = "最优能拟到多低"的金标准, 与被测后端精度无关
* 被测后端按各自原生精度 (kernel/operon fp32, scipy/torch/pysr fp64); fp32 后端的 tier 率
  = "fp32 追平 fp64 金标准的比例", 是有意义的诚实数字
* **fp32 是 kernel 的既定目标不是妥协** (EvoGP/数据本就 fp32, 无 fp64 信号可恢复, 下游只需排序)
* 推论: **tier-A (0.5%) 是 fp32 的数值天花板** — fp32 在最优档分辨不足, 不该当唯一主标尺;
  详见 `008/RERUN_A100.md §0`

### 3c. 选择保真度 (CO 服务"选择"的 use-relevant 指标)

* 动机: CO 这一算子的用途是给候选公式排序喂选择, 不是逼 fp32 追平 fp64 精确度
* 定义 (`runner.selection_fidelity`, 只对 eligible 且两侧有限的树):
  - **Spearman** = 后端 loss 排序 vs oracle loss 排序的全局名次相关
  - **top-K% 重合** = oracle 选出 loss 最低 K% 棵 vs 后端选出的交集比 (进化选择真正在意的粒度)
* 读法: top-25/50% 高 = 粗筛 (淘汰差的一半/留好的四分之一) 保真; top-10% 低多因 fp32 在最优档
  洗牌, 但 memetic 每代重评估, 不需单代精排最优档
* 合成 preset 另报**噪声地板** (已实现, 见 `NOISE_FLOOR.md`): `gen_synth` 落 `c_true` sidecar,
  harness 算 `loss_pop(pop, c_true)` = 注入噪声能量 = 统计下界 (走同一条 fp64 loss 路径). 报
  oracle / 各后端"够到地板"的比例 —— 是相对 tier 的**绝对兜底** (回答"oracle 自己是否最优").
  读法: 主推**不变量** oracle≈kernel (差 ≤2.3pp, init-无关); 绝对档位随 c_init 散布而变 (±30%),
  报数必带初值口径. oracle 够地板 <100% = 演化初值的基底吸引域, 非预算/精度 (实测)

## 4. 停机口径

* 各后端跑**自己的原生收敛准则** + 披露的迭代上限 (kernel: max_iter=50 内建;
  torch: 50 outer; scipy: max_nfev=200)
* 结果三分类: converged (本家准则) / iter-limit / failed, 图注披露各家准则
* 迭代数只作 per-backend 辅助列, **不跨后端比较** (LM 一步 ≠ BFGS 一步; scipy nfev 含 FD evals)
* time-to-target 曲线 = 二级指标, 只对暴露 per-iteration trace 的后端做

## 5. 计时口径

* `wall_e2e`: fit_pop(内存 pop) 进 → 常数回内存; 含该后端全部 marshal
  (kernel 含 pop.bin 落盘 + subprocess 启动 + 读回 — in-loop 真实成本)
* `wall_core`: 后端自报核心计算 (kernel = binary 自报; scipy = pool map 段; torch = fit 循环段)
* 两列分开报, 不混; e2e 是主列
* 计时前 1 次 dummy warmup (CUDA ctx / torch 编译 / Julia JIT), 披露
* 重复 ≥3 次取中位数 (冒烟可 1 次, 标注 smoke); 跑前 `nvidia-smi` 确认 GPU 无并发任务
* CPU 后端给满核, 进程/线程数披露
* 结果 json 记录: hostname, GPU 型号, git sha, 各 binary 路径 + mtime
* 本机 (5070 Ti laptop) 数字仅 sanity; 论文数字按 `008/RERUN_A100.md` 在 A100 重测

## 6. 输出指标 (面板, 每次全算; 写作选 lead 见 §0)

* **质量×吞吐**: tier-A / tier-B throughput = (#达档树) / wall_e2e, 单位 trees/s (frontier 主轴)
* **选择保真度** (§3c): Spearman + top-10/25/50% 重合 — CO 服务选择的 use-relevant 主指标
* 副: tier rate (%), 状态分布 (converged/iter-limit/failed/K0), per-backend 迭代中位数,
  wall_core 折算吞吐; 合成 preset 加噪声地板
* 不报: 裸 trees/s headline; 任何跨系统 recovery 比较
* json 存全套, md 出两张表 (质量×吞吐 + 选择保真度)
