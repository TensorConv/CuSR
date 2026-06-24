# docs 导航

**Start Here:** [`research/paper_plan.md`](research/paper_plan.md) —— 论文怎么拼、6 页怎么分、C1/C3 有哪些材料可选。

**写论文随身带(离线事实卡):** [`research/PAPER_FACTS.md`](research/PAPER_FACTS.md) —— 当前正确数字 / 红线 / 引用陷阱 / 诚实边界。记忆(`~/.claude/`)不随 repo 下载,关键事实都固化在这一份里。

## 现在在用的

**论文**
- [`PAPER.md`](PAPER.md) —— 待办 + 进度(「现在到哪了」)
- [`research/PAPER_FACTS.md`](research/PAPER_FACTS.md) —— **离线事实卡**:当前数字 / 红线 / 引用陷阱 / caveat
- [`research/paper_plan.md`](research/paper_plan.md) —— 入口:6 页怎么分、材料怎么选
- [`research/C2_draft.md`](research/C2_draft.md) —— C2 正文初稿(设计=完整、评测=骨架)
- [`research/contributions.md`](research/contributions.md) —— 三个贡献的细节(C1 workload / C2 kernel / C3 deployment)
- [`research/experiment_plan.md`](research/experiment_plan.md) —— 要跑哪些实验
- [`OUTLINE.md`](OUTLINE.md) —— 论文叙事 + 骨架(较全;和 paper_plan 有重叠)
- [`../experiments/e7_section2/FINDINGS.md`](../experiments/e7_section2/FINDINGS.md) —— section-2 每个数的出处 + 审计

**kernel(做 C2 时看)**
- [`kernel/OPTIMIZATION_BACKLOG.md`](kernel/OPTIMIZATION_BACKLOG.md) —— 优化路线(reverse-AD 是 #1)
- [`kernel/REVAD_V5.md`](kernel/REVAD_V5.md) —— reverse-AD v5 设计(C2 技术核心)
- [`kernel/TECH_DEBT.md`](kernel/TECH_DEBT.md) —— 已知问题
- [`kernel/INTERNAL.md`](kernel/INTERNAL.md) —— kernel 内部参考
- [`kernel/OPERON_ADAPTER_SPEC.md`](kernel/OPERON_ADAPTER_SPEC.md) —— Operon baseline
- [`kernel/INPROCESS_CO_PLAN.md`](kernel/INPROCESS_CO_PLAN.md) —— 进程内集成(C3 相关)
- `kernel/lm_algorithm.tex` / `kernel/fig_architecture.tex` —— 论文图

**参考 / 证据(需要时查)**
- [`characterization.md`](characterization.md) —— 真实快照统计(synth 数据据此校准)
- [`MIXED_PRECISION_PROBE.md`](MIXED_PRECISION_PROBE.md) —— 为什么 damping 没用(别再提)
- [`PROTOCOL.md`](PROTOCOL.md) —— 实验 / 计时协议
- [`SETUP.md`](SETUP.md) —— 环境搭建

## 旧的

[`archive/`](archive/) —— 旧的、被取代的、pilot 文档。**不是现在的准,别照着引。**
