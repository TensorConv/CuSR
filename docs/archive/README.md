# docs/archive —— 归档(2026-06-23)

这里是**旧的、被取代的、或者 pilot 阶段**的文档。**当前在用的不在这里。**

开工请看 [`../README.md`](../README.md),入口是 [`../research/paper_plan.md`](../research/paper_plan.md)。

**为什么归档:** 文档太多太杂,人和 LLM 都容易看错、照着旧结论引。这些都不是现在的准。需要的话 `git mv` 搬回去就行。

里面大致有:

- **旧策略/计划**:`research/co_benchmark_strategy.md`(45KB 老策略)、`research/co_benchmark_contribution_plan.md` —— 被 `research/contributions.md` + `research/paper_plan.md` 取代。
- **旧结果/验证**:`RESULTS_laptop.md`、`RESULTS_kernel_bridge.md`、`kernel/verify_report_*`(笔记本 / 旧快照)—— 吞吐已被 e6 取代,见 `experiments/RESULTS.md`。
- **pilot 发现**:`bloat_regression_finding.md`、`co_signal_finding.md`、`NOISE_FLOOR.md`。
- **旧设计**:`kernel/AD_JACOBIAN_V4.md` —— 已被 reverse-AD **v5** 取代(v5 设计见 [`../kernel/REVAD_V5.md`](../kernel/REVAD_V5.md))。
- **走读/README/杂项**:`kernel/walkthrough/`(逐步搭 kernel 的教学代码)、各 `*_README.md`、`OVERVIEW.md`、`REORG_REVIEW.md`、`kernel/RERUN_A100.md`、`kernel/evogp_inloop_notes.md`。
- **回头要用就搬回**:`judge_calibration.md` —— 做 C3 / 符号恢复评判时会用到。
