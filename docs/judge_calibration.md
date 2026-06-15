# 判定器标定(这把尺子量准没有)

记录:把"算不算找回真公式"的判定器量准,定 009 的容忍策略。结论先行,证据在后。
所有数字都是实测(脚本见本文件末尾的复现说明),不是凭记忆。

> 状态:**已定 + 已实现**(用户 2026-06-08 拍板,见 §5)。判定器落在 `judge.py`,TDD 在 `test_judge.py`(17 测试过)。

---

## 一句话

SRBench 那把尺子对**内部常数**不是尺度无关的:`|c|≳1` 时约 3 位有效数字(相对 ~1e-3),
`5e-4<|c|<1` 时退化成**绝对 ±5e-4**(对小常数相对容忍能到 ±10–25%),`|c|<~5e-4` 直接**抹成 0**。
我们整套 benchmark 的主张就是"内部非线性常数 × 真实区间(常数跨多个量级)"——
SRBench 这把尺子恰好在小常数那头要么白送、要么销毁。所以**不建议全盘照搬 SRBench 当主判定**。

---

## 1. SRBench 实际怎么判(源码,master 分支)

`experiment/symbolic_utils.py` + `experiment/assess_symbolic_model.py`:

```python
def round_floats(ex):                      # 对表达式里每个 Float 原子:
    for a in preorder_traversal(ex):
        if isinstance(a, Float):
            if abs(a) < 1e-4:  ex = ex.subs(a, Integer(0))     # 小于 1e-4 → 0
            else:              ex = ex.subs(a, Float(round(a,3), 3))  # round 到 3 位小数,存 3 位有效精度
    return ex

# 门控:r2_test > 0.5,然后:
sym_diff = round_floats(simplify(round_floats(true - pred), ratio=1))
sym_frac = round_floats(pred / true)
recovered = (str(sym_diff)=='0')  or  sym_diff.is_constant()  or  sym_frac.is_constant()
```

- `str(sym_diff)=='0'`:四舍五入后差为 0(精确命中)。
- `sym_diff.is_constant()`:差是个**常数** → 容忍**外层加性**常数。
- `sym_frac.is_constant()`:比是个常数 → 容忍**外层乘性**常数。
- 三者 OR;外层加、乘各容忍一个,但**同时缩放+平移(仿射)不算**(两条都不成立)。
- 关键点:`round_floats` **只动表达式系数、不碰数据 X** → SRBench 这把尺子是**域无关**的。

来源:
[symbolic_utils.py](https://github.com/cavalab/srbench/blob/master/experiment/symbolic_utils.py) ·
[assess_symbolic_model.py](https://github.com/cavalab/srbench/blob/master/experiment/assess_symbolic_model.py)

## 2. 实测:SRBench 尺子对内部常数的容忍画像

`sin(c·x0)` vs `sin((c+Δ)·x0)`,找 Δ 的找回边界:

| true `c` | 容忍 Δ(绝对) | 相对 | 谁在管 |
|---|---|---|---|
| 0.002 | ±5e-4 | **±25%** | 绝对 5e-4 地板 |
| 0.005 | ±5e-4 | **±10%** | 绝对 5e-4 地板 |
| 0.05 | ±5e-4 | ±1% | 绝对 5e-4 地板 |
| 0.5 | ±5e-4 | ±0.1% | 过渡 |
| 9.8 | ±0.005(±0.01 失败) | ~5e-4 | 3 位有效 |
| 1234 | 归到 1230 | ~1e-3 | 3 位有效 |

两个病理(对我们最致命):
1. **小常数过松**:`|c|<1` 是绝对 ±5e-4,c 越小相对容忍越爆炸(±10–25%)→ 小内部常数的题近乎白送 / 尺子伪命中。
2. **抹零悬崖**:`round(c,3)` 对 `|c|≲5e-4` 直接归 0(比显式的 `<1e-4` 阈值还早)→ 真·小常数被销毁(`exp(-0.00005·x)` 塌成 `1`)。SRSD-Feynman 改用 NED 正是为此。

## 3. 我们现有 bench judge(`bench/sources/evogp.py`)对比

- **sympy 分支**:精确——系数**不 round**,只把 `<1e-6` 的加性尾巴归零 → 内部常数差一点点就判失败。
- **numeric 分支**(`tol_rel=1e-4`):相对容忍,但**随采样域变**——有效频率容忍 ≈ `1e-4/|x|_max`。
  实测同一公式:`[-1,1]`~1e-4、`[0.5,2.5]`~3e-5、`[-50,50]`~1e-6(宽域比窄域严 ~100×)。
- **composite = 两者 OR**。
- 毛病:既不 round 又域耦合 → 跟 SRBench 不可比,且 headline 会随题目区间漂。它当年是为 exp007 的 Feynman 配的,不是为"内部常数 × 跨量级"设计的。

## 4. 选项

| 方案 | 内部容忍 | 优 | 劣 |
|---|---|---|---|
| **A 全盘照搬 SRBench** | round_floats 3 条 OR | 与 published 数字可比、域无关、省事 | 继承小常数过松 + 抹零悬崖——正打在我们主张的软肋上 |
| **B 009 自定义统一相对容忍**(倾向) | 对系数做**跨量级一致的相对误差**(如 1e-3),保留 numeric proxy 兜不透明改写,SRBench-faithful 另列做可比 | 尺度无关、科学正确、堵住两个病理;仍可比 | 要自己写一层薄判定 + 自证容忍值合理 |
| **C 保持现 bench composite** | 精确 ∨ 域耦合 | 现成 | 域耦合,不宜作主判定 |

- 不管选哪个,**NED**(连续树编辑距离)仍作副指标,兜小常数盲区(部分接近/被抹零的题在 NED 上仍有区分)。
- numeric proxy **不能丢**:SRBench 纯 sympy,会漏掉 exp007 真实碰到的不透明改写(`sin(α+π/2)=cos`、大系数代数)。composite =(sympy+round)∨ numeric-proxy。

## 5. 已定的决策(用户 2026-06-08)+ 实现

| 决策 | 选了 | 落在哪 |
|---|---|---|
| **主判定** | 009 自定义统一相对容忍(方案 B)+ SRBench-faithful 作可比列 | `judge.round_sig` / `judge.recovered` / `judge.recovered_srbench` |
| **内部容忍** | 相对 ~1e-3(`sig=3`,≈SRBench 大常数档,跨量级一致) | `judge.DEFAULT_SIG=3` |
| **外层常数** | 报**两列**:`strict`(不容忍外层)+ `lenient`(容忍,和竞品同条件,线性缩放是标准操作) | `judge.recovered(..., outer=...)` / `recovery_columns` |

实现要点(`judge.py`):
- `round_sig(expr, sig=3)`:每个 Float 原子 round 到 `sig` 位有效数字(相对,无绝对地板、无抹零)→ 跨量级一致 ~1e-3,堵住 §2 两个病理。实测 c∈[0.002, 1234] 行为一致。
- `recovered(cand, true, outer="lenient"|"strict", X=None)`:结构等价(sig-round 后差化简为 0 / 外层加性常数 / 外层乘性常数)。lenient 且给 X 时,结构判失败再退回 numeric proxy(`bench.check_recovery_numeric`)兜不透明改写(`sin(α+π/2)=cos` 等)。
- `recovered_srbench(cand, true)`:SRBench 原版(round_floats + 三条 OR),作可比列,**故意保留**它抹零小常数的盲区(`test_judge.py` 有一条 doc 它)。
  - **不是逐位忠实**(引用可比性时别overclaim):(a) frac 分支多做了一步 `simplify(p/t)`(SRBench 靠它自己的 `clean_pred_model` 预处理,我们没复刻),所以外层乘性这块比原版略强;(b) **没有 `r2_test>0.5` 门控**(判定器拿不到数据)——这是 symbolic-equivalence-only,不是 SRBench 完整的 "symbolic solution"。所以这列数字**不能直接对标 published 排行榜**。
  - `r2>0.5` 门控接 run 时**放 harness 层**(judge 拿不到 X/y)。它是 SRBench 防"数据撑不住的伪恢复"的闸,对 **numeric proxy 分支**尤其要紧——否则恢复率-算力曲线会高估。接入时必须补。
- `recovery_columns(...)` 一次给 `{strict, lenient, srbench}` 三列,供 benchmark 结果行。
- `seed_bench.score(...)` 默认走 `outer="lenient"`;`seed_bench.score_columns(...)` 给三列。

**仍待办**:`simplify` 在病态表达式上可能慢/卡 —— 扩到全量前给结构判定加超时(`bench` 的 `recheck` 脚本有 `_with_timeout` 可参照)。`sig=3` 在不同首位数字上有效容忍会在 ~1e-3 上下浮动(有效数字法的已知小毛病),可接受;真要严格统一相对阈再换"逐系数相对比对"。

---

## 复现

```bash
uv run python  # 见 git 历史 / 下方片段
```
`round_floats` 按 §1 复刻;边界扫描:固定 true `c`,对 `sin(c·x0)` 递增 Δ 找 `recovers(c, c+Δ)` 翻 False 的点。
SRBench 判定纯系数运算(不需 X);bench numeric 分支需 X(域耦合就出在这)。
