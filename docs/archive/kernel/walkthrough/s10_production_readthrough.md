# s10 — 通读生产版 `batch_lm.cu`

假设你已经走完 s1..s9,已经懂这一整套 LM:逆前缀 + 栈的解释器(s1/s2)、warp-per-problem
映射(s4)、shfl 蝶形归约(s5)、有限差分(FD)求 Jacobian(s6)、`JᵀJ`/`JᵀR` 拼装(s7)、
Cholesky 解 `K×K` 方程(s8)、host 端 accept/reject + λ 调度的整圈 LM 循环(s9)。

**这一级【没有任何新的 CUDA 概念】。** 所有 kernel、所有数值都和 s9 一字不差:
λ 初值 `1e-3`,accept 时 `λ*=0.1`(下限 `1e-12`),reject 时 `λ*=10`(`λ>1e12` 判失败),
`xtol=1e-5`,FD 步长 `eps_fd=1e-3`,收敛判据 `d_norm < xtol*(c_norm + xtol)`,`max_iter`
默认 50,warp-per-problem + shfl 全照旧。**凡是这些,本文一律说一句"和 s9 一样"就跳过。**

s9 是个 toy:树是写死在文件里的 3 个 fixture(powerlaw / quadratic / expsat),
对拍 `c_final ≈ c_true` 打印 PASS。生产版要做的是把这同一台 LM 引擎接到【真实输入输出】上,
并且【扛住真种群里那些会崩的树】。所以 s10 全部是 **host 端的工程加固**,一条条对应:

1. 从文件读一批树(loader),不再写死 fixture;
2. 给每棵树一个 0–4 的状态码(status),区分"收敛 / 跑满没收敛 / 几种崩法";
3. K=0 的树(没有常数可优化)开跑前直接跳过;
4. 初始 loss 就是 NaN 的树,直接判死,别浪费迭代;
5. **把 Cholesky 失败单独归一类 `status=4`**:s9 已经会读 `d_stat`、把 solve 失败当 reject
   (s8 承诺、s9 兑现);生产在此之上把"反复崩到 λ 爆"单独标成 `FAIL_CHOLESKY`,不混进别的死法;
6. 单步算出来的 `loss_try` 是 NaN,放弃这棵;
7. λ 一路涨爆,判失败;
8. 把结果写成两个二进制 blob:`status.bin` + `c_final.bin`;
9. 命令行参数 + 输出目录的默认值处理。

下面逐节讲,每条都钉在 `batch_lm.cu` 的真实行号上。

> 先约定两个大白话:
> - **status(状态码)** = 一个 `int`,记一棵树"最后落到哪个结局"。生产要对 1000 棵树
>   一次性跑,跑完得告诉调用方每棵的下场,所以需要一个码。
> - **blob** = 一坨没有结构的二进制字节,直接 `fwrite` 到文件;读的人按约定的长度和类型
>   自己切。我们不用 CSV/JSON,纯字节,Python 端 `numpy.fromfile` 一句就读回来。

---

## (1) 文件 loader:`pop.bin` 怎么解包成那几个并排数组

s9 里树是怎么来的?手写在文件里 —— `FIXTURES[]` 三个 `TreeSpec`,`main` 开头一段 `for`
循环把它们的 `nt/nv/ci` 按 `node_offset` 拼进并排大数组,把 `K` 累成 `c_offset`,
`c_init = c_true * 0.9`,`xs` 现场算,`ym` 用解析公式现场生成。**生产版把这一整段换成"读文件"。**

新概念为零:读完文件后,内存里的数据布局和 s9【完全一样】—— 还是那几个并排的大数组
(`nt_all` / `nv_all` / `ci_all`)+ 一个 `TreeMeta[]`。变的只是"这些数从哪来"。

**文件格式**(`pop_format.h:1-53`)。`pop.bin` 是一个写死小端序的纯二进制文件,头 64 字节是
`PopHeader`(`pop_format.h:32-43`),记了全局尺寸:`M_prob`(树数)、`total_nodes`(所有树节点数之和)、
`total_c`(所有树常数个数之和,= Σ K)、`N`(数据点数)、`n_vars`、`K_max`(最大的 K)、`max_stack`。
头后面依次是几大块并排数组(layout 见 `pop_format.h:6-16`):`nt_all` / `nv_all` / `ci_all`
各 `total_nodes` 个,然后 `metas`(每棵树一个 16 字节 `TreeMeta{node_offset, n_nodes, c_offset, K}`,
`pop_format.h:46-51`),然后 `c_init`(`total_c` 个 float,就是 s9 里手填的那个 c 起点)、
`xs`(`N*n_vars` 个)、`ym`(`M_prob*N` 个,真值 y)。

> 注意 `ym` 的布局是 `ym[m*N + i]` —— 每棵树存了一份自己的 y(`pop_format.h:14`)。
> 真实 EvoGP 跑时所有树共享同一份 y,dump 端复制了 M 份,MVP 不省这点空间。
> 这就是为什么 `residual_kernel` 里 `ym` 是 `[M_prob*N]` 而不是 `[N]`(`batch_lm.cu:131-135`)。

**怎么读**(`loader.c` + 在 `batch_lm.cu:270-273` 调用)。`load_pop_bin` 打开文件,先把 64 字节头
读进 `PopData.header`,然后做三道校验,任何一道不过就报错返回非 0:
- magic 必须 `== POP_MAGIC`(那 4 个字节 `'M''4''L''M'`,`loader.c:25-29`)—— 防止把别的文件喂进来;
- version 必须 `== POP_VERSION`(`loader.c:30-34`);
- 读完所有块后做一致性 sanity:`Σ metas[i].K` 必须等于 header 里的 `total_c`,
  `Σ metas[i].n_nodes` 必须等于 `total_nodes`(`loader.c:57-68`)。对不上说明文件内部矛盾,拒绝。

校验全过,`load_pop_bin` 给每块 `malloc` 一段 host 内存并 `fread` 填满
(`loader.c:37-55`),返回的 `PopData`(定义见 `loader.h:13-23`)就是 s9 里那堆 `h_*` 数组的
"从文件版"。`batch_lm.cu` 里把它们 `memcpy`/直接当 host 指针用,再像 s9 一样
`cudaMemcpy` 上 GPU(`batch_lm.cu:322-327`)。

**加固:运行时 guard**(`batch_lm.cu:280-289`)。s9 也有一条 `K_max > MAX_K` 的检查
(m3_4c:224-228),因为 `solve_kernel` 里 `float A[MAX_K*MAX_K]` 是编译期定死的栈数组,
真 K 超了就越界。生产版把这条保留,**再加一条对称的** `max_stack > MAX_STACK`(解释器栈
`float stack[MAX_STACK]` 同理),两条都触发就打印"增 MAX_K/MAX_STACK 重编"并 `return 2`。
这是文件来源不可控之后必须补的:fixture 时尺寸你自己定,读真文件时尺寸是别人给的,得先验。

**加固:更全的算子表**。s9 的解释器只认 11 个算子(m3_4c:34-37)。生产树来自真 EvoGP,
算子种类多得多,所以 `FuncE` 扩到了 20 多个(`batch_lm.cu:37-44`),解释器的 `switch`
也跟着补全(UFUNC 分支 `batch_lm.cu:71-88`,BFUNC 分支 `batch_lm.cu:89-107`,新增了
`TAN/SINH/COSH/TANH/INV/ABS` 和 `MAX/MIN/LT/GT/LE/GE` 这些)。算法没变,就是 case 多了。

> 一个要留意的设计点:两个 `switch` 的 `default` 都是 `r=0.0f` / `o=0.0f`
> (`batch_lm.cu:86, 104`),注释叫它 "unknown op -- silent failure 入口"
> (`batch_lm.cu:35-36`)。意思是:如果一棵树带了解算端不认识的算子,它【不会报错】,
> 而是悄悄当成 0 算下去 —— 结果多半发散或 NaN,最后落进下面那几个 fail 状态码。
> 这是生产取的折中:宁可让坏树自己崩进 status,也不让一个未知 op 把整批 1000 棵搞挂。

---

## (2) status 状态码 0–4,和内部 `h_finished` 的编码

s9 里"结局"只有两种:`h_finished[m]==1` 算收敛,`==2` 算失败(λ 爆),`==0` 没标过就是跑满
(m3_4c:417/429,以及末尾对拍)。生产要给调用方一个干净的状态码,所以定了 5 个
(`batch_lm.cu:50-60`):

| status | 名字 | 含义 |
|--------|------|------|
| 0 | `STATUS_CONVERGED` | `d_norm < xtol*(c_norm+xtol)`,正常收敛 |
| 1 | `STATUS_MAXITER` | 没收敛也没崩,`max_iter` 用完了 |
| 2 | `STATUS_FAIL_NAN` | loss 或 loss_try 出了 NaN/Inf(eval 把树推到奇点) |
| 3 | `STATUS_K0_SKIP` | K=0,根本没常数可优化 |
| 4 | `STATUS_FAIL_CHOLESKY` | Cholesky 反复失败,λ 爆 `> 1e12` |

**关键陷阱:循环里用的内部码 `h_finished`,和最后写出去的 `status_out`,不是同一套数!**

循环跑的时候,每棵树的实时状态记在 `h_finished[m]`(`batch_lm.cu:336` 分配)。它的内部编码
(注释在 `batch_lm.cu:432-438`)是:

- `0` = 还在跑(主循环每轮看 `==0` 决定继不继续跑这棵);
- `1` = 已收敛;
- `2` = FAIL_NAN;
- `3` = K=0 skip;
- `4` = FAIL_CHOLESKY。

> 源码里 `batch_lm.cu:432` 那句注释写着 "internal, == status_out 的 enum" —— **这句不准,别信它**。
> 看末尾真正的映射就知道 1 和 0 在出门时换了意思。

跑完循环后,在 `batch_lm.cu:502-511` 把内部 `h_finished` 翻译成外部 `status_out`:

```
h_finished==1  →  status_out = 0  (CONVERGED)
h_finished==2  →  status_out = 2  (FAIL_NAN)
h_finished==3  →  status_out = 3  (K0_SKIP)
h_finished==4  →  status_out = 4  (FAIL_CHOLESKY)
其它(==0)     →  status_out = 1  (MAXITER)
```

也就是说,**内部 1 和外部 0 互换、内部 0 翻成外部 1**,2/3/4 原样穿过。原因很简单:循环里
`h_finished==0` 是"非零才算结束"的哨兵(`!h_finished[m]` 当布尔用,见 `batch_lm.cu:363, 387, 418, 440`),
所以"还在跑"必须是 0;但对外 0 留给最常见、最该当默认的结局 CONVERGED。两套各自最方便,
中间隔一层翻译,所以读这份代码时千万别把循环里的 `h_finished` 数字直接当成最终 status 读。

---

## (3) K=0 预跳过

什么是 K=0?一棵树里【一个常数都没有】(比如 `x0 * x0`,纯结构,没有可调系数)。LM 是
"调常数找最优",没常数可调,跑它就是空转。

s9 的 fixture 三棵都有常数,没这个情况,所以 s9 没专门处理(它只在收敛判据里顺带提了一句
"K=0 树 d_norm==0 会立即收敛",m3_4c:410)。生产种群里一大票 K=0 的树,得在开跑前就摘掉。

`batch_lm.cu:341-345`:进主循环【之前】扫一遍,凡 `metas[m].K==0` 的,直接
`h_finished[m]=3`(K0_SKIP),并计个数 `n_k0_pre`。这样它从第一轮起就不满足 `!h_finished[m]`,
整个 LM 循环都不碰它,最后映射成 `status=3`。它的 `c_final` 就是原样的 `c_init`(没动过)。

---

## (4) 初始 loss 是 NaN:直接判死

新概念为零,纯防御。生产版在主循环【之前】先做一次完整的 eval → residual → loss
(`batch_lm.cu:350-358`,和 s9 的 init 段 m3_4c:301-309 一模一样),把每棵树在 `c_init`
处的 loss 拿到 host。

加固在 `batch_lm.cu:360-365`:扫一遍 `h_loss`,凡是 `!isfinite(h_loss[m])` 的(NaN 或 Inf),
直接 `h_finished[m]=2`(FAIL_NAN)。道理(注释 `batch_lm.cu:360-361`):**连起点 `c_init` 都
产生 NaN 的树,LM 救不了它** —— LM 只会沿着这个起点往下走,起点就坏,后面只会更坏。早判死,
省掉它后面所有迭代的算力。(注意这里先 `if (h_finished[m]) continue`,跳过已经被 K=0 摘掉的,
别覆盖它们的 status 3。)

这就是第一条"NaN funnel 进 status 2"的入口。后面 (6)(7) 还有两条,殊途同归。

---

## (5) Cholesky 失败这条路:`solve_kernel` 设 `status=-1`,host 必须去读 —— 否则假阳性"收敛"

**s9 已经把这条路的核心拦截写对了(对齐生产);生产在此之上多给"反复崩"一个专属状态码。
先讲清这条路的机关 —— 即使 s9 的 fixture 从不触发它,读生产代码也必须懂。**

机关:`solve_kernel` 在 Cholesky 分解时,如果某个对角元 `s <= 0`(矩阵在浮点下没保持正定、
数值崩了),会 `status[m] = -1`、把这棵树的 `delta` 全写 0,然后 `return`(`batch_lm.cu:205`,
和 s9 的 `solve_kernel` 一字不差)。**kernel 这边没问题,坑在 host 怎么用这个信号。**

陷阱:`delta` 全 0 时,收敛判据 `d_norm < xtol*(c_norm+xtol)` 里 `d_norm=0`,**无条件成立** ——
host 若不看 `status`,这棵崩掉的树就会被当成"步子小到收敛了"接受掉,打个假的 CONVERGED。
所以 host **必须**把 `d_stat` 拷回来、在收敛判据【之前】把它拦掉。

**s9 已经把这条拦截写对了**(你自己写的):`solve` 之后拷回 `h_solve_stat`,F 步开头先
`if (h_solve_stat[m] != 0)` 当 reject 处理 —— 正是 s8 承诺"solver 失败交给 s9 当拒绝"的兑现。
生产的三句在 `batch_lm.cu:371`(声明 + 注释"必须读…避免假阳性 converge")、`:413`(拷回)、
`:446-451`(最先检查),逻辑和 s9 完全一致。

> 历史插曲:这条路在原型 `m3_4c` 里是【故意】留着没修的 —— 注释写明 "⚠️ KNOWN BUG…留作
> m3 oracle 回归用,m4 已修"(`m3_4c:367-373`):`d_stat` 算了却从不拷回,崩树会假收敛;3 棵
> 良态 archetype 从不触发,真种群才暴露(2026-05-24)。s9 已按生产做法修好,不带这个 bug。

**那么生产相对 s9 真正【新增】的是什么?** —— 把"反复崩、λ 爆过 `1e12`"单独归成一个专属
终态 `status=4 FAIL_CHOLESKY`(`batch_lm.cu:449`):

```c
if (h_solve_stat[m] != 0) {     // Cholesky 崩了
    h_lam[m] *= 10.0f;          // 当作一次 reject:加阻尼(同 s9)
    h_rejected[m]++;
    if (h_lam[m] > 1e12f) h_finished[m] = 4;  // 生产新增:反复崩、λ 爆 → 专属 FAIL_CHOLESKY
    continue;                   // 不许往下走到 d_norm 那条判据
}
```

s9 这里偷懒:同样的情形它折进了 `h_finished=2`(和 λ 爆、NaN 混成一类)。生产要对 1000 棵树
事后统计,得分清"这棵是 Cholesky 崩死的"还是"别的原因死的",所以给它独立编号。

**这也是一处"PASS≠正确"的活教材**:s9 的 3 棵 fixture 良态,Cholesky 失败这条分支在 s9 里
【从来没被实际走到过】—— 它 PASS,不代表这条路对,只代表测试没覆盖到它。真 EvoGP 种群里
畸形树多,这条路天天走,所以生产才舍得为它单开一个状态码 + 专门写注释提醒"d_stat 必读"。

---

## (6) `loss_try` 是 NaN:放弃这棵

承接 (4),只是从"起点 NaN"挪到了"走一步之后 NaN"。

每一轮单步走完,算出试探解 `c_try` 处的 `loss_try`(`batch_lm.cu:415-429`,流程同 s9)。在
accept/reject 步,Cholesky 检查过了之后,紧接着检查 `batch_lm.cu:456-459`:
若 `!isfinite(h_loss_try[m])`,直接 `h_finished[m]=2`(FAIL_NAN),`continue`。

为什么是"放弃"而不是"reject 后重试"?注释讲得很清楚(`batch_lm.cu:453-455`):reject 的
常规做法是 `λ*=10` 再来一次,指望更小的步长躲开奇点。但这里 `c_try` 之所以 NaN,是因为
`δ` 把树 eval 推到了 Inf/NaN;而 `δ` 是由当前 `c` 处的 `J`/`r` 决定的,`c` 没被污染、`J`/`r`
也一样,所以下一轮 `λ*=10` 后会算出**方向相同、只是更短**的 `δ` —— 大概率还是把树推向同一个
奇点。与其原地打转耗 max_iter,不如直接判死。这是第二条 NaN funnel → status 2。

---

## (7) λ 一路涨爆 → 失败

这是普通 reject 分支(`batch_lm.cu:480-484`),整段连同那条收尾
`if (h_lam[m] > 1e12f) h_finished[m] = 2;` 都和 s9 一字不差,s10 在这条分支上**没动一行代码**:
loss 没降也没收敛 → `λ*=10`、`h_rejected[m]++`,λ 都涨到 `1e12`(reject 几乎全程压倒 accept,
这棵在当前起点附近怎么都下不去)就判死。s10 在这里加的不是代码,是状态码语义和下面那条警告:
内部 `h_finished=2` 出门映射成外部 FAIL_NAN,且别把这条 `:483` 的 λ 爆路径和 (5) 里 `:449`
那条 Cholesky λ 爆路径搞混。

> **务必和 (5) 分清楚:这是两条不同的 λ 爆路径,落在不同的状态码上。**
> - `batch_lm.cu:449`(Cholesky 失败分支里)λ 爆 → `h_finished=4`(FAIL_CHOLESKY);
> - `batch_lm.cu:483`(普通 reject 分支里)λ 爆 → `h_finished=2`(FAIL_NAN)。
>
> 后者归到 FAIL_NAN 桶,注释解释了取舍(`batch_lm.cu:483`):没崩 Cholesky 却一路 reject 到
> λ 爆,通常是 loss 面在这附近全是 NaN-ish 的烂地形,归进 NaN 桶比单开一类更诚实。

至此,status 2(FAIL_NAN)一共有三条进来的路,记住这个就不会在读代码时把它们搞混:
**初始 loss NaN(`:364`)、单步 loss_try NaN(`:457`)、普通 reject 把 λ 顶爆(`:483`)。**

---

## (8) 输出 blob:`status.bin` + `c_final.bin`

s9 没有输出文件,它在 `main` 末尾直接把 `c_final` vs `c_true` 打印出来对拍 PASS/FAIL
(m3_4c:449-475)。生产版【不对拍】(没有 `c_true` 可对,真种群哪来真值),而是把结果落盘
给下游(GP 主循环)读。

`batch_lm.cu:502-511`:先把内部 `h_finished` 翻译成外部 `status_out[]`(就是 (2) 讲的那张映射表),
顺手统计 `n_conv / n_maxiter / n_fail_nan / n_k0 / n_fail_chol` 各多少棵。

然后写两个 blob(`batch_lm.cu:513-517`,用辅助函数 `write_blob`,`batch_lm.cu:230-237`):
- `status.bin` = `status_out[]`,`M_prob` 个 `int32`,每棵树一个 0–4 的码;
- `c_final.bin` = `h_c`,`total_c` 个 `float32`,**按每棵树各自的 K jagged 拼起来**
  (第 m 棵的常数在 `[metas[m].c_offset, +K)`)。读的人拿 `metas` 里的 `c_offset`/`K` 切片。

`write_blob` 很朴素:`fopen("wb")` → `fwrite` 整块 → 校验写够字节数(短写就报错退出)→ `fclose`。
没有任何格式头,纯字节。Python 端 `numpy.fromfile(..., dtype=...)` 直接读回。

> 一个不起眼但真实的工程细节:拼路径的 buffer 是 `char buf[2048]`(`batch_lm.cu:513`),
> 注释说"`> dir_buf[1024] + "/c_final.bin"` 留余地,避 snprintf fortify 警告"。就是把目录
> (最长 1024)加文件名拼进去时,给编译器的 `_FORTIFY_SOURCE` 留够空间别报溢出警告。

跑完还会打印一行汇总和耗时(`batch_lm.cu:524-532`,`--quiet` 可关),包括 trees/s 吞吐;
计时用 `clock_gettime(CLOCK_MONOTONIC)`,起止点在 `batch_lm.cu:244-245` 和 `519-522`。

---

## (9) 命令行参数 + 输出目录默认值

s9 是 `int main()`,不吃参数。生产版是 `int main(int argc, char **argv)`,用法
(`batch_lm.cu:17-18, 239-243`):

```
./batch_lm <pop.bin> [out_dir] [--max-iter N] [--quiet]
```

参数解析在 `batch_lm.cu:246-254`:`argv[1]` 是必需的 `pop_path`;之后循环扫剩下的,
`--max-iter N` 覆盖默认的 50(`batch_lm.cu:248, 251`),`--quiet` 关掉打印
(`batch_lm.cu:249, 252`),任何不以 `-` 开头的位置参数当成 `out_dir`(`batch_lm.cu:253`)。

**out_dir 的默认值**(`batch_lm.cu:255-268`):没显式传 `out_dir` 时,从 `pop_path` 里取
dirname —— 找最后一个 `/`,把它前面那段拷进 `dir_buf` 当输出目录。所以
`./batch_lm data/pop.bin` 会把 `status.bin`/`c_final.bin` 写进 `data/`;
`./batch_lm pop.bin`(没有 `/`)则落到当前目录 `"."`。意图是:输出默认跟输入放一起,省得每次都打目录。

---

## 现在你能整页读懂 `batch_lm.cu` 了

把 s9 → s10 的全部增量收一下,全是 host 端工程,没有一个新 CUDA 概念:

- **输入端**:写死的 fixture → `load_pop_bin` 读 `pop.bin`(magic/version/sanity 三校验,
  `loader.c:25-68`),解包成和 s9 完全相同的并排数组 + `TreeMeta`;读真文件了,所以补了
  `K_max`/`max_stack` 两条运行时 guard(`:280-289`)和更全的算子表(`:37-44, 71-107`,
  未知 op 走 `default 0.0f` 静默失败)。
- **状态机**:5 个 status 码(`:50-60`);循环里用内部 `h_finished`(0=在跑、1=收敛、
  2/3/4=各种结局),出门时在 `:502-511` 翻译成外部 `status_out`(**内部 1↔外部 0、内部 0→外部 1**,
  别被 `:432` 那句注释骗)。
- **四种坏树都被接住**,各归其码:K=0 预跳过 → 3(`:341-345`);初始 loss NaN → 2(`:360-365`);
  Cholesky 崩(host 必读 `h_solve_stat` 才不会假阳性收敛,`:413, 446-451`),λ 爆 → 4(`:449`);
  loss_try NaN → 2(`:456-459`);普通 reject 把 λ 顶爆 → 2(`:483`)。NaN 三条路全汇进 status 2。
- **输出端**:不再 print 对拍,而是 `write_blob` 写 `status.bin` + `c_final.bin`(`:503-517`);
  CLI 参数 + out_dir 从 `pop_path` 取 dirname 默认(`:240-268`)。

中间那台 LM 引擎 —— 解释器、warp-per-problem、shfl 归约、FD Jacobian、`JᵀJ`/`JᵀR`、Cholesky、
λ 调度、收敛判据 —— **和你在 s1..s9 一路搭出来的【一模一样】**。生产版的全部价值,就在它周围
这一圈"喂得进真文件、扛得住坏树、吐得出可读结果"的硬壳。

---

## 自检(先自己答,再看下面)

1. 循环里看到某棵树 `h_finished[m] == 1`,它最后写进 `status.bin` 的码是几?`== 0` 又是几?
2. 有两处 `if (h_lam[m] > 1e12f) h_finished[m] = ...`,一处赋 4 一处赋 2。怎么记住哪处是哪个?
3. 如果有人把 `batch_lm.cu:413` 那句 `cudaMemcpy(h_solve_stat, ...)` 删掉,会复现什么 bug?
4. K=0 的树,它的 `c_final.bin` 里那段值是什么?

## 参考答案

1. **`h_finished==1` → status `0`(CONVERGED);`h_finished==0` → status `1`(MAXITER)。**
   两者在 `batch_lm.cu:506` 和 `:510` 的映射里互换了。原因:循环里用 `!h_finished[m]` 当
   "还没结束"的判断,所以"在跑"必须是 0;但对外要把最常见的好结局 CONVERGED 放成默认的 0。
   `:432` 注释说"internal == external"是错的,以 `:502-511` 的映射代码为准。

2. `batch_lm.cu:449` 在 **Cholesky 失败分支**里(`h_solve_stat[m] != 0` 那个 `if` 内),λ 爆判
   **4 = FAIL_CHOLESKY**;`batch_lm.cu:483` 在**普通 reject 分支**里(loss 没降),λ 爆判
   **2 = FAIL_NAN**。记法:"先崩 Cholesky 才爆的"才算 Cholesky 失败;纯粹下不去坡爆的,
   归进 NaN-ish 桶。

3. 复现 m3_4c 故意留着、而 s9/生产都已修好的那个假阳性 bug(m3_4c:367-373):`solve_kernel` 在 Cholesky breakdown 时把 δ 写 0、
   `status=-1`,但 host 读不到这个 -1,只看到 δ=0;收敛判据 `d_norm < xtol*(c_norm+xtol)` 被
   `d_norm=0` 无条件满足 → 一棵崩掉的树被误判成 CONVERGED。良态 fixture(s9)碰不到,真种群一跑就露。

4. 就是它的 `c_init`,原封不动。K=0 树在 `batch_lm.cu:341-345` 被预跳过,整个 LM 循环都没碰过它的
   `h_c`,而 `h_c` 一开始就是从 `pop.c_init` 拷来的(`batch_lm.cu:298`)。
