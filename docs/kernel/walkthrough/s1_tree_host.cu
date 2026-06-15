// s1_tree_host.cu — 一棵表达式树:怎么存,怎么求值。
//
// 这是整个 batch_lm.cu 的地基,而且它【和 GPU 完全无关】—— 所以我们先在
// 普通 CPU 代码里把它吃透,一个 CUDA 概念都不引入。下一级(s2)才把这同
// 一个函数原样搬上 GPU,那时你只需要操心"搬"这一件新事。
//
// 跑: ./build.sh s1     预期最后一行 PASS。
//
// 学完这一级,你应该能回答:
//   - 一棵树为什么用 3 个数组 + 一个 c[] 存,各存什么?
//   - "逆前缀(reverse prefix)+ 栈" 怎么把一串节点算成一个数?
//   - BFUNC 弹栈时,先弹出来的 l 到底是左操作数还是右操作数?(最易错)
// 答案在文件末尾 ## 自检。

#include <cstdio>
#include <cmath>

// ---------------------------------------------------------------------------
// 1. 节点种类 / 算子编号 —— 数值和生产代码 batch_lm.cu 严格一致。
//    (生产里还有一大堆算子,这一级 fixture 只用到 MUL 和 POW,先放够用的。)
// ---------------------------------------------------------------------------
enum NTypeE { N_VAR = 0, N_CONST = 1, N_UFUNC = 2, N_BFUNC = 3 };
// BFUNC = binary function(双目算子,吃 2 个操作数)
enum FuncE { F_ADD = 1, F_SUB = 2, F_MUL = 3, F_DIV = 4, F_POW = 6 };

#define MAX_STACK 64   // 栈深上限,和生产一致

// ---------------------------------------------------------------------------
// 2. 解释器:给定一棵树 + 一个 x + 一组常数 c,算出 y。
//    这个函数和 batch_lm.cu 里的 eval_tree_d 逻辑【一字不差】,只是那边带了
//    __device__(在 GPU 上跑)。s2 就是给它加上 __device__,别的不动。
//
//    存树的约定(前缀序 / prefix order,即"算子在前,操作数在后"):
//      nt[i]        第 i 个节点的【种类】(VAR/CONST/UFUNC/BFUNC)
//      nv[i]        第 i 个节点的【参数】:
//                     VAR   -> 变量编号(x 的第几维)
//                     CONST -> 占位,不用(真值在 c[] 里,见下)
//                     BFUNC -> 哪个算子(F_MUL / F_POW / ...)
//      const_idx[i] 若第 i 个节点是 CONST,它的真值是 c[const_idx[i]];
//                   否则填 -1。
//
//    为什么 CONST 的真值不直接存进 nv[],而要绕一层 c[] + const_idx?
//      —— 因为 LM 要反复【改常数】找最优。把常数单独放 c[],就能只换 c[]、
//         完全不动树结构。这是后面所有级的前提。
//
//    算法:【从最后一个节点倒着往前扫,配一个栈】。
//      碰到操作数(VAR/CONST)-> 把它的值压栈。
//      碰到算子(UFUNC/BFUNC)-> 从栈顶弹出它要的操作数,算完把结果压回去。
//    扫完,栈底 stack[0] 就是整棵树的值。
// ---------------------------------------------------------------------------
float eval_tree(const int *nt, const float *nv, int n_nodes,
                const int *const_idx, const float *x, const float *c)
{
    float stack[MAX_STACK];
    int sp = 0;  // stack pointer:栈里现在有几个数;压栈 stack[sp++],弹栈 stack[--sp]

#ifdef TRACE
    printf("eval_tree at x[0]=%.4f  (倒着扫 %d 个节点):\n", x[0], n_nodes);
#endif

    for (int i = n_nodes - 1; i >= 0; i--) {   // 倒着扫
        int t = nt[i];
        float v = nv[i];

        if (t == N_VAR) {
            stack[sp++] = x[(int)v];               // 压入 x 的第 v 维
        } else if (t == N_CONST) {
            stack[sp++] = c[const_idx[i]];         // 压入对应的常数
        } else if (t == N_BFUNC) {
            // 双目算子:弹 2 个。先弹出来的是 l,后弹的是 rv。
            // 【最易错的点】因为是倒着扫,先弹出来的 l 恰好是原式里的【左】
            // 操作数,后弹的 rv 是【右】操作数。所以非交换律的算子(减/除/
            // 幂)直接写 l OP rv 就对,不用调换。
            //   例: 原式 x ^ c1  存成 [POW, x, c1],倒扫到 POW 时栈顶是 x、
            //       次顶是 c1 -> l=x(底数), rv=c1(指数) -> powf(l,rv)=x^c1 ✓
            float l  = stack[--sp];
            float rv = stack[--sp];
            int fid = (int)v;
            float o;
            switch (fid) {
                case F_ADD: o = l + rv;        break;
                case F_SUB: o = l - rv;        break;   // 左减右,顺序要紧
                case F_MUL: o = l * rv;        break;
                case F_DIV: o = l / rv;        break;   // 左除以右
                case F_POW: o = powf(l, rv);   break;   // 底^指
                default:    o = 0.0f;          break;   // 不认识的算子 -> 0(生产里这是 silent failure 入口)
            }
            stack[sp++] = o;
        }
        // N_UFUNC(单目,如 sin/exp)这一级 fixture 用不到,s 后续级再加。

#ifdef TRACE
        // 单步可视化:处理完第 i 个节点后,打印当前整个栈
        static const char *nm[] = { "VAR", "CONST", "UFUNC", "BFUNC" };
        printf("  i=%d  %-5s  ->  栈(sp=%d): [", i, nm[t], sp);
        for (int s = 0; s < sp; s++) printf("%s%.4f", s ? ", " : "", stack[s]);
        printf("]\n");
#endif
    }
    return stack[0];
}

// ---------------------------------------------------------------------------
// 3. fixture:y = c0 * x^c1,c0=2.5, c1=1.3
//    前缀序写出来是:  MUL  c0  POW  x  c1   (共 5 个节点)
//
//    节点:  下标 0     1      2     3    4
//           ----------------------------------
//    nt:    BFUNC  CONST  BFUNC  VAR  CONST
//    nv:    F_MUL   --     F_POW  x0   --        (CONST 的 nv 是占位)
//    ci:     -1     0      -1     -1   1          (节点1->c[0], 节点4->c[1])
//    c:     {2.5, 1.3}
//
//    倒扫一遍验证一下(脑子里走一遍最有用):
//      i=4 CONST ci=1 -> 压 c[1]=1.3        栈: [1.3]
//      i=3 VAR   v=0  -> 压 x[0]            栈: [1.3, x]
//      i=2 BFUNC POW  -> l=x, rv=1.3        栈: [pow(x,1.3)]
//      i=1 CONST ci=0 -> 压 c[0]=2.5        栈: [pow(x,1.3), 2.5]
//      i=0 BFUNC MUL  -> l=2.5, rv=pow(...) 栈: [2.5*pow(x,1.3)]
//    stack[0] = 2.5 * x^1.3  ✓
// ---------------------------------------------------------------------------
int main(void)
{
    const int   nt[] = { N_BFUNC, N_CONST, N_BFUNC, N_VAR, N_CONST };
    const float nv[] = { F_MUL,   0.0f,    F_POW,   0.0f,  0.0f    };
    const int   ci[] = { -1,      0,       -1,      -1,    1       };
    const int   n_nodes = 5;
    float       c[]  = { 2.5f, 1.3f };

    // 几个测试用的 x(标量,n_vars=1),逐个对拍解析公式 2.5 * x^1.3
    const float xs[] = { 0.1f, 0.5f, 1.0f, 2.0f, 5.0f };
    const int   n_x  = 5;

    printf("fixture: y = c0 * x^c1,  c0=%.1f c1=%.1f\n\n", c[0], c[1]);

    int bad = 0;
    for (int i = 0; i < n_x; i++) {
        float xi = xs[i];
        float got = eval_tree(nt, nv, n_nodes, ci, &xi, c);  // 树解释器算的
        float ref = 2.5f * powf(xi, 1.3f);                   // 直接公式算的(oracle)
        float rel = fabsf(got - ref) / (fabsf(ref) + 1e-30f);
        bool ok = rel < 1e-5f;
        if (!ok) bad++;
        printf("  x=%.2f   tree=%.6f   ref=%.6f   rel=%.1e   %s\n",
               xi, got, ref, rel, ok ? "ok" : "MISMATCH");
    }

    printf("\n%s (%d/%d mismatches)\n", bad == 0 ? "PASS" : "FAIL", bad, n_x);
    return bad == 0 ? 0 : 1;
}

// ===========================================================================
// ## 自检(先自己答,再看下面)
//
//  1. nt / nv / const_idx / c 各存什么?为什么常数要单独放 c[] 而不塞进 nv[]?
//  2. 为什么是【倒着】扫节点(i 从 n_nodes-1 到 0)而不是正着?
//  3. BFUNC 里 `l = stack[--sp]; rv = stack[--sp];` —— l 是左操作数还是右?
//     把 fixture 改成 `x ^ c1`(去掉乘法),自己倒扫一遍确认 l 是底数。
//  4. 把 c[] 改成 {3.0, 2.0},不动树结构,结果该变成什么?(体会"改常数不动树")
//
// ## 参考答案
//
//  1. nt=节点种类,nv=节点参数(算子是哪个 / VAR 是第几维),const_idx=这个
//     CONST 节点取 c[] 的第几个,c=常数的真值。常数单独放 c[] 是为了让 LM
//     反复改常数时【只换 c[]、完全不碰树结构】—— 这是后面优化常数的前提。
//
//  2. 前缀序是"算子在前、操作数在后"。倒着扫,就保证【轮到一个算子时,它要
//     的操作数已经先被压进栈了】。正着扫的话碰到算子时操作数还没出现,栈是
//     空的,没法算。(这就是逆波兰求值的核心。)
//
//  3. l 是【左】操作数。倒扫时,原式左边的子树反而后压进栈、所以先弹出 ->
//     先弹的 l = 左。验证 `x^c1` = [POW, x, c1]:i=2 压 c1=1.3,i=1 压 x,
//     i=0 POW 弹出 l=x(底)、rv=1.3(指)-> powf(x,1.3) ✓。顺序对了,减/除/
//     幂这些非交换算子才不会算反。
//
//  4. c={3.0,2.0} -> y = 3.0 * x^2.0。比如 x=2 -> 3*4=12。树一个节点都没动,
//     只换了 c[] —— 这正是 LM 每次迭代在做的事。
// ===========================================================================
