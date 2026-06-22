// _probe_corpus_silent.cu — does safe_mul produce SILENTLY-WRONG finite Jacobian entries
// (Codex's removable-singularity concern) in the REAL workload?
//
// For every (tree,point,col): compute plain-multiply VJP (pre-fix) and the library's safe_mul VJP.
// "Affected" = plain is non-finite but safe is finite (safe_mul changed the answer).
// For affected elements, compare safe value against a robust in-domain FD oracle (central; falls back
// to whichever one-sided step stays finite). Flag SILENT-WRONG when oracle is finite & clearly nonzero
// & |safe - oracle| is a large relative gap (Codex's 0-instead-of-nonzero). Host-only, full corpus.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../revad_interp.cuh"   // eval_tree_vjp_d (SAFE) + eval_tree_val_host + enums
#include "../loader.h"

// plain-multiply VJP (pre-fix behaviour): identical to eval_tree_vjp_d but adj*partial (no annihilation).
static void plain_vjp(const int *nt, const float *nv, int n, const int *ci,
                      const float *x, const float *c, int K, float *out) {
    float sv[MAX_STACK]; float d1[MAX_NODES], d2[MAX_NODES]; int sp = 0;
    for (int i = n - 1; i >= 0; i--) {
        int t = nt[i]; float v = nv[i];
        if (t == N_VAR) sv[sp++] = x[(int)v];
        else if (t == N_CONST) sv[sp++] = c[ci[i]];
        else if (t == N_UFUNC) {
            int fid = (int)v; float a = sv[--sp], r, d;
            switch (fid) {
                case F_SIN: r=sinf(a); d=cosf(a); break;       case F_COS: r=cosf(a); d=-sinf(a); break;
                case F_TAN: r=tanf(a); d=1.0f+r*r; break;      case F_SINH: r=sinhf(a); d=coshf(a); break;
                case F_COSH: r=coshf(a); d=sinhf(a); break;    case F_TANH: r=tanhf(a); d=1.0f-r*r; break;
                case F_LOG: r=logf(a); d=1.0f/a; break;        case F_EXP: r=expf(a); d=r; break;
                case F_INV: r=1.0f/a; d=-r*r; break;           case F_NEG: r=-a; d=-1.0f; break;
                case F_ABS: r=fabsf(a); d=(a>0.f)?1.f:((a<0.f)?-1.f:0.f); break;
                case F_SQRT: r=sqrtf(a); d=1.0f/(2.0f*r); break; default: r=0; d=0;
            }
            d1[i]=d; sv[sp++]=r;
        } else if (t == N_BFUNC) {
            int fid=(int)v; float l=sv[--sp], rv=sv[--sp], o, dl, dr;
            switch (fid) {
                case F_ADD: o=l+rv; dl=1; dr=1; break;          case F_SUB: o=l-rv; dl=1; dr=-1; break;
                case F_MUL: o=l*rv; dl=rv; dr=l; break;         case F_DIV: o=l/rv; dl=1.0f/rv; dr=-o/rv; break;
                case F_POW: o=powf(l,rv); dl=rv*powf(l,rv-1.0f); dr=o*logf(l); break;
                case F_MAX: o=fmaxf(l,rv); dl=(l>=rv)?1.f:0.f; dr=(l>=rv)?0.f:1.f; break;
                case F_MIN: o=fminf(l,rv); dl=(l<=rv)?1.f:0.f; dr=(l<=rv)?0.f:1.f; break;
                case F_LT: o=(l<rv)?1.f:0.f; dl=0; dr=0; break; case F_GT: o=(l>rv)?1.f:0.f; dl=0; dr=0; break;
                case F_LE: o=(l<=rv)?1.f:0.f; dl=0; dr=0; break;case F_GE: o=(l>=rv)?1.f:0.f; dl=0; dr=0; break;
                default: o=0; dl=0; dr=0;
            }
            d1[i]=dl; d2[i]=dr; sv[sp++]=o;
        }
    }
    for (int k=0;k<K;k++) out[k]=0.0f;
    float sa[MAX_STACK]; int asp=0; sa[asp++]=1.0f;
    for (int i=0;i<n;i++) {
        int t=nt[i];
        if (t==N_VAR) --asp;
        else if (t==N_CONST) { float adj=sa[--asp]; out[ci[i]] += adj; }
        else if (t==N_UFUNC) { float adj=sa[--asp]; sa[asp++]=adj*d1[i]; }   // PLAIN multiply
        else if (t==N_BFUNC) { float adj=sa[--asp]; sa[asp++]=adj*d2[i]; sa[asp++]=adj*d1[i]; }
    }
}

// robust in-domain FD of column k: central; fall back to whichever one-sided step is finite.
static float robust_fd(const int *nt, const float *nv, int n, const int *ci,
                       const float *x, const float *c, int K, int k) {
    float ck=c[k]; float h=1e-3f*fabsf(ck); if (h<1e-4f) h=1e-4f;
    float cp[MAX_K]; for (int j=0;j<K;j++) cp[j]=c[j];
    cp[k]=ck+h; float fp=eval_tree_val_host(nt,nv,n,ci,x,cp);
    cp[k]=ck-h; float fm=eval_tree_val_host(nt,nv,n,ci,x,cp);
    float f0=eval_tree_val_host(nt,nv,n,ci,x,c);
    if (isfinite(fp) && isfinite(fm)) return (fp-fm)/(2*h);
    if (isfinite(fp) && isfinite(f0)) return (fp-f0)/h;     // one-sided +
    if (isfinite(fm) && isfinite(f0)) return (f0-fm)/h;     // one-sided -
    return NAN;
}

int main(int argc, char **argv) {
    const char *path = (argc>1)?argv[1]
        : "/home/weish/hao/CuSR/data/workload/synth/synth_inner-const-heavy_M4000_N1000_seed0.bin";
    PopData pop; if (load_pop_bin(path,&pop)) { fprintf(stderr,"load failed\n"); return 2; }
    const PopHeader *h=&pop.header; int N=h->N, nv_=h->n_vars;
    long long n=0, affected=0, oracle_finite=0, silent_wrong=0; int ex=0;
    for (int m=0;m<h->M_prob;m++) {
        TreeMeta mt=pop.metas[m]; int K=mt.K; if (K==0) continue;
        const int *nt=pop.nt+mt.node_offset; const float *nvv=pop.nv+mt.node_offset;
        const int *ci=pop.ci+mt.node_offset; const float *c=pop.c_init+mt.c_offset;
        for (int i=0;i<N;i++) {
            const float *x=pop.xs+(size_t)i*nv_;
            float gs[MAX_K], gp[MAX_K];
            eval_tree_vjp_d(nt,nvv,mt.n_nodes,ci,x,c,K,gs);   // SAFE
            plain_vjp(nt,nvv,mt.n_nodes,ci,x,c,K,gp);          // PLAIN
            for (int k=0;k<K;k++) {
                n++;
                bool aff = !isfinite(gp[k]) && isfinite(gs[k]);   // safe_mul changed nan -> finite
                if (!aff) continue;
                affected++;
                float fd = robust_fd(nt,nvv,mt.n_nodes,ci,x,c,K,k);
                if (!isfinite(fd)) continue;
                oracle_finite++;
                float rel = fabsf(gs[k]-fd)/(fabsf(fd)+1e-3f);
                if (fabsf(fd) > 1e-2f && rel > 0.1f) {            // oracle clearly nonzero & safe disagrees
                    silent_wrong++;
                    if (ex<10){ fprintf(stderr,"  SILENT-WRONG m=%d i=%d k=%d  safe=%g  FD=%g\n",m,i,k,gs[k],fd); ex++; }
                }
            }
        }
    }
    printf("[corpus_silent] pop=%s\n", path);
    printf("  elements                         = %lld\n", n);
    printf("  affected (plain nan -> safe finite) = %lld\n", affected);
    printf("  of affected, oracle FD finite    = %lld\n", oracle_finite);
    printf("  SILENT-WRONG (safe!=FD, FD nonzero) = %lld\n", silent_wrong);
    free_pop_data(&pop);
    if (silent_wrong==0) { printf("RESULT: no silent-wrong entries in this corpus\n"); return 0; }
    printf("RESULT: %lld silent-wrong entries — Codex's concern occurs here\n", silent_wrong);
    return 1;
}
