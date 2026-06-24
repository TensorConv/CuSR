# ncu profiling — SOL + warp-stall + FP-pipe (locked 1410MHz, GPU7)

compute/mem/dram/issue = SOL %; FMA/fp64 = FP-pipe util %; GIPS = inst/duration (unit-normalized); stalls = warps-stalled-per-issue (small-M reps).

| variant | regime | M | kernel | compute% | mem% | dram% | issue% | FMA% | fp64% | GIPS | top stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ad | early-gen | 2000 | ad_jacobian | 9.8 | 19.2 | 0.70 | 29.5 | 2.0 | 0.0 | 46.0 | wait:2.97, long_scoreboard:2.58, selected:1.00 |
| ad | early-gen | 2000 | eval | 21.6 | 17.0 | 0.10 | 35.4 | 4.1 | 0.0 | 105.8 | wait:2.91, long_scoreboard:1.56, branch_resolving:1.14 |
| ad | early-gen | 64000 | ad_jacobian | 35.1 | 72.9 | 5.37 | 34.5 | 7.2 | 0.0 | - | - |
| ad | early-gen | 64000 | eval | 76.6 | 61.8 | 5.05 | 76.4 | 14.6 | 0.0 | - | - |
| ad | inner-const-heavy | 2000 | ad_jacobian | 19.5 | 49.8 | 2.09 | 28.2 | 4.0 | 0.0 | 96.7 | long_scoreboard:7.31, wait:2.98, selected:1.00 |
| ad | inner-const-heavy | 2000 | eval | 41.8 | 32.0 | 1.33 | 47.6 | 7.2 | 0.0 | 206.0 | wait:2.99, long_scoreboard:1.50, branch_resolving:1.21 |
| ad | inner-const-heavy | 64000 | ad_jacobian | 30.3 | 77.0 | 9.39 | 29.8 | 6.1 | 0.0 | - | - |
| ad | inner-const-heavy | 64000 | eval | 84.5 | 65.2 | 2.65 | 84.4 | 14.6 | 0.0 | - | - |
| fusedfd | early-gen | 2000 | fd_jacobian_fused | 7.4 | 5.8 | 0.34 | 24.3 | 1.1 | 0.0 | 36.6 | wait:2.92, long_scoreboard:1.79, branch_resolving:1.13 |
| fusedfd | early-gen | 2000 | eval | 21.6 | 17.0 | 0.17 | 35.4 | 4.1 | 0.0 | 105.8 | wait:2.91, long_scoreboard:1.55, branch_resolving:1.14 |
| fusedfd | early-gen | 64000 | fd_jacobian_fused | 50.6 | 40.8 | 4.08 | 50.5 | 7.4 | 0.0 | - | - |
| fusedfd | early-gen | 64000 | eval | 76.5 | 61.2 | 5.14 | 76.4 | 14.6 | 0.0 | - | - |
| fusedfd | inner-const-heavy | 2000 | fd_jacobian_fused | 25.7 | 19.9 | 0.77 | 39.4 | 3.4 | 0.0 | 126.3 | wait:3.00, long_scoreboard:1.57, branch_resolving:1.22 |
| fusedfd | inner-const-heavy | 2000 | eval | 42.1 | 32.1 | 0.17 | 47.8 | 7.3 | 0.0 | 206.4 | wait:2.99, long_scoreboard:1.50, branch_resolving:1.21 |
| fusedfd | inner-const-heavy | 64000 | fd_jacobian_fused | 67.3 | 53.1 | 3.14 | 67.2 | 8.8 | 0.0 | - | - |
| fusedfd | inner-const-heavy | 64000 | eval | 84.2 | 65.1 | 2.64 | 84.1 | 14.6 | 0.0 | - | - |
| revad | early-gen | 2000 | rev_jacobian | 20.2 | 28.0 | 1.52 | 35.4 | 4.2 | 0.0 | 99.6 | wait:2.95, long_scoreboard:1.86, selected:1.00 |
| revad | early-gen | 2000 | eval | 21.6 | 17.0 | 0.15 | 35.4 | 4.1 | 0.0 | 105.8 | wait:2.91, long_scoreboard:1.56, branch_resolving:1.14 |
| revad | early-gen | 64000 | rev_jacobian | 52.0 | 73.6 | 8.48 | 51.9 | 10.7 | 0.0 | - | - |
| revad | early-gen | 64000 | eval | 75.9 | 61.3 | 5.19 | 75.7 | 14.5 | 0.0 | - | - |
| revad | inner-const-heavy | 2000 | rev_jacobian | 35.7 | 51.1 | 6.95 | 46.7 | 6.5 | 0.0 | 174.9 | wait:2.97, long_scoreboard:2.59, selected:1.00 |
| revad | inner-const-heavy | 2000 | eval | 42.0 | 32.1 | 0.18 | 47.7 | 7.3 | 0.0 | 206.3 | wait:2.99, long_scoreboard:1.50, branch_resolving:1.21 |
| revad | inner-const-heavy | 64000 | rev_jacobian | 50.3 | 75.8 | 22.78 | 50.3 | 9.1 | 0.0 | - | - |
| revad | inner-const-heavy | 64000 | eval | 84.2 | 65.3 | 2.76 | 84.1 | 14.6 | 0.0 | - | - |
