# Parameter-sensitivity (trees/s) — locked 1410 MHz

## throughput vs M (early-gen, N=1000)

| variant | M=1000 | M=4000 | M=16000 | M=64000 | M=256000 |
|---|---|---|---|---|---|
| fusedfd | 5503 | 19090 | 38711 | 49664 | 58882 |
| ad | 10408 | 30482 | 51655 | 69430 | 67780 |
| revad | 13821 | 36999 | 61731 | 77111 | 77327 |

## throughput vs N (early-gen, M=64000)

| variant | N=100 | N=1000 | N=10000 |
|---|---|---|---|
| fusedfd | 196236 | 49664 | · |
| ad | 220315 | 69430 | · |
| revad | 318209 | 77111 | · |
