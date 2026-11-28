| group | kernel | shape | core | old ns | new ns | new/old | old %peak | new %peak |
|---|---|---|---|---:|---:|---:|---:|---:|
| tile | packed (no FINISH: spill sums) | k=192 | Zen5 cpu2 | 579.4 | 580.1 | 1.0012 | 99.9 | 99.7 |
| tile | packed (no FINISH: spill sums) | k=192 | Zen5c cpu6 | 583.5 | 582.8 | 0.9988 | 99.6 | 99.8 |
| tile | packed FINISH(replace) | k=16 | Zen5 cpu2 | 51.4 | 51.4 | 1.0000 | 93.9 | 93.8 |
| tile | packed FINISH(replace) | k=16 | Zen5c cpu6 | 51.7 | 51.6 | 0.9981 | 93.7 | 93.8 |
| tile | packed FINISH(replace) | k=192 | Zen5 cpu2 | 583.9 | 583.7 | 0.9997 | 99.1 | 99.1 |
| tile | packed FINISH(replace) | k=192 | Zen5c cpu6 | 586.6 | 586.6 | 1.0000 | 99.1 | 99.1 |
| tile | packed FINISH(replace) | k=64 | Zen5 cpu2 | 196.6 | 196.4 | 0.9990 | 98.1 | 98.2 |
| tile | packed FINISH(replace) | k=64 | Zen5c cpu6 | 197.6 | 197.6 | 1.0000 | 98.1 | 98.0 |
| tile | packed FINISH|ADD | k=192 | Zen5 cpu2 | 583.6 | 584.0 | 1.0007 | 99.2 | 99.1 |
| tile | packed FINISH|ADD | k=192 | Zen5c cpu6 | 587.3 | 586.5 | 0.9986 | 99.0 | 99.1 |
| tile | packed RESUME|FINISH|ADD | k=192 | Zen5 cpu2 | 584.4 | 584.2 | 0.9997 | 99.0 | 99.0 |
| tile | packed RESUME|FINISH|ADD | k=192 | Zen5c cpu6 | 588.1 | 588.3 | 1.0003 | 98.8 | 98.8 |
| tile | stream full tile (B in place) | k=192 x1panel | Zen5 cpu2 | - | 581.5 | new-only | - | 99.5 |
| tile | stream full tile (B in place) | k=192 x1panel | Zen5c cpu6 | - | 583.9 | new-only | - | 99.5 |
| tile | stream full tile (B in place) | k=64 x1panel | Zen5 cpu2 | - | 193.7 | new-only | - | 99.5 |
| tile | stream full tile (B in place) | k=64 x1panel | Zen5c cpu6 | - | 195.5 | new-only | - | 99.1 |
| tile | stream full tile (B in place) | k=64 x4panel | Zen5 cpu2 | - | 780.7 | new-only | - | 98.8 |
| tile | stream full tile (B in place) | k=64 x4panel | Zen5c cpu6 | - | 786.2 | new-only | - | 98.6 |
| driver | packed driver (multiply) | 191x257x139 | Zen5 cpu2 | 481543.1 | 482843.1 | 1.0027 | 89.1 | 88.7 |
| driver | packed driver (multiply) | 191x257x139 | Zen5c cpu6 | 484124.2 | 486119.2 | 1.0041 | 88.9 | 88.6 |
| driver | packed driver (multiply) | 192x257x139 | Zen5 cpu2 | 489747.2 | 483393.0 | 0.9870 | 88.0 | 89.1 |
| driver | packed driver (multiply) | 192x257x139 | Zen5c cpu6 | 486293.4 | 493803.6 | 1.0154 | 89.0 | 87.6 |
| driver | packed driver (multiply) | 256x256x256 | Zen5 cpu2 | 1151750.5 | 1132279.0 | 0.9831 | 91.5 | 93.0 |
| driver | packed driver (multiply) | 256x256x256 | Zen5c cpu6 | 1157295.5 | 1157686.0 | 1.0003 | 91.4 | 91.3 |
| driver | packed driver (multiply) | 511x511x511 | Zen5 cpu2 | 8878458.0 | 8855575.0 | 0.9974 | 94.5 | 94.6 |
| driver | packed driver (multiply) | 511x511x511 | Zen5c cpu6 | 8970790.0 | 8958076.0 | 0.9986 | 93.8 | 93.9 |
| driver | packed driver (multiply) | 512x512x512 | Zen5 cpu2 | 9026373.0 | 8924314.0 | 0.9887 | 93.5 | 94.4 |
| driver | packed driver (multiply) | 512x512x512 | Zen5c cpu6 | 9120879.0 | 9052913.0 | 0.9925 | 92.9 | 93.5 |
| driver | stream driver (stream_rhs) | 191x257x139 | Zen5 cpu2 | - | 475399.2 | new-only | - | 90.1 |
| driver | stream driver (stream_rhs) | 191x257x139 | Zen5c cpu6 | - | 481555.8 | new-only | - | 89.3 |
| driver | stream driver (stream_rhs) | 192x257x139 | Zen5 cpu2 | - | 459360.6 | new-only | - | 93.7 |
| driver | stream driver (stream_rhs) | 192x257x139 | Zen5c cpu6 | - | 470407.4 | new-only | - | 91.9 |
| driver | stream driver (stream_rhs) | 256x256x256 | Zen5 cpu2 | - | 1130391.0 | new-only | - | 93.2 |
| driver | stream driver (stream_rhs) | 256x256x256 | Zen5c cpu6 | - | 1136577.5 | new-only | - | 93.0 |
| driver | stream driver (stream_rhs) | 511x511x511 | Zen5 cpu2 | - | 9348924.0 | new-only | - | 89.6 |
| driver | stream driver (stream_rhs) | 511x511x511 | Zen5c cpu6 | - | 9398767.0 | new-only | - | 89.5 |
| driver | stream driver (stream_rhs) | 512x512x512 | Zen5 cpu2 | - | 9387977.0 | new-only | - | 89.7 |
| driver | stream driver (stream_rhs) | 512x512x512 | Zen5c cpu6 | - | 9444953.0 | new-only | - | 89.6 |
| e2e | gemm Avx256 (forced) | 191x257x139 | Zen5 cpu2 | 449672.6 | 450023.2 | 1.0008 | 95.3 | 95.2 |
| e2e | gemm Avx256 (forced) | 191x257x139 | Zen5c cpu6 | 454784.6 | 453147.8 | 0.9964 | 94.7 | 94.9 |
| e2e | gemm Avx256 (forced) | 192x257x139 | Zen5 cpu2 | 451903.0 | 448187.4 | 0.9918 | 95.3 | 96.1 |
| e2e | gemm Avx256 (forced) | 192x257x139 | Zen5c cpu6 | 456445.2 | 451523.6 | 0.9892 | 94.8 | 95.7 |
| e2e | gemm Avx256 (forced) | 256x256x256 | Zen5 cpu2 | 1115087.0 | 1105755.0 | 0.9916 | 94.4 | 95.2 |
| e2e | gemm Avx256 (forced) | 256x256x256 | Zen5c cpu6 | 1120246.5 | 1105624.5 | 0.9869 | 94.5 | 95.7 |
| e2e | gemm Avx256 (forced) | 511x511x511 | Zen5 cpu2 | 8763404.0 | 8749247.0 | 0.9984 | 95.6 | 95.8 |
| e2e | gemm Avx256 (forced) | 511x511x511 | Zen5c cpu6 | 8913233.0 | 8909927.0 | 0.9996 | 94.5 | 94.4 |
| e2e | gemm Avx256 (forced) | 512x512x512 | Zen5 cpu2 | 8818746.0 | 8763454.0 | 0.9937 | 95.6 | 96.1 |
| e2e | gemm Avx256 (forced) | 512x512x512 | Zen5c cpu6 | 8944210.0 | 8914485.0 | 0.9967 | 94.7 | 94.9 |
| e2e | gemm Avx512 (host default) | 191x257x139 | Zen5 cpu2 | 437939.5 | 437828.1 | 0.9997 | 97.8 | 97.8 |
| e2e | gemm Avx512 (host default) | 191x257x139 | Zen5c cpu6 | 440786.0 | 440773.5 | 1.0000 | 97.7 | 97.6 |
| e2e | gemm Avx512 (host default) | 192x257x139 | Zen5 cpu2 | 438206.2 | 436875.1 | 0.9970 | 98.3 | 98.5 |
| e2e | gemm Avx512 (host default) | 192x257x139 | Zen5c cpu6 | 440417.9 | 440476.6 | 1.0001 | 98.2 | 98.1 |
| e2e | gemm Avx512 (host default) | 256x256x256 | Zen5 cpu2 | 1071335.5 | 1065685.0 | 0.9947 | 98.3 | 98.8 |
| e2e | gemm Avx512 (host default) | 256x256x256 | Zen5c cpu6 | 1078779.0 | 1072998.5 | 0.9946 | 98.1 | 98.7 |
| e2e | gemm Avx512 (host default) | 511x511x511 | Zen5 cpu2 | 8493230.0 | 8484113.0 | 0.9989 | 98.7 | 98.7 |
| e2e | gemm Avx512 (host default) | 511x511x511 | Zen5c cpu6 | 8575784.0 | 8567077.0 | 0.9990 | 98.1 | 98.2 |
| e2e | gemm Avx512 (host default) | 512x512x512 | Zen5 cpu2 | 8528436.0 | 8517295.0 | 0.9987 | 98.8 | 98.9 |
| e2e | gemm Avx512 (host default) | 512x512x512 | Zen5c cpu6 | 8623664.0 | 8606922.0 | 0.9981 | 98.3 | 98.3 |
