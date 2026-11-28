# Microkernel benchmark and first Zen 5 / Zen 5c baseline (#27)

Command (one pinned CPU; the bench refuses to start unless affinity is exactly one CPU):

```
BENCH_COMMIT=$(git rev-parse --short HEAD) taskset -c <cpu> \
  cargo bench --features bench-internals --bench microkernel [tile|edge|pack|driver|e2e]
```

Compile-only in CI (`cargo build --profile ci --features bench-internals --benches`); never run there.

## Definitions

- Peak per core = 2 FMA units × 4 doubles × 2 flops = **16 flop/cycle** (documented AVX2 FMA throughput) × the clock in the run. The clock is measured by a dependent-add chain on the pinned core and printed; here it was **1.99 GHz** (`cpuinfo_max_freq` is 2.0 GHz, boost off) → **31.8 GFLOP/s**. Two columns since `5a6171c`: **`%avx2`** is against this AVX2 peak for every row; **`%isa`** is against the row's own instruction-set peak: 16 flop/cycle for AVX2 rows (so `%isa == %avx2`) and 32 flop/cycle for AVX-512 rows (2 x 512-bit FMA, 63.6 GFLOP/s at 1.99 GHz). The old single `%peak` column was `%avx2`; tables above this section are unchanged and keep that name.

**On this part AVX-512 f64 runs at AVX2 throughput.** AVX-512 executes at 256-bit width here, so an AVX-512 row at `%isa` ≈ 49 is at the part's real f64 limit (`%avx2` ≈ 98), not at half of it. Measured: the AVX-512 tile reaches 31.2 GFLOP/s (97.9 % of the AVX2 peak) against 31.6 (99.1 %) for the AVX2 8x6 tile, so the tile is 1.2 points below it, not above. End to end, AVX-512 `gemm` is 3-5 % faster than forced AVX2 (below). Do not read a `%avx2` of 98.9 on an AVX-512 row as "near the AVX-512 peak"; read `%isa`.
- Repetitions: median of **31 samples**, each ≥ 2 ms of back-to-back calls, 100 ms warm-up. Table values below are the median of **3 rounds** of that median; round order alternated.
- Groups: `tile` (L1-resident packed tile, per flag), `edge` (fringe tiles), `pack` (packing alone, GB/s), `driver` (whole driver, scratch preallocated, no public dispatch and no CPU gate), `e2e` (public `gemm`, dispatch included).

## Host

`shinaoka-EVO-X1Pro`, AMD Ryzen AI 9 HX 470 (the host's own model string). Two core types, identified from sysfs and checked before every run by `run.sh` (L3 size of the pinned CPU must match; SMT sibling idle before and during):

| core | pinned CPU | sibling | L3 (sysfs) shared by |
|---|---|---|---|
| Zen 5 (full) | 2 | 14 | 16 MiB, cpus 0-3,12-15 |
| Zen 5c (compact) | 6 | 18 | 8 MiB, cpus 4-11,16-23 |

Both report the same max frequency (2.0 GHz) and the same 1 MiB L2/core in sysfs; the L3 grouping is the only distinguishing evidence the kernel exposes.

## What the numbers say

Same source bench in both trees: **old** = merge base `e688d4a` + bench (`f918b7f`), **new** = #26 head `3016c7e` + the same bench plus a visibility-only overlay (`pub(super)` on `stream_rhs::panels`) so the streamed tile and driver can be called without the Raptor Lake gate. The overlay is not part of #26.

Full table: `2026-11-28-microkernel-bench-data/table.md`; raw per-run output beside it.

| row | Zen 5 (cpu 2) | Zen 5c (cpu 6) |
|---|---|---|
| **kernel** packed tile, k=192 (code identical in both trees) old / new | 583.9 / 583.7 ns, 99.1 % peak | 586.6 / 586.6 ns, 99.1 % peak |
| kernel streamed tile (new only), k=192 / k=64 | 99.5 % / 99.5 % peak | 99.5 % / 99.1 % peak |
| **driver** streamed (new) ÷ packed (old), 191×257×139 | 0.987 | 0.995 |
| driver, 192×257×139 | 0.938 | 0.967 |
| driver, 256³ | 0.982 | 0.982 |
| driver, 511³ / 512³ | 1.053 / 1.040 | 1.048 / 1.036 |
| **e2e** `gemm` Avx512 (host default) new ÷ old, 191×257×139 … 512³ | 0.995 – 1.000 | 0.995 – 1.000 |
| e2e `gemm` Avx256 (forced) new ÷ old | 0.992 – 1.001 | 0.987 – 1.000 |

- The control rows (identical code in both trees: packed tile, packed driver) differ by up to ±1.5 % run to run, so differences under about 2 % are noise.
- On Zen 5 / 5c the streamed driver is never dispatched (gated to Raptor Lake model 0xB7) and `fits_packed_lhs` is a no-op for these shapes, so end-to-end old/new is the same code path: no regression, no change.
- The ordering of tile and driver results is the same on Zen 5 and Zen 5c (differences ≤ 3 %, the c core about 0.5–1 % slower in absolute time at the same measured clock). Both have the same L1/L2; they differ in L3 only, which these L1/L2-resident rows do not exercise. This pair therefore gives no evidence that the 8×6 tile's optimum differs between the two.
- Not measured: any Intel part (Haswell, Skylake client, Raptor Lake) — no other host was used. Whether #26's 191×257×139 attribution holds on Raptor Lake geometry (2 MiB L2) is unanswered here.

## Four configurations: tile vs B-streaming driver (maintainer request)

Config 1 = old tile + packed driver (merge base + bench). 2 = new `mul_add` tile in the packed driver. 3 = old intrinsics tile in the streamed-B driver. 4 = #26 as proposed. Configs 2 and 3 are local experiments (`exp-newtile`, `exp-oldtile` features on an overlay of #26 that swap only the inner `step`/`accumulate`; not pushed, not part of #26). Both variants pass `tests/fringe_bits.rs` (bit-identical to the scalar chain, so the tile swap does not change bits). Same bench source and `taskset` pin per run (cpu 2 = Zen 5, cpu 6 = Zen 5c, L3 checked before each run; sibling idle ≤ 2 %), 3 rounds with core and config order rotated, median of 31 samples each. Values are driver time ÷ config-1 time on the same core (below 1 is faster). Control (the packed driver, identical code, measured inside binaries 3 and 4 against binary 1): 0.994–1.019, so the noise floor is about ±2 %.

| shape | core | 1 old/packed (ns) | 2 new tile / packed | 3 old tile / **streamed B** | 4 new tile / streamed (PR) |
|---|---|---:|---:|---:|---:|
| 191×257×139 | Zen 5 | 480038 | 1.027 | **0.939** | 0.992 |
| 191×257×139 | Zen 5c | 485730 | 1.025 | **0.934** | 0.992 |
| 192×257×139 | Zen 5 | 482953 | 1.029 | **0.926** | 0.969 |
| 192×257×139 | Zen 5c | 493363 | 1.023 | **0.907** | 0.959 |
| 256³ | Zen 5 | 1118679 | 1.061 | **0.968** | 1.014 |
| 256³ | Zen 5c | 1152892 | 1.035 | **0.941** | 0.995 |
| 511³ | Zen 5 | 8821001 | 1.040 | **0.975** | 1.059 |
| 511³ | Zen 5c | 8972353 | 1.027 | **0.968** | 1.060 |
| 512³ | Zen 5 | 8954820 | 1.039 | **0.970** | 1.048 |
| 512³ | Zen 5c | 9065446 | 1.032 | **0.974** | 1.053 |
| 256×32×256 | Zen 5 / 5c | 143879 / 144258 | 1.016 / 1.018 | **0.945 / 0.950** | 0.968 / 0.975 |
| 256×512×256 | Zen 5 / 5c | 2234983 / 2272633 | 1.046 / 1.032 | **0.968 / 0.956** | 1.092 / 1.093 |
| 256×513×256 | Zen 5 / 5c | 2250472 / 2288082 | 1.047 / 1.033 | **0.963 / 0.952** | 1.089 / 1.091 |

What it shows (this host, 1T, Zen 5 and Zen 5c identical in direction):

- **Streaming B without packing helps on AMD**: config 3 is 2.5–9 % faster than baseline at every shape and every k tried (32 to 513), on both core types, a little more on the compact core at small and middle sizes. It is not Intel-specific, and not limited to large k.
- **The new `mul_add` tile hurts here**: config 2 is 1.5–6 % slower than baseline in the packed driver, and config 4 loses the streaming gain, ending 0–9 % slower than baseline at k ≥ 511 (+5–9 %) although config 3 is 2.5–3 % faster. So on Zen 5/5c the PR is a good driver with a worse tile.
- Mechanism not fully isolated. The packed tile alone is 3.5 % slower with `mul_add` at k=192 (620.6 vs 599.5 ns, Zen 5; the disassembly shows 2 ymm stack stores and loads in the `mul_add` version against none in the intrinsics version), which accounts for config 2. The streamed tile measured alone at k ≤ 192 does not show the config 3 vs 4 gap (580.6 vs 582.0 ns); the gap appears only inside the driver at k ≥ 512, which the tile rows do not reach. A k = 512 tile row would be the next step.
- Absolute tile times differ by up to 3 % between separately built binaries (packed tile k=192: 583.9 ns in the earlier build, 599.5 ns in this one, same source), so compare within a build, as the ratios above do.
- **Does the streamed path avoid the B pack?** Scratch written by one call (`scratch` group; `pa` = packed A, `pb` = packed B or dense tail rows): 191×257×139 packed driver writes 383 KiB of `pa` and 279 KiB of `pb`; streamed driver 385 KiB and **0 KiB**. 512³: packed 2552 KiB / 1145 KiB; streamed 896 / **0 KiB**. So the driver writes no packed B (its `pb` is used only for one or two dense tail rows; 191 and 512 have none). The RSS delta I also printed is noise (±70 KiB, allocator reuse), so I do not use it. At the public API level the allocation itself is **not** avoided: `gemm_impl` allocates the thread-local scratch (one L3 for each of packed A and packed B) before dispatching, whichever driver runs.
- **Bits at the fringe**: the streamed driver, which reads B in place, is bit-identical to the ordered scalar `mul_add` chain and to the padded-A product for 1666 products (m = 8..41, one- and two-row fringes included, k up to 513), so it equals the packed driver. The same holds for both experimental tile swaps.

## How the streamed driver was forced, and why

`gemm_driver()` returns `stream_rhs::multiply` only when `cpu_model() == Some(0xB7)` and CPUID leaf 0x1A reports a P-core. The test-only `PROBE` cannot select it: a positive force makes `haswell()` true and returns the *packed* `multiply`; a negative force returns `None`; and it is `#[cfg(test)]`, unreachable from a bench binary. The gate was not edited. The four configurations call `stream_rhs::multiply` directly through the bench wrapper (a `pub(super)` visibility change in a local overlay), with the same arguments `gemm_impl` passes. So "driver-only" is forced-on; an end-to-end `gemm` run with the streamed driver selected would need a hook in `gemm_driver()` itself and was not done.

## Does the cache-geometry story predict it?

Hypothesis: Raptor Cove's 48 KiB L1D, 2 MiB L2 and prefetcher keep up with a six-column column-major read, so packing B is pure loss; with a smaller L1 or weaker prefetch, packing should win. Both Zen 5 (cpu 2) and Zen 5c (cpu 6) report **48 KiB L1D and 1 MiB L2** (sysfs, `cache f64s : L1d 6144`), differing only in L3 (16 vs 8 MiB). So:

- The story predicts "streaming wins" on both, and no difference between them. It does win on both, and the two core types agree within about 2 %. That is consistent with the story, but not a test of it: no core here has a smaller L1.
- One manipulation of effective capacity is available: a pointer-chasing thrash on the SMT sibling (random read-modify-write over 64 KiB, `sibling-loaded/thrash.c`) takes a share of L1/L2 and load ports. With it, the baseline slows 4–5 %, and the config 3 ratios are essentially unchanged (0.907–0.978 loaded, 0.907–0.975 unloaded); no shape flips to packing. Config 4 improves relative to baseline at small sizes (0.93–0.97) and still loses 3–8 % at k ≥ 511. The new tile's penalty in config 2 grows (1.045–1.094 vs 1.016–1.061).
- This does not test the prefetcher part of the claim (the sibling shares the prefetcher), and the loader also contends for load/store ports and the front end, so it is a confounded stand-in for a smaller L1. The hypothesis is therefore **not decided here**: an Intel core with a smaller L1 or an older core without the strong prefetcher is needed.


## AVX-512 tile, its fringe, and `%isa` (#28 follow-up)

Commit `5a6171c`. Same host, pins and protocol as above (cpu 2 = Zen 5, cpu 6 = Zen 5c; sibling 0 % busy before and during every run, `avx512/idle.txt`; L3 checked before each run; median of 31 samples; **median of 3 rounds**, core order alternated). Measured clocks 1.992-1.995 GHz (Zen 5) and 1.979-1.982 GHz (Zen 5c); each run's own clock is its denominator. Raw: `2026-11-28-microkernel-bench-data/avx512/`; `aggregate.py` makes the table; `run.sh` runs it.

What was called: the generated AVX-512 f64 asm microkernels exactly as `gemm_impl` selects them: **48 x 4** (`F64_SIMD512x4[3]`, used when `nrows > 48`) and **24 x 8** (`F64_SIMD512x8[7]`, `nrows <= 48`), through one `millikernel_colmajor` call per tile, operands column-major in place (no packing), `bench_internals::avx512_tile`. This is not the AVX2 "packed tile" call: there is no separate pack step for these kernels in the bench, and one call carries the millikernel loop overhead. The bench checks every fringe call against a scalar sum (and that the rest of the tile is untouched) before timing it.

Findings:

- **Tile**: 48x4 and 24x8 are the same speed (31.2 GFLOP/s at k=192, 97.9 % of the AVX2 peak, 48.9 % of the AVX-512 nominal peak). The AVX2 8x6 packed tile is 99.1 %. So there is **no throughput gain from 512-bit shape on this part, and the 512-bit-shaped tile is not leaving more than about 2 points on the table** against the 16 flop/cycle ceiling.
- **Fringe**: a partial tile costs the same time as a full one at equal depth, so useful flop/s falls with the fraction of the tile used: 47x4 and 23x8 (one row short) 95.5 % and 93.8 % `%avx2`; 8-row and 16-row fringes about 96-98 %; fewer columns costs more (48x3 96.8 %, 48x1 45 %, 24x1 55 %); 1-row and 1x1 fringes 3-12 %. The AVX2 8x6 fringes in the `edge` group are the matching rows.
- **End to end**: AVX-512 `gemm` 31.0-31.6 GFLOP/s (95-99 % `%avx2`) against 29.7-30.5 (94-96 %) for forced AVX2: **3.3-4.7 % faster at the five square-ish shapes** on both core types. The tile alone does not explain it (the AVX-512 tile is slower than the AVX2 tile); that is a driver-level difference I did not isolate.
- **Zen 5 vs Zen 5c**: same ordering; the c core is 0.5-1.5 % slower in ns on tiles, within the clock difference of about 0.7 %.

clock GHz (measured, per run): {'2': [1.995, 1.992, 1.994], '6': [1.98, 1.979, 1.982]}

| group | kernel | shape | Zen 5 ns | GFLOP/s | %avx2 | %isa | Zen 5c ns | GFLOP/s | %avx2 | %isa |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tile | packed FINISH(replace) | k=16 | 51.3 | 29.96 | 93.9 | 93.9 | 51.7 | 29.73 | 93.9 | 93.9 |
| tile | packed FINISH(replace) | k=64 | 196.3 | 31.30 | 98.2 | 98.2 | 197.6 | 31.09 | 98.1 | 98.1 |
| tile | packed FINISH(replace) | k=192 | 582.9 | 31.62 | 99.1 | 99.1 | 587.1 | 31.39 | 99.2 | 99.2 |
| tile | packed FINISH|ADD | k=192 | 582.8 | 31.63 | 99.2 | 99.2 | 586.5 | 31.43 | 99.2 | 99.2 |
| tile | packed RESUME|FINISH|ADD | k=192 | 583.2 | 31.60 | 99.1 | 99.1 | 587.5 | 31.37 | 99.0 | 99.0 |
| tile | packed (no FINISH: spill sums) | k=192 | 580.4 | 31.76 | 99.5 | 99.5 | 583.5 | 31.59 | 99.7 | 99.7 |
| tile512 | avx512 48x4 replace | k=16 | 238.7 | 25.74 | 80.8 | 40.4 | 240.0 | 25.60 | 80.8 | 40.4 |
| tile512 | avx512 48x4 replace | k=64 | 817.4 | 30.07 | 94.2 | 47.1 | 821.7 | 29.91 | 94.4 | 47.2 |
| tile512 | avx512 48x4 replace | k=192 | 2362.3 | 31.21 | 97.8 | 48.9 | 2376.8 | 31.02 | 98.0 | 49.0 |
| tile512 | avx512 48x4 accumulate | k=192 | 2362.1 | 31.21 | 97.9 | 48.9 | 2376.2 | 31.03 | 98.0 | 49.0 |
| tile512 | avx512 48x4 fringe | 48x4 k=192 | 2372.6 | 31.08 | 97.4 | 48.7 | 2388.6 | 30.87 | 97.4 | 48.7 |
| tile512 | avx512 48x4 fringe | 47x4 k=192 | 2370.3 | 30.46 | 95.5 | 47.7 | 2386.3 | 30.25 | 95.5 | 47.7 |
| tile512 | avx512 48x4 fringe | 32x4 k=192 | 1581.3 | 31.08 | 97.4 | 48.7 | 1592.3 | 30.87 | 97.5 | 48.8 |
| tile512 | avx512 48x4 fringe | 16x4 k=192 | 803.6 | 30.58 | 96.0 | 48.0 | 808.1 | 30.41 | 96.1 | 48.0 |
| tile512 | avx512 48x4 fringe | 8x4 k=192 | 402.0 | 30.56 | 95.9 | 48.0 | 405.5 | 30.31 | 95.7 | 47.8 |
| tile512 | avx512 48x4 fringe | 1x4 k=192 | 466.6 | 3.29 | 10.3 | 5.2 | 468.9 | 3.28 | 10.3 | 5.2 |
| tile512 | avx512 48x4 fringe | 48x3 k=192 | 1789.9 | 30.89 | 96.8 | 48.4 | 1806.1 | 30.62 | 96.6 | 48.3 |
| tile512 | avx512 48x4 fringe | 48x1 k=192 | 1275.4 | 14.45 | 45.3 | 22.6 | 1301.6 | 14.16 | 44.7 | 22.3 |
| tile512 | avx512 48x4 fringe | 8x2 k=192 | 386.0 | 15.92 | 49.9 | 24.9 | 389.4 | 15.78 | 49.8 | 24.9 |
| tile512 | avx512 48x4 fringe | 1x1 k=192 | 386.9 | 0.99 | 3.1 | 1.6 | 391.1 | 0.98 | 3.1 | 1.5 |
| tile512 | avx512 24x8 replace | k=16 | 236.4 | 25.99 | 81.5 | 40.7 | 238.2 | 25.79 | 81.5 | 40.7 |
| tile512 | avx512 24x8 replace | k=64 | 814.2 | 30.19 | 94.6 | 47.3 | 819.5 | 29.99 | 94.7 | 47.4 |
| tile512 | avx512 24x8 replace | k=192 | 2361.4 | 31.22 | 97.9 | 48.9 | 2376.3 | 31.03 | 97.9 | 49.0 |
| tile512 | avx512 24x8 accumulate | k=192 | 2362.6 | 31.21 | 97.9 | 48.9 | 2377.2 | 31.01 | 98.0 | 49.0 |
| tile512 | avx512 24x8 fringe | 24x8 k=192 | 2361.0 | 31.23 | 97.9 | 48.9 | 2376.9 | 31.02 | 98.0 | 49.0 |
| tile512 | avx512 24x8 fringe | 23x8 k=192 | 2361.6 | 29.92 | 93.8 | 46.9 | 2376.9 | 29.73 | 93.8 | 46.9 |
| tile512 | avx512 24x8 fringe | 16x8 k=192 | 1577.3 | 31.16 | 97.7 | 48.8 | 1588.7 | 30.94 | 97.7 | 48.8 |
| tile512 | avx512 24x8 fringe | 8x8 k=192 | 791.6 | 31.04 | 97.4 | 48.7 | 796.6 | 30.85 | 97.3 | 48.6 |
| tile512 | avx512 24x8 fringe | 1x8 k=192 | 796.0 | 3.86 | 12.1 | 6.1 | 798.4 | 3.85 | 12.1 | 6.1 |
| tile512 | avx512 24x8 fringe | 24x4 k=192 | 1184.3 | 31.13 | 97.6 | 48.8 | 1191.7 | 30.93 | 97.7 | 48.9 |
| tile512 | avx512 24x8 fringe | 24x1 k=192 | 526.8 | 17.50 | 54.9 | 27.5 | 544.2 | 16.94 | 53.5 | 26.8 |
| tile512 | avx512 24x8 fringe | 8x2 k=192 | 386.3 | 15.91 | 49.9 | 25.0 | 387.5 | 15.85 | 50.0 | 25.0 |
| tile512 | avx512 24x8 fringe | 1x1 k=192 | 389.5 | 0.99 | 3.1 | 1.5 | 388.6 | 0.99 | 3.1 | 1.6 |
| e2e | gemm Avx512 (host default) | 191x257x139 | 436506.8 | 31.26 | 98.0 | 49.0 | 439822.9 | 31.03 | 97.8 | 48.9 |
| e2e | gemm Avx256 (forced) | 191x257x139 | 451088.9 | 30.25 | 94.8 | 94.8 | 454848.4 | 30.00 | 94.8 | 94.8 |
| e2e | gemm Avx512 (host default) | 192x257x139 | 436570.8 | 31.42 | 98.5 | 49.2 | 440761.0 | 31.12 | 98.2 | 49.1 |
| e2e | gemm Avx256 (forced) | 192x257x139 | 452201.0 | 30.34 | 95.1 | 95.1 | 456356.2 | 30.06 | 94.8 | 94.8 |
| e2e | gemm Avx512 (host default) | 256x256x256 | 1066050.5 | 31.48 | 98.7 | 49.3 | 1074111.0 | 31.24 | 98.7 | 49.3 |
| e2e | gemm Avx256 (forced) | 256x256x256 | 1116574.5 | 30.05 | 94.2 | 94.2 | 1122891.0 | 29.88 | 94.4 | 94.4 |
| e2e | gemm Avx512 (host default) | 511x511x511 | 8481026.0 | 31.47 | 98.6 | 49.3 | 8574130.0 | 31.12 | 98.2 | 49.1 |
| e2e | gemm Avx256 (forced) | 511x511x511 | 8768111.0 | 30.44 | 95.4 | 95.4 | 8930994.0 | 29.88 | 94.3 | 94.3 |
| e2e | gemm Avx512 (host default) | 512x512x512 | 8509289.0 | 31.55 | 99.0 | 49.5 | 8612711.0 | 31.17 | 98.5 | 49.2 |
| e2e | gemm Avx256 (forced) | 512x512x512 | 8803497.0 | 30.49 | 95.7 | 95.7 | 8977461.0 | 29.90 | 94.4 | 94.4 |
| e2e | gemm Avx512 (host default) | 256x32x256 | 138217.4 | 30.35 | 95.2 | 47.6 | 139124.1 | 30.15 | 95.1 | 47.6 |
| e2e | gemm Avx256 (forced) | 256x32x256 | 140256.2 | 29.90 | 93.7 | 93.7 | 141410.2 | 29.66 | 93.7 | 93.7 |
| e2e | gemm Avx512 (host default) | 256x512x256 | 2128754.0 | 31.52 | 98.9 | 49.4 | 2144895.0 | 31.29 | 98.8 | 49.4 |
| e2e | gemm Avx256 (forced) | 256x512x256 | 2198955.0 | 30.52 | 95.6 | 95.6 | 2220446.0 | 30.22 | 95.4 | 95.4 |
| e2e | gemm Avx512 (host default) | 256x513x256 | 2148481.0 | 31.30 | 98.1 | 49.1 | 2162728.0 | 31.09 | 98.1 | 49.0 |
| e2e | gemm Avx256 (forced) | 256x513x256 | 2225655.0 | 30.21 | 94.7 | 94.7 | 2258738.0 | 29.77 | 93.9 | 93.9 |
