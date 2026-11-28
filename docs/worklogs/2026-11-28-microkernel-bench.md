# Microkernel benchmark and first Zen 5 / Zen 5c baseline (#27)

Command (one pinned CPU; the bench refuses to start unless affinity is exactly one CPU):

```
BENCH_COMMIT=$(git rev-parse --short HEAD) taskset -c <cpu> \
  cargo bench --features bench-internals --bench microkernel [tile|edge|pack|driver|e2e]
```

Compile-only in CI (`cargo build --profile ci --features bench-internals --benches`); never run there.

## Definitions

- Peak per core = 2 FMA units × 4 doubles × 2 flops = **16 flop/cycle** (documented AVX2 FMA throughput) × the clock in the run. The clock is measured by a dependent-add chain on the pinned core and printed; here it was **1.99 GHz** (`cpuinfo_max_freq` is 2.0 GHz, boost off) → **31.8 GFLOP/s**. `%peak` always uses this AVX2 denominator, including the AVX-512 end-to-end rows (which are not bounded by it).
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
