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
