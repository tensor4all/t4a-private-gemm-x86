# Skylake client, the f32 packed tile, and the Intel Mac L3 probe

Follow-up to [#9](https://github.com/tensor4all/t4a-private-gemm-x86/pull/9), related to
[tensor4all/colmat-rs#208](https://github.com/tensor4all/colmat-rs/issues/208). The 8 × 6
classical f64 driver in #9 was measured on a Haswell server and its dispatch gate named
only the Haswell model IDs. This branch takes the same driver to a Skylake-generation
desktop and adds the f32 tile.

## Experiments

- **Gate the measured family, not one CPU model.** Skylake through Coffee Lake client
  (`0x4E`, `0x5E`, `0x8E`, `0x9E`) keep the 2-load-port, 32 KiB L1 / 256 KiB L2 core the
  tile is derived for. On an i9-9900K (Coffee Lake Refresh, AVX2+FMA) the packed f64
  driver gained 10–21% over 0.1.28 from `384^3` up once it was admitted: 512³ 52.5 →
  61.3, 1024³ 51.4 → 59.2, 2048³ 49.1 → 59.5 GFLOP/s, three alternating process sets,
  median of per-set medians. AVX-512 hosts never reach the gate: `gemm_impl` selects
  `InstrSet::Avx512` and the gate requires `Avx256`.
- **Move the work floor from `16 * l2` to `128 * l2` elements.** The packed driver lost
  7% at `128^3` and was level at `160^3` on the measured desktop; it wins from `192^3`
  up. `128 * l2` is 4.19M element-multiply-adds on the 256 KiB L2, just above `160^3`.
  The first `128^3` cell (0.940) now stays on the general driver.
- **Write the f32 tile (16 × 6) in Rust.** Twelve YMM accumulators, two vectors of A and
  six broadcasts of B; the packed layout, 512-depth arithmetic groups, cache panels and
  scale-then-add epilogue follow the f64 driver. The finish multiplies and adds with
  separate intrinsics, because a contracted FMA rounds differently. The gate-off/on
  test is bit-identical for f32 as well as f64. Paired 0.1.28 medians on the same host:
  512³ 110.8 → 128.8, 1024³ 105.9 → 125.6, 1536³ 107.6 → 122.4 GFLOP/s (1.13–1.19×),
  and 1024×128×1024 105.6 → 119.0 (1.13×).
- **Read the Intel Mac L3 size in the cache probe.** The Apple branch read L1 and L2 and
  left L3 zero, which the default clamped to 2 MiB. On a 16 MiB L3 that capped the
  column band and re-packed the A panel for every band: at `1536^3` the A pack was
  15.9% of the call and the product was 51.7 GFLOP/s, against 60.8 after
  `hw.l3cachesize` is read. Apple Silicon's `hw.perflevel0.l3cachesize` is the fallback.
- **Rejected.** Direct A reads (2× slower: the `lda` stride puts every depth step in one
  L1 set), a flat A panel (pack and tile cancel), a fused double-buffered A pack inside
  the tile loop (−8%: it evicts the current panel from L2), `PACK4`/`PACK8` and
  prefetch in the pack (pack itself −18%, public within ±1%), tile prefetch (60.9
  against 62.0), and a 2D cache-blocked compute loop (0.937 at `1024^3`; the i-outer /
  j-inner order streams the B block per i tile, and at `1536^3` the all-column raw
  partials need 18.9 MB of the 16 MiB scratch). A B-row padding probe produced a wrong
  checksum and is not a result.

## Verification

`cargo test --lib packed_avx2` covers both modules: 16 passed, including the f64 and f32
gate-off/gate-on bitwise comparisons, the multi-panel depth and tail shapes, odd edges
with padded leading dimensions, the finite near-maximum inputs, and the forced-gate
checks. `cargo test --test avx256_f64_depth` passes as in #9. The three-set public
campaign compared colmat with system OpenBLAS 0.3.34 on the i9-9900K, one thread, native
macOS, `-C target-cpu=skylake`, backends rotated per set, 7 repeats, 150 ms sample
floor. f64 colmat/OpenBLAS: 1.21 at `64^3`, 1.10 at `128^3`, 1.00 at `256^3`/`512^3`,
0.97 at `1024^3`, 0.96 at `1536^3`, and 1.25/1.32 at `2048^2x64`/`4096^2x64`; f32 is
1.11/1.05 at `64^3`/`128^3`, 0.98–0.99 at `256^3`/`512^3`, 0.94–0.95 at the large
squares, and 1.13/1.22 at the two shallow-depth large products.

The pure-Rust product therefore beats OpenBLAS on the small squares and the large
shallow-depth products and is at parity at `256^3`/`512^3`; the canonical large squares
stay 3–6% behind, and no performance claim is made for them here. The multithread
columns are not a claim (the Docker VM does not present a controlled core placement).
Host, harness and the three-set campaign: [tensor4all/colmat-rs#208](https://github.com/tensor4all/colmat-rs/issues/208).
