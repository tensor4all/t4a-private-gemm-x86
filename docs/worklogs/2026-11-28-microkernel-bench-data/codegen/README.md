# Tile code generation and `-C target-cpu` check (#26 / #27)

Question: is the `mul_add` formulation slower than the intrinsics one because of the instruction stream, and does that depend on the build flags?

Binaries (local overlay `overlay-on-3016c7e.patch` = #26 head `3016c7e` + the bench + two experiment features, **not pushed, not part of #26**):

- `B0` = features `bench-internals`: packed tile **intrinsics**, streamed tile **`mul_add`** (this is #26 as proposed; its packed tile is unchanged from the merge base).
- `B1` = `bench-internals,exp-newtile,exp-oldtile`: packed tile **`mul_add`**, streamed tile **intrinsics**.

Config 1 = B0 packed driver, 2 = B1 packed driver, 3 = B1 streamed driver, 4 = B0 streamed driver.
Each built with the same release profile and `RUSTFLAGS` unset, or `-C target-cpu=x86-64-v3 | raptorlake | znver5`, rustc 1.99.0.
Run: `run-codegen.sh` (cpu 2 = Zen 5, cpu 6 = Zen 5c, 3 rounds, sibling idle in `idle.txt`).

- `disasm/*.s`: whole functions (`packed_avx2::tile`, `stream_rhs::panels::<0|1|2>`), default and znver5. `summary-*.txt` = `loop.py` output for the hottest innermost loop (one iteration = 8 depth steps).
- `disasm-posted-binaries/`: the same functions from the four binaries behind the numbers already posted on #26 (`c1`..`c4`).
- `table-tile.md` (run 2), `table-tile-run1.md` (run 1), `table-driver-and-modes.md`: medians of 3 rounds.
- Hardware counters: **not collected.** `kernel.perf_event_paranoid` is 4 on this host and the session has no root, so `perf stat` has no events. Instruction counts below are derived from the disassembly, cycles from ns x measured clock.
