# Streamed tile rewritten with explicit AVX2 intrinsics (branch `fix/streamed-tile-intrinsics`, commit 8219ead, on top of #26 head 3016c7e)

- `streamed-tile-intrinsics-8219ead.patch`: the change (only `src/packed_avx2/stream_rhs.rs`).
- `avx512-scan.txt`: `scan512.py` over every `packed_avx2::` function of the bench binary, per `-C target-cpu`, for `B0` (#26 as is) and `N` (new tile). It flags `%zmm`, `%xmm/%ymm16-31`, mask registers `%k0-7`, and any instruction whose first encoded byte is `0x62` (EVEX).
- `table-four-config-and-modes.md`, `raw/`: four-configuration ratios and per-mode tile ratios, medians of 3 rounds, cpu 2 (Zen 5) and cpu 6 (Zen 5c), sibling idle. `N` binaries are built from the new branch's `stream_rhs.rs` plus the local bench overlay (visibility changes only; the overlay is the one in `../codegen/overlay-on-3016c7e.patch`).
- `stack-counts-new.txt`, `disasm/`: hot-loop summaries and full bodies.
