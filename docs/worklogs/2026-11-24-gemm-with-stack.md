# The packing memory from the caller: `gemm_with_stack` and `gemm_stack_bytes`

Issue: [t4a-private-gemm-x86#14](https://github.com/tensor4all/t4a-private-gemm-x86/issues/14),
the counterpart of [tensor4all/t4a-gemm#21](https://github.com/tensor4all/t4a-gemm/issues/21).

## What and why

`gemm_impl` takes its packing memory from a thread-local, `MEM`: one L3 for the packed lhs and one
for the packed rhs, in 4096-byte pages, allocated on a thread's first call (and allocated per call
by a nested call that finds it borrowed). colmat-next-rs's `matmul_with_scratch` promises to
allocate nothing beyond the caller's stack, from the first call, and runs its own threads with one
single-threaded call per participant. On x86-64 it calls this crate, so that promise could not
hold.

* `gemm_stack_bytes(dtype, instr, nrows, ncols, depth)` is the memory a call needs: the same pages
  `MEM` would hand it, plus 4095 bytes so that any `&mut [MaybeUninit<u8>]` can be aligned to a
  page. The size does not depend on the shape, because the drivers block the product to fit an
  L3-sized buffer. It takes the wider `nr` rounding (8, the narrow AVX-512 `f64` kernel), so it
  covers every register blocking and both orientations. It is 0 for an empty product.
* `gemm_with_stack` is `gemm` with one thread and a `stack` argument: the pages come from
  `stack`, and `MEM` is not touched. It panics, before writing, when `stack` is too short.

The body of `gemm_impl` became a closure over a page slice. The stack path hands it the pages from
`stack`; the existing path hands it `MEM`'s. Both slices have the same length, and the drivers
read their capacities from it (the packed AVX2 drivers derive `mc` and `nc` from them), so the
blocking, the driver chosen and the result are the same: bit-for-bit `gemm`'s with
`n_threads == 1`.

`gemm`, `gemm_with` and `gemm_with_scratch` are unchanged. With one thread the job tables of
`MilliScratch` are never built, so `gemm_with_stack` takes none.

## Not done

* A shape-dependent bound. A small product still asks for two L3s (for example 64 MiB with a
  32 MiB L3). Sizing the buffer by the shape would change the capacities the packed AVX2 drivers
  block by, so it would need its own bit-for-bit argument.
* More than one thread.

## Verification

`tests/with_stack.rs`, in CI (`.github/workflows/ci.yml`, "Packing memory from the caller").

* On a fresh thread, a first call allocates nothing. This is checked for each `DType`, each
  instruction set the runner has, row- and column-major `dst` and `lhs`, and shapes from
  `1 x 37 x 19` to an `lhs` of 96 MiB (above the L3-sized buffer). `512^3` is in the packed AVX2
  driver's range. The process-wide set-up (cache sizes, CPU features) runs before the counted
  call.
* The result matches the bits of `gemm` with one thread, for the same shapes, both
  orientations, both `lhs` layouts, both accumulations and the conjugations.
* A short stack panics.

The tests were written on an aarch64 host with no x86-64 machine available: they were compiled
for `x86_64-apple-darwin` there but not run. CI is their first run.
