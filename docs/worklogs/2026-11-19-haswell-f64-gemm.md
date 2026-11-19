# Haswell single-thread f64 matrix multiplication

[Issue 208](https://github.com/tensor4all/colmat-rs/issues/208) measures the time for callers of colmat's public matrix multiplication. SVD also makes internal GEMM calls. A separate trace of square SVDs at orders 511, 513 and 1024 identified both their logical shapes and the leading dimensions of their views. For example, the 1024 decomposition calls a 513 × 1025 × 513 product with all three leading dimensions equal to 1032. Measuring that layout separately matters because it changes cache access.

## Experiments

- Unrolling the existing 12 × 4 kernel four depth steps at a time saved about 2–3% on the three primary squares. It lost time on the 64-square scan. Its full output matched the old kernel bit-for-bit. The final candidate restores the original general kernel and specializes only the measured Haswell family. [First comparison](https://github.com/tensor4all/t4a-private-gemm-x86/pull/9#issuecomment-7001457667)
- Keeping the old driver and using an 8 × 6 tile with fixed-offset loads improved the 1024-square median from 64.11 to 59.12 ms in a four-round scan. System OpenBLAS took 50.11 ms. On the plain 8 × 6 variant, changing only the depth cap from 512 to 256 moved 1024 from 58.52 to 56.08 ms in a separate three-round scan. These probes motivated changing how the microkernel receives and reuses its input.
- The selected driver packs both inputs. The assembly prototype used twelve YMM accumulators, two vectors of A and two alternating broadcasts of B. Four ordered depth steps share a branch. A and B micropanels together occupy up to two thirds of L1. A row block takes up to half of L2. On the measured 32 KiB / 256 KiB machine this gives a depth cap of 192 and row blocks of 80 for full panels. The same calculation applies to rectangular and odd inputs. Padded edge microtiles use the same FMA loop and copy back only the valid output entries.
- The fast path runs inside the existing backend with its reusable thread-local buffers. It accepts full, single-thread, regular column-major f64 products on the Haswell family above the packing threshold. Indexed, triangular, diagonal and other execution modes use the general driver. The first packed candidate rounded a partial product after each cache panel. It also fused scaling with the destination update. The full colmat test suite caught a bitwise difference between serial and two-worker LU at order 1024. That candidate was dropped. [Superseded measurement](https://github.com/tensor4all/t4a-private-gemm-x86/pull/9#issuecomment-7001711260)
- In a three-round scan, forcing a single 512-depth packed tile took 61.38 ms at 1024³, compared with 57.31 ms for cache panels with raw carry and 56.76 ms for system OpenBLAS. The corrected kernel carries raw accumulator bits across cache panels in spare space in the existing A scratch. It scales each output after the same 512-depth group as the general kernel, then adds the rounded product to the destination. The arithmetic group size is shared by both drivers. The destination is read only for Add or subsequent arithmetic groups.

## Verification

The public comparison uses identical fresh inputs within each round, separate processes, a 250 ms warmup and five timed calls. The system LP64 reference is OpenBLAS 0.3.26 and the Julia ILP64 reference is 0.3.30. Each public GEMM checks the entire output against the system reference. Each full SVD checks three reconstruction entries. The focused kernel test checks every output entry exactly on dyadic inputs with row and column scale factors of 2⁻⁴⁵⁰ and 2⁴⁵⁰. It exercises multi-panel depths, Add and Replace, odd edges and padded leading dimensions. The random-data regression compares output bits against scalar FMA with the original 512-depth grouping, including non-unit alpha, Add and Replace. Additional tests cover small scratch bands and a finite result when adding two input entries would overflow. The final public sweep also checks full-output hashes against the old kernel. The full colmat library suite passed, including the LU test that rejected the first packed implementation.

The measurements use colmat 0ad25fe1, the old kernel at 8224897 and Rust 1.96 with a Haswell target. The Xeon E5-2699 v3 ran at one thread on CPU 8, with the observer on CPU 0. Both logical CPUs of the measured physical core passed idle checks at both ends of the run. Turbo was enabled and the last-level cache was shared. These are local reference measurements for a draft. The separate controlled merge-acceptance campaign is still outstanding.

## Assembly reference (9fadf8a)

Historical median times for one public multiplication in milliseconds on the Haswell host:

| Square | Previous colmat (ms) | Assembly (ms) | System OpenBLAS (ms) | Julia OpenBLAS (ms) |
|---:|---:|---:|---:|---:|
| 511 | 8.844 | 7.796 | 7.874 | 7.791 |
| 513 | 9.356 | 8.235 | 7.830 | 7.879 |
| 1024 | 75.051 | 61.266 | 58.711 | 58.713 |

[Assembly reference measurements](https://github.com/tensor4all/t4a-private-gemm-x86/pull/9#issuecomment-7001924855)

## Rust implementation

The current tile holds 48 classical dot products in a fixed Rust array. Each depth step uses f64::mul_add. The compiler assigns SIMD registers. The Haswell entry point enables AVX2 and FMA locally so a generic x86-64 build can generate hardware FMA on that path. The 512-depth rounding groups and the scratch layout follow the assembly reference. The Rust tile replaces the hand-written assembly in this PR. CPU dispatch remains limited to Haswell.

A feasibility scan compared the assembly, Rust intrinsics and plain Rust on identical seeded inputs in four balanced rounds. At 511³, 513³ and 1024³, the plain Rust prototype took 3.2% more, 3.4% less and 1.5% more time than assembly, respectively (median within-round comparisons). The 465 × 513 × 48 SVD update took 14.9% more. All 16 input cases produced the same full-output hash across the three implementations. The isolated tile also compiled to an ARM64 object for Apple Silicon. ARM integration and runtime performance are unverified.

The first integration (0f1b354) passed the three panel-update indicators as separate arguments and passed the output stride in elements. In the resulting Haswell timing binary, the compiler spilled vector values to the stack inside the dot product. Its first guarded 1024-square round took 102.516 ms against 52.020 ms for system OpenBLAS. The revision restores the measured prototype's single flags word and byte stride. The Haswell timing binary again has vector FMAs without stack spills in the depth loop. Both versions use the same Rust arithmetic. See the current PR description for the head-specific comparison in milliseconds. [Draft PR #9](https://github.com/tensor4all/t4a-private-gemm-x86/pull/9)

## Eight steps per depth loop

The generated four-step loop at efdb756 performed redundant bounds branches inside the loop. The revised loop rounds its limit down to a multiple of eight and indexes only the full groups of eight. The tail follows the same depth order. The generated Haswell depth loop keeps the twelve vector accumulators in registers.

The eight-step candidate improved the equally weighted four-square time ratio by 2.69% in the median paired round, with eight wins in eight comparisons to efdb756. The four orders were 511, 512, 513 and 1024. At 511 it led both OpenBLAS builds by absolute median, with six of eight paired wins against the system build and five of eight against Julia's build. The other three orders still had slower medians than both builds. A four-round guard showed improvements at 513 × 1025 × 513 and 2048 cubed, with changes between +0.23% and +2.12% for three smaller padded layouts. All timed Rust results matched efdb756 bit for bit.

The two-step variant was slower than efdb756 in all four square medians in the three-round screen. An eight-step variant with next-tile prefetch is undergoing a separate paired comparison. The release-shaped phase-specialization trial with Rust 1.84 compatible iterators was dropped after its 1024 median increased from 57.441 ms to 59.644 ms.

## Explicit vector arithmetic across compiler versions

The scalar Rust source in 473ca80 depends on LLVM to choose the vector grouping. On Haswell, Rust 1.99.1 generated vector spills in the depth loop. In a five-seed crossover with the same logical inputs, 1024 cubed took 228.538 ms on 1.99.1 and 63.471 ms on 1.96.0.

The revised step loads two vectors of A, broadcasts each B value, and updates the twelve accumulators through Rust AVX2/FMA intrinsics. The machine code from both tested compilers keeps the vector sums in registers. The source retains the existing depth groups, scaling, and final add. Every full-result digest matched the previous draft across the paired measurements.

On Rust 1.99.1, the median across eight fresh paired rounds improved by 73.15% across the four equally weighted orders. The 1024 median fell from 251.035 ms to 66.141 ms. System OpenBLAS took 62.075 ms and Julia's build took 62.694 ms. On Rust 1.96.0, the four-order paired median changed by −0.19%, with four wins in eight comparisons to the earlier draft. In both toolchains the 511 absolute median was lower than both libraries, while 512, 513 and 1024 remained slower.
