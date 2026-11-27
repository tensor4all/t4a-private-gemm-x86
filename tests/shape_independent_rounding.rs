//! A window of a product, computed by one call on the window alone, is bit for bit the same
//! entries of one call on the whole product (#24, tensor4all/colmat-next-rs#45).
//!
//! A caller that cuts a product into tiles and calls `gemm` once per tile, as colmat-next-rs's
//! thread team does, relies on this to give the result of one serial call. The choices of this
//! crate that depend on the output's shape are where it could break:
//!
//! - on AVX-512, `f64` takes the `48 x 4` kernel when `nrows > 48` and the `24 x 8` kernel
//!   otherwise, so a 24-row window and a 49-row or larger whole product take different kernels;
//! - on AVX2, `f64` and `f32` take the packed driver (`packed_avx2`, `packed_avx2_f32`) only from
//!   `m >= 32`, `n >= 24` and a size above a multiple of L2, so a small window takes the general
//!   driver while the whole product may take the packed one.
//!
//! The depths cross `DEPTH_BLOCK` (512) and its one-element tail (513, 1025). Each instruction
//! set the host lacks is skipped with a line on stdout.

use private_gemm_x86::{Accum, DType, DstKind, IType, InstrSet, gemm};

fn instr_sets() -> Vec<InstrSet> {
	let mut out = Vec::new();
	if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
		out.push(InstrSet::Avx256);
	} else {
		println!("skip: no AVX2/FMA on this host");
	}
	if is_x86_feature_detected!("avx512f") {
		out.push(InstrSet::Avx512);
	} else {
		println!("skip: no AVX-512F on this host");
	}
	out
}

/// Bytes per element.
fn size_of(dtype: DType) -> usize {
	match dtype {
		DType::F32 => 4,
		DType::F64 | DType::C32 => 8,
		DType::C64 => 16,
	}
}

/// Raw storage of `len` elements of `dtype`, filled with values in `[-1, 1)`.
fn values(dtype: DType, len: usize, seed: u64) -> Vec<u64> {
	let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
	let mut next = || {
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		(x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
	};
	let words = (len * size_of(dtype)).div_ceil(8).max(1);
	(0..words)
		.map(|_| match dtype {
			DType::F64 | DType::C64 => next().to_bits(),
			DType::F32 | DType::C32 => (next() as f32).to_bits() as u64 | ((next() as f32).to_bits() as u64) << 32,
		})
		.collect()
}

/// The bytes of element `i` of `buf`.
fn element(dtype: DType, buf: &[u64], i: usize) -> &[u8] {
	let size = size_of(dtype);
	let bytes = unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len() * 8) };
	&bytes[i * size..(i + 1) * size]
}

/// `dst := alpha * lhs * rhs` on one thread, every operand column-major: `lhs` (`m x k`) with
/// column stride `lhs_cs` from element `lhs_at`, `rhs` (`k x n`) with column stride `k` from
/// element `rhs_at`, `dst` (`m x n`) packed.
#[allow(clippy::too_many_arguments)]
fn product(
	dtype: DType,
	instr: InstrSet,
	(m, n, k): (usize, usize, usize),
	lhs: &[u64],
	lhs_at: usize,
	lhs_cs: usize,
	rhs: &[u64],
	rhs_at: usize,
	alpha: &[u64],
) -> Vec<u64> {
	let mut dst = vec![0u64; (m * n * size_of(dtype)).div_ceil(8).max(1)];
	let size = size_of(dtype);
	let null = core::ptr::null::<()>();
	unsafe {
		gemm(
			dtype,
			IType::U64,
			instr,
			m,
			n,
			k,
			dst.as_mut_ptr().cast(),
			1,
			m as isize,
			null,
			null,
			DstKind::Full,
			Accum::Replace,
			lhs.as_ptr().cast::<u8>().add(lhs_at * size).cast(),
			1,
			lhs_cs as isize,
			false,
			null,
			0,
			rhs.as_ptr().cast::<u8>().add(rhs_at * size).cast(),
			1,
			k as isize,
			false,
			alpha.as_ptr().cast(),
			1,
		);
	}
	dst
}

/// The window of `rows x cols` at `(i0, j0)` of the `m x n x k` product, computed alone, against
/// the same entries of the whole product.
fn window_matches_whole(dtype: DType, instr: InstrSet, (m, n, k): (usize, usize, usize), (i0, rows): (usize, usize), (j0, cols): (usize, usize)) {
	let lhs = values(dtype, m * k, 1);
	let rhs = values(dtype, k * n, 2);
	let alpha = values(dtype, 1, 3);
	let whole = product(dtype, instr, (m, n, k), &lhs, 0, m, &rhs, 0, &alpha);
	let window = product(dtype, instr, (rows, cols, k), &lhs, i0, m, &rhs, j0 * k, &alpha);
	for j in 0..cols {
		for i in 0..rows {
			assert_eq!(
				element(dtype, &window, i + rows * j),
				element(dtype, &whole, (i0 + i) + m * (j0 + j)),
				"{dtype:?} {instr:?} {m}x{n}x{k}: window {rows}x{cols} at ({i0}, {j0}), element ({i}, {j}) differs from the whole product",
			);
		}
	}
}

/// The `f64` AVX-512 kernel choice (`nrows > 48`): a `24 x 24` window of a product of 96 or 49
/// rows.
#[test]
fn f64_window_across_the_avx512_kernel_choice() {
	for instr in instr_sets() {
		for (shape, at_rows, at_cols) in [
			((96, 96, 1025), (24, 24), (8, 24)),
			((49, 96, 1024), (24, 24), (8, 24)),
			((96, 96, 513), (24, 24), (8, 24)),
			((96, 96, 512), (0, 24), (40, 24)),
		] {
			window_matches_whole(DType::F64, instr, shape, at_rows, at_cols);
		}
	}
}

/// The same windows for the other types, which take one kernel whatever the shape.
#[test]
fn other_types_window_matches_whole() {
	for instr in instr_sets() {
		for dtype in [DType::F32, DType::C32, DType::C64] {
			for (shape, at_rows, at_cols) in [((96, 96, 1025), (24, 24), (8, 24)), ((49, 96, 513), (24, 24), (8, 24))] {
				window_matches_whole(dtype, instr, shape, at_rows, at_cols);
			}
		}
	}
}

/// Large enough products that the whole may take the AVX2 packed driver while a `16 x 16`
/// window cannot (`m >= 32` and `n >= 24` fail). Whether the whole takes it depends on the
/// host's L2 and CPU family (`floor_factor`), so this test does not assert which path it took.
#[test]
fn window_of_a_product_large_enough_for_the_packed_driver() {
	for instr in instr_sets() {
		for dtype in [DType::F64, DType::F32] {
			window_matches_whole(dtype, instr, (512, 256, 1025), (40, 16), (100, 16));
		}
	}
}

/// Near the compact-driver L2 limit a whole product and its smaller window can
/// choose different output traversals. Include a full and a partial depth block.
#[test]
fn f64_window_near_the_l2_panel_boundary() {
	if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
		for depth in [501, 512, 513, 1025] {
			window_matches_whole(DType::F64, InstrSet::Avx256, (487, 419, depth), (11, 57), (19, 47));
		}
	}
}

#[test]
fn overwrite_windows_cover_staged_panels_and_fringe_tiles() {
	if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
		return;
	}
	let alpha = [1.0_f64.to_bits()];
	let dtype = DType::F64;
	let instr = InstrSet::Avx256;
	for (m, n, k) in [
		(511, 511, 511),
		(512, 512, 512),
		(513, 513, 513),
		(463, 59, 509),
		(501, 1003, 1025),
		(473, 419, 504),
		(259, 515, 507),
		(497, 52, 512),
	] {
		let lda = m + 3;
		let lhs = values(dtype, lda * k, 71);
		let rhs = values(dtype, k * n, 19);
		let whole = product(dtype, instr, (m, n, k), &lhs, 0, lda, &rhs, 0, &alpha);
		for i0 in [0, 217.min(m - 13), m - 13] {
			for j0 in [0, n / 2, n - 11] {
				let window = product(dtype, instr, (13, 11, k), &lhs, i0, lda, &rhs, j0 * k, &alpha);
				for j in 0..11 {
					for i in 0..13 {
						assert_eq!(
							window[i + j * 13],
							whole[(i + i0) + (j + j0) * m],
							"shape {m}x{k}x{n} at {},{}",
							i0 + i,
							j0 + j
						);
					}
				}
			}
		}
	}
}
