//! Sequential, packed 8 × 6 classical f64 GEMM for contiguous matrix rows.
use crate::{Accum, DEPTH_BLOCK};
use core::arch::x86_64::{_mm256_mul_pd, _mm256_add_pd, _mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};
#[cfg(test)]
use core::cell::Cell;

#[cfg(feature = "bench-internals")]
pub(crate) mod bench;

const MR: usize = 8;
const NR: usize = 6;
const ADD: usize = 1;
const RESUME: usize = 2;
const FINISH: usize = 4;

// Test-only view of the dispatch decision: the gate answer can be forced and
// the number of times the driver is entered counted. Both are per thread, so
// parallel tests neither see another test's override nor its calls.
#[cfg(test)]
thread_local! {
	static PROBE: (Cell<i8>, Cell<usize>) = const { (Cell::new(0), Cell::new(0)) };
}

// 0 leaves the gate alone, 1 accepts every product, -1 rejects every product.
#[cfg(test)]
pub(super) fn force_gate(state: i8) {
	PROBE.with(|(force, _)| force.set(state));
}

#[cfg(test)]
pub(super) fn gate_hits() -> usize {
	PROBE.with(|(_, hits)| hits.get())
}

// Other AVX2 cores have different load ports and cache geometry. Scope this
// implementation to the Intel family it was measured on.
//
// The measured host is Haswell (0x3C/0x3F/0x45/0x46; 0x45/0x46 are
// Haswell-ULT/GT3e). Skylake through Coffee Lake client (0x4E, 0x5E, 0x8E,
// 0x9E) keep the same 2-load-port, 32 KiB L1 / 256 KiB L2 core this tile is
// derived for, so they are admitted as the same family and measured separately
// (the Skylake-generation desktop in gemm-skylake-lab). AVX-512 hosts never
// reach this gate: the caller chooses `InstrSet` and this gate requires
// `Avx256`; none of the admitted model IDs has AVX-512.
pub(super) fn haswell() -> bool {
	#[cfg(test)]
	match PROBE.with(|(force, _)| force.get()) {
		0 => {},
		force => return force > 0,
	}
	cpu_admitted()
}

/// The `GenuineIntel` model ID, or `None` when this is not an Intel family-6
/// part.
fn cpu_model() -> Option<u8> {
	let cpuid = raw_cpuid::CpuId::new();
	cpuid.get_vendor_info()?.as_str().eq("GenuineIntel").then_some(())?;
	let feature = cpuid.get_feature_info()?;
	(feature.family_id() == 6).then(|| feature.model_id())
}

/// The shared CPU admission: the Haswell-style core and the instruction set the
/// packed driver compiles with.
pub(super) fn cpu_admitted() -> bool {
	static ADMITTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*ADMITTED.get_or_init(|| {
		if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
			return false;
		}
		cpu_model().is_some_and(admitted_model)
	})
}

/// The work floor in units of L2 elements. Haswell keeps the floor measured in
/// #9; the Skylake client family was measured at 128 on the i9-9900K.
pub(super) fn floor_factor() -> usize {
	static FLOOR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
	*FLOOR.get_or_init(|| {
		if cpu_model().is_some_and(|m| matches!(m, 0x4E | 0x5E | 0x8E | 0x9E)) {
			128
		} else {
			16
		}
	})
}

/// Intel model IDs with the Haswell-style core this tile was derived for.
#[inline]
pub(super) fn admitted_model(model: u8) -> bool {
	matches!(
		model,
		// Haswell
		0x3C | 0x3F | 0x45 | 0x46 |
		// Skylake / Kaby Lake / Coffee Lake client
		0x4E | 0x5E | 0x8E | 0x9E
	)
}

// Each of the 48 accumulators follows the same depth order as the classical dot.
// Keep the lane grouping explicit so the compiler keeps all twelve sums in YMM registers.
// SAFETY: the caller must enable AVX2 and FMA.
#[inline(always)]
unsafe fn step(sums: &mut [[f64; MR]; NR], a: &[f64; MR], b: &[f64; NR]) {
	let first = _mm256_loadu_pd(a.as_ptr());
	let second = _mm256_loadu_pd(a.as_ptr().add(4));
	for (col, &bj) in sums.iter_mut().zip(b) {
		let factor = _mm256_set1_pd(bj);
		let low = _mm256_fmadd_pd(first, factor, _mm256_loadu_pd(col.as_ptr()));
		let high = _mm256_fmadd_pd(second, factor, _mm256_loadu_pd(col.as_ptr().add(4)));
		_mm256_storeu_pd(col.as_mut_ptr(), low);
		_mm256_storeu_pd(col.as_mut_ptr().add(4), high);
	}
}

// A and B each contain depth full packed rows. RESUME loads the raw sums
// saved by the preceding panel. FINISH needs a full writable output tile;
// ADD also requires its entries to be initialized. Partial tiles use an
// edge buffer in the caller. Keep the three update bits in one argument.
#[cfg_attr(target_arch = "x86_64", target_feature(enable = "avx2,fma"))]
#[inline(never)]
unsafe fn tile(a: *const f64, b: *const f64, c: *mut f64, ldc_bytes: usize, depth: usize, alpha: f64, partial: *mut f64, flags: usize) {
	let mut sums = if flags & RESUME == 0 {
		[[0.; MR]; NR]
	} else {
		*partial.cast::<[[f64; MR]; NR]>()
	};
	let a = core::slice::from_raw_parts(a.cast::<[f64; MR]>(), depth);
	let b = core::slice::from_raw_parts(b.cast::<[f64; NR]>(), depth);
	let mut p = 0;
	let limit = depth / 8 * 8;
	while p < limit {
		for offset in 0..8 {
			// p and limit are multiples of eight. All eight indices are
			// below limit, which does not exceed either packed slice length.
			step(&mut sums, a.get_unchecked(p + offset), b.get_unchecked(p + offset));
		}
		p += 8;
	}
	for (ai, bi) in a[limit..].iter().zip(&b[limit..]) {
		step(&mut sums, ai, bi);
	}
	if flags & FINISH == 0 {
		partial.cast::<[[f64; MR]; NR]>().write(sums);
	} else {
		// Scale and add as two operations, like the general driver's
		// epilogue: a fused alpha*value + C would round differently.
		let factor = _mm256_set1_pd(alpha);
		for (j, col) in sums.iter().enumerate() {
			let dst = c.byte_add(j * ldc_bytes);
			let mut i = 0;
			while i + 4 <= MR {
				let scaled = _mm256_mul_pd(factor, _mm256_loadu_pd(col.as_ptr().add(i)));
				if flags & ADD == 0 {
					_mm256_storeu_pd(dst.add(i), scaled);
				} else {
					_mm256_storeu_pd(dst.add(i), _mm256_add_pd(scaled, _mm256_loadu_pd(dst.add(i))));
				}
				i += 4;
			}
			while i < MR {
				let slot = dst.add(i);
				let scaled = alpha * col[i];
				slot.write(if flags & ADD == 0 { scaled } else { scaled + slot.read() });
				i += 1;
			}
		}
	}
}

#[inline]
unsafe fn pack_lhs(a: *const f64, lda: usize, pa: *mut f64, rows: usize, depth: usize) {
	for i in (0..rows).step_by(MR) {
		let height = (rows - i).min(MR);
		let src = a.add(i);
		let dst = pa.add(i * depth);
		if height == MR {
			for p in 0..depth {
				core::ptr::copy_nonoverlapping(src.add(p * lda), dst.add(p * MR), MR);
			}
		} else {
			for p in 0..depth {
				core::ptr::copy_nonoverlapping(src.add(p * lda), dst.add(p * MR), height);
				core::ptr::write_bytes(dst.add(p * MR + height), 0, MR - height);
			}
		}
	}
}

#[inline]
unsafe fn pack_rhs(b: *const f64, ldb: usize, pb: *mut f64, cols: usize, depth: usize) {
	for j in (0..cols).step_by(NR) {
		let width = (cols - j).min(NR);
		let src = b.add(j * ldb);
		let dst = pb.add(j * depth);
		if width == NR {
			for p in 0..depth {
				for t in 0..NR {
					*dst.add(p * NR + t) = *src.add(p + t * ldb);
				}
			}
		} else {
			for p in 0..depth {
				for t in 0..width {
					*dst.add(p * NR + t) = *src.add(p + t * ldb);
				}
				core::ptr::write_bytes(dst.add(p * NR + width), 0, NR - width);
			}
		}
	}
}
// B is `L^T`: its depth index is a row of `L` (stride `ldb`), its column index
// is the next row (stride 1), the opposite of `pack_rhs`. The trailing
// Cholesky update needs this because both operands are the same panel.
#[inline]
unsafe fn pack_rhs_t(b: *const f64, ldb: usize, pb: *mut f64, cols: usize, depth: usize) {
	for j in (0..cols).step_by(NR) {
		let width = (cols - j).min(NR);
		let src = b.add(j * ldb);
		let dst = pb.add(j * depth);
		for p in 0..depth {
			for t in 0..width {
				*dst.add(p * NR + t) = *src.add(t + p * ldb);
			}
			for t in width..NR {
				*dst.add(p * NR + t) = 0.;
			}
		}
	}
}


// Cache capacities and scratch lengths are in f64 elements. Row and column
// panel sizes are rounded down to whole microtiles. Inputs may have padding.
// SAFETY: inputs describe positive-stride column-major f64 views with valid
// matrix extents, C is disjoint and writable (initialized for Add), and both
// scratch regions are disjoint, writable and have the specified capacities.
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn multiply(
	m: usize,
	n: usize,
	k: usize,
	a: *const f64,
	lda: usize,
	b: *const f64,
	ldb: usize,
	c: *mut f64,
	ldc: usize,
	alpha: f64,
	beta: Accum,
	pa: *mut f64,
	alen: usize,
	pb: *mut f64,
	blen: usize,
	l1: usize,
	l2: usize,
) -> bool {
	#[cfg(test)]
	PROBE.with(|(_, hits)| hits.set(hits.get() + 1));
	// Fit one A and one B micro-panel in two thirds of L1. A row block
	// then occupies up to half of L2. Haswell's 32/256 KiB gives kc=192, mc=80.
	let kc = (l1 / (MR + NR) * 2 / 3 / 16 * 16).max(16).min(k);
	let mc = (l2 / kc / 2 / MR * MR).min(alen / kc / MR * MR);
	let row_pitch = m.next_multiple_of(MR);
	let mut nc = blen / kc / NR * NR;
	// The remainder of the A scratch holds raw accumulator tiles between
	// cache panels. Fit an integral column band before touching the output.
	let partials = pa.add(mc * kc);
	let split = k.min(DEPTH_BLOCK) > kc;
	if split {
		nc = nc.min((alen - mc * kc) / row_pitch / NR * NR);
	}
	if mc == 0 || nc == 0 {
		return false;
	}
	// Only valid source rows/columns are read. Padding in the packed edge
	// tiles is initialized here; destination padding is never accessed.
	// Keep the final depth-one arithmetic group separate and update C by column.
	let tail = k > 1 && k % DEPTH_BLOCK == 1;
	let packed_k = k - usize::from(tail);
	for group in (0..packed_k).step_by(DEPTH_BLOCK) {
		let group_depth = (packed_k - group).min(DEPTH_BLOCK);
		let add = group != 0 || beta == Accum::Add;
		for j0 in (0..n).step_by(nc) {
			let cols = (n - j0).min(nc);
			for offset in (0..group_depth).step_by(kc) {
				let k0 = group + offset;
				let depth = (group_depth - offset).min(kc);
				let last = offset + depth == group_depth;
				let flags = usize::from(add) * ADD | usize::from(offset != 0) * RESUME | usize::from(last) * FINISH;
				pack_rhs(b.add(k0 + j0 * ldb), ldb, pb, cols, depth);
				for i0 in (0..m).step_by(mc) {
					let rows = (m - i0).min(mc);
					pack_lhs(a.add(i0 + k0 * lda), lda, pa, rows, depth);
					for j in (0..cols).step_by(NR) {
						let width = (cols - j).min(NR);
						for i in (0..rows).step_by(MR) {
							let height = (rows - i).min(MR);
							let ap = pa.add(i * depth);
							let bp = pb.add(j * depth);
							let out = c.add(i0 + i + (j0 + j) * ldc);
							let partial = if split {
								partials.add(j * row_pitch + (i0 + i) * NR)
							} else {
								core::ptr::null_mut()
							};
							if !last || (height == MR && width == NR) {
								tile(ap, bp, out, 8 * ldc, depth, alpha, partial, flags);
							} else {
								let mut edge = [0.; MR * NR];
								if add {
									for col in 0..width {
										core::ptr::copy_nonoverlapping(out.add(col * ldc), edge.as_mut_ptr().add(col * MR), height);
									}
								}
								tile(ap, bp, edge.as_mut_ptr(), 8 * MR, depth, alpha, partial, flags);
								for col in 0..width {
									core::ptr::copy_nonoverlapping(edge.as_ptr().add(col * MR), out.add(col * ldc), height);
								}
							}
						}
					}
				}
			}
		}
	}
	if tail {
		add_rank_one(m, n, a.add(packed_k * lda), b.add(packed_k), ldb, c, ldc, alpha);
	}
	true
}

// C is initialized by earlier arithmetic groups. The column views are disjoint
// and valid for the stated dimensions, as in multiply.
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn add_rank_one(m: usize, n: usize, a: *const f64, b: *const f64, ldb: usize, c: *mut f64, ldc: usize, alpha: f64) {
	use core::arch::x86_64::{_mm256_add_pd, _mm256_mul_pd, _mm256_setzero_pd};
	let factor = _mm256_set1_pd(alpha);
	let zero = _mm256_setzero_pd();
	let vector_end = m / 4 * 4;
	for j in 0..n {
		let rhs = b.add(j * ldb).read();
		let rhs_vector = _mm256_set1_pd(rhs);
		let dst = c.add(j * ldc);
		for i in (0..vector_end).step_by(4) {
			let value = _mm256_fmadd_pd(_mm256_loadu_pd(a.add(i)), rhs_vector, zero);
			let scaled = _mm256_mul_pd(factor, value);
			let out = dst.add(i);
			_mm256_storeu_pd(out, _mm256_add_pd(scaled, _mm256_loadu_pd(out)));
		}
		for i in vector_end..m {
			let value = a.add(i).read().mul_add(rhs, 0.0);
			let out = dst.add(i);
			out.write(alpha * value + out.read());
		}
	}
}

/// The depth cap of the lower update: one depth panel, the `kc` that `multiply`
/// derives from an admitted core's 32 KiB L1 (32 KiB of f64 is 4096 elements,
/// `4096 / (MR + NR) * 2 / 3 / 16 * 16 == 192`, then bounded by `k`). Both the
/// driver below and the public wrapper in `lib.rs` refuse a deeper call through
/// this constant, so the two limits cannot drift apart; a change to `multiply`'s
/// `kc` formula has to change this too.
pub(super) const SYRK_DEPTH_PANEL: usize = 32768 / 8 / (MR + NR) * 2 / 3 / 16 * 16;

/// f64 elements `pack_lhs` writes for `rows` rows and depth `k`: whole `MR`-row
/// tiles, so a partial tile still writes `MR` slots per depth.
#[inline]
pub(super) const fn lhs_pack_len(rows: usize, k: usize) -> usize {
	(rows.saturating_add(MR - 1) / MR)
		.saturating_mul(MR)
		.saturating_mul(k)
}

// Lower-triangular rank-k update `C -= L L^T`, the trailing update of a blocked
// Cholesky factorization: `L` is `n x k` column-major with row stride 1, `C` is
// `n x n` column-major with row stride 1 and only its lower triangle is read or
// written. `alpha` scales the product and `beta` adds it to `C` or replaces it.
//
// `pa` and `pb` are the caller's scratch: `lhs_pack_len(rows, k)` and `NR * k` f64
// (the packer writes whole `MR`-row tiles, so a partial tile still writes `MR`
// slots per depth).
//
// One row block `[i0, i0 + rows)` of `C` is one unit of work: it packs that
// block of `L` once and updates its rows against every column block at or below
// the diagonal. Blocks are independent (disjoint rows) and each entry's sum over
// the depth is the same one a single call makes, so a team can hand out blocks
// and reach the same bits.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn syrk_lower_block(
	n: usize,
	k: usize,
	l: *const f64,
	lda: usize,
	c: *mut f64,
	ldc: usize,
	alpha: f64,
	beta: Accum,
	i0: usize,
	rows: usize,
	pa: *mut f64,
	pb: *mut f64,
) -> bool {
	// One depth panel: the Cholesky block updates use k = 16..=192.
	if k == 0 || k > SYRK_DEPTH_PANEL || lda == 0 || ldc == 0 || rows == 0 || i0 + rows > n {
		return false;
	}
	let flags = if beta == Accum::Add { ADD | FINISH } else { FINISH };
	pack_lhs(l.add(i0), lda, pa, rows, k);
	let j_end = Ord::min(n, i0 + rows);
	for j0 in (0..j_end).step_by(NR) {
		let cols = (n - j0).min(NR);
		pack_rhs_t(l.add(j0), lda, pb, cols, k);
		for i in (0..rows).step_by(MR) {
			let height = (rows - i).min(MR);
			let r0 = i0 + i;
			if r0 + height <= j0 {
				continue; // every entry is above the diagonal
			}
			let ap = pa.add(i * k);
			if r0 >= j0 + cols && height == MR && cols == NR {
				tile(
					ap,
					pb,
					c.add(r0 + j0 * ldc),
					8 * ldc,
					k,
					alpha,
					core::ptr::null_mut(),
					flags,
				);
			} else {
				let mut edge = [0.; MR * NR];
				if beta == Accum::Add {
					for col in 0..cols {
						for row in 0..height {
							if r0 + row >= j0 + col {
								edge[col * MR + row] = *c.add(r0 + row + (j0 + col) * ldc);
							}
						}
					}
				}
				tile(ap, pb, edge.as_mut_ptr(), 8 * MR, k, alpha, core::ptr::null_mut(), flags);
				for col in 0..cols {
					for row in 0..height {
						if r0 + row >= j0 + col {
							*c.add(r0 + row + (j0 + col) * ldc) = edge[col * MR + row];
						}
					}
				}
			}
		}
	}
	true
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn admitted_models_cover_the_measured_families() {
		assert!(admitted_model(0x3F)); // Haswell-EP, the measured host
		assert!(admitted_model(0x9E)); // Coffee Lake desktop (i9-9900K)
		assert!(admitted_model(0x4E)); // Skylake client
		assert!(!admitted_model(0x55)); // Skylake-SP: AVX-512 dispatch, never here
		assert!(!admitted_model(0x8F)); // Sapphire Rapids
	}

	#[test]
	fn depth_one_suffix_preserves_order_and_padding() {
		if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
			return;
		}
		let (m, n, k) = (9, 7, DEPTH_BLOCK + 1);
		let (lda, ldb, ldc) = (11, k + 3, 13);
		let a: Vec<_> = (0..lda * k).map(|i| ((i % 17) as f64 - 8.) / 7.).collect();
		let b: Vec<_> = (0..ldb * n).map(|i| ((i % 13) as f64 - 6.) / 11.).collect();
		let alpha = -0.27;
		let mut left = [0.; 8192];
		let mut right = [0.; 8192];
		for beta in [Accum::Replace, Accum::Add] {
			let mut c = vec![3.; ldc * n];
			assert!(unsafe {
				multiply(
					m,
					n,
					k,
					a.as_ptr(),
					lda,
					b.as_ptr(),
					ldb,
					c.as_mut_ptr(),
					ldc,
					alpha,
					beta,
					left.as_mut_ptr(),
					left.len(),
					right.as_mut_ptr(),
					right.len(),
					4096,
					32768,
				)
			});
			for j in 0..n {
				for i in 0..m {
					let mut dot = 0.;
					for p in 0..DEPTH_BLOCK {
						dot = a[i + p * lda].mul_add(b[p + j * ldb], dot);
					}
					let first = alpha * dot;
					let first = if beta == Accum::Add { first + 3. } else { first };
					let tail = a[i + DEPTH_BLOCK * lda].mul_add(b[DEPTH_BLOCK + j * ldb], 0.);
					let expected = alpha * tail + first;
					assert_eq!(c[i + ldc * j].to_bits(), expected.to_bits());
				}
				assert!(c[m + ldc * j..ldc * (j + 1)].iter().all(|&v| v == 3.));
			}
		}
	}

	#[test]
	fn bounded_panels_and_column_tail() {
		if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
			return;
		}
		let (m, n, k) = (17usize, 13, 2 * DEPTH_BLOCK + 1);
		let a: Vec<_> = (0..m * k).map(|p| ((p % 11) as f64 - 5.) / 7.).collect();
		let b: Vec<_> = (0..n * k).map(|p| ((p % 9) as f64 - 4.) / 11.).collect();
		let alpha = -0.27;
		// Exactly one packed tile per side. Each loop must cross its panel
		// boundary. Sentinel slots surrounding the capacity are never scratch.
		let left_len = MR * 16 + m.next_multiple_of(MR) * NR;
		let mut left = vec![19.; left_len + 2];
		let mut right = vec![23.; NR * 16 + 2];
		let mut c = vec![1.; m * n];
		let mut run = |alen, blen| unsafe {
			multiply(
				m,
				n,
				k,
				a.as_ptr(),
				m,
				b.as_ptr(),
				k,
				c.as_mut_ptr(),
				m,
				alpha,
				Accum::Add,
				left.as_mut_ptr().add(1),
				alen,
				right.as_mut_ptr().add(1),
				blen,
				16,
				256,
			)
		};
		assert!(!run(MR * 16 - 1, NR * 16));
		assert!(!run(MR * 16, NR * 16 - 1));
		assert!(run(left_len, NR * 16));
		for j in 0..n {
			for i in 0..m {
				let mut expected = 1.0;
				for start in (0..k).step_by(DEPTH_BLOCK) {
					let dot = (start..k.min(start + DEPTH_BLOCK))
						.fold(0.0, |sum, p| a[i + p * m].mul_add(b[p + j * k], sum));
					expected += alpha * dot;
				}
				assert_eq!(c[i + j * m].to_bits(), expected.to_bits(), "row={i}, col={j}");
			}
		}
		assert_eq!((left[0], left[left.len() - 1]), (19., 19.));
		assert_eq!((right[0], right[right.len() - 1]), (23., 23.));
	}
	// Adding two A entries would overflow, while every classical product is
	// exactly two. Exercise this case through multiple packed depth panels.
	#[test]
	fn large_finite_inputs_with_small_multipliers() {
		if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
			return;
		}
		let (m, n, k) = (9, 7, 35);
		let a = vec![2.0_f64.powi(1023); m * k];
		let b = vec![f64::MIN_POSITIVE; k * n];
		let mut c = vec![f64::NAN; m * n];
		let mut left = [0.; MR * 16 + 2 * MR * NR];
		let mut right = [0.; NR * 16];
		let used = unsafe {
			multiply(
				m,
				n,
				k,
				a.as_ptr(),
				m,
				b.as_ptr(),
				k,
				c.as_mut_ptr(),
				m,
				1.,
				Accum::Replace,
				left.as_mut_ptr(),
				left.len(),
				right.as_mut_ptr(),
				right.len(),
				16,
				256,
			)
		};
		assert!(used);
		assert!(c.iter().all(|&value| value == 2. * k as f64));
	}

	// A full, single-thread, contiguous column-major f64 product: the only
	// shape the packed driver accepts.
	#[allow(clippy::too_many_arguments)]
	fn product(m: usize, n: usize, k: usize, a: &[f64], lda: usize, b: &[f64], ldb: usize, c: &mut [f64], ldc: usize, alpha: f64, beta: Accum) {
		unsafe {
			crate::gemm(
				crate::DType::F64,
				crate::IType::U64,
				crate::InstrSet::Avx256,
				m,
				n,
				k,
				c.as_mut_ptr().cast(),
				1,
				ldc as isize,
				core::ptr::null(),
				core::ptr::null(),
				crate::DstKind::Full,
				beta,
				a.as_ptr().cast(),
				1,
				lda as isize,
				false,
				core::ptr::null(),
				0,
				b.as_ptr().cast(),
				1,
				ldb as isize,
				false,
				(&alpha as *const f64).cast(),
				1,
			)
		}
	}

	fn avx2_fma() -> bool {
		is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
	}

	// The gate also demands 128 scalar updates per L2 element, so the
	// qualifying depth depends on the host's reported L2 size.
	fn qualifying_depth(m: usize, n: usize) -> usize {
		let cache = *crate::CACHE_INFO;
		let l2 = cache[1].cache_bytes / core::mem::size_of::<f64>();
		(crate::packed_avx2::floor_factor() * l2 + 1).div_ceil(m * n)
	}

	// The gate is the only way into this driver, so a qualifying product must
	// reach it and a product below its shape thresholds must not. The answer is
	// forced, because the host is not always the measured Haswell family.
	#[test]
	fn haswell_gate_is_reached_by_a_qualifying_product() {
		if !avx2_fma() {
			return;
		}
		let (m, n) = (89, 73);
		let k = qualifying_depth(m, n);
		let a: Vec<_> = (0..m * k).map(|p| (p % 17) as f64 / 7.).collect();
		let b: Vec<_> = (0..k * n).map(|p| (p % 13) as f64 / 11.).collect();
		let mut c = vec![0.; m * n];
		force_gate(1);
		let before = gate_hits();
		product(m, n, k, &a, m, &b, k, &mut c, m, -0.75, Accum::Replace);
		assert_eq!(gate_hits(), before + 1, "a qualifying product reaches the driver");
		// m below 32 keeps the same product out of the packed driver.
		let (m, n) = (16, 8);
		let a: Vec<_> = (0..m * k).map(|p| (p % 17) as f64 / 7.).collect();
		let b: Vec<_> = (0..k * n).map(|p| (p % 13) as f64 / 11.).collect();
		let mut c = vec![0.; m * n];
		product(m, n, k, &a, m, &b, k, &mut c, m, -0.75, Accum::Replace);
		assert_eq!(gate_hits(), before + 1, "a product below the gate's row threshold stays out");
		force_gate(0);
	}

	// Both drivers group the arithmetic the same way, so the gate decision must
	// not change one bit of the output.
	#[test]
	fn gate_off_matches_gate_on_bit_for_bit() {
		if !avx2_fma() {
			return;
		}
		let (m, n) = (89, 73);
		let first = qualifying_depth(m, n);
		// The smallest depth the gate accepts, one more full depth block, 513
		// (one past a depth block) and the smallest accepted depth that ends in
		// a depth-one tail, so the compared depths keep both shapes whatever L2
		// the host reports.
		let tail = first.div_ceil(DEPTH_BLOCK) * DEPTH_BLOCK + 1;
		let mut depths = vec![first, first + DEPTH_BLOCK, 513, tail];
		depths.sort_unstable();
		depths.retain(|&k| k >= first);
		depths.dedup();
		let mut seed = 0x42B5E91u64;
		let mut sample = || {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			(seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
		};
		for k in depths {
			let a: Vec<_> = (0..m * k).map(|_| sample()).collect();
			let b: Vec<_> = (0..k * n).map(|_| sample()).collect();
			for beta in [Accum::Replace, Accum::Add] {
				let initial = vec![0.125; m * n];
				let (mut on, mut off) = (initial.clone(), initial);
				force_gate(1);
				let hits = gate_hits();
				product(m, n, k, &a, m, &b, k, &mut on, m, -0.75, beta);
				assert_eq!(gate_hits(), hits + 1, "the packed driver ran with the gate on");
				force_gate(-1);
				let hits = gate_hits();
				product(m, n, k, &a, m, &b, k, &mut off, m, -0.75, beta);
				assert_eq!(gate_hits(), hits, "the general driver ran with the gate off");
				force_gate(0);
				for j in 0..n {
					for i in 0..m {
						assert_eq!(on[i + j * m].to_bits(), off[i + j * m].to_bits(), "k={k}, i={i}, j={j}, beta={beta:?}");
					}
				}
			}
		}
	}

	// The packed driver's own scratch, as the caller allocates it: one row block
	// of A panels plus the raw accumulator tiles, and one B column panel.
	#[allow(clippy::too_many_arguments)]
	fn run_packed(
		m: usize,
		n: usize,
		k: usize,
		a: &[f64],
		lda: usize,
		b: &[f64],
		ldb: usize,
		c: &mut [f64],
		ldc: usize,
		alpha: f64,
		beta: Accum,
	) -> bool {
		let mut pa = vec![0.; 96 * 1024];
		let mut pb = vec![0.; 32 * 1024];
		unsafe {
			multiply(
				m,
				n,
				k,
				a.as_ptr(),
				lda,
				b.as_ptr(),
				ldb,
				c.as_mut_ptr(),
				ldc,
				alpha,
				beta,
				pa.as_mut_ptr(),
				pa.len(),
				pb.as_mut_ptr(),
				pb.len(),
				4096,
				32768,
			)
		}
	}

	// Moved here from `tests/packed_avx2.rs`, where an integration test cannot
	// reach the force flag and so only tested the general driver on a host that
	// is not the measured family. These depths are below the dispatch gate's
	// shape rule, so the driver is called directly and has to report that it ran.
	#[test]
	fn forced_packed_driver_preserves_scaled_entries_and_padded_odd_edges() {
		if !avx2_fma() {
			return;
		}
		let (m, n) = (89usize, 73usize);
		let (lda, ldc) = (m + 3, m + 7);
		for k in [191, 192, 193, 385, 513] {
			let ldb = k + 5;
			let mut a = vec![f64::NAN; lda * k];
			let mut b = vec![f64::NAN; ldb * n];
			for p in 0..k {
				for i in 0..m {
					a[i + p * lda] = ((i + 3 * p) % 9) as f64 - 4.;
					a[i + p * lda] *= 2.0f64.powi(if i % 2 == 0 { -450 } else { 450 });
				}
				for j in 0..n {
					b[p + j * ldb] = ((2 * j + p) % 7) as f64 - 3.;
					b[p + j * ldb] *= 2.0f64.powi(if j % 2 == 0 { 450 } else { -450 });
				}
			}
			for mode in [Accum::Replace, Accum::Add] {
				let alpha = -0.75_f64;
				let mut c = vec![42.; ldc * n];
				for j in 0..n {
					for i in 0..m {
						c[i + j * ldc] = if mode == Accum::Add {
							2.0f64.powi(match (i % 2, j % 2) {
								(0, 1) => -900,
								(1, 0) => 900,
								_ => 0,
							})
						} else {
							f64::NAN
						};
					}
				}
				let initial = c.clone();
				assert!(
					run_packed(m, n, k, &a, lda, &b, ldb, &mut c, ldc, alpha, mode),
					"the packed driver ran, k={k}, mode={mode:?}"
				);
				for j in 0..n {
					for i in 0..m {
						let mut want = initial[i + j * ldc];
						for start in (0..k).step_by(512) {
							let sum = (start..k.min(start + 512)).fold(0.0, |v, p| a[i + p * lda].mul_add(b[p + j * ldb], v));
							let scaled = sum * alpha;
							if start == 0 && mode == Accum::Replace {
								want = scaled;
							} else {
								want += scaled;
							}
						}
						assert_eq!(c[i + j * ldc], want, "i={i}, j={j}, k={k}, mode={mode:?}");
					}
					assert!(c[j * ldc + m..(j + 1) * ldc].iter().all(|x| *x == 42.));
				}
			}
		}
	}

	// The random-data half of the same move: the original 191..1025 sweep must
	// equal scalar FMA over the driver's own 512-depth groups.
	#[test]
	fn forced_packed_driver_matches_the_scalar_depth_groups_bit_for_bit() {
		if !avx2_fma() {
			return;
		}
		let (m, n) = (89, 73);
		let mut seed = 0x42B5E91u64;
		let mut sample = || {
			seed ^= seed << 13;
			seed ^= seed >> 7;
			seed ^= seed << 17;
			(seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
		};
		for k in [191usize, 193, 511, 512, 513, 1025] {
			let a: Vec<_> = (0..m * k).map(|_| sample()).collect();
			let b: Vec<_> = (0..k * n).map(|_| sample()).collect();
			for mode in [Accum::Add, Accum::Replace] {
				let alpha = -0.75_f64;
				let mut c = vec![0.125_f64; m * n];
				assert!(
					run_packed(m, n, k, &a, m, &b, k, &mut c, m, alpha, mode),
					"the packed driver ran, k={k}, mode={mode:?}"
				);
				for j in 0..n {
					for i in 0..m {
						let mut expected = 0.125;
						for start in (0..k).step_by(512) {
							let dot = (start..k.min(start + 512)).fold(0.0, |sum, p| a[i + p * m].mul_add(b[p + j * k], sum));
							let scaled = dot * alpha;
							if start == 0 && mode == Accum::Replace {
								expected = scaled;
							} else {
								expected += scaled;
							}
						}
						assert_eq!(c[i + j * m].to_bits(), expected.to_bits(), "k={k}, i={i}, j={j}, mode={mode:?}");
					}
				}
			}
		}
	}
	/// Runs one block of the lower update and returns the result next to the
	/// scalar expectation, using exactly the documented scratch extent with a
	/// canary past it.
	fn syrk_block_case(
		n: usize,
		k: usize,
		i0: usize,
		rows: usize,
		alpha: f64,
		beta: Accum,
	) -> Option<(Vec<f64>, Vec<f64>)> {
		let lda = n + 3;
		let ldc = n + 5;
		let l: Vec<f64> = (0..lda * k).map(|i| ((i % 17) as f64 - 8.) / 7.).collect();
		// the strict upper triangle and the rows outside the block carry canaries
		let mut c = vec![0.; ldc * n];
		for j in 0..n {
			for i in 0..n {
				c[i + j * ldc] = ((i * 5 + j) % 11) as f64 / 3. - 4.;
			}
		}
		let before = c.clone();
		let (pa_len, pb_len) = (lhs_pack_len(rows, k), NR * k);
		let mut scratch = vec![0.; pa_len + pb_len + 8];
		for v in &mut scratch[pa_len + pb_len..] {
			*v = -1234.5;
		}
		let (pa, pb) = scratch.split_at_mut(pa_len);
		let ran = unsafe {
			syrk_lower_block(
				n,
				k,
				l.as_ptr(),
				lda,
				c.as_mut_ptr(),
				ldc,
				alpha,
				beta,
				i0,
				rows,
				pa.as_mut_ptr(),
				pb.as_mut_ptr(),
			)
		};
		if !ran {
			return None;
		}
		assert!(
			scratch[pa_len + pb_len..].iter().all(|&v| v == -1234.5),
			"the packer stayed inside the documented scratch, n={n}, k={k}, rows={rows}"
		);
		// outside the block, and the strict upper triangle, keep their bits
		for j in 0..n {
			for i in 0..n {
				let touched = (i0..i0 + rows).contains(&i) && i >= j;
				if !touched {
					assert_eq!(
						c[i + j * ldc].to_bits(),
						before[i + j * ldc].to_bits(),
						"untouched n={n}, k={k}, i0={i0}, rows={rows}, i={i}, j={j}"
					);
				}
			}
		}
		let mut want = before.clone();
		for j in 0..n {
			for i in (j..n).filter(|i| (i0..i0 + rows).contains(i)) {
				let mut dot = 0.;
				for p in 0..k {
					dot = l[i + p * lda].mul_add(l[j + p * lda], dot);
				}
				want[i + j * ldc] = match beta {
					Accum::Add => before[i + j * ldc] + alpha * dot,
					Accum::Replace => alpha * dot,
				};
			}
		}
		Some((c, want))
	}

	#[test]
	fn syrk_block_matches_the_scalar_lower_update() {
		if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
			return;
		}
		for (n, k) in [(13usize, 5usize), (96, 64), (97, 192)] {
			for beta in [Accum::Add, Accum::Replace] {
				for alpha in [-1.0, 0.75] {
					// whole range, a block narrower than the microkernels, and blocks
					// of 1..=7 rows so every partial-tile path is taken
					let mut blocks = vec![(0usize, n)];
					blocks.push((n / 3, n - n / 3));
					for r in [1usize, 3, 5, 7] {
						if r < n {
							blocks.push((n - r, r));
						}
					}
					for (i0, rows) in blocks {
						let (got, want) = syrk_block_case(n, k, i0, rows, alpha, beta)
							.unwrap_or_else(|| panic!("the driver ran, n={n}, k={k}, rows={rows}"));
						for j in 0..n {
							for i in 0..n {
								let t = got[i + j * ldc_of(n)] - want[i + j * ldc_of(n)];
								assert!(
									t.abs() <= 1e-12 * want[i + j * ldc_of(n)].abs().max(1.),
									"n={n}, k={k}, i0={i0}, rows={rows}, beta={beta:?}: {} vs {}",
									got[i + j * ldc_of(n)],
									want[i + j * ldc_of(n)]
								);
							}
						}
					}
				}
			}
		}
		// a rejected call must leave the destination bit-unchanged and must not
		// allocate: one element past the depth panel is outside the driver
		let n = 32;
		let k = SYRK_DEPTH_PANEL + 1;
		let lda = n;
		let ldc = n;
		let l: Vec<f64> = (0..lda * k).map(|i| (i % 7) as f64).collect();
		let mut c = vec![1.5; ldc * n];
		let before = c.clone();
		let ran = unsafe {
			crate::syrk_lower_block(
				n,
				k,
				l.as_ptr(),
				lda,
				c.as_mut_ptr(),
				ldc,
				-1.0,
				Accum::Add,
				0,
				n,
			)
		};
		assert!(!ran, "k = {k} is outside the driver");
		assert!(c.iter().zip(&before).all(|(a, b)| a.to_bits() == b.to_bits()));
	}

	/// The test cases above use `ldc = n + 5`.
	fn ldc_of(n: usize) -> usize {
		n + 5
	}

	// The public entry's success path: whole range in one call against three
	// blocks, the way a team splits the trailing update. The test-only override
	// skips only the model admission, so this runs on every AVX2+FMA host and
	// the entry's scratch sizing, block validation and `Accum` plumbing are
	// exercised where the production gate does and where it does not.
	#[test]
	fn syrk_entry_whole_range_and_partitions_are_bit_identical() {
		if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
			return;
		}
		let _forced_on = ForcedGate::new(1);
		assert!(haswell());
		for (n, k) in [(96usize, 64usize), (97, 48)] {
			let lda = n + 3;
			let ldc = n + 5;
			let l: Vec<f64> = (0..lda * k).map(|i| ((i % 13) as f64 - 6.) / 5.).collect();
			let base: Vec<f64> = (0..ldc * n)
				.map(|i| if i % ldc <= (i / ldc) { ((i % 7) as f64) / 3. } else { 9.5 })
				.collect();
			let mut whole = base.clone();
			let mut split = base.clone();
			assert!(unsafe {
				crate::syrk_lower_block(n, k, l.as_ptr(), lda, whole.as_mut_ptr(), ldc, -1.0, Accum::Add, 0, n)
			});
			let step = n / 3;
			let mut i0 = 0;
			while i0 < n {
				let rows = (n - i0).min(step).max(1);
				assert!(unsafe {
					crate::syrk_lower_block(
						n, k, l.as_ptr(), lda, split.as_mut_ptr(), ldc, -1.0, Accum::Add, i0, rows,
					)
				}, "block i0={i0}, rows={rows}");
				i0 += rows;
			}
			assert!(
				whole.iter().zip(&split).all(|(a, b)| a.to_bits() == b.to_bits()),
				"the row blocks are bit-identical to the whole call, n={n}, k={k}"
			);
		}
	}

	// The scratch is the per-thread buffer, grown once and reused: a repeated
	// call, and a later smaller one, must not reallocate it.
	#[test]
	fn syrk_scratch_is_reused_across_calls() {
		if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
			return;
		}
		let _forced_on = ForcedGate::new(1);
		let (n, k) = (96usize, 64usize);
		let (lda, ldc) = (n + 3, n + 5);
		let l: Vec<f64> = (0..lda * k).map(|i| ((i % 13) as f64 - 6.) / 5.).collect();
		let mut c = vec![0.5f64; ldc * n];
		assert!(unsafe {
			crate::syrk_lower_block(n, k, l.as_ptr(), lda, c.as_mut_ptr(), ldc, -1.0, Accum::Replace, 0, n)
		});
		let first = crate::syrk_scratch_probe();
		assert!(
			first.0 >= lhs_pack_len(n, k) + 6 * k,
			"the scratch holds the packed block plus the RHS panel"
		);
		assert!(unsafe {
			crate::syrk_lower_block(n, k, l.as_ptr(), lda, c.as_mut_ptr(), ldc, -1.0, Accum::Add, 0, n)
		});
		assert_eq!(crate::syrk_scratch_probe(), first, "a repeated call must reuse the buffer");
		// a smaller shape is served from the same allocation
		assert!(unsafe { crate::syrk_lower_block(n, 16, l.as_ptr(), lda, c.as_mut_ptr(), ldc, -1.0, Accum::Add, 0, n) });
		assert_eq!(
			crate::syrk_scratch_probe(),
			first,
			"a smaller shape must not reallocate the buffer"
		);
	}

	/// Forces the gate for the current thread and restores it on drop.
	struct ForcedGate;
	impl ForcedGate {
		fn new(state: i8) -> Self {
			force_gate(state);
			ForcedGate
		}
	}
	impl Drop for ForcedGate {
		fn drop(&mut self) {
			force_gate(0);
		}
	}

	// The refusal path, covered on every host: with the gate off the entry
	// returns false and leaves `c` (triangles included) untouched.
	#[test]
	fn syrk_entry_refuses_and_writes_nothing_when_the_gate_is_off() {
		let _off = ForcedGate::new(-1);
		assert!(!haswell());
		let (n, k) = (96usize, 64usize);
		let (lda, ldc) = (n + 3, n + 5);
		let l: Vec<f64> = (0..lda * k).map(|i| ((i % 13) as f64 - 6.) / 5.).collect();
		let base: Vec<f64> = (0..ldc * n).map(|i| (i % 11) as f64 + 0.25).collect();
		for beta in [Accum::Add, Accum::Replace] {
			let mut c = base.clone();
			let ran = unsafe { crate::syrk_lower_block(n, k, l.as_ptr(), lda, c.as_mut_ptr(), ldc, -1.0, beta, 0, n) };
			assert!(!ran, "a gate-excluded host must be refused");
			assert!(
				c.iter().zip(&base).all(|(a, b)| a.to_bits() == b.to_bits()),
				"a refused call must not write c"
			);
		}
	}

	// On a host the real gate excludes (AMD, unlisted Intel), the unforced entry
	// refuses too; where the gate admits, it runs.
	#[test]
	fn syrk_entry_follows_the_production_gate() {
		let (n, k) = (32usize, 8usize);
		let l = vec![1.0f64; n * k];
		let mut c = vec![0.0f64; n * n];
		let ran = unsafe { crate::syrk_lower_block(n, k, l.as_ptr(), n, c.as_mut_ptr(), n, 1.0, Accum::Replace, 0, n) };
		assert_eq!(ran, haswell());
		if !ran {
			assert!(c.iter().all(|&v| v == 0.0));
		}
	}

}
