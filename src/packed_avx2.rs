//! Sequential, packed 8 × 6 classical f64 GEMM for contiguous matrix rows.
use crate::{Accum, DEPTH_BLOCK};
use core::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_set1_pd, _mm256_storeu_pd};
#[cfg(test)]
use core::cell::Cell;

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
pub(super) fn haswell() -> bool {
	#[cfg(test)]
	match PROBE.with(|(force, _)| force.get()) {
		0 => {},
		force => return force > 0,
	}
	static HASWELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*HASWELL.get_or_init(|| {
		let cpuid = raw_cpuid::CpuId::new();
		cpuid.get_vendor_info().is_some_and(|v| v.as_str() == "GenuineIntel")
			&& cpuid
				.get_feature_info()
				.is_some_and(|f| f.family_id() == 6 && matches!(f.model_id(), 0x3C | 0x3F | 0x45 | 0x46))
	})
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
		for (j, col) in sums.iter().enumerate() {
			let dst = c.byte_add(j * ldc_bytes);
			for (i, &value) in col.iter().enumerate() {
				let slot = dst.add(i);
				let scaled = alpha * value;
				slot.write(if flags & ADD == 0 { scaled } else { scaled + slot.read() });
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

#[cfg(test)]
mod tests {
	use super::*;

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

	// The gate also demands sixteen scalar updates per L2 element, so the
	// qualifying depth depends on the host's reported L2 size.
	fn qualifying_depth(m: usize, n: usize) -> usize {
		let cache = *crate::CACHE_INFO;
		let l2 = cache[1].cache_bytes / core::mem::size_of::<f64>();
		(16 * l2).div_ceil(m * n)
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
}
