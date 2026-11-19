//! Sequential, packed 8 × 6 classical f64 GEMM for contiguous matrix rows.
use crate::{Accum, DEPTH_BLOCK};
use core::arch::asm;

const MR: usize = 8;
const NR: usize = 6;

// Other AVX2 cores have different load ports and cache geometry. Scope this
// implementation to the Intel family it was measured on.
pub(super) fn haswell() -> bool {
	static HASWELL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
	*HASWELL.get_or_init(|| {
		let cpuid = raw_cpuid::CpuId::new();
		cpuid.get_vendor_info().is_some_and(|v| v.as_str() == "GenuineIntel")
			&& cpuid
				.get_feature_info()
				.is_some_and(|f| f.family_id() == 6 && matches!(f.model_id(), 0x3C | 0x3F | 0x45 | 0x46))
	})
}

// Two A vectors and two alternating B broadcasts leave twelve registers for
// the output. Each accumulator visits depth indices in increasing order.
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn tile(a: *const f64, b: *const f64, c: *mut f64, ldc: usize, depth: usize, alpha: f64, partial: *mut f64, flags: usize) {
	asm!(
		include_str!("packed_avx2.s"),
		a=inout(reg) a=>_, b=inout(reg) b=>_, c=inout(reg) c=>_, ld=in(reg) ldc,
		k=inout(reg) depth=>_, count=out(reg) _, alpha=in(reg) alpha.to_bits(), flags=in(reg) flags, partial=in(reg) partial,
		out("ymm0") _, out("ymm1") _, out("ymm2") _, out("ymm3") _, out("ymm4") _, out("ymm5") _,
		out("ymm6") _, out("ymm7") _, out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
		out("ymm12") _, out("ymm13") _, out("ymm14") _, out("ymm15") _, options(nostack)
	);
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
	for group in (0..k).step_by(DEPTH_BLOCK) {
		let group_depth = (k - group).min(DEPTH_BLOCK);
		let add = group != 0 || beta == Accum::Add;
		for j0 in (0..n).step_by(nc) {
			let cols = (n - j0).min(nc);
			for offset in (0..group_depth).step_by(kc) {
				let k0 = group + offset;
				let depth = (group_depth - offset).min(kc);
				let last = offset + depth == group_depth;
				let flags = usize::from(add) | (usize::from(offset != 0) << 1) | (usize::from(last) << 2);
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
	true
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn bounded_panels_and_column_tail() {
		if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
			return;
		}
		let (m, n, k) = (17usize, 13, 35);
		let a: Vec<_> = (0..m * k).map(|p| (p % 11) as f64 - 5.).collect();
		let b: Vec<_> = (0..n * k).map(|p| (p % 9) as f64 - 4.).collect();
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
				-0.5,
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
				let dot: f64 = (0..k).map(|p| a[i + p * m] * b[p + j * k]).sum();
				assert_eq!(c[i + j * m], 1. - 0.5 * dot);
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
}
