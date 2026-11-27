//! Column-major overwrite GEMM on Raptor Lake P-cores. The full 8 × 6 tiles
//! read B in place. While their twelve accumulators are live, spare load/store
//! bandwidth packs A for subsequent tiles.
use super::{Accum, DEPTH_BLOCK, FINISH, MR, NR, add_rank_one, pack_lhs, pack_rhs, tile};
use core::arch::x86_64::*;

const UNROLL: usize = 8;
const COMPUTE: usize = 0;
const PACK_PANEL: usize = 1;
const PACK_SLAB: usize = 2;

/// Same view and scratch requirements as the parent packed driver. Check all
/// capacities before the first write so a caller can fall back on refusal.
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
	// Handle one arithmetic group and its optional terminal rank-one update.
	// The general driver owns accumulation across multiple complete groups.
	if k < UNROLL || k > DEPTH_BLOCK + 1 || alpha != 1.0 || beta != Accum::Replace {
		return false;
	}
	let depth = k.min(DEPTH_BLOCK);
	let pitch = MR * depth;
	if depth * NR > l1 * 2 / 3 || pitch > alen || depth * NR > blen {
		return false;
	}
	let ways = crate::cache::CACHE_INFO[1].associativity;
	let ways_room = if ways >= 4 { l2 / ways } else { 0 };
	let block_rows = ((l2 / 2 - ways_room) / pitch).max(1).min(alen / pitch) * MR;
	let two_buffers = m > block_rows && n / NR > UNROLL && block_rows <= alen / depth / 2;
	let full_cols = n / NR * NR;
	let stage_from = full_cols.saturating_sub(UNROLL * NR);
	let row_bounds = |start: usize| {
		let len = (m - start).min(block_rows);
		// Overwrite produces the same dot on overlapping rows. Let a short
		// final block end with a complete tile when it has a predecessor.
		let overlap = if start >= MR && len > 0 { (MR - len % MR) % MR } else { 0 };
		(start - overlap, len + overlap)
	};
	for start in (0..m).step_by(block_rows) {
		let (i0, rows) = row_bounds(start);
		let slot = start / block_rows % 2;
		let (next_i, next_rows) = if two_buffers {
			row_bounds(start.saturating_add(block_rows).min(m))
		} else {
			(m, 0)
		};
		let current = if two_buffers { pa.add(slot * block_rows * depth) } else { pa };
		let next = if two_buffers { pa.add((1 - slot) * block_rows * depth) } else { pa };
		let seed = (start == 0 || !two_buffers) && rows >= 2 * MR && n >= NR;
		let full_rows = rows / MR * MR;
		if seed {
			pack_lhs(a.add(i0), lda, current, MR, depth);
			if full_rows < rows {
				pack_lhs(a.add(i0 + full_rows), lda, current.add(full_rows * depth), rows - full_rows, depth);
			}
		} else if start == 0 || !two_buffers {
			pack_lhs(a.add(i0), lda, current, rows, depth);
		}
		if next_rows != 0 {
			let next_full = next_rows / MR * MR;
			if next_full < next_rows {
				pack_lhs(a.add(next_i + next_full), lda, next.add(next_full * depth), next_rows - next_full, depth);
			}
			for i in (0..next_full).step_by(MR) {
				for p in (depth / UNROLL * UNROLL)..depth {
					core::ptr::copy_nonoverlapping(a.add(next_i + i + p * lda), next.add(i * depth + p * MR), MR);
				}
			}
		}
		for j in (0..n).step_by(NR) {
			let width = (n - j).min(NR);
			let right = b.add(j * ldb);
			let out = c.add(i0 + j * ldc);
			let rank = if k > depth {
				Some((a.add(i0 + depth * lda), right.add(depth)))
			} else {
				None
			};
			let done = if width == NR {
				let full = rows / MR;
				let staged = if next_rows >= MR && j >= stage_from {
					let count = next_rows / MR;
					let p = (j - stage_from) / NR * (depth / UNROLL);
					panels::<PACK_SLAB>(
						depth,
						current,
						right,
						out,
						ldb,
						ldc,
						count,
						a.add(next_i + p * lda),
						lda,
						next.add(p * MR),
						rank,
					);
					count
				} else if seed && j == 0 {
					let count = full - 1;
					panels::<PACK_PANEL>(depth, current, right, out, ldb, ldc, count, a.add(i0 + MR), lda, current.add(pitch), rank);
					count
				} else {
					0
				};
				if staged < full {
					let row = staged * MR;
					panels::<COMPUTE>(
						depth,
						current.add(row * depth),
						right,
						out.add(row),
						ldb,
						ldc,
						full - staged,
						core::ptr::null(),
						lda,
						core::ptr::null_mut(),
						rank.map(|(left, right)| (left.add(row), right)),
					);
				}
				full * MR
			} else {
				match width {
					1 => narrow_columns::<1, 6>(current, right, out, depth, ldb, ldc, rows),
					2 => narrow_columns::<2, 3>(current, right, out, depth, ldb, ldc, rows),
					3 => narrow_columns::<3, 2>(current, right, out, depth, ldb, ldc, rows),
					4 => narrow_columns::<4, 1>(current, right, out, depth, ldb, ldc, rows),
					5 => narrow_columns::<5, 1>(current, right, out, depth, ldb, ldc, rows),
					_ => unreachable!(),
				}
			};
			if done < rows {
				pack_rhs(right, ldb, pb, width, depth);
				for i in (done..rows).step_by(MR) {
					let height = (rows - i).min(MR);
					let mut edge = [[0.0; MR]; NR];
					tile(
						current.add(i * depth),
						pb,
						edge.as_mut_ptr().cast(),
						MR * 8,
						depth,
						alpha,
						core::ptr::null_mut(),
						FINISH,
					);
					for (col, panel) in edge[..width].iter().enumerate() {
						core::ptr::copy_nonoverlapping(panel.as_ptr(), out.add(i + col * ldc), height);
					}
				}
			}
			if let Some((left, rhs)) = rank {
				let first = if width == NR { done } else { 0 };
				if first < rows {
					add_rank_one(rows - first, width, left.add(first), rhs, ldb, out.add(first), ldc, alpha);
				}
			}
		}
	}
	true
}

/// Consume complete packed A panels while keeping only valid output columns live.
#[target_feature(enable = "avx2,fma")]
unsafe fn narrow_columns<const COLS: usize, const PANELS: usize>(
	a: *const f64,
	b: *const f64,
	c: *mut f64,
	depth: usize,
	ldb: usize,
	ldc: usize,
	rows: usize,
) -> usize {
	use core::arch::x86_64::*;
	let group = MR * PANELS;
	let complete = rows / group * group;
	for i in (0..complete).step_by(group) {
		let mut sums = [[[_mm256_setzero_pd(); 2]; PANELS]; COLS];
		for p in 0..depth {
			let factors: [__m256d; COLS] = core::array::from_fn(|j| _mm256_set1_pd(*b.add(j * ldb + p)));
			for t in 0..PANELS {
				let ap = a.add((i + t * MR) * depth + p * MR);
				let low = _mm256_loadu_pd(ap);
				let high = _mm256_loadu_pd(ap.add(4));
				for j in 0..COLS {
					sums[j][t][0] = _mm256_fmadd_pd(low, factors[j], sums[j][t][0]);
					sums[j][t][1] = _mm256_fmadd_pd(high, factors[j], sums[j][t][1]);
				}
			}
		}
		for j in 0..COLS {
			for t in 0..PANELS {
				let dest = c.add(i + j * ldc + t * MR);
				_mm256_storeu_pd(dest, sums[j][t][0]);
				_mm256_storeu_pd(dest.add(4), sums[j][t][1]);
			}
		}
	}
	complete
}

macro_rules! accumulate_8x6 {
	($depth:literal) => {
		concat!(
			"vmovupd ymm12, [{a} + 64*(",
			$depth,
			")]\n",
			"vmovupd ymm13, [{a} + 64*(",
			$depth,
			") + 32]\n",
			"vbroadcastsd ymm14, [{b0} + {idx} + 8*(",
			$depth,
			")]\n",
			"vbroadcastsd ymm15, [{b1} + {idx} + 8*(",
			$depth,
			")]\n",
			"vfmadd231pd ymm0, ymm12, ymm14\n",
			"vfmadd231pd ymm1, ymm13, ymm14\n",
			"vbroadcastsd ymm14, [{b2} + {idx} + 8*(",
			$depth,
			")]\n",
			"vfmadd231pd ymm2, ymm12, ymm15\n",
			"vfmadd231pd ymm3, ymm13, ymm15\n",
			"vbroadcastsd ymm15, [{b3} + {idx} + 8*(",
			$depth,
			")]\n",
			"vfmadd231pd ymm4, ymm12, ymm14\n",
			"vfmadd231pd ymm5, ymm13, ymm14\n",
			"vbroadcastsd ymm14, [{b4} + {idx} + 8*(",
			$depth,
			")]\n",
			"vfmadd231pd ymm6, ymm12, ymm15\n",
			"vfmadd231pd ymm7, ymm13, ymm15\n",
			"vbroadcastsd ymm15, [{b5} + {idx} + 8*(",
			$depth,
			")]\n",
			"vfmadd231pd ymm8, ymm12, ymm14\n",
			"vfmadd231pd ymm9, ymm13, ymm14\n",
			"vfmadd231pd ymm10, ymm12, ymm15\n",
			"vfmadd231pd ymm11, ymm13, ymm15\n",
		)
	};
}

/// The hot loop is shared by all three packing modes. Slab mode copies
/// one source row fragment per unrolled group, completing a panel in eight column tiles.
#[target_feature(enable = "avx2,fma")]
unsafe fn panels<const MODE: usize>(
	k: usize,
	packed: *const f64,
	b: *const f64,
	out: *mut f64,
	ldb: usize,
	ldc: usize,
	panels: usize,
	source: *const f64,
	lda: usize,
	destination: *mut f64,
	rank: Option<(*const f64, *const f64)>,
) {
	let bulk = k / UNROLL * UNROLL;
	let b = b.wrapping_add(bulk);
	for panel in 0..panels {
		let a = packed.add(panel * MR * k);
		let c = out.add(panel * MR);
		let v0: __m256d;
		let v1: __m256d;
		let v2: __m256d;
		let v3: __m256d;
		let v4: __m256d;
		let v5: __m256d;
		let v6: __m256d;
		let v7: __m256d;
		let v8: __m256d;
		let v9: __m256d;
		let v10: __m256d;
		let v11: __m256d;

		macro_rules! dot { ($step_copy:expr, $block_copy:expr, $tail_copy:expr; $($operands:tt)*) => {{
core::arch::asm!(
            "vxorpd ymm0, ymm0, ymm0",
            "vxorpd ymm1, ymm1, ymm1",
            "vxorpd ymm2, ymm2, ymm2",
            "vxorpd ymm3, ymm3, ymm3",
            "vxorpd ymm4, ymm4, ymm4",
            "vxorpd ymm5, ymm5, ymm5",
            "vxorpd ymm6, ymm6, ymm6",
            "vxorpd ymm7, ymm7, ymm7",
            "vxorpd ymm8, ymm8, ymm8",
            "vxorpd ymm9, ymm9, ymm9",
            "vxorpd ymm10, ymm10, ymm10",
            "vxorpd ymm11, ymm11, ymm11",
            ".p2align 5",
            "2:",
            ".irp q,0,1,2,3,4,5,6,7",
            accumulate_8x6!("\\q"),
            $step_copy,
            ".endr",
            $block_copy,
            "add {a}, 512",
            "add {idx}, 64",
            "jnz 2b",
            "test {tail}, {tail}",
            "jz 4f",
            "3:",
            accumulate_8x6!("0"),
            $tail_copy,
            "add {a},64",
            "add {idx},8",
            "dec {tail}",
            "jnz 3b",
            "4:",
            $($operands)*
            a=inout(reg) a=>_,idx=inout(reg) (-(bulk as isize)*8)=>_,tail=inout(reg) (k-bulk)=>_,
            b0=in(reg) b,
            b1=in(reg) b.wrapping_add(ldb),
            b2=in(reg) b.wrapping_add(2*ldb),
            b3=in(reg) b.wrapping_add(3*ldb),
            b4=in(reg) b.wrapping_add(4*ldb),
            b5=in(reg) b.wrapping_add(5*ldb),
            out("ymm0") v0, out("ymm1") v1, out("ymm2") v2, out("ymm3") v3, out("ymm4") v4, out("ymm5") v5, out("ymm6") v6, out("ymm7") v7, out("ymm8") v8, out("ymm9") v9, out("ymm10") v10, out("ymm11") v11,
            out("ymm12") _,out("ymm13") _,out("ymm14") _,out("ymm15") _,options(nostack),
        );
        }}; }

		macro_rules! copy_line {
			($offset:literal) => {
				concat!(
					"vmovupd ymm12,[{src}]\nvmovupd ymm13,[{src}+32]\n",
					"vmovupd [{dst}+",
					$offset,
					"],ymm12\nvmovupd [{dst}+",
					$offset,
					"+32],ymm13\n",
					"add {src},{stride}"
				)
			};
		}
		if MODE == COMPUTE {
			dot!("", "", "";);
		} else {
			let src = source.add(panel * MR);
			let dst = destination.add(panel * MR * k);
			if MODE == PACK_PANEL {
				dot!(copy_line!("64*\\q"), "add {dst},512", concat!(copy_line!("0"), "\nadd {dst},64");
                    src=inout(reg) src=>_, dst=inout(reg) dst=>_, stride=in(reg) (lda * 8),);
			} else {
				dot!("", concat!(copy_line!("0"), "\nadd {dst},64"), "";
                    src=inout(reg) src=>_, dst=inout(reg) dst=>_, stride=in(reg) (lda * 8),);
			}
		}
		let mut cols = [[v0, v1], [v2, v3], [v4, v5], [v6, v7], [v8, v9], [v10, v11]];
		if let Some((rank_a, rank_b)) = rank {
			let src = rank_a.add(panel * MR);
			let left = [_mm256_loadu_pd(src), _mm256_loadu_pd(src.add(4))];
			for (j, col) in cols.iter_mut().enumerate() {
				let right = _mm256_set1_pd(*rank_b.add(j * ldb));
				for (val, a) in col.iter_mut().zip(left) {
					let product = _mm256_fmadd_pd(a, right, _mm256_setzero_pd());
					*val = _mm256_add_pd(product, *val);
				}
			}
		}
		for (j, col) in cols.into_iter().enumerate() {
			let dst = c.add(j * ldc);
			_mm256_storeu_pd(dst, col[0]);
			_mm256_storeu_pd(dst.add(4), col[1]);
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn streamed_tiles_match_the_arithmetic_groups_and_preserve_padding() {
		if !std::is_x86_feature_detected!("avx2") || !std::is_x86_feature_detected!("fma") {
			return;
		}
		let sentinel = 73.0_f64;
		for (m, k, n) in [(67, 257, 55), (81, 511, 57), (65, 512, 58), (73, 513, 59), (47, 135, 24)] {
			let (lda, ldb, ldc) = (m + 3, k + 1, m + 5);
			let a: Vec<_> = (0..lda * k).map(|i| (f64::from((i * 19 % 127) as u32) - 62.) / 17.).collect();
			let b: Vec<_> = (0..ldb * n).map(|i| (f64::from((i * 23 % 103) as u32) - 51.) / 13.).collect();
			let mut c = vec![sentinel; 1 + ldc * n];
			let mut pa = vec![0.; 2 * MR * 4 * DEPTH_BLOCK];
			let mut pb = vec![0.; NR * DEPTH_BLOCK];
			let run = |c: &mut [f64], pa_len, pb_len, pa: &mut [f64], pb: &mut [f64]| unsafe {
				multiply(
					m,
					n,
					k,
					a.as_ptr(),
					lda,
					b.as_ptr(),
					ldb,
					c.as_mut_ptr().add(1),
					ldc,
					1.,
					Accum::Replace,
					pa.as_mut_ptr(),
					pa_len,
					pb.as_mut_ptr(),
					pb_len,
					6144,
					32768,
				)
			};
			let (alen, blen) = (pa.len(), pb.len());
			assert!(!run(&mut c, 0, blen, &mut pa, &mut pb));
			assert!(!run(&mut c, alen, 0, &mut pa, &mut pb));
			assert!(c.iter().all(|&x| x == sentinel));
			assert!(run(&mut c, alen, blen, &mut pa, &mut pb));
			assert_eq!(c[0], sentinel);
			for j in 0..n {
				for i in 0..m {
					let mut dot = 0.;
					for p in 0..k.min(DEPTH_BLOCK) {
						dot = a[i + p * lda].mul_add(b[p + j * ldb], dot);
					}
					if k > DEPTH_BLOCK {
						dot = a[i + DEPTH_BLOCK * lda].mul_add(b[DEPTH_BLOCK + j * ldb], 0.) + dot;
					}
					assert_eq!(c[1 + i + j * ldc].to_bits(), dot.to_bits(), "{m}x{k}x{n} at {i},{j}");
				}
				assert!(c[1 + m + j * ldc..1 + (j + 1) * ldc].iter().all(|&x| x == sentinel));
			}
		}
	}
}
