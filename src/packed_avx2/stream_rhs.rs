//! Column-major overwrite GEMM on Raptor Lake P-cores. The full 8 × 6 tiles
//! read B in place. While their twelve accumulators are live, spare load/store
//! bandwidth packs A for subsequent tiles.
use super::{Accum, DEPTH_BLOCK, MR, NR, add_rank_one, pack_lhs};

// Number of explicit depth steps in the full-tile loop below.
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
	if m < MR || k < UNROLL || k > DEPTH_BLOCK + 1 || alpha != 1.0 || beta != Accum::Replace {
		return false;
	}
	let depth = k.min(DEPTH_BLOCK);
	let pitch = MR * depth;
	if depth * NR > l1 * 2 / 3 || pitch > alen {
		return false;
	}
	// With just one or two residual rows a padded eight-row tile wastes
	// most of its FMAs. Pack those rows densely for the last row block.
	let fringe = m % MR;
	let tail = if m >= 2 * MR && fringe <= MR / 4 { fringe } else { 0 };
	let body = m - tail;
	if tail * depth > blen {
		return false;
	}
	if tail != 0 {
		for p in 0..depth {
			core::ptr::copy_nonoverlapping(a.add(body + p * lda), pb.add(p * tail), tail);
		}
	}
	let ways = crate::cache::CACHE_INFO[1].associativity;
	let ways_room = if ways >= 4 { l2 / ways } else { 0 };
	let block_rows = ((l2 / 2 - ways_room) / pitch).max(1).min(alen / pitch) * MR;
	// End an incomplete M tile with a full, overlapping overwrite tile.
	// This also avoids repacking B for every column tile when A fits in L2.
	let block_rows = block_rows.min(body / MR * MR);
	let two_buffers = body > block_rows && n / NR > UNROLL && block_rows <= alen / depth / 2;
	let full_cols = n / NR * NR;
	let stage_from = full_cols.saturating_sub(UNROLL * NR);
	let row_bounds = |start: usize| {
		let len = (body - start).min(block_rows);
		// Overwrite produces the same dot on overlapping rows. Let a short
		// final block end with a complete tile when it has a predecessor.
		let overlap = (MR - len % MR) % MR;
		(start - overlap, len + overlap)
	};
	for start in (0..body).step_by(block_rows) {
		let (i0, rows) = row_bounds(start);
		let slot = start / block_rows % 2;
		let (next_i, next_rows) = if two_buffers {
			row_bounds(start.saturating_add(block_rows).min(body))
		} else {
			(body, 0)
		};
		let current = if two_buffers { pa.add(slot * block_rows * depth) } else { pa };
		let next = if two_buffers { pa.add((1 - slot) * block_rows * depth) } else { pa };
		let seed = (start == 0 || !two_buffers) && rows >= 2 * MR && n >= NR;
		if seed {
			pack_lhs(a.add(i0), lda, current, MR, depth);
		} else if start == 0 || !two_buffers {
			pack_lhs(a.add(i0), lda, current, rows, depth);
		}
		if next_rows != 0 {
			for i in (0..next_rows).step_by(MR) {
				for p in (depth / UNROLL * UNROLL)..depth {
					core::ptr::copy_nonoverlapping(a.add(next_i + i + p * lda), next.add(i * depth + p * MR), MR);
				}
			}
		}
		let joint = tail != 0 && i0 + rows == body;
		let regular = rows - if joint { MR } else { 0 };
		if joint && seed {
			pack_lhs(a.add(body - MR), lda, current.add(regular * depth), MR, depth);
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
			if width == NR {
				let columns = core::array::from_fn(|j| right.add(j * ldb));
				let full = regular / MR;
				let staged = if next_rows >= MR && j >= stage_from {
					let count = next_rows / MR;
					let p = (j - stage_from) / NR * (depth / UNROLL);
					panels::<PACK_SLAB>(
						depth,
						current,
						&columns,
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
					panels::<PACK_PANEL>(
						depth,
						current,
						&columns,
						out,
						ldb,
						ldc,
						count,
						a.add(i0 + MR),
						lda,
						current.add(pitch),
						rank,
					);
					count
				} else {
					0
				};
				if staged < full {
					let row = staged * MR;
					panels::<COMPUTE>(
						depth,
						current.add(row * depth),
						&columns,
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
			} else {
				match width {
					1 => narrow_columns::<1, 6>(current, right, out, depth, ldb, ldc, regular),
					2 => narrow_columns::<2, 3>(current, right, out, depth, ldb, ldc, regular),
					3 => narrow_columns::<3, 2>(current, right, out, depth, ldb, ldc, regular),
					4 => narrow_columns::<4, 1>(current, right, out, depth, ldb, ldc, regular),
					5 => narrow_columns::<5, 1>(current, right, out, depth, ldb, ldc, regular),
					_ => unreachable!(),
				}
				if let Some((left, rhs)) = rank {
					add_rank_one(regular, width, left, rhs, ldb, out, ldc, alpha);
				}
			}
			if joint {
				let packed = current.add(regular * depth);
				let last = rank.map(|(_, b)| (a.add(body - MR + depth * lda), b));
				for at in (0..width).step_by(3) {
					let b = right.add(at * ldb);
					let c = out.add(regular + at * ldc);
					let end = last.map(|(a, b)| (a, b.add(at * ldb)));
					match (tail, (width - at).min(3)) {
						(1, 1) => joint_rows::<1, 1>(depth, packed, pb, b, ldb, c, ldc, end),
						(1, 2) => joint_rows::<1, 2>(depth, packed, pb, b, ldb, c, ldc, end),
						(1, 3) => joint_rows::<1, 3>(depth, packed, pb, b, ldb, c, ldc, end),
						(2, 1) => joint_rows::<2, 1>(depth, packed, pb, b, ldb, c, ldc, end),
						(2, 2) => joint_rows::<2, 2>(depth, packed, pb, b, ldb, c, ldc, end),
						(2, 3) => joint_rows::<2, 3>(depth, packed, pb, b, ldb, c, ldc, end),
						_ => unreachable!(),
					}
				}
			}
		}
	}
	true
}

#[target_feature(enable = "avx2,fma")]
unsafe fn joint_rows<const R: usize, const W: usize>(
	depth: usize,
	packed: *const f64,
	tail: *const f64,
	b: *const f64,
	ldb: usize,
	c: *mut f64,
	ldc: usize,
	rank: Option<(*const f64, *const f64)>,
) {
	let mut full = [[0.0; MR]; W];
	let mut extra = [[0.0; W]; R];
	for p in 0..depth {
		let right: [f64; W] = core::array::from_fn(|j| *b.add(p + j * ldb));
		super::step(&mut full, &*packed.add(p * MR).cast::<[f64; MR]>(), &right);
		for (r, sums) in extra.iter_mut().enumerate() {
			let l = *tail.add(p * R + r);
			for (sum, &v) in sums.iter_mut().zip(&right) {
				*sum = l.mul_add(v, *sum);
			}
		}
	}
	for (j, col) in full.iter().enumerate() {
		let dst = c.add(j * ldc);
		if let Some((left, right)) = rank {
			let v = *right.add(j * ldb);
			for (i, &sum) in col.iter().enumerate() {
				dst.add(i).write((*left.add(i)).mul_add(v, 0.0) + sum);
			}
			for (r, row) in extra.iter().enumerate() {
				dst.add(MR + r).write((*left.add(MR + r)).mul_add(v, 0.0) + row[j]);
			}
		} else {
			core::ptr::copy_nonoverlapping(col.as_ptr(), dst, MR);
			for (r, row) in extra.iter().enumerate() {
				dst.add(MR + r).write(row[j]);
			}
		}
	}
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
) {
	let group = MR * PANELS;
	let bulk = rows / group * group;
	narrow_groups::<COLS, PANELS>(a, b, c, depth, ldb, ldc, bulk);
	if bulk < rows {
		narrow_groups::<COLS, 1>(a.add(bulk * depth), b, c.add(bulk), depth, ldb, ldc, rows - bulk);
	}
}

#[target_feature(enable = "avx2,fma")]
unsafe fn narrow_groups<const COLS: usize, const PANELS: usize>(
	a: *const f64,
	b: *const f64,
	c: *mut f64,
	depth: usize,
	ldb: usize,
	ldc: usize,
	rows: usize,
) {
	let group = MR * PANELS;
	for i in (0..rows).step_by(group) {
		let mut sums = [[[0.0; MR]; COLS]; PANELS];
		for p in 0..depth {
			let right = core::array::from_fn(|j| *b.add(p + j * ldb));
			for (t, panel) in sums.iter_mut().enumerate() {
				let left = &*a.add((i + t * MR) * depth + p * MR).cast::<[f64; MR]>();
				super::step(panel, left, &right);
			}
		}
		for (t, panel) in sums.iter().enumerate() {
			for (j, col) in panel.iter().enumerate() {
				core::ptr::copy_nonoverlapping(col.as_ptr(), c.add(i + t * MR + j * ldc), MR);
			}
		}
	}
}

/// Each output follows the same increasing-depth FMA chain. Lanes in a
/// column are independent and can be vectorized together.
#[inline(always)]
fn accumulate<const COLS: usize>(sums: &mut [[f64; MR]; COLS], left: &[f64; MR], right: &[f64; COLS]) {
	for (col, &r) in sums.iter_mut().zip(right) {
		for (sum, &l) in col.iter_mut().zip(left) {
			*sum = l.mul_add(r, *sum);
		}
	}
}

/// The hot loop is shared by all three packing modes. Slab mode copies
/// one source row fragment per unrolled group, completing a panel in eight column tiles.
#[inline(never)]
#[target_feature(enable = "avx2,fma")]
unsafe fn panels<const MODE: usize>(
	k: usize,
	packed: *const f64,
	columns: &[*const f64; NR],
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
	let rank = rank.map(|(a, b)| {
		let values: [f64; NR] = core::array::from_fn(|j| *b.add(j * ldb));
		(a, values)
	});
	for panel in 0..panels {
		let a = packed.add(panel * MR * k);
		let c = out.add(panel * MR);
		let src = source.wrapping_add(panel * MR);
		let dst = destination.wrapping_add(panel * MR * k);
		let mut cols = [[0.0_f64; MR]; NR];
		macro_rules! step {
			($index:expr) => {{
				let p = $index;
				let left = &*a.add(p * MR).cast::<[f64; MR]>();
				let right = core::array::from_fn(|j| *columns[j].add(p));
				if MODE == COMPUTE {
					super::step(&mut cols, left, &right);
				} else {
					accumulate(&mut cols, left, &right);
				}
				if MODE == PACK_PANEL {
					core::ptr::copy_nonoverlapping(src.add(p * lda), dst.add(p * MR), MR);
				}
			}};
		}
		for p in (0..bulk).step_by(UNROLL) {
			// Keep one branch per packing group, including in fat-LTO builds.
			step!(p);
			step!(p + 1);
			step!(p + 2);
			step!(p + 3);
			if MODE == PACK_SLAB {
				let i = p / UNROLL;
				core::ptr::copy_nonoverlapping(src.add(i * lda), dst.add(i * MR), MR);
			}
			step!(p + 4);
			step!(p + 5);
			step!(p + 6);
			step!(p + 7);
		}
		for p in bulk..k {
			step!(p);
		}
		if let Some((rank_a, right)) = &rank {
			let left = &*rank_a.add(panel * MR).cast::<[f64; MR]>();
			for (j, (col, &r)) in cols.iter().zip(right).enumerate() {
				let updated: [f64; MR] = core::array::from_fn(|i| left[i].mul_add(r, 0.) + col[i]);
				core::ptr::copy_nonoverlapping(updated.as_ptr(), c.add(j * ldc), MR);
			}
		} else {
			for (j, col) in cols.iter().enumerate() {
				core::ptr::copy_nonoverlapping(col.as_ptr(), c.add(j * ldc), MR);
			}
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
		for (m, k, n) in [
			(9, 8, 1),
			(18, 65, 37),
			(66, 513, 61),
			(39, 63, 25),
			(63, 65, 61),
			(67, 257, 55),
			(81, 511, 57),
			(65, 512, 58),
			(73, 513, 59),
			(47, 135, 24),
		] {
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
			if m >= 2 * MR && (1..=MR / 4).contains(&(m % MR)) {
				assert!(!run(&mut c, alen, (m % MR) * k.min(DEPTH_BLOCK) - 1, &mut pa, &mut pb));
			}

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
