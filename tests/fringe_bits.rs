//! Bit identity of fringe handling (#27, checks #26's claim). Needs `--features bench-internals`.
//!
//! For every driver the tree offers (always the packed driver; the streamed driver when present):
//!  1. each output equals the ordered scalar `mul_add` chain, bit for bit;
//!  2. the product of an `m`-row A equals the first `m` rows of the product of the same A padded
//!     with extra rows to a multiple of eight, so fringe rows round exactly as full-tile rows.
#![cfg(feature = "bench-internals")]
use private_gemm_x86::bench_internals as k;
use private_gemm_x86::Accum;

fn rng(seed: &mut u64) -> f64 {
	*seed ^= *seed << 13;
	*seed ^= *seed >> 7;
	*seed ^= *seed << 17;
	(*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

fn reference(m: usize, n: usize, kk: usize, a: &[f64], lda: usize, b: &[f64]) -> Vec<f64> {
	let mut c = vec![0.0; m * n];
	for j in 0..n {
		for i in 0..m {
			let main = kk.min(512);
			let mut dot = 0.0f64;
			for p in 0..main {
				dot = a[i + p * lda].mul_add(b[p + j * kk], dot);
			}
			if kk == 513 {
				dot = a[i + 512 * lda].mul_add(b[512 + j * kk], 0.0) + dot;
			}
			c[i + j * m] = dot;
		}
	}
	c
}

#[test]
fn fringe_rows_round_like_full_tiles() {
	if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
		return;
	}
	let (l1, l2, l3) = k::cache_elems();
	let mut pa = vec![0.0f64; l3.next_multiple_of(16)];
	let mut pb = vec![0.0f64; l3.next_multiple_of(k::TILE_NR)];
	let mut drivers: Vec<(&str, k::Driver)> = vec![("packed", k::packed_driver())];
	if let Some(d) = k::stream_driver() {
		drivers.push(("stream", d));
	}
	let mut seed = 0xC0FFEEu64;
	let (mut checked, mut refused) = (0usize, 0usize);
	for (name, d) in &drivers {
		for &kk in &[8usize, 9, 64, 139, 257, 512, 513] {
			for &n in &[1usize, 5, 6, 7, 13, 24, 25] {
				for m in 8usize..=41 {
					let mp = m.next_multiple_of(8);
					let a: Vec<f64> = (0..mp * kk).map(|_| rng(&mut seed)).collect();
					let b: Vec<f64> = (0..kk * n).map(|_| rng(&mut seed)).collect();
					let mut c = vec![f64::NAN; m * n];
					let mut cp = vec![f64::NAN; mp * n];
					// the m-row view of the padded A: same leading dimension, fewer rows
					let ok = unsafe { d(m, n, kk, a.as_ptr(), mp, b.as_ptr(), kk, c.as_mut_ptr(), m, 1.0, Accum::Replace, pa.as_mut_ptr(), pa.len(), pb.as_mut_ptr(), pb.len(), l1, l2) };
					let okp = unsafe { d(mp, n, kk, a.as_ptr(), mp, b.as_ptr(), kk, cp.as_mut_ptr(), mp, 1.0, Accum::Replace, pa.as_mut_ptr(), pa.len(), pb.as_mut_ptr(), pb.len(), l1, l2) };
					if !ok || !okp {
						refused += 1;
						continue;
					}
					checked += 1;
					let want = reference(m, n, kk, &a, mp, &b);
					for j in 0..n {
						for i in 0..m {
							let got = c[i + j * m].to_bits();
							assert_eq!(got, want[i + j * m].to_bits(), "{name}: scalar mismatch m={m} n={n} k={kk} ({i},{j})");
							assert_eq!(got, cp[i + j * mp].to_bits(), "{name}: padded-A mismatch m={m} n={n} k={kk} ({i},{j})");
						}
					}
				}
			}
		}
		eprintln!("{name}: checked {checked} products, driver refused {refused}");
	}
	assert!(checked > 0);
}
