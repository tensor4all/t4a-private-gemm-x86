use private_gemm_x86::{Accum, DType, DstKind, IType, InstrSet, gemm};
use std::ptr::null;

// Dyadic entries make every dot exact. Opposite row and column exponents
// span 900 powers of two and make cross-row contamination visible.
#[test]
fn classical_product_preserves_scaled_entries_and_padded_odd_edges() {
	if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
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
			unsafe {
				gemm(
					DType::F64,
					IType::U64,
					InstrSet::Avx256,
					m,
					n,
					k,
					c.as_mut_ptr().cast(),
					1,
					ldc as isize,
					null(),
					null(),
					DstKind::Full,
					mode,
					a.as_ptr().cast(),
					1,
					lda as isize,
					false,
					null(),
					0,
					b.as_ptr().cast(),
					1,
					ldb as isize,
					false,
					(&alpha as *const f64).cast(),
					1,
				);
			}
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

#[test]
fn arithmetic_groups_match_the_general_kernel_bit_for_bit() {
	if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
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
			unsafe {
				gemm(
					DType::F64,
					IType::U64,
					InstrSet::Avx256,
					m,
					n,
					k,
					c.as_mut_ptr().cast(),
					1,
					m as isize,
					null(),
					null(),
					DstKind::Full,
					mode,
					a.as_ptr().cast(),
					1,
					m as isize,
					false,
					null(),
					0,
					b.as_ptr().cast(),
					1,
					k as isize,
					false,
					(&alpha as *const f64).cast(),
					1,
				);
			}
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
