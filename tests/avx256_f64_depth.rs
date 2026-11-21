use private_gemm_x86::{F64_SIMD256, MicrokernelInfo, Position, call_microkernel};
use std::ptr::null;

#[test]
fn packed_full_tile_preserves_depth_order_and_remainders() {
	if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
		return;
	}
	let mut seed = 13199u64;
	let mut sample = || {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		(seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
	};
	for k in (1..=9).chain([511, 512, 513]) {
		let mut a: Vec<_> = (0..12 * k).map(|_| sample()).collect();
		let mut b: Vec<_> = (0..4 * k).map(|_| sample()).collect();
		for add in [false, true] {
			let alpha = 1.0_f64;
			let mut c = [if add { 1.5 } else { f64::NAN }; 48];
			let info = MicrokernelInfo {
				flags: usize::from(add),
				depth: k,
				lhs_rs: 8,
				lhs_cs: 96,
				rhs_rs: 32,
				rhs_cs: 8,
				alpha: (&alpha as *const f64).cast(),
				ptr: c.as_mut_ptr().cast(),
				rs: 8,
				cs: 96,
				row_idx: null(),
				col_idx: null(),
				diag_ptr: null(),
				diag_stride: 0,
			};
			// Full packed 12-by-4 tile. All strides in MicrokernelInfo are bytes.
			unsafe {
				call_microkernel(
					F64_SIMD256[3],
					null(),
					a.as_mut_ptr().cast(),
					null(),
					b.as_mut_ptr().cast(),
					12,
					4,
					&info,
					&mut Position { row: 0, col: 0 },
				);
			}
			for j in 0..4 {
				for i in 0..12 {
					let dot = (0..k).fold(0.0, |acc, p| a[12 * p + i].mul_add(b[4 * p + j], acc));
					let want = if add { dot + 1.5 } else { dot };
					assert_eq!(c[i + 12 * j].to_bits(), want.to_bits(), "k={k}, row={i}, col={j}, add={add}");
				}
			}
		}
	}
}
