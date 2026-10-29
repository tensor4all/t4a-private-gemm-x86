//! A product with a diagonal, `dst = alpha * A * diag(D) * B`, must pair depth index `t` with
//! `D[t]` whatever the signs of the depth strides of `A`, `B` and `D`.
//!
//! When the kernel's left operand has a negative depth stride, `gemm` walks the depth backwards
//! by reversing the left operand's columns and the right operand's rows. Before the fix it did
//! not reverse the diagonal with them, so `t` met `D[depth - 1 - t]`. The kernel's left operand
//! is `B` transposed when the destination is row-major (the operands are swapped first), so a
//! negative stride on either operand reaches that branch. Each case is compared with the same
//! product where the diagonal is folded into `A` beforehand and no diagonal is passed.

#![cfg(target_arch = "x86_64")]

use private_gemm_x86::{Accum, DType, DstKind, IType, InstrSet, gemm};

trait Scalar: Copy {
	const DTYPE: DType;
	/// Relative tolerance against the largest entry of the reference.
	const TOL: f64;
	fn new(re: f64, im: f64) -> Self;
	/// `self * d` for a real `d`.
	fn scale(self, d: f64) -> Self;
	fn re(self) -> f64;
	fn dist(self, other: Self) -> f64;
	fn abs(self) -> f64;
}

impl Scalar for f32 {
	const DTYPE: DType = DType::F32;
	const TOL: f64 = 1e-4;
	fn new(re: f64, _: f64) -> Self {
		re as f32
	}
	fn scale(self, d: f64) -> Self {
		self * d as f32
	}
	fn re(self) -> f64 {
		self as f64
	}
	fn dist(self, other: Self) -> f64 {
		(self - other).abs() as f64
	}
	fn abs(self) -> f64 {
		self.abs() as f64
	}
}

impl Scalar for f64 {
	const DTYPE: DType = DType::F64;
	const TOL: f64 = 1e-12;
	fn new(re: f64, _: f64) -> Self {
		re
	}
	fn scale(self, d: f64) -> Self {
		self * d
	}
	fn re(self) -> f64 {
		self
	}
	fn dist(self, other: Self) -> f64 {
		(self - other).abs()
	}
	fn abs(self) -> f64 {
		self.abs()
	}
}

impl Scalar for [f32; 2] {
	const DTYPE: DType = DType::C32;
	const TOL: f64 = 1e-4;
	fn new(re: f64, im: f64) -> Self {
		[re as f32, im as f32]
	}
	fn scale(self, d: f64) -> Self {
		[self[0] * d as f32, self[1] * d as f32]
	}
	fn re(self) -> f64 {
		self[0] as f64
	}
	fn dist(self, other: Self) -> f64 {
		(self[0] - other[0]).hypot(self[1] - other[1]) as f64
	}
	fn abs(self) -> f64 {
		self[0].hypot(self[1]) as f64
	}
}

impl Scalar for [f64; 2] {
	const DTYPE: DType = DType::C64;
	const TOL: f64 = 1e-12;
	fn new(re: f64, im: f64) -> Self {
		[re, im]
	}
	fn scale(self, d: f64) -> Self {
		[self[0] * d, self[1] * d]
	}
	fn re(self) -> f64 {
		self[0]
	}
	fn dist(self, other: Self) -> f64 {
		(self[0] - other[0]).hypot(self[1] - other[1])
	}
	fn abs(self) -> f64 {
		self[0].hypot(self[1])
	}
}

/// Deterministic values in `[-1, 1)`, distinct enough that a reversed diagonal shows up.
fn value(seed: usize) -> f64 {
	((seed.wrapping_mul(2654435761) >> 7) % 257) as f64 / 128.5 - 1.0
}

/// A strided view over a column-major buffer: the pointer to logical entry `(0, 0)` and the
/// row and column strides, with the depth axis reversed in storage when `reversed`.
struct View<T> {
	buf: Vec<T>,
	offset: usize,
	rs: isize,
	cs: isize,
}

impl<T: Scalar> View<T> {
	/// `rows x cols`, logical entry `(i, j)` taken from `f(i, j)`. `depth_is_cols` says which
	/// axis is the depth axis (columns of `A`, rows of `B`).
	fn new(rows: usize, cols: usize, depth_is_cols: bool, reversed: bool, f: impl Fn(usize, usize) -> T) -> Self {
		let mut buf = vec![T::new(0.0, 0.0); rows * cols];
		for j in 0..cols {
			for i in 0..rows {
				let (si, sj) = match (depth_is_cols, reversed) {
					(true, true) => (i, cols - 1 - j),
					(false, true) => (rows - 1 - i, j),
					_ => (i, j),
				};
				buf[si + sj * rows] = f(i, j);
			}
		}
		let (rs, cs, offset) = match (depth_is_cols, reversed) {
			(true, true) => (1, -(rows as isize), (cols - 1) * rows),
			(false, true) => (-1, rows as isize, rows - 1),
			_ => (1, rows as isize, 0),
		};
		Self { buf, offset, rs, cs }
	}

	fn ptr(&self) -> *const () {
		self.buf[self.offset..].as_ptr() as *const ()
	}
}

/// `dst = alpha * A * diag(D) * B` through `gemm`, `dst` column-major or row-major.
unsafe fn product<T: Scalar>(
	instr: InstrSet,
	m: usize,
	n: usize,
	k: usize,
	a: &View<T>,
	diag: Option<&View<T>>,
	b: &View<T>,
	row_major_dst: bool,
) -> Vec<T> {
	let alpha = T::new(0.75, -0.25);
	let mut dst = vec![T::new(0.0, 0.0); m * n];
	let (dst_rs, dst_cs) = if row_major_dst { (n as isize, 1) } else { (1, m as isize) };
	let (diag_ptr, diag_stride) = match diag {
		Some(d) => (d.ptr(), d.rs),
		None => (core::ptr::null(), 0),
	};
	gemm(
		T::DTYPE,
		IType::U64,
		instr,
		m,
		n,
		k,
		dst.as_mut_ptr() as *mut (),
		dst_rs,
		dst_cs,
		core::ptr::null(),
		core::ptr::null(),
		DstKind::Full,
		Accum::Replace,
		a.ptr(),
		a.rs,
		a.cs,
		false,
		diag_ptr,
		diag_stride,
		b.ptr(),
		b.rs,
		b.cs,
		false,
		&alpha as *const T as *const (),
		1,
	);
	dst
}

/// Every sign combination of the depth strides of `A`, `B` and `D`, both destination layouts.
/// Returns the failing cases.
fn sweep<T: Scalar>(instr: InstrSet) -> Vec<String> {
	let mut failures = Vec::new();
	// k > 512 runs more than one depth block, so the diagonal pointer also advances between
	// blocks; n >= 65 is where faer#113 showed up.
	for (m, n, k) in [(1, 1, 1), (7, 5, 3), (13, 70, 66), (70, 13, 67), (37, 9, 600)] {
		let a_val = |i: usize, t: usize| T::new(value(3 * (i + m * t)), value(3 * (i + m * t) + 1));
		let b_val = |t: usize, j: usize| T::new(value(5 * (t + k * j) + 2), value(5 * (t + k * j) + 3));
		// Positive and well away from zero, so a reversed diagonal changes the result.
		let d_val = |t: usize| 0.5 + (t % 11) as f64 / 4.0 + value(7 * t + 4).abs() / 8.0;

		for row_major_dst in [false, true] {
			for rev_a in [false, true] {
				for rev_b in [false, true] {
					for rev_d in [false, true] {
						let a = View::new(m, k, true, rev_a, a_val);
						let b = View::new(k, n, false, rev_b, b_val);
						// The diagonal is a `k x 1` column: its row stride is its stride.
						// Complex dtypes read the real part of each entry.
						let d = View::new(k, 1, false, rev_d, |t, _| T::new(d_val(t), 0.0));
						let folded = View::new(m, k, true, rev_a, |i, t| a_val(i, t).scale(d_val(t)));

						let got = unsafe { product(instr, m, n, k, &a, Some(&d), &b, row_major_dst) };
						let want = unsafe { product(instr, m, n, k, &folded, None, &b, row_major_dst) };

						let scale = want.iter().map(|x| x.abs()).fold(1.0, f64::max);
						// A NaN distance is kept (`f64::max` would drop it), so it fails the check below.
						let err = got
							.iter()
							.zip(&want)
							.map(|(g, w)| g.dist(*w))
							.fold(0.0, |e, d| if d.is_nan() || d > e { d } else { e });
						if !(err <= T::TOL * scale) {
							failures.push(format!(
                                "{:?} {:?} m={m} n={n} k={k} row_major_dst={row_major_dst} rev_a={rev_a} rev_b={rev_b} rev_d={rev_d}: err={err:e} (scale {scale:e}, first {} vs {})",
                                instr, T::DTYPE, got[0].re(), want[0].re(),
                            ));
						}
					}
				}
			}
		}
	}
	failures
}

fn check(instr: InstrSet) {
	let mut failures = sweep::<f32>(instr);
	failures.extend(sweep::<f64>(instr));
	failures.extend(sweep::<[f32; 2]>(instr));
	failures.extend(sweep::<[f64; 2]>(instr));
	assert!(failures.is_empty(), "{} failing cases:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn avx256_diag_follows_reversed_depth() {
	if !std::arch::is_x86_feature_detected!("avx2") || !std::arch::is_x86_feature_detected!("fma") {
		eprintln!("avx2/fma unavailable; the 256-bit path cannot run here");
		return;
	}
	check(InstrSet::Avx256);
}

#[test]
fn avx512_diag_follows_reversed_depth() {
	if !std::arch::is_x86_feature_detected!("avx512f") {
		eprintln!("avx512f unavailable; the 512-bit path cannot run here");
		return;
	}
	check(InstrSet::Avx512);
}
