//! `gemm_with_scratch` must agree with the paths it does not change, and one borrowed
//! workspace must survive a sequence of different shapes.
//!
//! The reference is `gemm_with` on the same fork (the same kernel, job partition and packing,
//! only the storage of the job tables differs), so equality is asserted bit for bit; the serial
//! `gemm` is the independent reference, checked with the tolerances of `tests/fork.rs`.

#![cfg(all(target_arch = "x86_64", feature = "rayon"))]

use private_gemm_x86::{
	Accum, DType, DstKind, Fork, IType, InstrSet, MilliScratch, gemm, gemm_with, gemm_with_scratch,
};

/// Runs `f(0)`, `f(1)`, ... on the calling thread.
struct SeqFork(usize);

impl Fork for SeqFork {
	fn width(&self) -> usize {
		self.0
	}
	fn run(&self, f: &(dyn Fn(usize) + Sync)) {
		(0..self.0).for_each(f);
	}
}

/// Participant 0 is the caller, the others are scoped threads started per `run`.
struct ScopeFork(usize);

impl Fork for ScopeFork {
	fn width(&self) -> usize {
		self.0
	}
	fn run(&self, f: &(dyn Fn(usize) + Sync)) {
		std::thread::scope(|s| {
			for j in 1..self.0 {
				s.spawn(move || f(j));
			}
			f(0);
		});
	}
}

trait Scalar: Copy + Default + Send + Sync + PartialEq {
	fn from_f64(x: f64) -> Self;
	fn to_f64(self) -> f64;
	fn tol() -> f64;
}

macro_rules! scalar {
	($t:ty, $tol:expr) => {
		impl Scalar for $t {
			fn from_f64(x: f64) -> Self {
				x as $t
			}
			fn to_f64(self) -> f64 {
				self as f64
			}
			fn tol() -> f64 {
				$tol
			}
		}
	};
}
// complex values are carried as interleaved `f32` / `f64` pairs by the same scalar types
scalar!(f32, 2e-5);
scalar!(f64, 1e-12);

/// Deterministic pseudo-random values in (-1, 1).
fn fill<S: Scalar>(len: usize, seed: u64) -> Vec<S> {
	let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
	(0..len)
		.map(|_| {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			S::from_f64((x >> 11) as f64 / (1u64 << 52) as f64 - 1.0)
		})
		.collect()
}

#[derive(Clone, Copy, Debug)]
struct Case {
	m: usize,
	n: usize,
	k: usize,
	row_major_dst: bool,
}

enum Exec<'a> {
	Serial,
	Fork(&'a dyn Fork),
	/// A fork plus the workspace the job tables are borrowed from.
	Scratch(&'a dyn Fork, &'a mut [u8]),
}

/// Run one product with scalar type `S` carrying elements of `S::W` scalars; return dst.
unsafe fn product<S: Scalar>(dtype: DType, instr: InstrSet, c: Case, exec: Exec) -> Vec<S> {
	let w = if matches!(dtype, DType::C32 | DType::C64) { 2 } else { 1 };
	let Case { m, n, k, row_major_dst } = c;
	let a = fill::<S>(m * k * w, 1);
	let b = fill::<S>(k * n * w, 2);
	let alpha: Vec<S> = if w == 2 { vec![S::from_f64(0.75), S::from_f64(-0.25)] } else { vec![S::from_f64(0.75)] };
	// A non-zero destination would only matter for `Add`; kept so the bytes are not all zero.
	let mut dst = fill::<S>(m * n * w, 3);
	let (rs, cs) = if row_major_dst { (n as isize, 1) } else { (1, m as isize) };
	let (d, it, i, mm, nn, kk) = (dtype, IType::U64, instr, m, n, k);
	let (p, rs_, cs_) = (dst.as_mut_ptr() as *mut (), rs, cs);
	let (l, lrs, lcs) = (a.as_ptr() as *const (), 1isize, m as isize);
	let (r, rrs, rcs) = (b.as_ptr() as *const (), 1isize, k as isize);
	let al = alpha.as_ptr() as *const ();
	let null = std::ptr::null::<()>();
	unsafe {
		match exec {
			Exec::Serial => gemm(d, it, i, mm, nn, kk, p, rs_, cs_, null, null, DstKind::Full, Accum::Replace, l, lrs, lcs, false, null, 0, r, rrs, rcs, true, al, 1),
			Exec::Fork(f) => gemm_with(d, it, i, mm, nn, kk, p, rs_, cs_, null, null, DstKind::Full, Accum::Replace, l, lrs, lcs, false, null, 0, r, rrs, rcs, true, al, f),
			Exec::Scratch(f, mem) => gemm_with_scratch(
				d, it, i, mm, nn, kk, p, rs_, cs_, null, null, DstKind::Full, Accum::Replace, l, lrs, lcs, false, null, 0, r, rrs, rcs, true, al, f,
				MilliScratch::new(mem),
			),
		}
	}
	dst
}

fn instrs() -> Vec<InstrSet> {
	let mut v = vec![InstrSet::Avx256];
	if is_x86_feature_detected!("avx512f") {
		v.push(InstrSet::Avx512);
	}
	v
}

fn cases() -> Vec<Case> {
	// (m, n, k): odd edges, tall, wide, square, k > 512; both destination layouts (a row-major
	// destination makes the kernel swap the two dimensions, which must not confuse the tables).
	let mut v = Vec::new();
	for (m, n, k) in [(37, 41, 29), (301, 64, 40), (64, 301, 40), (200, 200, 600), (513, 129, 33)] {
		for row_major_dst in [false, true] {
			v.push(Case { m, n, k, row_major_dst });
		}
	}
	v
}

/// Bytes a workspace needs for the largest product of `cases`.
fn biggest_workspace(cases: &[Case]) -> usize {
	cases.iter().map(|c| MilliScratch::required(c.m, c.n)).max().unwrap()
}

/// One workspace, sized once for the largest product, reused for every dtype and shape of the
/// sweep (a smaller shape must reuse it, a larger one after it must still be correct): the
/// scratch path is bit-identical to `gemm_with` on the same fork, and matches the serial
/// reference within the dtype's epsilon.
#[test]
fn one_workspace_serves_a_sequence_of_shapes_and_dtypes() {
	let fork = SeqFork(3);
	let mut mem = vec![0u8; biggest_workspace(&cases())];
	for instr in instrs() {
		for c in cases() {
			check::<f64>(DType::F64, instr, c, &fork, &mut mem);
			check::<f32>(DType::F32, instr, c, &fork, &mut mem);
			check::<f64>(DType::C64, instr, c, &fork, &mut mem);
			check::<f32>(DType::C32, instr, c, &fork, &mut mem);
		}
	}
}

fn check<S: Scalar>(dtype: DType, instr: InstrSet, c: Case, fork: &dyn Fork, mem: &mut [u8]) {
	let want_fork = unsafe { product::<S>(dtype, instr, c, Exec::Fork(fork)) };
	let want = unsafe { product::<S>(dtype, instr, c, Exec::Serial) };
	let got = unsafe { product::<S>(dtype, instr, c, Exec::Scratch(fork, mem)) };
	assert!(got == want_fork, "{c:?} {dtype:?} {instr:?}: not the same as gemm_with");
	let scale = (c.k as f64).sqrt().max(1.0);
	for (i, (g, w)) in got.iter().zip(&want).enumerate() {
		let (g, w) = (g.to_f64(), w.to_f64());
		assert!((g - w).abs() <= S::tol() * scale * (1.0 + w.abs()), "{c:?} {dtype:?} {instr:?} at {i}: {g} vs {w}");
	}
}

/// The same workspace under a fork whose participants really run concurrently: the tables are
/// shared between threads there, which the sequential fork cannot exercise.
#[test]
fn concurrent_participants_share_the_tables() {
	let fork = ScopeFork(4);
	let mut mem = vec![0u8; biggest_workspace(&cases())];
	for instr in instrs() {
		for c in cases() {
			check_concurrent::<f64>(DType::F64, instr, c, &fork, &mut mem);
			check_concurrent::<f32>(DType::F32, instr, c, &fork, &mut mem);
			check_concurrent::<f64>(DType::C64, instr, c, &fork, &mut mem);
			check_concurrent::<f32>(DType::C32, instr, c, &fork, &mut mem);
		}
	}
}

fn check_concurrent<S: Scalar>(dtype: DType, instr: InstrSet, c: Case, fork: &dyn Fork, mem: &mut [u8]) {
	let got = unsafe { product::<S>(dtype, instr, c, Exec::Scratch(fork, mem)) };
	let want = unsafe { product::<S>(dtype, instr, c, Exec::Serial) };
	let scale = (c.k as f64).sqrt().max(1.0);
	for (i, (g, w)) in got.iter().zip(&want).enumerate() {
		let (g, w) = (g.to_f64(), w.to_f64());
		assert!((g - w).abs() <= S::tol() * scale * (1.0 + w.abs()), "{c:?} {dtype:?} {instr:?} at {i}: {g} vs {w}");
	}
}

/// A workspace too short for the product is not an error: the call falls back to allocating
/// the tables and the result is unchanged.
#[test]
fn short_workspace_falls_back() {
	let fork = SeqFork(4);
	let mut mem = [0u8; 0];
	for instr in instrs() {
		for c in cases() {
			let want = unsafe { product::<f64>(DType::F64, instr, c, Exec::Fork(&fork)) };
			let got = unsafe { product::<f64>(DType::F64, instr, c, Exec::Scratch(&fork, &mut mem)) };
			assert!(got == want, "{c:?} {instr:?}");
		}
	}
}

/// Width 1 runs inline without forking and without touching the tables.
#[test]
fn width_one_with_a_scratch_matches_serial() {
	let fork = SeqFork(1);
	let mut mem = vec![0u8; biggest_workspace(&cases())];
	for instr in instrs() {
		for c in cases() {
			let want = unsafe { product::<f64>(DType::F64, instr, c, Exec::Serial) };
			let got = unsafe { product::<f64>(DType::F64, instr, c, Exec::Scratch(&fork, &mut mem)) };
			assert!(got == want, "{c:?} {instr:?}");
		}
	}
}
