//! A warm parallel product from a caller-owned workspace allocates nothing.
//!
//! The counter is a global allocator in this test binary, which holds exactly one `#[test]`
//! (no other test thread allocates), and the fork runs its participants on the calling thread
//! one after another (no thread is spawned and spindle's pool is not touched), so what the
//! counter sees is the kernel's own allocation. Run it on its own:
//! `cargo test --test job_alloc -- --nocapture --test-threads=1`.

#![cfg(all(target_arch = "x86_64", feature = "rayon"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use private_gemm_x86::{Accum, DType, DstKind, Fork, IType, InstrSet, MilliScratch, gemm_with, gemm_with_scratch};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		ALLOCS.fetch_add(1, Ordering::Relaxed);
		unsafe { System.alloc(layout) }
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		ALLOCS.fetch_add(1, Ordering::Relaxed);
		unsafe { System.alloc_zeroed(layout) }
	}
	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		ALLOCS.fetch_add(1, Ordering::Relaxed);
		unsafe { System.realloc(ptr, layout, new_size) }
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn allocations() -> usize {
	ALLOCS.load(Ordering::Relaxed)
}

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

#[derive(Clone, Copy, Debug)]
struct Case {
	m: usize,
	n: usize,
	k: usize,
	/// A row-major destination makes the kernel swap the two dimensions.
	row_major: bool,
}

/// One `dst := alpha * lhs * rhs` with `beta = Replace`, on `fork` and, when given, with the
/// job tables borrowed from `scratch`.
#[allow(clippy::too_many_arguments)]
unsafe fn product(c: Case, fork: &dyn Fork, scratch: Option<&mut [u8]>, dst: &mut [f64], lhs: &[f64], rhs: &[f64], alpha: &[f64]) {
	let (m, n, k) = (c.m, c.n, c.k);
	let (rs, cs) = if c.row_major { (n as isize, 1) } else { (1, m as isize) };
	let null = std::ptr::null::<()>();
	let (d, it, instr) = (DType::F64, IType::U64, InstrSet::Avx256);
	let (p, l, r) = (dst.as_mut_ptr() as *mut (), lhs.as_ptr() as *const (), rhs.as_ptr() as *const ());
	let (lrs, lcs, rrs, rcs) = (1isize, m as isize, 1isize, k as isize);
	let alpha = alpha.as_ptr() as *const ();
	unsafe {
		match scratch {
			Some(mem) => gemm_with_scratch(
				d, it, instr, m, n, k, p, rs, cs, null, null, DstKind::Full, Accum::Replace, l, lrs, lcs, false, null, 0, r, rrs, rcs, true, alpha,
				fork, MilliScratch::new(mem),
			),
			None => gemm_with(d, it, instr, m, n, k, p, rs, cs, null, null, DstKind::Full, Accum::Replace, l, lrs, lcs, false, null, 0, r, rrs, rcs, true, alpha, fork),
		}
	}
}

fn values(len: usize, seed: u64) -> Vec<f64> {
	let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
	(0..len)
		.map(|_| {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			(x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
		})
		.collect()
}

#[test]
fn warm_scratch_allocates_nothing() {
	let fork = SeqFork(4);
	for c in [
		// square, column-major: the ordinary shape
		Case { m: 200, n: 200, k: 64, row_major: false },
		// one row, row-major: the kernel swaps the dimensions, so the row and column tables
		// need more entries than the caller's `m` and `n` suggest
		Case { m: 1, n: 301, k: 17, row_major: true },
	] {
		let (m, n, k) = (c.m, c.n, c.k);
		let mut lhs = values(m * k, 1);
		let mut rhs = values(k * n, 2);
		let mut dst = values(m * n, 3);
		let alpha = [0.75];
		let mut mem = vec![0u8; MilliScratch::required(m, n)];
		let mut run = |scratch: Option<&mut [u8]>| unsafe {
			let (l, r, a) = (&mut lhs, &mut rhs, &alpha);
			product(c, &fork, scratch, &mut dst, l, r, a)
		};

		// warm: first use grows the thread-local packing buffer and any other one-time state
		run(Some(&mut mem));
		run(Some(&mut mem));
		let before = allocations();
		run(Some(&mut mem));
		let with_scratch = allocations() - before;

		// the same product without a workspace: what a call allocates when the caller has
		// nothing to lend it
		run(None);
		let before = allocations();
		run(None);
		let fallback = allocations() - before;

		// and one participant: the parallel stages are never built
		let single = SeqFork(1);
		let mut run_single = |scratch: Option<&mut [u8]>| unsafe {
			let (l, r, a) = (&mut lhs, &mut rhs, &alpha);
			product(c, &single, scratch, &mut dst, l, r, a)
		};
		run_single(Some(&mut mem));
		let before = allocations();
		run_single(Some(&mut mem));
		let width_one = allocations() - before;

		println!("{c:?}: scratch {with_scratch}, fallback {fallback}, width 1 {width_one} allocations");
		assert_eq!(with_scratch, 0, "{c:?}: a warm call with a workspace must not allocate");
		assert!(fallback >= 1, "{c:?}: the fallback is expected to allocate the job tables");
		assert_eq!(width_one, 0, "{c:?}: width 1 must not allocate the job tables");
	}
}
