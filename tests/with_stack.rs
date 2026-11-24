//! `gemm_with_stack` and `gemm_stack_bytes` (#14): with the memory `gemm_stack_bytes` asks for,
//! a first call on a fresh thread allocates nothing, and the result is bit-for-bit that of
//! `gemm` with one thread.
//!
//! A counting global allocator counts the allocations of the thread that marks itself. The
//! process-wide one-time set-up (the cache sizes, the CPU feature checks) is done once on the
//! main thread first: it is not per thread, so it is not what this test is about. Each
//! instruction set the host lacks is skipped with a line on stdout. Run it with
//! `cargo test --release --test with_stack -- --nocapture --test-threads=1` (unoptimised, the
//! 96 MiB shapes take most of an hour).

#![cfg(target_arch = "x86_64")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use private_gemm_x86::{Accum, DType, DstKind, IType, InstrSet, gemm, gemm_stack_bytes, gemm_with_stack};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
	static COUNTED: Cell<bool> = const { Cell::new(false) };
}

fn counted() -> bool {
	COUNTED.try_with(|c| c.get()).unwrap_or(false)
}

unsafe impl GlobalAlloc for Counting {
	unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
		if counted() {
			ALLOCS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.alloc(layout) }
	}
	unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
		if counted() {
			ALLOCS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.alloc_zeroed(layout) }
	}
	unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
		if counted() {
			ALLOCS.fetch_add(1, Ordering::Relaxed);
		}
		unsafe { System.realloc(ptr, layout, new_size) }
	}
	unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
		unsafe { System.dealloc(ptr, layout) }
	}
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// The tests share one counter: one at a time.
static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The instruction sets this host runs, with a skip line for the others.
fn instr_sets() -> Vec<InstrSet> {
	let mut out = Vec::new();
	if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
		out.push(InstrSet::Avx256);
	} else {
		println!("skip: no AVX2/FMA on this host");
	}
	if is_x86_feature_detected!("avx512f") {
		out.push(InstrSet::Avx512);
	} else {
		println!("skip: no AVX-512F on this host");
	}
	out
}

const DTYPES: [DType; 4] = [DType::F32, DType::F64, DType::C32, DType::C64];

/// Bytes per element, and whether the element is a pair.
fn elem(dtype: DType) -> (usize, bool) {
	match dtype {
		DType::F32 => (4, false),
		DType::F64 => (8, false),
		DType::C32 => (8, true),
		DType::C64 => (16, true),
	}
}

/// A product: `dst` row-major or column-major (a row-major one makes the kernel transpose the
/// product), `lhs` row-major or column-major.
#[derive(Clone, Copy, Debug)]
struct Case {
	m: usize,
	n: usize,
	k: usize,
	dst_row_major: bool,
	lhs_row_major: bool,
	beta: Accum,
	conj_lhs: bool,
	conj_rhs: bool,
}

/// Raw storage of `len` elements of `dtype`, filled with small values.
struct Buf {
	words: Vec<u64>,
}

impl Buf {
	fn new(dtype: DType, len: usize, seed: u64) -> Self {
		let (size, _) = elem(dtype);
		let n_words = (len * size).div_ceil(8).max(1);
		let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
		let mut next = || {
			x ^= x << 13;
			x ^= x >> 7;
			x ^= x << 17;
			(x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
		};
		let words = (0..n_words)
			.map(|_| match dtype {
				DType::F64 | DType::C64 => next().to_bits(),
				DType::F32 | DType::C32 => (next() as f32).to_bits() as u64 | ((next() as f32).to_bits() as u64) << 32,
			})
			.collect();
		Self { words }
	}
	fn ptr(&self) -> *const () {
		self.words.as_ptr() as *const ()
	}
	fn ptr_mut(&mut self) -> *mut () {
		self.words.as_mut_ptr() as *mut ()
	}
}

struct Operands {
	dst: Buf,
	lhs: Buf,
	rhs: Buf,
	alpha: Buf,
}

impl Operands {
	fn new(dtype: DType, c: Case) -> Self {
		Self {
			dst: Buf::new(dtype, c.m * c.n, 3),
			lhs: Buf::new(dtype, c.m * c.k, 1),
			rhs: Buf::new(dtype, c.k * c.n, 2),
			alpha: Buf::new(dtype, 1, 4),
		}
	}
}

/// `dst := [dst +] alpha * lhs * rhs` with `gemm` on one thread (`stack` `None`) or with
/// `gemm_with_stack`.
unsafe fn product(dtype: DType, instr: InstrSet, c: Case, ops: &mut Operands, stack: Option<&mut [MaybeUninit<u8>]>) {
	let (m, n, k) = (c.m, c.n, c.k);
	let (drs, dcs) = if c.dst_row_major { (n as isize, 1) } else { (1, m as isize) };
	let (lrs, lcs) = if c.lhs_row_major { (k as isize, 1) } else { (1, m as isize) };
	let (rrs, rcs) = (1isize, k as isize);
	let null = std::ptr::null::<()>();
	let (_, cplx) = elem(dtype);
	let (conj_lhs, conj_rhs) = if cplx { (c.conj_lhs, c.conj_rhs) } else { (false, false) };
	let (d, l, r, a) = (ops.dst.ptr_mut(), ops.lhs.ptr(), ops.rhs.ptr(), ops.alpha.ptr());
	unsafe {
		match stack {
			None => gemm(
				dtype, IType::U64, instr, m, n, k, d, drs, dcs, null, null, DstKind::Full, c.beta, l, lrs, lcs, conj_lhs, null, 0, r, rrs, rcs,
				conj_rhs, a, 1,
			),
			Some(stack) => gemm_with_stack(
				dtype, IType::U64, instr, m, n, k, d, drs, dcs, null, null, DstKind::Full, c.beta, l, lrs, lcs, conj_lhs, null, 0, r, rrs, rcs,
				conj_rhs, a, stack,
			),
		}
	}
}

fn stack_for(dtype: DType, instr: InstrSet, c: Case) -> Vec<MaybeUninit<u8>> {
	let len = gemm_stack_bytes(dtype, instr, c.m, c.n, c.k);
	(0..len).map(|_| MaybeUninit::uninit()).collect()
}

/// Shapes below the packing memory (small, the column-major millikernel path, the packed
/// AVX2 driver's range) and above it: an `lhs` of about 96 MiB, more than the L3 the buffer
/// is sized from on any current x86-64 host.
fn shapes(dtype: DType) -> Vec<(usize, usize, usize)> {
	let (size, _) = elem(dtype);
	let k = 512;
	let tall = (96 << 20) / (k * size);
	vec![(1, 37, 19), (24, 32, 40), (64, 64, 64), (300, 200, 150), (512, 512, 512), (tall, 48, k)]
}

fn case(m: usize, n: usize, k: usize, dst_row_major: bool, lhs_row_major: bool) -> Case {
	Case {
		m,
		n,
		k,
		dst_row_major,
		lhs_row_major,
		beta: Accum::Replace,
		conj_lhs: false,
		conj_rhs: false,
	}
}

/// The process-wide set-up, once, before any thread is counted.
fn warm_up_process(instrs: &[InstrSet]) {
	for &instr in instrs {
		for dtype in DTYPES {
			let c = case(64, 64, 64, false, false);
			let mut ops = Operands::new(dtype, c);
			let mut stack = stack_for(dtype, instr, c);
			unsafe { product(dtype, instr, c, &mut ops, Some(&mut stack)) };
		}
	}
}

#[test]
fn first_call_on_a_fresh_thread_allocates_nothing() {
	let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
	let instrs = instr_sets();
	warm_up_process(&instrs);
	for &instr in &instrs {
		for dtype in DTYPES {
			for (m, n, k) in shapes(dtype) {
				for (dst_row_major, lhs_row_major) in [(false, false), (true, false), (false, true)] {
					let c = case(m, n, k, dst_row_major, lhs_row_major);
					let mut ops = Operands::new(dtype, c);
					let mut stack = stack_for(dtype, instr, c);
					let n_alloc = std::thread::scope(|s| {
						s.spawn(|| {
							let before = ALLOCS.load(Ordering::Relaxed);
							COUNTED.with(|f| f.set(true));
							unsafe { product(dtype, instr, c, &mut ops, Some(&mut stack)) };
							COUNTED.with(|f| f.set(false));
							ALLOCS.load(Ordering::Relaxed) - before
						})
						.join()
						.unwrap()
					});
					assert_eq!(n_alloc, 0, "{dtype:?} {instr:?} {c:?}");
				}
			}
		}
	}
}

/// `gemm_with_stack` gives the bits of `gemm` with one thread, for every shape above, both
/// orientations, both layouts of `lhs`, both accumulations and the conjugations.
#[test]
fn matches_gemm_bit_for_bit() {
	let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
	for instr in instr_sets() {
		for dtype in DTYPES {
			for (m, n, k) in shapes(dtype) {
				for (i, (dst_row_major, lhs_row_major)) in [(false, false), (true, false), (false, true), (true, true)].into_iter().enumerate() {
					let c = Case {
						beta: if i % 2 == 0 { Accum::Replace } else { Accum::Add },
						conj_lhs: i == 1 || i == 3,
						conj_rhs: i == 2 || i == 3,
						..case(m, n, k, dst_row_major, lhs_row_major)
					};
					let mut want = Operands::new(dtype, c);
					unsafe { product(dtype, instr, c, &mut want, None) };
					let mut got = Operands::new(dtype, c);
					let mut stack = stack_for(dtype, instr, c);
					unsafe { product(dtype, instr, c, &mut got, Some(&mut stack)) };
					assert!(want.dst.words == got.dst.words, "{dtype:?} {instr:?} {c:?}");
				}
			}
		}
	}
}

#[test]
#[should_panic(expected = "shorter than gemm_stack_bytes")]
fn a_short_stack_panics() {
	let instr = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
		InstrSet::Avx256
	} else {
		panic!("shorter than gemm_stack_bytes (no AVX2 on this host; nothing to run)");
	};
	let c = case(64, 64, 64, false, false);
	let mut ops = Operands::new(DType::F64, c);
	let mut stack = vec![MaybeUninit::uninit(); 64];
	unsafe { product(DType::F64, instr, c, &mut ops, Some(&mut stack)) };
}
