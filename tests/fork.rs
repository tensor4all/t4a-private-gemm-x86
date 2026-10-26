//! `gemm_with` on a caller-supplied [`Fork`] must agree with `gemm`.
//!
//! The reference is `gemm` on one thread (no fork at all); `gemm_with` runs on three forks: a
//! sequential one (valid because no participant waits for another), a `std::thread::scope` one,
//! and a recording one that proves the kernel stays on the participants it was handed and calls
//! `run` only from the calling thread. Both instruction sets are passed explicitly so an AVX-512
//! host still covers the 256-bit path.

#![cfg(all(target_arch = "x86_64", feature = "rayon"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread::ThreadId;

use private_gemm_x86::{gemm, gemm_with, Accum, DType, DstKind, Fork, IType, InstrSet};

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

/// Like `ScopeFork`, and records which thread ran each participant and how often `run` was called.
struct RecordFork {
    width: usize,
    caller: ThreadId,
    runs: AtomicUsize,
    seen: Mutex<Vec<(usize, ThreadId)>>,
    foreign_run: AtomicUsize,
}

impl Fork for RecordFork {
    fn width(&self) -> usize {
        self.width
    }
    fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        self.runs.fetch_add(1, Ordering::Relaxed);
        if std::thread::current().id() != self.caller {
            self.foreign_run.fetch_add(1, Ordering::Relaxed);
        }
        let record = |j: usize| {
            self.seen.lock().unwrap().push((j, std::thread::current().id()));
            f(j);
        };
        std::thread::scope(|s| {
            for j in 1..self.width {
                let record = &record;
                s.spawn(move || record(j));
            }
            record(0);
        });
    }
}

trait Scalar: Copy + Default + Send + Sync {
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
    kind: DstKind,
    beta: Accum,
    indexed: bool,
    row_major_dst: bool,
}

#[derive(Clone, Copy)]
enum Exec<'a> {
    Reference,
    Fork(&'a dyn Fork),
}

/// Run one product with scalar type `S` carrying elements of `S::W` scalars; return dst.
unsafe fn product<S: Scalar>(dtype: DType, instr: InstrSet, c: Case, exec: Exec) -> Vec<S> {
    let w = if matches!(dtype, DType::C32 | DType::C64) { 2 } else { 1 };
    let Case { m, n, k, kind, beta, indexed, row_major_dst } = c;
    let a = fill::<S>(m * k * w, 1);
    let b = fill::<S>(k * n * w, 2);
    let alpha: Vec<S> = if w == 2 { vec![S::from_f64(0.75), S::from_f64(-0.25)] } else { vec![S::from_f64(0.75)] };
    // The destination starts non-zero so Add and the untouched triangle are both checked.
    let mut dst = fill::<S>(m * n * w, 3);
    let (rs, cs) = if row_major_dst { (n as isize, 1) } else { (1, m as isize) };
    let row_idx: Vec<u64> = (0..m as u64).rev().collect();
    let col_idx: Vec<u64> = (0..n as u64).rev().collect();
    let (ri, ci) = if indexed { (row_idx.as_ptr() as *const (), col_idx.as_ptr() as *const ()) } else { (std::ptr::null(), std::ptr::null()) };
    let args = |dst: &mut Vec<S>| {
        (
            dtype,
            IType::U64,
            instr,
            m,
            n,
            k,
            dst.as_mut_ptr() as *mut (),
            rs,
            cs,
            ri,
            ci,
            kind,
            beta,
            a.as_ptr() as *const (),
            1isize,
            m as isize,
            false,
            std::ptr::null::<()>(),
            0isize,
            b.as_ptr() as *const (),
            1isize,
            k as isize,
            true,
            alpha.as_ptr() as *const (),
        )
    };
    let (d, it, i, mm, nn, kk, p, rs_, cs_, ri_, ci_, kd, bt, l, lrs, lcs, cl, rd, ds, r, rrs, rcs, cr, al) = args(&mut dst);
    unsafe {
        match exec {
            Exec::Reference => gemm(d, it, i, mm, nn, kk, p, rs_, cs_, ri_, ci_, kd, bt, l, lrs, lcs, cl, rd, ds, r, rrs, rcs, cr, al, 1),
            Exec::Fork(f) => gemm_with(d, it, i, mm, nn, kk, p, rs_, cs_, ri_, ci_, kd, bt, l, lrs, lcs, cl, rd, ds, r, rrs, rcs, cr, al, f),
        }
    }
    dst
}

fn check<S: Scalar>(dtype: DType, instr: InstrSet, c: Case, fork: &dyn Fork, what: &str) {
    let want = unsafe { product::<S>(dtype, instr, c, Exec::Reference) };
    let got = unsafe { product::<S>(dtype, instr, c, Exec::Fork(fork)) };
    let scale = (c.k as f64).sqrt().max(1.0);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        let (g, w) = (g.to_f64(), w.to_f64());
        assert!((g - w).abs() <= S::tol() * scale * (1.0 + w.abs()), "{what}: {c:?} {dtype:?} {instr:?} at {i}: {g} vs {w}");
    }
}

fn instrs() -> Vec<InstrSet> {
    let mut v = vec![InstrSet::Avx256];
    if is_x86_feature_detected!("avx512f") {
        v.push(InstrSet::Avx512);
    }
    v
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    // (m, n, k): small, tall, wide (packing stage skipped for wide), square, odd edges, k > 512.
    for (m, n, k) in [(1, 1, 1), (7, 5, 3), (37, 41, 29), (130, 70, 90), (301, 64, 40), (64, 301, 40), (200, 200, 600), (513, 129, 33)] {
        for kind in [DstKind::Full, DstKind::Lower, DstKind::Upper] {
            for beta in [Accum::Replace, Accum::Add] {
                for indexed in [false, true] {
                    // triangular + indexed is exercised on square-ish products only
                    if kind != DstKind::Full && !indexed && m != n && (m, n) != (130, 70) {
                        continue;
                    }
                    for row_major_dst in [false, true] {
                        v.push(Case { m, n, k, kind, beta, indexed, row_major_dst });
                    }
                }
            }
        }
    }
    v
}

fn sweep(fork: &dyn Fork, what: &str) {
    for instr in instrs() {
        for c in cases() {
            check::<f32>(DType::F32, instr, c, fork, what);
            check::<f64>(DType::F64, instr, c, fork, what);
            check::<f32>(DType::C32, instr, c, fork, what);
            check::<f64>(DType::C64, instr, c, fork, what);
        }
    }
}

#[test]
fn sequential_fork_matches_gemm() {
    for width in [2, 3, 8] {
        sweep(&SeqFork(width), "seq");
    }
}

#[test]
fn scoped_thread_fork_matches_gemm() {
    for width in [2, 4, 7] {
        sweep(&ScopeFork(width), "scope");
    }
}

#[test]
fn width_one_runs_inline_without_forking() {
    let fork = RecordFork { width: 1, caller: std::thread::current().id(), runs: AtomicUsize::new(0), seen: Mutex::new(Vec::new()), foreign_run: AtomicUsize::new(0) };
    let c = Case { m: 200, n: 200, k: 100, kind: DstKind::Full, beta: Accum::Replace, indexed: false, row_major_dst: false };
    check::<f64>(DType::F64, instrs()[0], c, &fork, "width1");
    assert_eq!(fork.runs.load(Ordering::Relaxed), 0);
}

#[test]
fn kernel_only_uses_the_participants_it_was_given() {
    let me = std::thread::current().id();
    let fork = RecordFork { width: 4, caller: me, runs: AtomicUsize::new(0), seen: Mutex::new(Vec::new()), foreign_run: AtomicUsize::new(0) };
    let c = Case { m: 300, n: 300, k: 200, kind: DstKind::Full, beta: Accum::Replace, indexed: false, row_major_dst: false };
    unsafe { product::<f64>(DType::F64, instrs()[0], c, Exec::Fork(&fork)) };
    assert!(fork.runs.load(Ordering::Relaxed) >= 1, "the kernel never forked");
    assert_eq!(fork.foreign_run.load(Ordering::Relaxed), 0, "run called from a thread other than the caller");
    let seen = fork.seen.lock().unwrap();
    // every participant index ran exactly once per `run`, always on the caller for j = 0
    let runs = fork.runs.load(Ordering::Relaxed);
    for j in 0..4 {
        assert_eq!(seen.iter().filter(|(i, _)| *i == j).count(), runs, "participant {j}");
    }
    assert!(seen.iter().filter(|(i, _)| *i == 0).all(|(_, t)| *t == me));
    // and nothing ran on a thread that is not a participant thread of a `run` (spindle's pool)
    assert!(seen.iter().all(|(j, _)| *j < 4));
}

/// A product tall enough that the right-hand side is packed (the first parallel stage, split
/// over the depth) as well as computed (the job queue): two forks per product.
#[test]
fn tall_product_forks_for_packing_and_compute() {
    let me = std::thread::current().id();
    for instr in instrs() {
        let fork = RecordFork { width: 3, caller: me, runs: AtomicUsize::new(0), seen: Mutex::new(Vec::new()), foreign_run: AtomicUsize::new(0) };
        let c = Case { m: 200_000, n: 9, k: 16, kind: DstKind::Full, beta: Accum::Replace, indexed: false, row_major_dst: false };
        check::<f64>(DType::F64, instr, c, &fork, "tall");
        assert!(fork.runs.load(Ordering::Relaxed) >= 2, "{instr:?}: packing and compute stages should both fork");
        assert_eq!(fork.foreign_run.load(Ordering::Relaxed), 0);
    }
}

/// A fork need not be `Sync` (a team token is exclusive authority held by one thread).
struct CellFork {
    width: usize,
    runs: std::cell::Cell<usize>,
}

impl Fork for CellFork {
    fn width(&self) -> usize {
        self.width
    }
    fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        self.runs.set(self.runs.get() + 1);
        (0..self.width).for_each(f);
    }
}

#[test]
fn a_fork_that_is_not_sync_works() {
    let fork = CellFork { width: 3, runs: std::cell::Cell::new(0) };
    let c = Case { m: 150, n: 90, k: 70, kind: DstKind::Lower, beta: Accum::Add, indexed: false, row_major_dst: false };
    check::<f64>(DType::F64, instrs()[0], c, &fork, "cell");
    assert!(fork.runs.get() >= 1);
}
