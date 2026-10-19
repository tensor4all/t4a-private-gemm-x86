//! Minimal reproducer: `DstKind::Upper` with the 256-bit kernel writes outside the
//! upper triangle.
//!
//! c64, m = n = 3, k = 2, beta = Replace, no row/col index, no threads. The `(2, 0)`
//! entry is below the diagonal, so `DstKind::Upper` must leave it alone. The AVX-512
//! path leaves it at its initial value; the AVX-256 path overwrites it with the value
//! that belongs to `(0, 1)`.
//!
//!   $ cargo run --example triangular_dst_avx256
//!   Avx512: (2,0) preserved = true
//!   Avx256: (2,0) preserved = false
//!
//! Found against private-gemm-x86 0.1.20 and still present in 0.1.22.

use private_gemm_x86::{gemm, Accum, DType, DstKind, IType, InstrSet};

type C64 = [f64; 2];

fn c(re: f64, im: f64) -> C64 { [re, im] }

unsafe fn run(instr: InstrSet, label: &str) {
    let (m, n, k) = (3usize, 3usize, 2usize);
    let a: Vec<C64> = (0..m).flat_map(|i| (0..k).map(move |kk| c(0.5 + i as f64, 0.25 + kk as f64))).collect();
    let b: Vec<C64> = (0..k).flat_map(|kk| (0..n).map(move |j| c(1.0 + j as f64, -0.5 - kk as f64))).collect();
    let d: Vec<f64> = (0..k).map(|kk| 1.25 + kk as f64).collect();
    let alpha = c(0.75, -0.25);

    let mut c_mat = vec![c(7.0, 2.0); m * n];   // sentinel, column-major
    gemm(
        DType::C64, IType::U64, instr,
        m, n, k,
        c_mat.as_mut_ptr() as *mut (), 1, m as isize,
        core::ptr::null(), core::ptr::null(),
        DstKind::Upper,
        Accum::Replace,
        a.as_ptr() as *const (), 1, m as isize, true,   // conj_lhs
        d.as_ptr() as *const (), 1isize,
        b.as_ptr() as *const (), 1, k as isize, false,
        &alpha as *const C64 as *const (),
        1,
    );
    for i in 0..m {
        println!("{label}: row{i} = [{:?}, {:?}, {:?}]", c_mat[i], c_mat[i + m], c_mat[i + 2 * m]);
    }
    let untouched = c_mat[2] == c(7.0, 2.0);   // (row 2, col 0) in column-major
    println!("   {label}: (2,0) preserved = {untouched}  (value {:?})\n", c_mat[2]);
}

fn main() {
    unsafe {
        run(InstrSet::Avx512, "Avx512");
        run(InstrSet::Avx256, "Avx256");
    }
}
