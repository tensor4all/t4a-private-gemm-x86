//! Benchmark-only wrappers around the private AVX2 f64 kernels (feature `bench-internals`).
//! Each wrapper forwards unchanged; no kernel code lives here. The same module compiles
//! in trees with and without the streamed driver so one `benches/microkernel.rs` serves both.
use super::*;

pub const TILE_MR: usize = MR;
pub const TILE_NR: usize = NR;
pub const F_ADD: usize = ADD;
pub const F_RESUME: usize = RESUME;
pub const F_FINISH: usize = FINISH;

pub type Driver = unsafe fn(
	usize, usize, usize, *const f64, usize, *const f64, usize, *mut f64, usize,
	f64, Accum, *mut f64, usize, *mut f64, usize, usize, usize,
) -> bool;

/// Packed 8 x 6 tile. `ldc_bytes` is the byte stride of the output columns.
///
/// # Safety
/// Same contract as the private `tile`; the CPU must have AVX2 and FMA.
pub unsafe fn tile_packed(a: *const f64, b: *const f64, c: *mut f64, ldc_bytes: usize, depth: usize, alpha: f64, partial: *mut f64, flags: usize) {
	tile(a, b, c, ldc_bytes, depth, alpha, partial, flags)
}

/// A partial tile as the driver computes it: the packed tile into an edge buffer, then a
/// copy of the valid `height x width` corner to `out`.
///
/// # Safety
/// `out` must be writable for the valid corner; the CPU must have AVX2 and FMA.
pub unsafe fn tile_edge(a: *const f64, b: *const f64, out: *mut f64, ldc: usize, height: usize, width: usize, depth: usize, add: bool) {
	let mut edge = [0.; MR * NR];
	let flags = usize::from(add) * ADD | FINISH;
	if add {
		for col in 0..width {
			core::ptr::copy_nonoverlapping(out.add(col * ldc), edge.as_mut_ptr().add(col * MR), height);
		}
	}
	tile(a, b, edge.as_mut_ptr(), 8 * MR, depth, 1.0, core::ptr::null_mut(), flags);
	for col in 0..width {
		core::ptr::copy_nonoverlapping(edge.as_ptr().add(col * MR), out.add(col * ldc), height);
	}
}

/// # Safety
/// As the private `pack_lhs`.
pub unsafe fn pack_lhs_f64(a: *const f64, lda: usize, pa: *mut f64, rows: usize, depth: usize) {
	pack_lhs(a, lda, pa, rows, depth)
}
/// # Safety
/// As the private `pack_rhs`.
pub unsafe fn pack_rhs_f64(b: *const f64, ldb: usize, pb: *mut f64, cols: usize, depth: usize) {
	pack_rhs(b, ldb, pb, cols, depth)
}
/// # Safety
/// As the private `pack_rhs_t`.
pub unsafe fn pack_rhs_t_f64(b: *const f64, ldb: usize, pb: *mut f64, cols: usize, depth: usize) {
	pack_rhs_t(b, ldb, pb, cols, depth)
}

/// The packed-panel driver (`multiply`), called without the CPU-model gate.
pub fn packed_driver() -> Driver {
	multiply
}

/// The streamed driver of the tree, if it has one, called without the CPU-model gate.
pub fn stream_driver() -> Option<Driver> {
	STREAM_DRIVER
}

/// Full streamed 8 x 6 tile loop over `panels` complete packed A panels, B read in place:
/// `(depth, packed_a, b, ldb, out, ldc, panels)`.
pub type StreamTile = unsafe fn(usize, *const f64, *const f64, usize, *mut f64, usize, usize);

pub fn stream_tile() -> Option<StreamTile> {
	STREAM_TILE
}

const STREAM_DRIVER: Option<Driver> = None;
const STREAM_TILE: Option<StreamTile> = None;

/// Detected `(L1d, L2, L3)` capacities in f64 elements, as `gemm` passes them to the driver.
pub fn cache_elems() -> (usize, usize, usize) {
	let c = *crate::cache::CACHE_INFO;
	(c[0].cache_bytes / 8, c[1].cache_bytes / 8, c[2].cache_bytes / 8)
}
