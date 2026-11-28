//! Microkernel, packing, driver and end-to-end benchmark for the AVX2 f64 GEMM tiles (#27).
//!
//! Run (one pinned core, nothing else on its SMT sibling):
//!
//! ```text
//! BENCH_COMMIT=$(git rev-parse --short HEAD) \
//!   taskset -c 2 cargo bench --features bench-internals --bench microkernel
//! ```
//!
//! Method: every row is a median over `SAMPLES` samples; one sample times `iters` back-to-back
//! calls, with `iters` calibrated so a sample lasts at least `MIN_SAMPLE`. Pack buffers, inputs
//! and drivers' scratch are allocated and touched before timing; only the named call is timed.
//!
//! Peak: `FLOPS_PER_CYCLE = 2 FMA units x 4 doubles x 2 flops = 16` per core (documented AVX2
//! FMA throughput of Haswell..Raptor Lake P-core and Zen 4/5). The denominator uses the clock
//! in `BENCH_GHZ` when set; otherwise the clock measured here by a dependent-add chain
//! (one add per cycle). The clock actually used is printed. No fraction is printed without one.
use private_gemm_x86::bench_internals as k;
use private_gemm_x86::{Accum, DType, DstKind, IType, InstrSet};
use std::hint::black_box;
use std::time::{Duration, Instant};

const SAMPLES: usize = 31;
const MIN_SAMPLE: Duration = Duration::from_millis(2);
const FLOPS_PER_CYCLE: f64 = 16.0;

struct Ctx {
	ghz: f64,
	rows: Vec<String>,
}

/// Median ns per call of `f`.
fn time<F: FnMut()>(mut f: F) -> f64 {
	// warm-up: fill caches, reach steady frequency
	let t0 = Instant::now();
	while t0.elapsed() < Duration::from_millis(100) {
		f();
	}
	let mut iters = 1usize;
	loop {
		let t = Instant::now();
		for _ in 0..iters {
			f();
		}
		if t.elapsed() >= MIN_SAMPLE {
			break;
		}
		iters *= 2;
	}
	let mut v: Vec<f64> = (0..SAMPLES)
		.map(|_| {
			let t = Instant::now();
			for _ in 0..iters {
				f();
			}
			t.elapsed().as_secs_f64() * 1e9 / iters as f64
		})
		.collect();
	v.sort_by(|a, b| a.partial_cmp(b).unwrap());
	v[SAMPLES / 2]
}

impl Ctx {
	/// A row whose ISA peak is the AVX2 peak (`%avx2 == %isa`).
	fn row(&mut self, group: &str, name: &str, shape: &str, ns: f64, flops: f64) {
		self.row_isa(group, name, shape, ns, flops, FLOPS_PER_CYCLE);
	}
	/// `isa_fpc` is the flop/cycle the row's instruction set can issue: 16 for AVX2, 32 for AVX-512
	/// (2 x 512-bit FMA). `%avx2` is always against 16 flop/cycle; `%isa` against `isa_fpc`.
	fn row_isa(&mut self, group: &str, name: &str, shape: &str, ns: f64, flops: f64, isa_fpc: f64) {
		let gf = flops / ns;
		let pct = gf / (self.ghz * FLOPS_PER_CYCLE) * 100.0;
		let isa = gf / (self.ghz * isa_fpc) * 100.0;
		let line = format!("{group:<8} {name:<34} {shape:<16} {ns:>12.1} {gf:>9.2} {pct:>7.1}% {isa:>7.1}%");
		println!("{line}");
		self.rows.push(line);
	}
	fn bytes(&mut self, group: &str, name: &str, shape: &str, ns: f64, bytes: f64) {
		println!("{group:<8} {name:<34} {shape:<16} {ns:>12.1} {:>9.2} GB/s", bytes / ns);
	}
}

fn rng(seed: &mut u64) -> f64 {
	*seed ^= *seed << 13;
	*seed ^= *seed >> 7;
	*seed ^= *seed << 17;
	(*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

fn vec(n: usize, seed: &mut u64) -> Vec<f64> {
	(0..n).map(|_| rng(seed)).collect()
}

/// Cycles-per-second of this core from a dependent integer-add chain (latency 1).
fn calibrate_ghz() -> f64 {
	fn chain(n: u64) -> u64 {
		let mut x = 0u64;
		for _ in 0..n / 8 {
			unsafe {
				core::arch::asm!(
					"add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
					"add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
					x = inout(reg) x, options(nomem, nostack)
				);
			}
		}
		x
	}
	let n = 400_000_000u64;
	black_box(chain(n / 10));
	let mut best = f64::MAX;
	for _ in 0..5 {
		let t = Instant::now();
		black_box(chain(black_box(n)));
		best = best.min(t.elapsed().as_secs_f64());
	}
	n as f64 / best / 1e9
}

fn cpu_name() -> String {
	raw_cpuid::CpuId::new()
		.get_processor_brand_string()
		.map(|b| b.as_str().trim().to_string())
		.unwrap_or_else(|| "unknown".into())
}

fn sh(cmd: &str) -> String {
	std::process::Command::new("sh")
		.args(["-c", cmd])
		.output()
		.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
		.unwrap_or_default()
}

extern "C" {
	fn sched_getcpu() -> i32;
}

fn main() {
	if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
		println!("SKIP: no AVX2+FMA");
		return;
	}
	// `cargo bench` passes `--bench`; a bare filter argument selects groups by prefix.
	let filter: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with('-')).collect();
	let want = |g: &str| filter.is_empty() || filter.iter().any(|f| g.starts_with(f.as_str()));

	// Verify the pin instead of trusting it: exactly one allowed CPU, and record what the kernel
	// says about that CPU (L3 size and sharing group distinguish Zen 5 from Zen 5c).
	let allowed = sh("grep Cpus_allowed_list /proc/self/status | cut -f2");
	let cpu: usize = match allowed.parse() {
		Ok(c) => c,
		Err(_) => {
			eprintln!("REFUSING TO RUN: affinity is '{allowed}', not a single CPU. Use taskset -c <cpu>.");
			std::process::exit(2);
		}
	};
	let sys = |f: &str| sh(&format!("cat /sys/devices/system/cpu/cpu{cpu}/{f} 2>/dev/null"));
	let core_line = format!(
		"cpu{cpu} (smt sibling {}), L3 {} shared by {}, L2 {}",
		sys("topology/thread_siblings_list"), sys("cache/index3/size"), sys("cache/index3/shared_cpu_list"), sys("cache/index2/size")
	);
	let measured = calibrate_ghz();
	let ghz = std::env::var("BENCH_GHZ").ok().and_then(|s| s.parse().ok()).unwrap_or(measured);
	let (l1, l2, l3) = k::cache_elems();
	println!("host       : {}", sh("hostname"));
	println!("cpu        : {}", cpu_name());
	println!("affinity   : {allowed} (verified single CPU)");
	println!("core       : {core_line}");
	println!("ran on cpu : {} (sched_getcpu at start)", unsafe { sched_getcpu() });
	println!("commit     : {}", std::env::var("BENCH_COMMIT").unwrap_or_else(|_| "unrecorded".into()));
	println!("boost      : {} (1 = turbo/boost on)", sh("cat /sys/devices/system/cpu/cpufreq/boost /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null | tr '\\n' ' '"));
	println!("cache f64s : L1d {l1}  L2 {l2}  L3 {l3}");
	println!("clock      : measured {measured:.3} GHz (dependent-add chain), used for peak {ghz:.3} GHz");
	println!("peak/core  : %avx2 = {FLOPS_PER_CYCLE} flop/cycle x {ghz:.3} GHz = {:.2} GFLOP/s; %isa = {} flop/cycle for AVX-512 rows = {:.2} GFLOP/s (AVX2 rows: same as %avx2)", FLOPS_PER_CYCLE * ghz, 2.0 * FLOPS_PER_CYCLE, 2.0 * FLOPS_PER_CYCLE * ghz);
	println!("method     : median of {SAMPLES} samples, each >= {MIN_SAMPLE:?} of back-to-back calls, 100 ms warm-up");
	println!("stream drv : {}   stream tile: {}", k::stream_driver().is_some(), k::stream_tile().is_some());
	println!();
	println!("{:<8} {:<34} {:<16} {:>12} {:>9} {:>8} {:>8}", "group", "kernel", "shape", "ns/call", "GFLOP/s", "%avx2", "%isa");
	let mut c = Ctx { ghz, rows: vec![] };
	let mut seed = 0x9E3779B97F4A7C15u64;
	let (mr, nr) = (k::TILE_MR, k::TILE_NR);

	if want("tile") {
		// Packed tile, L1-resident: A is k*8, B is k*6, all packed.
		for depth in [16usize, 64, 192] {
			let a = vec(depth * mr, &mut seed);
			let b = vec(depth * nr, &mut seed);
			let mut out = vec![0.0; mr * nr];
			let mut partial = vec![0.0; mr * nr];
			let flops = (2 * mr * nr * depth) as f64;
			let ns = time(|| unsafe {
				k::tile_packed(black_box(a.as_ptr()), black_box(b.as_ptr()), out.as_mut_ptr(), 8 * mr, depth, 1.0, partial.as_mut_ptr(), k::F_FINISH);
				black_box(&mut out);
			});
			c.row("tile", "packed FINISH(replace)", &format!("k={depth}"), ns, flops);
		}
		let depth = 192usize;
		let a = vec(depth * mr, &mut seed);
		let b = vec(depth * nr, &mut seed);
		let mut out = vec![0.5; mr * nr];
		let mut partial = vec![0.25; mr * nr];
		let flops = (2 * mr * nr * depth) as f64;
		for (name, flags) in [
			("packed FINISH|ADD", k::F_FINISH | k::F_ADD),
			("packed RESUME|FINISH|ADD", k::F_RESUME | k::F_FINISH | k::F_ADD),
			("packed (no FINISH: spill sums)", 0),
		] {
			let ns = time(|| unsafe {
				k::tile_packed(black_box(a.as_ptr()), black_box(b.as_ptr()), out.as_mut_ptr(), 8 * mr, depth, 1.0, partial.as_mut_ptr(), flags);
				black_box(&mut out);
			});
			c.row("tile", name, &format!("k={depth}"), ns, flops);
		}
		// Streamed tile (only in trees that have it): A packed, B column-major in place.
		if let Some(st) = k::stream_tile() {
			for (depth, panels) in [(64usize, 1usize), (192, 1), (64, 4)] {
				let a = vec(depth * mr * panels, &mut seed);
				let ldb = depth;
				let b = vec(depth * nr, &mut seed);
				let mut out = vec![0.0; mr * panels * nr];
				let flops = (2 * mr * nr * depth * panels) as f64;
				let ns = time(|| unsafe {
					st(depth, black_box(a.as_ptr()), black_box(b.as_ptr()), ldb, out.as_mut_ptr(), mr * panels, panels);
					black_box(&mut out);
				});
				c.row("tile", "stream full tile (B in place)", &format!("k={depth} x{panels}panel"), ns, flops);
			}
		}
	}

	if want("tile512") && k::avx512_available() {
		// AVX-512 f64 asm microkernels as `gemm_impl` calls them: 48 x 4 (nrows > 48) and 24 x 8 (nrows <= 48).
		// Operands column-major in place, one tile per call. Peak for %isa: 32 flop/cycle.
		let isa = 2.0 * FLOPS_PER_CYCLE;
		for (wide, mr5, nr5) in [(false, 48usize, 4usize), (true, 24, 8)] {
			let label = if wide { "avx512 24x8" } else { "avx512 48x4" };
			for depth in [16usize, 64, 192] {
				let a = vec(depth * mr5, &mut seed);
				let b = vec(depth * nr5, &mut seed);
				let mut out = vec![0.0; mr5 * nr5];
				let flops = (2 * mr5 * nr5 * depth) as f64;
				let ns = time(|| unsafe {
					k::avx512_tile(wide, mr5, nr5, depth, black_box(a.as_ptr()), black_box(b.as_ptr()), out.as_mut_ptr(), false);
					black_box(&mut out);
				});
				c.row_isa("tile512", &format!("{label} replace"), &format!("k={depth}"), ns, flops, isa);
			}
			let depth = 192usize;
			let a = vec(depth * mr5, &mut seed);
			let b = vec(depth * nr5, &mut seed);
			let mut out = vec![0.5; mr5 * nr5];
			let flops = (2 * mr5 * nr5 * depth) as f64;
			let ns = time(|| unsafe {
				k::avx512_tile(wide, mr5, nr5, depth, black_box(a.as_ptr()), black_box(b.as_ptr()), out.as_mut_ptr(), true);
				black_box(&mut out);
			});
			c.row_isa("tile512", &format!("{label} accumulate"), &format!("k={depth}"), ns, flops, isa);
			// Fringe: useful flops only, same depth, against the full tile above.
			let mut out = vec![0.0; mr5 * nr5];
			let fr: &[(usize, usize)] = if wide { &[(24, 8), (23, 8), (16, 8), (8, 8), (1, 8), (24, 4), (24, 1), (8, 2), (1, 1)] } else { &[(48, 4), (47, 4), (32, 4), (16, 4), (8, 4), (1, 4), (48, 3), (48, 1), (8, 2), (1, 1)] };
			for &(h, w) in fr {
				let flops = (2 * h * w * depth) as f64;
				// Check the call contract before timing: valid corner equals the scalar sum, rest untouched.
				out.iter_mut().for_each(|x| *x = f64::NAN);
				unsafe { k::avx512_tile(wide, h, w, depth, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), false) };
				for j in 0..w {
					for i in 0..h {
						let r: f64 = (0..depth).map(|p| a[i + mr5 * p] * b[p + depth * j]).sum();
						assert!((out[i + mr5 * j] - r).abs() <= 1e-9 * r.abs().max(1.0), "avx512 tile {label} {h}x{w}: wrong value at ({i},{j})");
					}
				}
				let ns = time(|| unsafe {
					k::avx512_tile(wide, h, w, depth, black_box(a.as_ptr()), black_box(b.as_ptr()), out.as_mut_ptr(), false);
					black_box(&mut out);
				});
				c.row_isa("tile512", &format!("{label} fringe"), &format!("{h}x{w} k={depth}"), ns, flops, isa);
			}
		}
	}

	if want("edge") {
		// Fringe cost against a full tile at the same depth (packed tile into an edge buffer + copy).
		let depth = 192usize;
		let a = vec(depth * mr, &mut seed);
		let b = vec(depth * nr, &mut seed);
		let mut out = vec![0.0; mr * nr];
		for (h, w) in [(8usize, 6usize), (7, 6), (4, 6), (2, 6), (1, 6), (8, 3), (8, 1), (2, 3), (1, 1)] {
			let flops = (2 * h * w * depth) as f64; // useful flops only
			let ns = time(|| unsafe {
				k::tile_edge(black_box(a.as_ptr()), black_box(b.as_ptr()), out.as_mut_ptr(), mr, h, w, depth, false);
				black_box(&mut out);
			});
			c.row("edge", "packed tile + edge copy", &format!("{h}x{w} k={depth}"), ns, flops);
		}
	}

	if want("pack") {
		// Typical panels: mc x kc left block, kc x nc right block (Haswell-like blocking) and a fringe row count.
		let kc = 192usize;
		let lda = 1032usize;
		let src = vec(lda * (kc + 260), &mut seed); // pack_rhs_t strides j by ldb: needs (cols + kc) rows
		let mut dst = vec![0.0; 256 * kc.max(240)];
		for rows in [80usize, 191, 256] {
			let ns = time(|| unsafe {
				k::pack_lhs_f64(black_box(src.as_ptr()), lda, dst.as_mut_ptr(), rows, kc);
				black_box(&mut dst);
			});
			c.bytes("pack", "pack_lhs", &format!("{rows}x{kc}"), ns, (rows * kc * 16) as f64);
		}
		for cols in [48usize, 240, 258] {
			let ns = time(|| unsafe {
				k::pack_rhs_f64(black_box(src.as_ptr()), lda, dst.as_mut_ptr(), cols, kc);
				black_box(&mut dst);
			});
			c.bytes("pack", "pack_rhs", &format!("{kc}x{cols}"), ns, (cols * kc * 16) as f64);
			let ns = time(|| unsafe {
				k::pack_rhs_t_f64(black_box(src.as_ptr()), lda, dst.as_mut_ptr(), cols, kc);
				black_box(&mut dst);
			});
			c.bytes("pack", "pack_rhs_t", &format!("{kc}x{cols}"), ns, (cols * kc * 16) as f64);
		}
	}

	// Driver-only: the whole driver (pack + traversal + tiles) with scratch preallocated, no public dispatch.
	// End-to-end: the public `gemm` (dispatch included), default instruction set and forced AVX2.
	let shapes: &[(usize, usize, usize)] = &[(191, 257, 139), (192, 257, 139), (256, 256, 256), (511, 511, 511), (512, 512, 512), (256, 32, 256), (256, 512, 256), (256, 513, 256)];
	let pa_len = l3.next_multiple_of(16);
	let pb_len = l3.next_multiple_of(nr);
	let mut pa = vec![0.0f64; pa_len];
	let mut pb = vec![0.0f64; pb_len];
	for &(m, kk, n) in shapes {
		let a = vec(m * kk, &mut seed);
		let b = vec(kk * n, &mut seed);
		let mut out = vec![0.0; m * n];
		let flops = (2 * m * n * kk) as f64;
		let shape = format!("{m}x{kk}x{n}");
		if want("driver") {
			let drivers: Vec<(&str, k::Driver)> = std::iter::once(("packed driver (multiply)", k::packed_driver()))
				.chain(k::stream_driver().map(|d| ("stream driver (stream_rhs)", d)))
				.collect();
			for (name, d) in drivers {
				// Refusal is a result too: report it instead of timing a no-op.
				let ok = unsafe { d(m, n, kk, a.as_ptr(), m, b.as_ptr(), kk, out.as_mut_ptr(), m, 1.0, Accum::Replace, pa.as_mut_ptr(), pa_len, pb.as_mut_ptr(), pb_len, l1, l2) };
				if !ok {
					println!("{:<8} {name:<34} {shape:<16} refused", "driver");
					continue;
				}
				let ns = time(|| unsafe {
					black_box(d(m, n, kk, black_box(a.as_ptr()), m, black_box(b.as_ptr()), kk, out.as_mut_ptr(), m, 1.0, Accum::Replace, pa.as_mut_ptr(), pa_len, pb.as_mut_ptr(), pb_len, l1, l2));
					black_box(&mut out);
				});
				c.row("driver", name, &shape, ns, flops);
			}
		}
		if want("e2e") {
			let mut instrs = vec![("gemm Avx256 (forced)", InstrSet::Avx256)];
			if is_x86_feature_detected!("avx512f") {
				instrs.insert(0, ("gemm Avx512 (host default)", InstrSet::Avx512));
			}
			for (name, instr) in instrs {
				let alpha = 1.0f64;
				let ns = time(|| unsafe {
					private_gemm_x86::gemm(
						DType::F64, IType::U64, instr, m, n, kk,
						out.as_mut_ptr().cast(), 1, m as isize, core::ptr::null(), core::ptr::null(), DstKind::Full,
						Accum::Replace,
						black_box(a.as_ptr()).cast(), 1, m as isize, false,
						core::ptr::null(), 0,
						black_box(b.as_ptr()).cast(), 1, kk as isize, false,
						(&alpha as *const f64).cast(), 1,
					);
					black_box(&mut out);
				});
				// peak for Avx512 would double; report against the AVX2 denominator and say so.
				c.row_isa("e2e", name, &shape, ns, flops, if matches!(instr, InstrSet::Avx512) { 2.0 * FLOPS_PER_CYCLE } else { FLOPS_PER_CYCLE });
			}
		}
	}
	if want("scratch") {
		// Which scratch does each driver write? Buffers are calloc'd and untouched, so RSS growth
		// counts only pages the call writes. `pa` holds packed A, `pb` packed B (or the dense tail rows).
		println!();
		println!("scratch  driver                             shape            pa_written_KiB pb_written_KiB  rss_delta_KiB");
		for &(m, kk, n) in &[(191usize, 257usize, 139usize), (512, 512, 512)] {
			let a = vec(m * kk, &mut seed);
			let b = vec(kk * n, &mut seed);
			let drivers: Vec<(&str, k::Driver)> = std::iter::once(("packed driver (multiply)", k::packed_driver()))
				.chain(k::stream_driver().map(|d| ("stream driver (stream_rhs)", d)))
				.collect();
			for (name, d) in drivers {
				let mut out = vec![f64::NAN; m * n];
				let mut spa = vec![0.0f64; pa_len];
				let mut spb = vec![0.0f64; pb_len];
				let rss = || -> i64 { sh("awk '/^VmRSS/{print $2}' /proc/self/status").parse().unwrap_or(-1) };
				let r0 = rss();
				let ok = unsafe { d(m, n, kk, a.as_ptr(), m, b.as_ptr(), kk, out.as_mut_ptr(), m, 1.0, Accum::Replace, spa.as_mut_ptr(), pa_len, spb.as_mut_ptr(), pb_len, l1, l2) };
				let r1 = rss();
				let w = |v: &[f64]| v.iter().filter(|x| **x != 0.0).count() * 8 / 1024;
				println!("scratch  {name:<34} {:<16} {:>14} {:>14} {:>14}{}", format!("{m}x{kk}x{n}"), w(&spa), w(&spb), r1 - r0, if ok { "" } else { "  (refused)" });
			}
		}
	}
	println!();
	println!("%avx2 = GFLOP/s / ({FLOPS_PER_CYCLE} flop/cycle x {ghz:.3} GHz), the AVX2 FMA peak, for every row. %isa = the same against the row's own ISA peak ({} flop/cycle for AVX-512 rows); equal to %avx2 on AVX2 rows.", 2.0 * FLOPS_PER_CYCLE);
}
