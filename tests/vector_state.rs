//! The kernels must return to Rust with a clean upper vector state (#31).
//!
//! The generated code uses `ymm`/`zmm` registers and never executes `vzeroupper`; the clear is
//! done at the Rust call boundary, in the one `call_kernel!` macro in `src/lib.rs`. Two checks
//! keep it there:
//!
//! * runtime: after every public entry that runs a kernel, for every dtype and every instruction
//!   set the host has, `XGETBV(1)` (`XINUSE`) must have `YMM_Hi128` (bit 2) and `ZMM_Hi256`
//!   (bit 6) clear, on the calling thread and on every participant of a `gemm_with` fork, read
//!   at the end of the participant's job (where a consumer's next scalar code would run);
//! * source: no indirect call into assembly may appear in `src/` outside that macro.
//!
//! `call_microkernel`, `pack_lhs` and the millikernels are reached through `gemm*`, and every
//! one of them calls assembly only through the macro, which the source check enforces.
//!
//! Bit 7 (`Hi16_ZMM`) is not cleared by `vzeroupper` and is out of scope.
//!
//! A host without AVX-512 prints `SKIP avx512` for that path; set
//! `VECTOR_STATE_REQUIRE_AVX512=1` to make that a failure instead. A CPU without
//! `XGETBV` with `ECX = 1` prints `SKIP xgetbv(1)` and the runtime tests return.
//! Run with `cargo test --release --test vector_state -- --nocapture`.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::__cpuid_count;
use std::mem::MaybeUninit;
use std::sync::Mutex;

use private_gemm_x86::{
	Accum, DType, DstKind, Fork, IType, InstrSet, MilliScratch, gemm, gemm_stack_bytes, gemm_with, gemm_with_scratch, gemm_with_stack,
	syrk_lower_block,
};

const YMM_HI128: u64 = 1 << 2;
const ZMM_HI256: u64 = 1 << 6;
const UPPER: u64 = YMM_HI128 | ZMM_HI256;

/// `CPUID.(EAX=0DH,ECX=1):EAX[2]`: `XGETBV` with `ECX = 1` returns `XINUSE`.
fn xinuse_supported() -> bool {
	unsafe { __cpuid_count(0xD, 1).eax & (1 << 2) != 0 }
}

fn xinuse() -> u64 {
	let (lo, hi): (u32, u32);
	unsafe { std::arch::asm!("xgetbv", in("ecx") 1u32, out("eax") lo, out("edx") hi, options(nomem, nostack, preserves_flags)) };
	(hi as u64) << 32 | lo as u64
}

fn instrs() -> Vec<InstrSet> {
	let mut v = vec![InstrSet::Avx256];
	if is_x86_feature_detected!("avx512f") {
		v.push(InstrSet::Avx512);
	} else if std::env::var_os("VECTOR_STATE_REQUIRE_AVX512").is_some_and(|v| v == "1") {
		panic!("VECTOR_STATE_REQUIRE_AVX512=1 but this host has no AVX-512");
	} else {
		println!("SKIP avx512: this host lacks avx512f, only the AVX2 path is exercised");
	}
	v
}

const DTYPES: [(DType, usize); 4] = [(DType::F32, 4), (DType::F64, 8), (DType::C32, 8), (DType::C64, 16)];

/// Participant 0 is the caller, the others are scoped threads. Every participant reads `XINUSE`
/// right after its job.
struct CheckFork {
	width: usize,
	seen: Mutex<Vec<(usize, u64)>>,
}

impl Fork for CheckFork {
	fn width(&self) -> usize {
		self.width
	}
	fn run(&self, f: &(dyn Fn(usize) + Sync)) {
		let job = |j: usize| {
			f(j);
			self.seen.lock().unwrap().push((j, xinuse() & UPPER));
		};
		std::thread::scope(|s| {
			for j in 1..self.width {
				let job = &job;
				s.spawn(move || job(j));
			}
			job(0);
		});
	}
}

#[derive(Clone, Copy, Debug)]
enum Entry {
	Gemm,
	GemmWith,
	GemmWithScratch,
	GemmWithStack,
}

const SHAPE: (usize, usize, usize) = (256, 256, 128);
const WIDTH: usize = 4;

/// Runs one product through `entry`; returns the `XINUSE` upper bits per participant.
fn run(entry: Entry, dtype: DType, sizeof: usize, instr: InstrSet) -> Vec<(String, u64)> {
	let (m, n, k) = SHAPE;
	let a = vec![0x3Fu8; m * k * sizeof];
	let b = vec![0x3Fu8; k * n * sizeof];
	let mut dst = vec![0u8; m * n * sizeof];
	let mut alpha = vec![0u8; 16];
	match dtype {
		DType::F32 | DType::C32 => alpha[..4].copy_from_slice(&1.0f32.to_ne_bytes()),
		DType::F64 | DType::C64 => alpha[..8].copy_from_slice(&1.0f64.to_ne_bytes()),
	}
	let fork = CheckFork {
		width: WIDTH,
		seen: Mutex::new(Vec::new()),
	};
	let (d, a, b, al) = (
		dst.as_mut_ptr() as *mut (),
		a.as_ptr() as *const (),
		b.as_ptr() as *const (),
		alpha.as_ptr() as *const (),
	);
	macro_rules! args {
		($call:ident $(, $extra:expr)*) => {
			$call(dtype, IType::U64, instr, m, n, k, d, 1, m as isize, std::ptr::null(), std::ptr::null(), DstKind::Full, Accum::Replace,
				a, 1, m as isize, false, std::ptr::null(), 0, b, 1, k as isize, false, al $(, $extra)*)
		};
	}
	unsafe {
		match entry {
			Entry::Gemm => args!(gemm, 1),
			Entry::GemmWith => args!(gemm_with, &fork),
			Entry::GemmWithScratch => {
				let mut mem = vec![0u8; MilliScratch::required(m, n)];
				args!(gemm_with_scratch, &fork, MilliScratch::new(&mut mem))
			},
			Entry::GemmWithStack => {
				let mut stack = vec![MaybeUninit::<u8>::uninit(); gemm_stack_bytes(dtype, instr, m, n, k)];
				args!(gemm_with_stack, &mut stack)
			},
		}
	}
	// the caller, after the entry returned: this is the boundary a consumer sees
	let mut out = vec![("caller after return".to_string(), xinuse() & UPPER)];
	for (j, bits) in fork.seen.into_inner().unwrap() {
		out.push((format!("participant {j} at end of job"), bits));
	}
	out
}

fn check_entry(entry: Entry) {
	if !xinuse_supported() {
		println!("SKIP xgetbv(1): CPUID.(0DH,1):EAX[2] is clear on this host");
		return;
	}
	let mut dirty = Vec::new();
	let mut checked = 0;
	for instr in instrs() {
		for (dtype, sizeof) in DTYPES {
			for (who, bits) in run(entry, dtype, sizeof, instr) {
				checked += 1;
				if bits != 0 {
					dirty.push(format!(
						"{entry:?} {dtype:?} {instr:?} {who}: XINUSE upper bits {bits:#x} (bit2 YMM_Hi128={}, bit6 ZMM_Hi256={})",
						bits & YMM_HI128 != 0,
						bits & ZMM_HI256 != 0
					));
				}
			}
		}
	}
	println!("{entry:?}: {checked} (dtype, isa, thread) observations, {} dirty", dirty.len());
	assert!(
		dirty.is_empty(),
		"kernels returned with a dirty upper vector state:\n{}",
		dirty.join("\n")
	);
}

#[test]
fn gemm_leaves_clean_upper_vector_state() {
	check_entry(Entry::Gemm);
}

#[test]
fn gemm_with_leaves_clean_upper_vector_state_on_every_participant() {
	check_entry(Entry::GemmWith);
}

#[test]
fn gemm_with_scratch_leaves_clean_upper_vector_state_on_every_participant() {
	check_entry(Entry::GemmWithScratch);
}

#[test]
fn gemm_with_stack_leaves_clean_upper_vector_state() {
	check_entry(Entry::GemmWithStack);
}

#[test]
fn syrk_lower_block_leaves_clean_upper_vector_state() {
	if !xinuse_supported() {
		println!("SKIP xgetbv(1)");
		return;
	}
	let (n, k) = (64usize, 32usize);
	let l = vec![1.0f64; n * k];
	let mut c = vec![0.0f64; n * n];
	let ok = unsafe { syrk_lower_block(n, k, l.as_ptr(), n, c.as_mut_ptr(), n, 1.0, Accum::Replace, 0, n) };
	if !ok {
		println!("SKIP syrk_lower_block: host is not Haswell-class, the entry declined");
		return;
	}
	assert_eq!(xinuse() & UPPER, 0, "syrk_lower_block returned with a dirty upper vector state");
}

/// Positive control: the reader does see dirty upper state, so a clean result above means
/// something. Dirty `ymm` (and `zmm` where available) on purpose and read it back.
#[test]
fn xgetbv_reader_sees_dirty_upper_state() {
	if !xinuse_supported() {
		println!("SKIP xgetbv(1)");
		return;
	}
	// all-ones is not the init state, so the processor must track it
	unsafe { std::arch::asm!("vpcmpeqd ymm0, ymm0, ymm0", out("ymm0") _, options(nomem, nostack)) };
	let bits = xinuse();
	assert!(bits & YMM_HI128 != 0, "ymm0 written but YMM_Hi128 not in use: {bits:#x}");
	unsafe { std::arch::asm!("vzeroupper", options(nomem, nostack)) };
	assert_eq!(xinuse() & YMM_HI128, 0, "vzeroupper did not clear YMM_Hi128");
	if is_x86_feature_detected!("avx512f") {
		unsafe { std::arch::asm!("vpternlogd zmm0, zmm0, zmm0, 0xff", out("zmm0") _, options(nomem, nostack)) };
		assert!(xinuse() & ZMM_HI256 != 0, "zmm0 written but ZMM_Hi256 not in use");
		unsafe { std::arch::asm!("vzeroupper", options(nomem, nostack)) };
		assert_eq!(xinuse() & UPPER, 0);
	}
}

#[test]
fn every_call_into_assembly_goes_through_call_kernel() {
	let mut files = Vec::new();
	for e in std::fs::read_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/src")).unwrap() {
		let p = e.unwrap().path();
		if p.extension().is_some_and(|x| x == "rs") {
			files.push(p);
		}
	}
	assert!(!files.is_empty());
	let mut bad = Vec::new();
	let mut macro_seen = false;
	for p in files {
		let text = std::fs::read_to_string(&p).unwrap();
		let mut in_macro = false;
		let mut macro_text = String::new();
		for (i, line) in text.lines().enumerate() {
			if line.contains("macro_rules! call_kernel") {
				in_macro = true;
				macro_seen = true;
			}
			if in_macro {
				macro_text.push_str(line);
				macro_text.push('\n');
				if line.trim() == "// end call_kernel" {
					in_macro = false;
					let call = macro_text.find("\"call r10\"").expect("call_kernel! has no call");
					let clear = macro_text.find("\"vzeroupper\"").expect("call_kernel! lost its vzeroupper");
					assert!(clear > call, "vzeroupper must follow the call");
				}
				continue;
			}
			let code = line.trim();
			if code.starts_with("//") {
				continue;
			}
			// the asm may also be a multi-line string whose line is just `call r10`
			for pat in ["call r", "call [", "call q", "call {"] {
				if code.contains(pat) {
					bad.push(format!("{}:{}: {code}", p.display(), i + 1));
				}
			}
		}
		assert!(!in_macro, "unterminated call_kernel macro in {}", p.display());
	}
	assert!(macro_seen, "macro_rules! call_kernel not found in src/");
	assert!(bad.is_empty(), "calls into assembly outside call_kernel!:\n{}", bad.join("\n"));
}
