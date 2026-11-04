#[derive(Default, Debug, Copy, Clone)]
pub struct CacheInfo {
	pub associativity: usize,
	pub cache_bytes: usize,
	pub cache_line_bytes: usize,
}

/// Runs the process-wide cache probe now instead of inside the first product (faer#232).
///
/// The probe is a `OnceLock`, so this is idempotent and later calls do nothing; a program that
/// wants the cost at start-up rather than in its first parallel region calls this first.
pub fn init() {
	let _ = *CACHE_INFO;
}

fn cache_info() -> Option<[CacheInfo; 3]> {
	if !cfg!(miri) {
		{
			#[cfg(all(target_os = "linux", feature = "std"))]
			{
				use std::fs;
				use std::path::Path;

				/// The CPUs this process may run on, from `/proc/self/status`'s `Cpus_allowed_list`
				/// (the `sched_getaffinity` set, without a `libc` dependency).
				fn allowed_cpus() -> Option<Vec<usize>> {
					let status = fs::read_to_string("/proc/self/status").ok()?;
					let list = status.lines().find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
					parse_cpu_list(list.trim())
				}

				fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
					let mut cpus = Vec::new();
					for item in list.split(',') {
						match item.split_once('-') {
							Some((start, end)) => {
								let start = start.trim().parse::<usize>().ok()?;
								let end = end.trim().parse::<usize>().ok()?;
								cpus.extend(start..=end);
							},
							None => cpus.push(item.trim().parse::<usize>().ok()?),
						}
					}
					(!cpus.is_empty()).then_some(cpus)
				}

				fn read_trimmed(path: impl AsRef<Path>) -> Option<String> {
					fs::read_to_string(path).ok().map(|s| s.trim().to_string())
				}

				fn package_of(cpu: usize) -> Option<String> {
					read_trimmed(Path::new("/sys/devices/system/cpu").join(format!("cpu{cpu}/topology/physical_package_id")))
				}

				/// One allowed CPU per physical package: the cache description of a package is read
				/// from a single CPU of it, so the cost does not grow with the CPU count, and a
				/// host whose packages differ still reports every package.
				///
				/// The packages of a machine are contiguous CPU ranges, so the first and the last
				/// allowed CPU decide it in two reads; only when they disagree is every allowed CPU
				/// looked at, which a single-socket host never pays.
				fn package_representatives(cpus: &[usize]) -> Vec<usize> {
					let (&first, &last) = (cpus.first().unwrap_or(&0), cpus.last().unwrap_or(&0));
					if package_of(first) == package_of(last) {
						return vec![first];
					}
					let mut packages: Vec<Option<String>> = Vec::new();
					let mut representatives = Vec::new();
					for &cpu in cpus {
						let package = package_of(cpu);
						if !packages.contains(&package) {
							packages.push(package);
							representatives.push(cpu);
						}
					}
					representatives
				}

				/// The online CPUs, from `/sys/devices/system/cpu/online`.
				fn online_cpus() -> Option<usize> {
					parse_cpu_list(&read_trimmed("/sys/devices/system/cpu/online")?).map(|cpus| cpus.len())
				}

				/// The threads per core of `cpu`, from its `thread_siblings_list`.
				fn threads_per_core(cpu: usize) -> Option<usize> {
					let path = Path::new("/sys/devices/system/cpu").join(format!("cpu{cpu}/topology/thread_siblings_list"));
					parse_cpu_list(&read_trimmed(path)?).map(|cpus| cpus.len())
				}

				/// The machine's physical cores, without `/proc/cpuinfo` (`num_cpus::get_physical`
				/// reads and parses all of it, which is most of the probe's cost on a large host,
				/// faer#232).
				///
				/// Two sysfs reads -- the online CPUs and the threads per core -- settle it while
				/// every online core has the same number of threads, and the result then does not
				/// grow with the CPU count. Otherwise the distinct (`physical_package_id`,
				/// `core_id`) pairs of the allowed CPUs are counted, two reads per CPU.
				fn physical_cores(allowed: &[usize]) -> Option<usize> {
					if let (Some(online), Some(threads)) = (online_cpus(), threads_per_core(allowed.first().copied().unwrap_or(0))) {
						return Some(online / threads.max(1)).filter(|cores| *cores > 0);
					}
					let mut pairs: Vec<(String, String)> = Vec::new();
					for &cpu in allowed {
						let topology = Path::new("/sys/devices/system/cpu").join(format!("cpu{cpu}/topology"));
						let pair = (
							read_trimmed(topology.join("physical_package_id"))?,
							read_trimmed(topology.join("core_id"))?,
						);
						if !pairs.contains(&pair) {
							pairs.push(pair);
						}
					}
					(!pairs.is_empty()).then_some(pairs.len())
				}

				/// `level`, the cache description and the width of `shared_cpu_list` of one
				/// `cache/index*` directory, reading only the six files the probe needs.
				fn read_index(index: &Path) -> Option<(usize, CacheInfo, usize)> {
					if !matches!(read_trimmed(index.join("type"))?.as_str(), "Data" | "Unified") {
						return None;
					}
					let level = read_trimmed(index.join("level"))?.parse::<usize>().ok()?;
					let mut cache_info = CacheInfo {
						associativity: 8,
						cache_bytes: 0,
						cache_line_bytes: 64,
					};
					if let Some(line) = read_trimmed(index.join("coherency_line_size")) {
						cache_info.cache_line_bytes = line.parse().ok()?;
					}
					if let Some(ways) = read_trimmed(index.join("ways_of_associativity")) {
						cache_info.associativity = ways.parse().ok()?;
					}
					let size = read_trimmed(index.join("size"))?;
					cache_info.cache_bytes = if let Some(size) = size.strip_suffix('G') {
						size.parse::<usize>().ok()? * 1024 * 1024 * 1024
					} else if let Some(size) = size.strip_suffix('M') {
						size.parse::<usize>().ok()? * 1024 * 1024
					} else if let Some(size) = size.strip_suffix('K') {
						size.parse::<usize>().ok()? * 1024
					} else {
						size.parse::<usize>().ok()?
					};

					let mut shared_count = 0;
					for item in read_trimmed(index.join("shared_cpu_list"))?.split(',') {
						match item.split_once('-') {
							Some((start, end)) => {
								let (start, end) = (start.parse::<usize>().ok()?, end.parse::<usize>().ok()?);
								shared_count += end + 1 - start;
							},
							None => shared_count += 1,
						}
					}
					Some((level, cache_info, shared_count))
				}

				fn try_cache_info_linux() -> Result<[CacheInfo; 3], std::io::Error> {
					let mut all_info = [CacheInfo {
						associativity: 8,
						cache_bytes: 0,
						cache_line_bytes: 64,
					}; 3];

					// the cache description of one CPU per package, not of every CPU
					let cpus = allowed_cpus().unwrap_or_else(|| vec![0]);
					let mut l1_shared_count = 1;
					for cpu in package_representatives(&cpus) {
						let cache = Path::new("/sys/devices/system/cpu").join(format!("cpu{cpu}/cache"));
						if !cache.is_dir() {
							continue;
						}
						for index in fs::read_dir(cache)? {
							let index = index?.path();
							if !index.file_name().and_then(|f| f.to_str()).is_some_and(|f| f.starts_with("index")) {
								continue;
							}
							let Some((level, cache_info, shared_count)) = read_index(&index) else {
								continue;
							};
							if level == 1 {
								l1_shared_count = shared_count;
							}
							// a level above L3 (an L4 on some hosts) has no slot here
							if level > 0 && level <= 3 && cache_info.cache_line_bytes >= all_info[level - 1].cache_line_bytes {
								all_info[level - 1].associativity = cache_info.associativity;
								all_info[level - 1].cache_line_bytes = cache_info.cache_line_bytes;
								all_info[level - 1].cache_bytes = cache_info.cache_bytes / shared_count.max(1);
							}
						}
					}
					for info in &mut all_info {
						info.cache_bytes *= l1_shared_count.max(1);
					}
					all_info[2].cache_bytes *= physical_cores(&cpus).unwrap_or_else(num_cpus::get_physical);

					Ok(all_info)
				}
				if let Ok(info) = try_cache_info_linux() {
					return Some(info);
				}

				if let Ok(lscpu) = std::process::Command::new("lscpu")
					.arg("-B")
					.arg("-C=type,level,ways,coherency-size,one-size")
					.output()
				{
					if lscpu.status.success() {
						if let Ok(lscpu) = String::from_utf8(lscpu.stdout).as_deref() {
							let mut info = CACHE_INFO_DEFAULT;
							for line in lscpu.lines().skip(1) {
								let mut iter = line.split_whitespace();
								if let [Some(cache_type), Some(level), Some(ways), Some(coherency_size), Some(one_size)] =
									[iter.next(), iter.next(), iter.next(), iter.next(), iter.next()]
								{
									if let "Data" | "Unified" = cache_type {
										let level = level.parse::<usize>().unwrap();
										let ways = ways.parse::<usize>().unwrap();
										let coherency_size = coherency_size.parse::<usize>().unwrap();
										let one_size = one_size.parse::<usize>().unwrap();

										info[level - 1].associativity = ways;
										info[level - 1].cache_line_bytes = coherency_size;
										info[level - 1].cache_bytes = one_size;
									}
								}
							}
							return Some(info);
						}
					}
				}
			}
			#[cfg(all(target_vendor = "apple", feature = "std"))]
			{
				use sysctl::{Ctl, Sysctl};

				let mut all_info = CACHE_INFO_DEFAULT;
				if let Ok(l1) = Ctl::new("hw.perflevel0.l1dcachesize").and_then(|ctl| ctl.value_string()) {
					if let Ok(l1) = l1.parse::<usize>() {
						all_info[0].cache_bytes = l1;
					}
				}
				if let (Ok(physicalcpu), Ok(cpusperl2), Ok(l2)) = (
					Ctl::new("hw.perflevel0.physicalcpu").and_then(|ctl| ctl.value_string()),
					Ctl::new("hw.perflevel0.cpusperl2").and_then(|ctl| ctl.value_string()),
					Ctl::new("hw.perflevel0.l2cachesize").and_then(|ctl| ctl.value_string()),
				) {
					if let (Ok(_physicalcpu), Ok(cpusperl2), Ok(l2)) = (physicalcpu.parse::<usize>(), cpusperl2.parse::<usize>(), l2.parse::<usize>())
					{
						all_info[1].cache_bytes = l2 / cpusperl2;
					}
				}
				all_info[2].cache_bytes = 0;
				return Some(all_info);
			}
		}

		#[cfg(any(
			all(target_arch = "x86", not(target_env = "sgx"), target_feature = "sse"),
			all(target_arch = "x86_64", not(target_env = "sgx"))
		))]
		{
			use raw_cpuid::CpuId;
			let cpuid = CpuId::new();

			if let Some(vf) = cpuid.get_vendor_info() {
				let vf = vf.as_str();
				if vf == "GenuineIntel" {
					if let Some(cparams) = cpuid.get_cache_parameters() {
						// not sure why, intel cpus seem to prefer smaller mc
						let mut info = [CacheInfo {
							cache_bytes: 0,
							associativity: 0,
							cache_line_bytes: 64,
						}; 3];

						for cache in cparams {
							use raw_cpuid::CacheType::*;
							match cache.cache_type() {
								Null | Instruction | Reserved => continue,
								Data | Unified => {
									let level = cache.level() as usize;
									let associativity = cache.associativity();
									let nsets = cache.sets();
									let cache_line_bytes = cache.coherency_line_size();
									if level > 0 && level < 4 {
										let info = &mut info[level - 1];
										info.cache_line_bytes = cache_line_bytes;
										info.associativity = associativity;
										info.cache_bytes = associativity * nsets * cache_line_bytes;
									}
								},
							}
						}
						return Some(info);
					}
				}

				if vf == "AuthenticAMD" {
					if let Some(l1) = cpuid.get_l1_cache_and_tlb_info() {
						if let Some(l23) = cpuid.get_l2_l3_cache_and_tlb_info() {
							let compute_info = |associativity: raw_cpuid::Associativity, cache_kb: usize, cache_line_bytes: u8| -> CacheInfo {
								let cache_bytes = cache_kb as usize * 1024;
								let cache_line_bytes = cache_line_bytes as usize;

								use raw_cpuid::Associativity::*;
								let associativity = match associativity {
									Unknown | Disabled => {
										return CacheInfo {
											associativity: 0,
											cache_bytes: 0,
											cache_line_bytes: 64,
										};
									},
									FullyAssociative => cache_bytes / cache_line_bytes,
									DirectMapped => 1,
									NWay(n) => n as usize,
								};

								CacheInfo {
									associativity,
									cache_bytes,
									cache_line_bytes,
								}
							};
							return Some([
								compute_info(l1.dcache_associativity(), l1.dcache_size() as usize, l1.dcache_line_size()),
								compute_info(l23.l2cache_associativity(), l23.l2cache_size() as usize, l23.l2cache_line_size()),
								compute_info(l23.l3cache_associativity(), l23.l3cache_size() as usize * 512, l23.l3cache_line_size()),
							]);
						}
					}
				}
			}
		}
	}
	None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static CACHE_INFO_DEFAULT: [CacheInfo; 3] = [
	CacheInfo {
		associativity: 8,
		cache_bytes: 32 * 1024, // 32KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 256 * 1024, // 256KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 2 * 1024 * 1024, // 2MiB
		cache_line_bytes: 64,
	},
];

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
static CACHE_INFO_DEFAULT: [CacheInfo; 3] = [
	CacheInfo {
		associativity: 8,
		cache_bytes: 64 * 1024, // 64KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 512 * 1024, // 512KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 4 * 1024 * 1024, // 4MiB
		cache_line_bytes: 64,
	},
];

#[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64", target_arch = "x86", target_arch = "x86_64")))]
static CACHE_INFO_DEFAULT: [CacheInfo; 3] = [
	CacheInfo {
		associativity: 8,
		cache_bytes: 16 * 1024, // 16KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 512 * 1024, // 512KiB
		cache_line_bytes: 64,
	},
	CacheInfo {
		associativity: 8,
		cache_bytes: 1024 * 1024, // 1MiB
		cache_line_bytes: 64,
	},
];

pub struct CacheInfoDeref;

impl core::ops::Deref for CacheInfoDeref {
	type Target = [CacheInfo; 3];

	#[cfg(feature = "std")]
	#[inline]
	fn deref(&self) -> &Self::Target {
		static CACHE_INFO: std::sync::OnceLock<[CacheInfo; 3]> = std::sync::OnceLock::new();

		CACHE_INFO.get_or_init(|| {
			let mut val = cache_info().unwrap_or(CACHE_INFO_DEFAULT);
			val[0].cache_bytes = Ord::max(val[0].cache_bytes, CACHE_INFO_DEFAULT[0].cache_bytes);

			val[1].cache_bytes = Ord::max(val[1].cache_bytes, CACHE_INFO_DEFAULT[1].cache_bytes);
			val[1].cache_bytes = Ord::max(val[1].cache_bytes, val[0].cache_bytes.checked_mul(4).unwrap());

			val[2].cache_bytes = Ord::max(val[2].cache_bytes, CACHE_INFO_DEFAULT[2].cache_bytes);
			val[2].cache_bytes = Ord::max(val[2].cache_bytes, val[1].cache_bytes.checked_mul(4).unwrap());
			val
		})
	}

	#[cfg(not(feature = "std"))]
	#[inline]
	fn deref(&self) -> &Self::Target {
		&CACHE_INFO_DEFAULT
	}
}

pub static CACHE_INFO: CacheInfoDeref = CacheInfoDeref;

#[cfg(test)]
mod tests {
	use super::*;

	/// The probe is a `OnceLock`: `init` warms it once and later products do not repeat it, and
	/// the levels the kernel reads are usable whatever the host reports (the deref floors them).
	#[test]
	fn init_warms_the_probe_once_and_the_levels_are_usable() {
		init();
		let warm = *CACHE_INFO;
		init();
		let again = *CACHE_INFO;
		for level in 0..3 {
			assert!(warm[level].cache_bytes > 0, "level {}: no size", level + 1);
			assert!(warm[level].cache_line_bytes > 0, "level {}: no line size", level + 1);
			assert_eq!(
				(warm[level].cache_bytes, warm[level].associativity, warm[level].cache_line_bytes),
				(again[level].cache_bytes, again[level].associativity, again[level].cache_line_bytes),
				"level {}: a second init changed the description",
				level + 1
			);
		}
		assert!(
			warm[0].cache_bytes <= warm[1].cache_bytes && warm[1].cache_bytes <= warm[2].cache_bytes,
			"the levels are not ordered: {warm:?}"
		);
	}
}
