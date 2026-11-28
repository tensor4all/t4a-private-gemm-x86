#!/bin/bash
# Same four binaries, benchmark on cpu N, thrash loader on its SMT sibling.
cd /home/shinaoka/work/gemm27/results4
declare -A L3=( [2]=16384K [6]=8192K ); declare -A SIB=( [2]=14 [6]=18 )
CFG=(c1-old-old c2-newtile-oldpacked c3-oldtile-stream c4-new-new)
for r in 0 1 2; do
 for cpu in 2 6; do
  [ "$(cat /sys/devices/system/cpu/cpu$cpu/cache/index3/size)" = "${L3[$cpu]}" ] || { echo mismatch >> loaded/errors.txt; continue; }
  taskset -c ${SIB[$cpu]} /tmp/w27/ld/thrash 65536 & LP=$!
  sleep 1; echo "r$r cpu$cpu loader pid $LP on cpu $(ps -o psr= -p $LP)" >> loaded/loader.txt
  for i in 0 1 2 3; do c=${CFG[$(( (i + r) % 4 ))]}
    BENCH_COMMIT=$c+sibling-loaded taskset -c $cpu ./$c --bench driver > loaded/$c-cpu$cpu-r$r.txt 2>&1
  done
  kill $LP; wait $LP 2>/dev/null
 done
done
echo done > loaded/finished
