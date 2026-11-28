#!/bin/bash
# Pinned runs: {full Zen5 cpu2, Zen5c cpu6} x {old, new} x ROUNDS, order alternated per round.
cd /home/shinaoka/work/gemm27/results
declare -A L3=( [2]=16384K [6]=8192K )
declare -A SIB=( [2]=14 [6]=18 )
busy() { awk -v c="cpu$1" '$1==c{print $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat; }
for r in 1 2 3; do
  if [ $((r%2)) = 1 ]; then cores="2 6"; trees="old new"; else cores="6 2"; trees="new old"; fi
  for cpu in $cores; do
    for t in $trees; do
      got=$(cat /sys/devices/system/cpu/cpu$cpu/cache/index3/size)
      [ "$got" = "${L3[$cpu]}" ] || { echo "L3 mismatch cpu$cpu: $got" >> errors.txt; continue; }
      read b0 t0 < <(busy ${SIB[$cpu]}); read c0 u0 < <(busy $cpu); sleep 2
      read b1 t1 < <(busy ${SIB[$cpu]}); read c1 u1 < <(busy $cpu)
      echo "pre-run r$r cpu$cpu($got) $t: sibling cpu${SIB[$cpu]} busy $((100*(b1-b0)/(t1-t0)))%  target busy $((100*(c1-c0)/(u1-u0)))%" >> idle.txt
      BENCH_COMMIT=$([ $t = old ] && echo old-main-e688d4a+bench@f918b7f || echo new-pr26-3016c7e+overlay@8060521) taskset -c $cpu ./bench-$t --bench > $t-cpu$cpu-r$r.txt 2>&1
      read b2 t2 < <(busy ${SIB[$cpu]})
      echo "post-run r$r cpu$cpu $t: sibling busy over run $((100*(b2-b1)/(t2-t1)))%" >> idle.txt
    done
  done
done
echo done > finished
