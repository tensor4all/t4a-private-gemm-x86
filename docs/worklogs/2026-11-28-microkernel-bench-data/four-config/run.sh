#!/bin/bash
# Four configurations x {Zen5 cpu2, Zen5c cpu6} x 3 rounds; core order and config order rotate per round.
cd /home/shinaoka/work/gemm27/results4
declare -A L3=( [2]=16384K [6]=8192K ); declare -A SIB=( [2]=14 [6]=18 )
CFG=(c1-old-old c2-newtile-oldpacked c3-oldtile-stream c4-new-new)
busy() { awk -v c="cpu$1" '$1==c{print $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat; }
for r in 0 1 2; do
  cores="2 6"; [ $r = 1 ] && cores="6 2"
  for cpu in $cores; do
    for i in 0 1 2 3; do
      c=${CFG[$(( (i + r) % 4 ))]}
      got=$(cat /sys/devices/system/cpu/cpu$cpu/cache/index3/size)
      [ "$got" = "${L3[$cpu]}" ] || { echo "L3 mismatch cpu$cpu: $got" >> errors.txt; continue; }
      read b0 t0 < <(busy ${SIB[$cpu]}); sleep 2; read b1 t1 < <(busy ${SIB[$cpu]})
      echo "pre r$r cpu$cpu($got) $c: sibling cpu${SIB[$cpu]} busy $((100*(b1-b0)/(t1-t0)))%" >> idle.txt
      BENCH_COMMIT=$c taskset -c $cpu ./$c --bench tile driver scratch > $c-cpu$cpu-r$r.txt 2>&1
      read b2 t2 < <(busy ${SIB[$cpu]}); echo "post r$r cpu$cpu $c: sibling busy over run $((100*(b2-b1)/(t2-t1)))%" >> idle.txt
    done
  done
done
echo done > finished
