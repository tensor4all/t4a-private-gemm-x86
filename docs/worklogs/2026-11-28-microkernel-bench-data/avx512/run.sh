#!/bin/bash
# AVX-512 tile/fringe + e2e rows: {Zen 5 cpu2, Zen 5c cpu6} x 3 rounds, core order alternated.
# usage: run.sh <bench-binary> <commit-label> <out-dir>
BIN=$1; LABEL=$2; OUT=$3; mkdir -p "$OUT"
declare -A L3=( [2]=16384K [6]=8192K )
declare -A SIB=( [2]=14 [6]=18 )
busy() { awk -v c="cpu$1" '$1==c{print $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat; }
for r in 1 2 3; do
  if [ $((r%2)) = 1 ]; then cores="2 6"; else cores="6 2"; fi
  for cpu in $cores; do
    got=$(cat /sys/devices/system/cpu/cpu$cpu/cache/index3/size)
    [ "$got" = "${L3[$cpu]}" ] || { echo "L3 mismatch cpu$cpu: $got" >> "$OUT/errors.txt"; continue; }
    read b0 t0 < <(busy ${SIB[$cpu]}); sleep 2
    read b1 t1 < <(busy ${SIB[$cpu]})
    echo "pre-run r$r cpu$cpu($got): sibling cpu${SIB[$cpu]} busy $((100*(b1-b0)/(t1-t0)))%" >> "$OUT/idle.txt"
    BENCH_COMMIT=$LABEL taskset -c $cpu "$BIN" --bench tile e2e > "$OUT/cpu$cpu-r$r.txt" 2>&1
    read b2 t2 < <(busy ${SIB[$cpu]})
    echo "post-run r$r cpu$cpu: sibling busy over run $((100*(b2-b1)/(t2-t1)))%" >> "$OUT/idle.txt"
  done
done
