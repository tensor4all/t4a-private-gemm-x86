#!/bin/bash
# Tile timing for {B0,B1} x {target-cpu} x {Zen 5 cpu2, Zen 5c cpu6} x 3 rounds. Order rotated per round.
# B0 = packed tile intrinsics + streamed tile mul_add (the PR); B1 = packed tile mul_add + streamed tile intrinsics.
# usage: run-codegen.sh <bin-dir> <out-dir>
BIN=$1; OUT=$2; mkdir -p "$OUT"
declare -A L3=( [2]=16384K [6]=8192K )
declare -A SIB=( [2]=14 [6]=18 )
busy() { awk -v c="cpu$1" '$1==c{print $2+$3+$4+$7+$8, $2+$3+$4+$5+$6+$7+$8}' /proc/stat; }
for r in 1 2 3; do
  if [ $((r%2)) = 1 ]; then cores="2 6"; vars="B0 B1 N"; else cores="6 2"; vars="N B1 B0"; fi
  for cpu in $cores; do
    got=$(cat /sys/devices/system/cpu/cpu$cpu/cache/index3/size)
    [ "$got" = "${L3[$cpu]}" ] || { echo "L3 mismatch cpu$cpu: $got" >> "$OUT/errors.txt"; continue; }
    for tc in default x86-64-v3 znver5 raptorlake; do
      for v in $vars; do
        read b0 t0 < <(busy ${SIB[$cpu]}); sleep 1
        read b1 t1 < <(busy ${SIB[$cpu]})
        echo "pre r$r cpu$cpu $tc $v: sibling busy $((100*(b1-b0)/(t1-t0)))%" >> "$OUT/idle.txt"
        taskset -c $cpu "$BIN/mk-$tc-$v" --bench tile driver > "$OUT/$tc-$v-cpu$cpu-r$r.txt" 2>&1
        read b2 t2 < <(busy ${SIB[$cpu]})
        echo "post r$r cpu$cpu $tc $v: sibling busy over run $((100*(b2-b1)/(t2-t1)))%" >> "$OUT/idle.txt"
      done
    done
  done
done
echo finished > "$OUT/finished"
