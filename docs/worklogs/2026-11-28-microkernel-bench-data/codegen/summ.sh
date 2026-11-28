#!/bin/bash
# usage: summ.sh  (run from the an/ directory)
for cpu in default x86-64-v3 znver5 raptorlake; do
  for v in B0 B1; do
    for f in tile panels0; do
      if [ "$f" = tile ]; then
        if [ "$v" = B0 ]; then arith=intrinsics; else arith=mul_add; fi
      else
        if [ "$v" = B0 ]; then arith=mul_add; else arith=intrinsics; fi
      fi
      echo "== $cpu $f $arith"
      python3 -I loop.py "$f-$cpu-$v.s" | grep -E "^loop|fma by|stack-rel|^vbroad|FMAs between|accumulator regs" | cut -c1-160
    done
  done
done
