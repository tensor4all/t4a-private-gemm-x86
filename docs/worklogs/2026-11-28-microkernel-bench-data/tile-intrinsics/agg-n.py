import re, glob, statistics as st, sys, os
d = sys.argv[1]
R = {}
for f in sorted(glob.glob(os.path.join(d, '*-cpu*-r*.txt'))):
    m = re.match(r'(.+)-(B0|B1|N)-cpu(\d+)-r(\d+)\.txt$', os.path.basename(f))
    tc, v, cpu, r = m.groups()
    for line in open(f):
        x = re.match(r'(tile|driver)\s+(.+?)\s{2,}(\S.*?)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+%\s*$', line.rstrip())
        if x:
            R.setdefault((tc, v, cpu, x.group(1), x.group(2), x.group(3)), []).append(float(x.group(4)))
med = {k: st.median(v) for k, v in R.items()}
tcs = ['default', 'x86-64-v3', 'raptorlake', 'znver5']
shapes = ['191x257x139', '192x257x139', '256x256x256', '511x511x511', '512x512x512', '256x32x256', '256x512x256', '256x513x256']
P, S = 'packed driver (multiply)', 'stream driver (stream_rhs)'
def g(tc, v, cpu, name, sh): return med.get((tc, v, cpu, 'driver', name, sh))
def pair(f):
    out = []
    for cpu in ('2', '6'):
        try: out.append(f'{f(cpu):.3f}')
        except Exception: out.append('-')
    return ' / '.join(out)
print('**Driver, time ÷ config 1 (intrinsics tile, packed driver; built with the same flags, same core).** Zen 5 / Zen 5c. Config 4 = #26 as proposed; **N** = the new intrinsics tile.\n')
print('| target-cpu | shape | cfg 2 | cfg 3 | cfg 4 (#26) | **N** | N ÷ cfg 4 | N ÷ cfg 3 |')
print('|---|---|---|---|---|---|---|---|')
for tc in tcs:
    for sh in shapes:
        c1 = lambda cpu: g(tc, 'B0', cpu, P, sh)
        row = [pair(lambda c: g(tc, 'B1', c, P, sh) / c1(c)),
               pair(lambda c: g(tc, 'B1', c, S, sh) / c1(c)),
               pair(lambda c: g(tc, 'B0', c, S, sh) / c1(c)),
               pair(lambda c: g(tc, 'N', c, S, sh) / c1(c)),
               pair(lambda c: g(tc, 'N', c, S, sh) / g(tc, 'B0', c, S, sh)),
               pair(lambda c: g(tc, 'N', c, S, sh) / g(tc, 'B1', c, S, sh))]
        print(f'| {tc} | {sh} | ' + ' | '.join(row) + ' |')
print()
print('Control (identical code, packed driver): N ÷ B0 packed driver, Zen 5 / Zen 5c, over all shapes: ' + ', '.join(
    f'{tc} {min(g(tc,"N",c,P,s)/g(tc,"B0",c,P,s) for c in ("2","6") for s in shapes):.3f}–{max(g(tc,"N",c,P,s)/g(tc,"B0",c,P,s) for c in ("2","6") for s in shapes):.3f}' for tc in tcs))
print()
print('**Streamed tile by mode**, ns per call, ratios to intrinsics (B1) — `mul_add` (B0, #26) and the new intrinsics tile (N) — Zen 5 / Zen 5c:\n')
print('| target-cpu | mode | k=192 x1: B0 ÷ B1 | N ÷ B1 | k=512 x1: B0 ÷ B1 | N ÷ B1 | k=512 x4: B0 ÷ B1 | N ÷ B1 |')
print('|---|---|---|---|---|---|---|---|')
for tc in tcs:
    for mode in ('COMPUTE', 'PACK_PANEL', 'PACK_SLAB'):
        cells = []
        for sh in ('k=192 x1panel', 'k=512 x1panel', 'k=512 x4panel'):
            t = lambda v, c: med[(tc, v, c, 'tile', f'stream mode {mode}', sh)]
            cells += [pair(lambda c: t('B0', c) / t('B1', c)), pair(lambda c: t('N', c) / t('B1', c))]
        print(f'| {tc} | {mode} | ' + ' | '.join(cells) + ' |')
