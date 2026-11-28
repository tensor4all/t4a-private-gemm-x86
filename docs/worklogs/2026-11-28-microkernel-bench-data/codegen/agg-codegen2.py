import re, glob, statistics as st, sys, os
d = sys.argv[1]
R = {}
for f in sorted(glob.glob(os.path.join(d, '*-cpu*-r*.txt'))):
    m = re.match(r'(.+)-(B[01])-cpu(\d+)-r(\d+)\.txt$', os.path.basename(f))
    tc, v, cpu, r = m.groups()
    for line in open(f):
        x = re.match(r'(tile|driver)\s+(.+?)\s{2,}(\S.*?)\s+([\d.]+)\s+([\d.]+)\s+[\d.]+%\s*$', line.rstrip())
        if x:
            g, name, shape, ns = x.group(1), x.group(2), x.group(3), float(x.group(4))
            R.setdefault((tc, v, cpu, g, name, shape), []).append(ns)
med = {k: st.median(v) for k, v in R.items()}
tcs = ['default', 'x86-64-v3', 'raptorlake', 'znver5']
shapes = ['191x257x139', '192x257x139', '256x256x256', '511x511x511', '512x512x512', '256x32x256', '256x512x256', '256x513x256']
def g(tc, v, cpu, name, shape):
    return med.get((tc, v, cpu, 'driver', name, shape))
P, S = 'packed driver (multiply)', 'stream driver (stream_rhs)'
print('**Driver, four configurations, per `-C target-cpu`.** Each cell is time ÷ config 1 built with the same flags on the same core (below 1 is faster). Config 1 = intrinsics tile, packed driver; 2 = `mul_add` tile, packed driver; 3 = intrinsics tile, streamed driver; 4 = `mul_add` tile, streamed driver (#26).\n')
print('| target-cpu | shape | cfg 2 Zen 5 / 5c | cfg 3 Zen 5 / 5c | cfg 4 Zen 5 / 5c | cfg 4 ÷ cfg 3 Zen 5 / 5c |')
print('|---|---|---|---|---|---|')
for tc in tcs:
    for sh in shapes:
        cells = []
        c43 = []
        for which in ('2', '3', '4'):
            pair = []
            for cpu in ('2', '6'):
                c1 = g(tc, 'B0', cpu, P, sh)
                c = {'2': g(tc, 'B1', cpu, P, sh), '3': g(tc, 'B1', cpu, S, sh), '4': g(tc, 'B0', cpu, S, sh)}[which]
                pair.append(f'{c / c1:.3f}' if c1 and c else '-')
            cells.append(' / '.join(pair))
        pair = []
        for cpu in ('2', '6'):
            c3 = g(tc, 'B1', cpu, S, sh); c4 = g(tc, 'B0', cpu, S, sh)
            pair.append(f'{c4 / c3:.3f}' if c3 and c4 else '-')
        cells.append(' / '.join(pair))
        print(f'| {tc} | {sh} | ' + ' | '.join(cells) + ' |')
print()
print('**Streamed tile in the driver\'s three modes**, ns per call, `mul_add` ÷ intrinsics (Zen 5 / Zen 5c):\n')
print('| target-cpu | mode | k=192 x1 | k=512 x1 | k=512 x4 |')
print('|---|---|---|---|---|')
for tc in tcs:
    for mode in ('COMPUTE', 'PACK_PANEL', 'PACK_SLAB'):
        cells = []
        for sh in ('k=192 x1panel', 'k=512 x1panel', 'k=512 x4panel'):
            pair = []
            for cpu in ('2', '6'):
                mu = med.get((tc, 'B0', cpu, 'tile', f'stream mode {mode}', sh)); it = med.get((tc, 'B1', cpu, 'tile', f'stream mode {mode}', sh))
                pair.append(f'{mu / it:.3f}' if mu and it else '-')
            cells.append(' / '.join(pair))
        print(f'| {tc} | {mode} | ' + ' | '.join(cells) + ' |')
