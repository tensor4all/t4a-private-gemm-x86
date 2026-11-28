import re, glob, statistics as st, sys, os
d = sys.argv[1]
rows = {}
clk = {}
for f in sorted(glob.glob(os.path.join(d, '*-cpu*-r*.txt'))):
    m = re.match(r'(.+)-(B[01])-cpu(\d+)-r(\d+)\.txt$', os.path.basename(f))
    tc, v, cpu, r = m.groups()
    for line in open(f):
        x = re.match(r'tile\s+(packed FINISH\(replace\)|stream full tile \(B in place\))\s+(k=\d+(?: x\d+panel)?)\s+([\d.]+)', line)
        if x:
            kind = 'packed' if x.group(1).startswith('packed') else 'stream'
            rows.setdefault((tc, v, cpu, kind, x.group(2)), []).append(float(x.group(3)))
        c = re.match(r'clock\s+: measured ([\d.]+)', line)
        if c: clk.setdefault(cpu, []).append(float(c.group(1)))
med = {k: st.median(v) for k, v in rows.items()}
rng = {k: (min(v), max(v)) for k, v in rows.items()}
tcs = ['default', 'x86-64-v3', 'raptorlake', 'znver5']
print('clock GHz per run:', {k: sorted(set(v)) for k, v in clk.items()})
print()
def table(kind, madd_v, intr_v, depths, title):
    print(f'**{title}**: ns per call (median of 3 rounds), and `mul_add` ÷ intrinsics\n')
    print('| target-cpu | k | Zen 5 intrinsics | Zen 5 mul_add | ratio | Zen 5c intrinsics | Zen 5c mul_add | ratio |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for tc in tcs:
        for dp in depths:
            cells = []
            for cpu in ('2', '6'):
                i = med.get((tc, intr_v, cpu, kind, dp)); m = med.get((tc, madd_v, cpu, kind, dp))
                cells += [f'{i:.1f}', f'{m:.1f}', f'{m / i:.3f}'] if i and m else ['-'] * 3
            print(f'| {tc} | {dp} | ' + ' | '.join(cells) + ' |')
    print()
table('packed', 'B1', 'B0', ['k=64', 'k=192', 'k=512'], 'Packed tile (8x6), k loop')
table('stream', 'B0', 'B1', ['k=64 x1panel', 'k=192 x1panel', 'k=512 x1panel', 'k=513 x1panel', 'k=512 x4panel'], 'Streamed tile (B in place)')
