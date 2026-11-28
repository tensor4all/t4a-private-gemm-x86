import re, glob, statistics as st, sys
# median over rounds of ns/call; GFLOP/s, %avx2, %isa recomputed from the median ns and the per-row flops
rows = {}
clk = {}
for f in sorted(glob.glob('cpu*-r*.txt')):
    cpu = re.match(r'cpu(\d+)-', f).group(1)
    for line in open(f):
        m = re.match(r'(tile512|tile|e2e)\s+(.+?)\s{2,}(\S.*?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)%$', line.rstrip())
        if m:
            g, name, shape, ns, gf, pa, pi = m.groups()
            rows.setdefault((g, name, shape, cpu), []).append((float(ns), float(gf), float(pa), float(pi)))
        c = re.match(r'clock\s+: measured ([\d.]+)', line)
        if c: clk.setdefault(cpu, []).append(float(c.group(1)))
print('clock GHz (measured, per run):', {k: v for k, v in clk.items()})
print()
print('| group | kernel | shape | Zen 5 ns | GFLOP/s | %avx2 | %isa | Zen 5c ns | GFLOP/s | %avx2 | %isa |')
print('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
seen = []
for k in rows:
    if k[:3] not in seen: seen.append(k[:3])
for g, name, shape in seen:
    cells = []
    for cpu in ('2', '6'):
        r = rows.get((g, name, shape, cpu))
        if not r: cells += ['-'] * 4; continue
        med = [st.median(x[i] for x in r) for i in range(4)]
        cells += [f'{med[0]:.1f}', f'{med[1]:.2f}', f'{med[2]:.1f}', f'{med[3]:.1f}']
    print(f'| {g} | {name} | {shape} | ' + ' | '.join(cells) + ' |')
