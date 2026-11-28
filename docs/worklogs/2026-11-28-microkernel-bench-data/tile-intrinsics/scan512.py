import re, subprocess, sys, collections
# usage: scan512.py binary [symbol-substring ...]   (default substring: packed_avx2::)
b = sys.argv[1]; subs = sys.argv[2:] or ['packed_avx2::']
out = subprocess.run(['objdump', '-d', '-C', '-M', 'att', b], capture_output=True, text=True).stdout
funcs = collections.OrderedDict(); cur = None
for line in out.splitlines():
    m = re.match(r'^[0-9a-f]+ <(.+)>:$', line)
    if m: cur = m.group(1); funcs[cur] = []; continue
    m = re.match(r'^\s+([0-9a-f]+):\t([0-9a-f ]+?)\t(.*)$', line)
    if m and cur: funcs[cur].append((m.group(1), m.group(2).split(), m.group(3)))
    elif cur and re.match(r'^\s+[0-9a-f]+:\t[0-9a-f ]+$', line):  # continuation bytes
        pass
tot = collections.Counter(); rows = []
for name, ins in funcs.items():
    if not any(s in name for s in subs): continue
    c = collections.Counter()
    for a, by, txt in ins:
        c['instructions'] += 1
        if by and by[0] == '62': c['EVEX-encoded'] += 1
        if '%zmm' in txt: c['zmm'] += 1
        if re.search(r'%[xy]mm(1[6-9]|2\d|3[01])\b', txt): c['xmm/ymm16-31'] += 1
        if re.search(r'%k[0-7]\b', txt): c['mask-k'] += 1
    rows.append((name, c)); tot.update(c)
bad = [(n, c) for n, c in rows if c['EVEX-encoded'] or c['zmm'] or c['xmm/ymm16-31'] or c['mask-k']]
print(f"{b.split('/')[-1]}: {len(rows)} functions matching {subs}, {tot['instructions']} instructions; "
      f"EVEX {tot['EVEX-encoded']}, zmm {tot['zmm']}, xmm/ymm16-31 {tot['xmm/ymm16-31']}, mask {tot['mask-k']}")
for n, c in bad[:12]:
    print(f"   {n[:90]}: EVEX {c['EVEX-encoded']} zmm {c['zmm']} hi-regs {c['xmm/ymm16-31']} mask {c['mask-k']}")
