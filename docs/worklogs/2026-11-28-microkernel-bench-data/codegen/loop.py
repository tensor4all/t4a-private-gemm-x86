import re, sys, collections
# usage: loop.py file.s  -> finds the innermost backward-branch loop with the most FMAs and summarises one iteration
lines = []
for l in open(sys.argv[1]):
    m = re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)$', l.rstrip())
    if m: lines.append((int(m.group(1), 16), m.group(2), m.group(3)))
addr = {a: i for i, (a, _, _) in enumerate(lines)}
loops = []
for i, (a, op, args) in enumerate(lines):
    if op.startswith('j'):
        m = re.match(r'([0-9a-f]+)\b', args)
        if m:
            t = int(m.group(1), 16)
            if t < a and t in addr:
                body = lines[addr[t]:i + 1]
                nf = sum(1 for x in body if 'vfmadd' in x[1])
                if nf: loops.append((addr[t], i, nf, body))
inner = [l for l in loops if not any(o is not l and o[0] >= l[0] and o[1] <= l[1] for o in loops)]
if '--all' in sys.argv:
    for l in loops: print('loop', hex(lines[l[0]][0]), '..', hex(lines[l[1]][0]), 'fma', l[2], 'len', len(l[3]), 'innermost' if l in inner else '')
inner.sort(key=lambda l: -l[2])
nf, body = inner[0][2], inner[0][3]
print(f'loop {body[0][0]:x}..{body[-1][0]:x}: {len(body)} instructions per iteration')
ops = collections.Counter(re.sub(r'\d+$', '', x[1]) if False else x[1] for x in body)
for k, v in sorted(ops.items(), key=lambda kv: -kv[1]): print(f'  {v:4d} {k}')
def reg(args, last=True):
    parts = [p for p in re.split(r',(?![^()]*\))', args)]
    return parts[-1].strip() if parts else ''
fm = [(j, x) for j, x in enumerate(body) if 'vfmadd' in x[1]]
widths = collections.Counter()
for j, x in fm:
    d = reg(x[2]); widths['ymm' if '%ymm' in d else 'xmm' if '%xmm' in d else 'zmm' if '%zmm' in d else '?'] += 1
print('fma by dest width:', dict(widths), 'scalar-sd/ss FMAs:', sum(1 for j, x in fm if x[1].endswith('sd') or x[1].endswith('ss')))
stack = [x for x in body if '(%rsp)' in x[2] or '(%rbp)' in x[2]]
print('stack-relative accesses in loop:', len(stack))
for x in stack[:6]: print('   ', f'{x[0]:x}', x[1], x[2])
print('vbroadcastsd:', sum(1 for x in body if x[1] == 'vbroadcastsd'), ' vpermilpd/vpermpd/vshufpd/vunpck:', sum(1 for x in body if re.match(r'v(permil|perm|shuf|unpck|insert|extract)', x[1])))
loads = sum(1 for x in body if x[1] in ('vmovupd', 'vmovapd') and re.search(r'\(.*\),', x[2]) is not None and x[2].split(',')[-1].startswith('%') and x[2].startswith(('-', '0', '(')))
print('vector loads (vmovupd/apd from memory):', loads, ' stores:', sum(1 for x in body if x[1] in ('vmovupd','vmovapd','vmovsd') and not x[2].split(',')[-1].strip().startswith('%')))
# same-destination FMA distance (in FMAs and in instructions) around the loop (cyclic)
dest = collections.defaultdict(list)
for j, x in fm: dest[reg(x[2])].append(j)
fm_idx = {j: n for n, (j, _) in enumerate(fm)}
gaps_f, gaps_i = [], []
for r, js in dest.items():
    for a, b in zip(js, js[1:] + [js[0] + len(body)]):
        gaps_f.append((fm_idx[b % len(body)] - fm_idx[a]) % len(fm) or len(fm)); gaps_i.append(b - a)
print('accumulator regs written by FMA:', len(dest), ' FMAs per iteration:', len(fm))
print('FMAs between consecutive writes to same dest: min', min(gaps_f), 'max', max(gaps_f), ' (instructions: min', min(gaps_i), 'max', max(gaps_i), ')')
