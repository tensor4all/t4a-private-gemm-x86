import re,glob,statistics as st,collections,sys
d=collections.defaultdict(list)
for f in sorted(glob.glob('*-cpu*-r*.txt')):
    tree,cpu,_=re.match(r'(old|new)-cpu(\d+)-r(\d)',f).groups()
    for l in open(f):
        m=re.match(r'(tile|edge|driver|e2e)\s+(.+?)\s{2,}(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)%',l)
        if m: d[(m[1],m[2].strip(),m[3].strip(),cpu,tree)].append((float(m[4]),float(m[5]),float(m[6])))
med=lambda v,i:st.median(x[i] for x in v)
def row(g,k,s,cpu):
    o=d.get((g,k,s,cpu,'old'));n=d.get((g,k,s,cpu,'new'))
    return o,n
print("| group | kernel | shape | core | old ns | new ns | new/old | old %peak | new %peak |\n|---|---|---|---|---:|---:|---:|---:|---:|")
for g in ['tile','driver','e2e']:
    for k in sorted({(a[1],a[2]) for a in d if a[0]==g}):
        for cpu,nm in (('2','Zen5 cpu2'),('6','Zen5c cpu6')):
            o,n=row(g,k[0],k[1],cpu)
            if g=='edge':continue
            if o and n: print(f"| {g} | {k[0]} | {k[1]} | {nm} | {med(o,0):.1f} | {med(n,0):.1f} | {med(n,0)/med(o,0):.4f} | {med(o,2):.1f} | {med(n,2):.1f} |")
            elif n: print(f"| {g} | {k[0]} | {k[1]} | {nm} | - | {med(n,0):.1f} | new-only | - | {med(n,2):.1f} |")
