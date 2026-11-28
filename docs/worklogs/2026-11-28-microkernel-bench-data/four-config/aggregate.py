import re,glob,statistics as st,collections
d=collections.defaultdict(list)
for f in sorted(glob.glob('c*-cpu*-r*.txt')):
    c,cpu,_=re.match(r'(c\d)-.*-cpu(\d+)-r(\d)',f).groups()
    for l in open(f):
        m=re.match(r'(tile|driver)\s+(.+?)\s{2,}(\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)%',l)
        if m: d[(m[1],m[2].strip(),m[3].strip(),cpu,c)].append(float(m[4]))
med=lambda g,k,s,cpu,c: st.median(d[(g,k,s,cpu,c)]) if d.get((g,k,s,cpu,c)) else None
shapes=['191x257x139','192x257x139','256x256x256','511x511x511','512x512x512','256x32x256','256x512x256','256x513x256']
print("driver time relative to config 1 (old tile, packed driver); median of 3 rounds")
print("| shape | core | 1 old/packed (ns) | 2 new tile/packed | 3 old tile/stream | 4 new tile/stream |\n|---|---|---:|---:|---:|---:|")
for s in shapes:
    for cpu,nm in (('2','Zen5 cpu2'),('6','Zen5c cpu6')):
        b=med('driver','packed driver (multiply)',s,cpu,'c1')
        r2=med('driver','packed driver (multiply)',s,cpu,'c2')
        r3=med('driver','stream driver (stream_rhs)',s,cpu,'c3')
        r4=med('driver','stream driver (stream_rhs)',s,cpu,'c4')
        # control: packed driver in c3/c4 is the same code as c1 (noise floor)
        ctl=[med('driver','packed driver (multiply)',s,cpu,c)/b for c in ('c3','c4')]
        print(f"| {s} | {nm} | {b:.0f} | {r2/b:.3f} | {r3/b:.3f} | {r4/b:.3f} | ctl {ctl[0]:.3f}/{ctl[1]:.3f}")
