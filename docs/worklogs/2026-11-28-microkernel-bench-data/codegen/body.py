import re,sys
lines=[]
for l in open(sys.argv[1]):
    m=re.match(r'\s*([0-9a-f]+):\s+(\S+)\s*(.*)$',l.rstrip())
    if m: lines.append((int(m.group(1),16),m.group(2),m.group(3)))
start=int(sys.argv[2],16); end=int(sys.argv[3],16)
for a,o,r in lines:
    if start<=a<=end: print(f'{o} {re.sub(r"<.*>","",r).strip()}')
