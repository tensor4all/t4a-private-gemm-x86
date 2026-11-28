// L1/L2 pressure from an SMT sibling: random read-modify-write over a 64 KiB buffer (> a 48 KiB L1D share).
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc,char**argv){ size_t n=(argc>1?atol(argv[1]):65536)/8; volatile uint64_t*b=aligned_alloc(64,n*8); for(size_t i=0;i<n;i++)b[i]=i; uint64_t x=88172645463325252ull,s=0; for(;;){ for(int k=0;k<1<<20;k++){ x^=x<<13;x^=x>>7;x^=x<<17; size_t i=x%n; s+=b[i]; b[i]=s; } } }
