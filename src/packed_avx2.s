// a and b point to packed micropanels. Each depth step produces 48 FMAs.
.macro colmat_packed_8x6_step t
vmovupd ymm12, [{a}+64*\t+0]
vmovupd ymm13, [{a}+64*\t+32]
vbroadcastsd ymm14, [{b}+48*\t+0]
vbroadcastsd ymm15, [{b}+48*\t+8]
vfmadd231pd ymm0, ymm12, ymm14
vfmadd231pd ymm1, ymm13, ymm14
vbroadcastsd ymm14, [{b}+48*\t+16]
vfmadd231pd ymm2, ymm12, ymm15
vfmadd231pd ymm3, ymm13, ymm15
vbroadcastsd ymm15, [{b}+48*\t+24]
vfmadd231pd ymm4, ymm12, ymm14
vfmadd231pd ymm5, ymm13, ymm14
vbroadcastsd ymm14, [{b}+48*\t+32]
vfmadd231pd ymm6, ymm12, ymm15
vfmadd231pd ymm7, ymm13, ymm15
vbroadcastsd ymm15, [{b}+48*\t+40]
vfmadd231pd ymm8, ymm12, ymm14
vfmadd231pd ymm9, ymm13, ymm14
vfmadd231pd ymm10, ymm12, ymm15
vfmadd231pd ymm11, ymm13, ymm15
.endm
.irp r,0,1,2,3,4,5,6,7,8,9,10,11
    vxorpd ymm\r, ymm\r, ymm\r
.endr
mov {count}, {k}
shr {count}, 2
jz 4f
.p2align 5
2:
.irp t,0,1,2,3
    colmat_packed_8x6_step \t
.endr
add {a}, 256
add {b}, 192
dec {count}
jnz 2b
4:
and {k}, 3
jz 6f
5:
colmat_packed_8x6_step 0
add {a}, 64
add {b}, 48
dec {k}
jnz 5b
6:
vmovq xmm12, {alpha}
vbroadcastsd ymm12, xmm12
test {add}, {add}
jnz 8f
.irp r,0,1,2,3,4,5,6,7,8,9,10,11
    vmulpd ymm13, ymm12, ymm\r
    .if (\r % 2) == 0
        vmovupd [{c}], ymm13
    .else
        vmovupd [{c}+32], ymm13
        add {c}, {ld}
    .endif
.endr
jmp 9f
8:
.irp r,0,1,2,3,4,5,6,7,8,9,10,11
    .if (\r % 2) == 0
        vfmadd213pd ymm\r, ymm12, [{c}]
        vmovupd [{c}], ymm\r
    .else
        vfmadd213pd ymm\r, ymm12, [{c}+32]
        vmovupd [{c}+32], ymm\r
        add {c}, {ld}
    .endif
.endr
9:
.purgem colmat_packed_8x6_step
