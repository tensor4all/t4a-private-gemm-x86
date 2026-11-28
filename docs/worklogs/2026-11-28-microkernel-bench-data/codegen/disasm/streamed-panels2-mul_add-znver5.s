0000000000235a00 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_>:
  235a00:	push   %rbp
  235a01:	push   %r15
  235a03:	push   %r14
  235a05:	push   %r13
  235a07:	push   %r12
  235a09:	push   %rbx
  235a0a:	mov    %rcx,-0x10(%rsp)
  235a0f:	mov    0x58(%rsp),%rcx
  235a14:	mov    %rsi,%r10
  235a17:	mov    0x48(%rsp),%rsi
  235a1c:	mov    %r9,-0x70(%rsp)
  235a21:	mov    %rdi,%rax
  235a24:	mov    (%rcx),%r9
  235a27:	mov    %rsi,-0x50(%rsp)
  235a2c:	mov    %eax,%esi
  235a2e:	and    $0x3f8,%esi
  235a34:	mov    %rsi,-0x18(%rsp)
  235a39:	mov    %r9,-0x20(%rsp)
  235a3e:	cmp    $0x1,%r9
  235a42:	jne    235a85 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x85>
  235a44:	mov    0x8(%rcx),%rsi
  235a48:	mov    0x10(%rcx),%rcx
  235a4c:	mov    %rsi,-0x40(%rsp)
  235a51:	mov    %r8,%rsi
  235a54:	shl    $0x4,%rsi
  235a58:	vmovsd (%rcx,%r8,8),%xmm4
  235a5e:	vmovsd (%rcx),%xmm5
  235a62:	vmovsd (%rcx,%rsi,1),%xmm3
  235a67:	lea    (%r8,%r8,2),%rsi
  235a6b:	vmovsd (%rcx,%rsi,8),%xmm2
  235a70:	lea    (%r8,%r8,4),%rsi
  235a74:	shl    $0x5,%r8
  235a78:	vmovsd (%rcx,%r8,1),%xmm1
  235a7e:	vmovsd (%rcx,%rsi,8),%xmm0
  235a83:	jmp    235a85 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x85>
  235a85:	mov    -0x70(%rsp),%rsi
  235a8a:	mov    %rax,%r9
  235a8d:	shr    $0x3,%r9
  235a91:	mov    (%rdx),%r14
  235a94:	mov    0x8(%rdx),%r15
  235a98:	mov    0x10(%rdx),%r12
  235a9c:	mov    0x18(%rdx),%r13
  235aa0:	mov    0x20(%rdx),%rbp
  235aa4:	mov    0x28(%rdx),%rdx
  235aa8:	mov    0x50(%rsp),%r11
  235aad:	shlq   $0x3,-0x50(%rsp)
  235ab3:	vbroadcastsd %xmm0,%zmm0
  235ab9:	vbroadcastsd %xmm1,%zmm1
  235abf:	vbroadcastsd %xmm2,%zmm2
  235ac5:	vbroadcastsd %xmm3,%zmm3
  235acb:	vbroadcastsd %xmm4,%zmm4
  235ad1:	vbroadcastsd %xmm5,%zmm5
  235ad7:	mov    %rax,%rdi
  235ada:	shl    $0x6,%rax
  235ade:	vxorpd %xmm6,%xmm6,%xmm6
  235ae2:	mov    %r9,-0x28(%rsp)
  235ae7:	mov    %rax,-0x38(%rsp)
  235aec:	xor    %eax,%eax
  235aee:	mov    %rsi,%rcx
  235af1:	shl    $0x4,%rcx
  235af5:	mov    %rcx,-0x58(%rsp)
  235afa:	lea    0x0(,%rsi,8),%rcx
  235b02:	shl    $0x5,%rsi
  235b06:	mov    %rsi,-0x68(%rsp)
  235b0b:	lea    0x1c0(%r10),%rsi
  235b12:	lea    (%rcx,%rcx,2),%r8
  235b16:	lea    (%rcx,%rcx,4),%rcx
  235b1a:	mov    %r8,-0x60(%rsp)
  235b1f:	mov    %r9,%r8
  235b22:	shl    $0x9,%r8
  235b26:	shl    $0x6,%r9
  235b2a:	mov    %rcx,-0x30(%rsp)
  235b2f:	add    %r10,%r8
  235b32:	mov    0x40(%rsp),%r10
  235b37:	mov    %r10,-0x48(%rsp)
  235b3c:	jmp    235c3b <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x23b>
  235b41:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  235b50:	mov    %rbx,%rax
  235b53:	lea    0x0(,%rbx,8),%r10
  235b5b:	mov    -0x40(%rsp),%rbx
  235b60:	mov    -0x70(%rsp),%r8
  235b65:	vmovupd (%rbx,%r10,8),%zmm13
  235b6c:	vfmadd132pd %zmm5,%zmm6,%zmm13
  235b72:	vaddpd %zmm12,%zmm13,%zmm12
  235b78:	vmovupd %zmm12,(%rcx)
  235b7e:	vmovupd (%rbx,%r10,8),%zmm12
  235b85:	vfmadd132pd %zmm4,%zmm6,%zmm12
  235b8b:	vaddpd %zmm11,%zmm12,%zmm11
  235b91:	vmovupd %zmm11,(%rcx,%r8,8)
  235b98:	mov    -0x58(%rsp),%r8
  235b9d:	vmovupd (%rbx,%r10,8),%zmm11
  235ba4:	vfmadd132pd %zmm3,%zmm6,%zmm11
  235baa:	vaddpd %zmm10,%zmm11,%zmm10
  235bb0:	vmovupd %zmm10,(%rcx,%r8,1)
  235bb7:	mov    -0x60(%rsp),%r8
  235bbc:	vmovupd (%rbx,%r10,8),%zmm10
  235bc3:	vfmadd132pd %zmm2,%zmm6,%zmm10
  235bc9:	vaddpd %zmm9,%zmm10,%zmm9
  235bcf:	vmovupd %zmm9,(%rcx,%r8,1)
  235bd6:	mov    -0x68(%rsp),%r8
  235bdb:	vmovupd (%rbx,%r10,8),%zmm9
  235be2:	vfmadd132pd %zmm1,%zmm6,%zmm9
  235be8:	vaddpd %zmm8,%zmm9,%zmm8
  235bee:	vmovupd %zmm8,(%rcx,%r8,1)
  235bf5:	vmovupd (%rbx,%r10,8),%zmm8
  235bfc:	vfmadd132pd %zmm0,%zmm6,%zmm8
  235c02:	vaddpd %zmm7,%zmm8,%zmm7
  235c08:	mov    -0x30(%rsp),%r8
  235c0d:	addq   $0x40,-0x48(%rsp)
  235c13:	inc    %rax
  235c16:	vmovupd %zmm7,(%rcx,%r8,1)
  235c1d:	mov    -0x38(%rsp),%rcx
  235c22:	mov    -0x8(%rsp),%r8
  235c27:	add    %rcx,%rsi
  235c2a:	add    %rcx,%r11
  235c2d:	add    %rcx,%r8
  235c30:	cmp    0x38(%rsp),%rax
  235c35:	je     235f03 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x503>
  235c3b:	mov    %rax,%rbx
  235c3e:	cmpq   $0x0,-0x28(%rsp)
  235c44:	mov    -0x50(%rsp),%rax
  235c49:	vxorpd %xmm7,%xmm7,%xmm7
  235c4d:	vxorpd %xmm8,%xmm8,%xmm8
  235c52:	vxorpd %xmm9,%xmm9,%xmm9
  235c57:	vxorpd %xmm10,%xmm10,%xmm10
  235c5c:	vxorpd %xmm11,%xmm11,%xmm11
  235c61:	vxorpd %xmm12,%xmm12,%xmm12
  235c66:	je     235e58 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x458>
  235c6c:	mov    -0x48(%rsp),%r10
  235c71:	xor    %ecx,%ecx
  235c73:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  235c80:	vmovupd -0x1c0(%rsi,%rcx,8),%zmm13
  235c88:	vmovupd -0x180(%rsi,%rcx,8),%zmm14
  235c90:	vmovupd -0x140(%rsi,%rcx,8),%zmm15
  235c98:	vmovupd -0x100(%rsi,%rcx,8),%zmm16
  235ca0:	vfmadd231pd (%r14,%rcx,1){1to8},%zmm13,%zmm12
  235ca7:	vfmadd231pd (%r15,%rcx,1){1to8},%zmm13,%zmm11
  235cae:	vfmadd231pd (%r12,%rcx,1){1to8},%zmm13,%zmm10
  235cb5:	vfmadd231pd 0x0(%r13,%rcx,1){1to8},%zmm13,%zmm9
  235cbd:	vfmadd231pd 0x0(%rbp,%rcx,1){1to8},%zmm13,%zmm8
  235cc5:	vfmadd231pd (%rdx,%rcx,1){1to8},%zmm13,%zmm7
  235ccc:	vmovups (%r10),%zmm13
  235cd2:	add    %rax,%r10
  235cd5:	vfmadd231pd 0x8(%r14,%rcx,1){1to8},%zmm14,%zmm12
  235cdd:	vfmadd231pd 0x8(%r15,%rcx,1){1to8},%zmm14,%zmm11
  235ce5:	vfmadd231pd 0x8(%r12,%rcx,1){1to8},%zmm14,%zmm10
  235ced:	vfmadd231pd 0x8(%r13,%rcx,1){1to8},%zmm14,%zmm9
  235cf5:	vfmadd231pd 0x8(%rbp,%rcx,1){1to8},%zmm14,%zmm8
  235cfd:	vfmadd231pd 0x8(%rdx,%rcx,1){1to8},%zmm14,%zmm7
  235d05:	vfmadd231pd 0x10(%r14,%rcx,1){1to8},%zmm15,%zmm12
  235d0d:	vfmadd231pd 0x10(%r15,%rcx,1){1to8},%zmm15,%zmm11
  235d15:	vfmadd231pd 0x10(%r12,%rcx,1){1to8},%zmm15,%zmm10
  235d1d:	vfmadd231pd 0x10(%r13,%rcx,1){1to8},%zmm15,%zmm9
  235d25:	vfmadd231pd 0x10(%rbp,%rcx,1){1to8},%zmm15,%zmm8
  235d2d:	vfmadd231pd 0x10(%rdx,%rcx,1){1to8},%zmm15,%zmm7
  235d35:	vfmadd231pd 0x18(%r14,%rcx,1){1to8},%zmm16,%zmm12
  235d3d:	vfmadd231pd 0x18(%r15,%rcx,1){1to8},%zmm16,%zmm11
  235d45:	vfmadd231pd 0x18(%r12,%rcx,1){1to8},%zmm16,%zmm10
  235d4d:	vfmadd231pd 0x18(%r13,%rcx,1){1to8},%zmm16,%zmm9
  235d55:	vfmadd231pd 0x18(%rbp,%rcx,1){1to8},%zmm16,%zmm8
  235d5d:	vfmadd231pd 0x18(%rdx,%rcx,1){1to8},%zmm16,%zmm7
  235d65:	vmovups %zmm13,(%r11,%rcx,1)
  235d6c:	vmovupd -0xc0(%rsi,%rcx,8),%zmm13
  235d74:	vmovupd -0x80(%rsi,%rcx,8),%zmm14
  235d7c:	vmovupd -0x40(%rsi,%rcx,8),%zmm15
  235d84:	vmovupd (%rsi,%rcx,8),%zmm16
  235d8b:	vfmadd231pd 0x20(%r14,%rcx,1){1to8},%zmm13,%zmm12
  235d93:	vfmadd231pd 0x20(%r15,%rcx,1){1to8},%zmm13,%zmm11
  235d9b:	vfmadd231pd 0x20(%r12,%rcx,1){1to8},%zmm13,%zmm10
  235da3:	vfmadd231pd 0x20(%r13,%rcx,1){1to8},%zmm13,%zmm9
  235dab:	vfmadd231pd 0x20(%rbp,%rcx,1){1to8},%zmm13,%zmm8
  235db3:	vfmadd231pd 0x20(%rdx,%rcx,1){1to8},%zmm13,%zmm7
  235dbb:	vfmadd231pd 0x28(%r14,%rcx,1){1to8},%zmm14,%zmm12
  235dc3:	vfmadd231pd 0x28(%r15,%rcx,1){1to8},%zmm14,%zmm11
  235dcb:	vfmadd231pd 0x28(%r12,%rcx,1){1to8},%zmm14,%zmm10
  235dd3:	vfmadd231pd 0x28(%r13,%rcx,1){1to8},%zmm14,%zmm9
  235ddb:	vfmadd231pd 0x28(%rbp,%rcx,1){1to8},%zmm14,%zmm8
  235de3:	vfmadd231pd 0x28(%rdx,%rcx,1){1to8},%zmm14,%zmm7
  235deb:	vfmadd231pd 0x30(%r14,%rcx,1){1to8},%zmm15,%zmm12
  235df3:	vfmadd231pd 0x30(%r15,%rcx,1){1to8},%zmm15,%zmm11
  235dfb:	vfmadd231pd 0x30(%r12,%rcx,1){1to8},%zmm15,%zmm10
  235e03:	vfmadd231pd 0x30(%r13,%rcx,1){1to8},%zmm15,%zmm9
  235e0b:	vfmadd231pd 0x30(%rbp,%rcx,1){1to8},%zmm15,%zmm8
  235e13:	vfmadd231pd 0x30(%rdx,%rcx,1){1to8},%zmm15,%zmm7
  235e1b:	vfmadd231pd 0x38(%r14,%rcx,1){1to8},%zmm16,%zmm12
  235e23:	vfmadd231pd 0x38(%r15,%rcx,1){1to8},%zmm16,%zmm11
  235e2b:	vfmadd231pd 0x38(%r12,%rcx,1){1to8},%zmm16,%zmm10
  235e33:	vfmadd231pd 0x38(%r13,%rcx,1){1to8},%zmm16,%zmm9
  235e3b:	vfmadd231pd 0x38(%rbp,%rcx,1){1to8},%zmm16,%zmm8
  235e43:	vfmadd231pd 0x38(%rdx,%rcx,1){1to8},%zmm16,%zmm7
  235e4b:	add    $0x40,%rcx
  235e4f:	cmp    %rcx,%r9
  235e52:	jne    235c80 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x280>
  235e58:	mov    %r8,-0x8(%rsp)
  235e5d:	mov    %r8,%rcx
  235e60:	mov    -0x18(%rsp),%r8
  235e65:	mov    %rdi,%rax
  235e68:	mov    %r8,%r10
  235e6b:	cmp    %rdi,%r8
  235e6e:	je     235eae <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x4ae>
  235e70:	vmovupd (%rcx),%zmm13
  235e76:	add    $0x40,%rcx
  235e7a:	vfmadd231pd (%r14,%r10,8){1to8},%zmm13,%zmm12
  235e81:	vfmadd231pd (%r15,%r10,8){1to8},%zmm13,%zmm11
  235e88:	vfmadd231pd (%r12,%r10,8){1to8},%zmm13,%zmm10
  235e8f:	vfmadd231pd 0x0(%r13,%r10,8){1to8},%zmm13,%zmm9
  235e97:	vfmadd231pd 0x0(%rbp,%r10,8){1to8},%zmm13,%zmm8
  235e9f:	vfmadd231pd (%rdx,%r10,8){1to8},%zmm13,%zmm7
  235ea6:	inc    %r10
  235ea9:	cmp    %rax,%r10
  235eac:	jb     235e70 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x470>
  235eae:	mov    %rbx,%rcx
  235eb1:	shl    $0x6,%rcx
  235eb5:	add    -0x10(%rsp),%rcx
  235eba:	cmpb   $0x0,-0x20(%rsp)
  235ebf:	jne    235b50 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x150>
  235ec5:	mov    -0x70(%rsp),%r10
  235eca:	mov    -0x58(%rsp),%r8
  235ecf:	vmovupd %zmm12,(%rcx)
  235ed5:	mov    %rbx,%rax
  235ed8:	vmovupd %zmm11,(%rcx,%r10,8)
  235edf:	vmovupd %zmm10,(%rcx,%r8,1)
  235ee6:	mov    -0x60(%rsp),%r8
  235eeb:	vmovupd %zmm9,(%rcx,%r8,1)
  235ef2:	mov    -0x68(%rsp),%r8
  235ef7:	vmovupd %zmm8,(%rcx,%r8,1)
  235efe:	jmp    235c08 <_RINvNtNtCs2C9t30boYQ8_16private_gemm_x8611packed_avx210stream_rhs6panelsKj2_EB6_+0x208>
  235f03:	pop    %rbx
  235f04:	pop    %r12
  235f06:	pop    %r13
  235f08:	pop    %r14
  235f0a:	pop    %r15
  235f0c:	pop    %rbp
  235f0d:	vzeroupper
  235f10:	ret
  235f11:	int3
  235f12:	int3
  235f13:	int3
  235f14:	int3
  235f15:	int3
  235f16:	int3
  235f17:	int3
  235f18:	int3
  235f19:	int3
  235f1a:	int3
  235f1b:	int3
  235f1c:	int3
  235f1d:	int3
  235f1e:	int3
  235f1f:	int3
