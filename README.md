# Optimize GEMM

Efficient Generalized Matrix Multiplication (GEMM) optimization, referring to https://github.com/flame/how-to-optimize-gemm.
This is the assignment for my Computer Architecture lesson during my Ph.D.

I did some improvements based on flame's GEMM optimization:
- Use AVX2 rather than AVX. So we can process four double numbers (256-bits) with a single instruction.
- Use OpenMP to implement multi-thread computation.
- Add some simple cache optimization for 1x4 cases.

The codes should be compiled under Linux, with the command:

```bash
gcc -mavx -fopen gemm_test.c -o gemm_main
```
or add -O3 option for faster execution:

```bash
gcc -mavx -fopen -O3 gemm_test.c -o gemm_main
```
