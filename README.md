# Optimize GEMM

Compile the C code

```bash
gcc -mavx -fopen gemm_test.c -o gemm_main
```
or add -O3 option for faster execution

```bash
gcc -mavx -fopen -O3 gemm_test.c -o gemm_main
```
