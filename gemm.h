#include "gemm_utils.h"

#define min(xx,yy) (((xx) < (yy)) ? (xx) : (yy))

void gemm_trivial(int m, int n, int k, double *a, int lda,
                                       double *b, int ldb,
                                       double *c, int ldc)
{
    // trivial matrix multiplication
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            for (int r = 0; r < k; r++) {
                C(i, j) += A(i, r) * B(r, j);
            }
        }
    }
}

void gemm_cache(int m, int n, int k, double *a, int lda,
                                       double *b, int ldb,
                                       double *c, int ldc)
{
    // optimize the order of matrix multiplication
    for (int j = 0; j < n; j++) {
        for (int r = 0; r < k; r++) {
            register double brj = B(r, j);
            for (int i = 0; i < m; i++){
                C(i, j) += A(i, r) * brj;
            }
        }
    }
}

void gemm_1x4(int m, int n, int k, double *a, int lda,
                                   double *b, int ldb,
                                   double *c, int ldc)
{
    for (int j = 0; j < n; j += 4) {
        for (int r = 0; r < k; r++) {
            dot1x4_cache(m, &A(0,r), lda, &B(r,j), ldb, &C(0,j), ldc);
        }
    }
}

void gemm_1x4_avx(int m, int n, int k, double *a, int lda,
                                      double *b, int ldb,
                                      double *c, int ldc)
{
    for (int j = 0; j < n; j += 4) {
        for (int r = 0; r < k; r++) {
            dot1x4_cache_avx(m, &A(0,r), lda, &B(r,j), ldb, &C(0,j), ldc);
        }
    }
}

void gemm_4x4(int m, int n, int k, double *a, int lda,
                                   double *b, int ldb,
                                   double *c, int ldc)
{
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < k; i += 4) {
            dot4x4(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
        }
    }
}

void gemm_4x4_avx(int m, int n, int k, double *a, int lda,
                                       double *b, int ldb,
                                       double *c, int ldc)
{
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            dot4x4_avx(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
        }
    }
}

void gemm_4x4_ik(int m, int n, int k, double *a, int lda,
                                      double *b, int ldb,
                                      double *c, int ldc)
{
    for (int rr = 0; rr < k; rr += ikk) {
        int jb = min(k - rr, ikk);
        for (int ii = 0; ii < m; ii += ikm) {
            int ib = min(m - ii, ikm);
            inner_kernel(ib, n, jb, &A(ii, rr), lda, &B(rr, 0), ldb, &C(ii, 0), ldc);
        }
    }
}

#define ikm 128
#define ikk 128
void gemm_pack_memory(int m, int n, int k, double *a, int lda,
                                      double *b, int ldb,
                                      double *c, int ldc)
{
    if (m < 2*ikk) {
        gemm_1x4(m, n, k, a, lda, b, ldb, c, ldc);
        return;
    }
    for (int l = 0; l < k; l += ikk) {
        int lb = min(k - l, ikk);
#ifdef _OPENMP
    // multi-thread via openmp
    #pragma omp parallel for schedule(dynamic)
#endif
        for (int s = 0; s < m; s += ikm) {
            int sb = min(m - s, ikm);
            inner_kernel_packAB(sb, n, lb, &A(s, l), lda, &B(l, 0), ldb, &C(s, 0), ldc);
        }
    }
}