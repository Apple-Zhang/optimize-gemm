#include "gemm_utils.h"

void trivial_gemm(int m, int n, int k, double *a, int lda,
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

void my_gemm_1x4(int m, int n, int k, double *a, int lda,
                                      double *b, int ldb,
                                      double *c, int ldc)
{
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i++) {
            dot1x4(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
        }
    }
}

void my_gemm_4x4(int m, int n, int k, double *a, int lda,
                                      double *b, int ldb,
                                      double *c, int ldc)
{
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            dot4x4(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
        }
    }
}

void my_gemm_4x4_avx(int m, int n, int k, double *a, int lda,
                                      double *b, int ldb,
                                      double *c, int ldc)
{
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            dot4x4_avx(k, &A(i,0), lda, &B(0,j), ldb, &C(i,j), ldc);
        }
    }
}

#define min(xx,yy) (((xx) < (yy)) ? (xx) : (yy))

void my_gemm_4x4_ik(int m, int n, int k, double *a, int lda,
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

void my_gemm_fin(int m, int n, int k, double *a, int lda,
                                      double *b, int ldb,
                                      double *c, int ldc)
{
    for (int rr = 0; rr < k; rr += ikk) {
        int jb = min(k - rr, ikk);
#ifdef _OPENMP
    // multi-thread with openmp
    #pragma omp parallel for schedule(dynamic,1)
#endif
        for (int ii = 0; ii < m; ii += ikm) {
            int ib = min(m - ii, ikm);
            inner_kernel_packAB(ib, n, jb, &A(ii, rr), lda, &B(rr, 0), ldb, &C(ii, 0), ldc);
        }
    }
}