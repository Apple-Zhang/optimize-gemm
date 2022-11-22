#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <omp.h>
#include <string.h>
#include "gemm.h"

// complile with:
// gcc -mavx -fopenmp -O3 gemm_test.c -o gemm_test

#include <sys/time.h>

void rand_matrix(double *a, int m, int n, int lda) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            a[i + j*lda] = 2.0 * drand48() - 1.0;
        }
    }
}

void print_matrix(const double *a, int m, int n, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.6f ", a[i + j*lda]);
        }
        putchar('\n');
    }
    putchar('\n');
}

// compare two matrices
int matrix_compare_bits(const double *x, const double *y, int m, int n, int ldx, int ldy) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            // if (x[i + j*ldx] != y[i + j*ldy]) {
            if (fabs(x[i + j*ldx] - y[i + j*ldy]) > 1e-10) {
                union {
                    double d;
                    long   l;
                } tmp1, tmp2;
                tmp1.d = x[i + j*ldx];
                tmp2.d = y[i + j*ldy];
                printf("%.6f != %.6f\n %lx v.s. %lx\n", tmp1.d, tmp2.d, tmp1.l, tmp2.l);
                return 0;
            } 
        }
    }
    return 1;
}

double matrix_compare(int m, int n, const double *a, int lda, const double *b, int ldb)
{
    int i, j;
    double err = 0.0, absdiff;

    for (int j = 0; j < n; j++) {   
        for (int i = 0; i < m; i++) {
            absdiff = fabs(a[i + j*lda] - b[i + j*ldb]);
            err = (absdiff > err ? absdiff : err);
        }
    }

    return err;
}


int main(int argc, char *argv[]) {
    int m, n, k;
    if (argc >= 2) {
        m = atoi(argv[1]);
        if (!m) {
            return -1;
        }
#ifdef _OPENMP
        int t;
        if (argc > 2) {
            t = atoi(argv[2]);
        }
        else {
            t = 2;
        }
        omp_set_num_threads(t);
        printf("#thread -> %d/%d\n", t, omp_get_num_procs());
#endif
    }
    else {
        m = 1024;
    }
    n = k = m;

    int lda, ldb, ldc;

    // make padding
    if (m % 4) {
        lda = m+4 - m%4;
        ldb = k+4 - k%4;
        ldc = m+4 - m%4;
    }
    else {
        lda = m;
        ldb = k;
        ldc = m;
    }

    double *a = (double *)memalign(32, sizeof(double) * lda*(k+3));
    double *b = (double *)memalign(32, sizeof(double) * ldb*(n+3));
    double *c = (double *)memalign(32, sizeof(double) * ldc*(n+3));
    double *d = (double *)memalign(32, sizeof(double) * ldc*(n+3));

    struct timeval start, end;
    double t1, t2;

    // clear up the array with zeros
    memset(a, 0, sizeof(double)*lda*(k+3));
    memset(b, 0, sizeof(double)*ldb*(n+3));
    memset(c, 0, sizeof(double)*ldc*(n+3));
    memset(d, 0, sizeof(double)*ldc*(n+3));

    // init matrix
    rand_matrix(a, m, k, lda);
    rand_matrix(b, k, n, ldb);

    /* compute the correct answer */ 
    gettimeofday(&start, NULL);
    gemm_trivial(m, n, k, a, lda, b, ldb, c, ldc);
    gettimeofday(&end, NULL);
    t1 = ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6;
    printf("Trivial:\t%.6f sec\n", t1);

    /* Cache */ 
    gettimeofday(&start, NULL);
    gemm_cache(m, n, k, a, lda, b, ldb, d, ldc);
    gettimeofday(&end, NULL);
    t2 = ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6;
    printf("Cache:\t\t%.6f sec\n", t2);

    /* Cache-1x4 */ 
    memset(d, 0, sizeof(double)*ldc*(n+3));
    gettimeofday(&start, NULL);
    gemm_1x4(m, n, k, a, lda, b, ldb, d, ldc);
    gettimeofday(&end, NULL);
    t2 = ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6;
    printf("1x4:\t\t%.6f sec\n", t2);

    /* Cache-4x4 */ 
    memset(d, 0, sizeof(double)*ldc*(n+3));
    gettimeofday(&start, NULL);
    gemm_4x4(m, n, k, a, lda, b, ldb, d, ldc);
    gettimeofday(&end, NULL);
    t2 = ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6;
    printf("4x4:\t\t%.6f sec\n", t2);

    /* Cache-4x4-avx */ 
    memset(d, 0, sizeof(double)*ldc*(n+3));
    gettimeofday(&start, NULL);
    gemm_4x4_avx(m, n, k, a, lda, b, ldb, d, ldc);
    gettimeofday(&end, NULL);
    t2 = ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6;
    printf("4x4-avx:\t%.6f sec\n", t2);

    /* Cache-4x4-ik */ 
    memset(d, 0, sizeof(double)*ldc*(n+3));
    gettimeofday(&start, NULL);
    gemm_4x4_ik(m, n, k, a, lda, b, ldb, d, ldc);
    gettimeofday(&end, NULL);
    t2 = ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6;
    printf("Block:\t\t%.6f sec\n", t2);

    /* packOpenMP */ 
    memset(d, 0, sizeof(double)*ldc*(n+3));
    gettimeofday(&start, NULL);
    gemm_pack_memory(m, n, k, a, lda, b, ldb, d, ldc);
    gettimeofday(&end, NULL);
    t2 = ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6;
    printf("packOpenMP:\t%.6f sec\n", t2);

    // check the result
    printf("\nmax elem-wise error = %g\n", matrix_compare(m, n, c, ldc, d, ldc));
    if (!matrix_compare_bits(c, d, m, n, ldc, ldc)) {
        puts("ERROR");
    }
    else {
        puts("CORRECT");
        printf("final speedup: %.6fx\n", t1/t2);
    }

    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}
