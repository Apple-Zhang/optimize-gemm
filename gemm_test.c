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

void print_matrix(double *a, int m, int n, int lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.6f ", a[i + j*lda]);
        }
        putchar('\n');
    }
    putchar('\n');
}

// compare two matrices
int matrix_compare_bits(double *x, double *y, int m, int n, int ldx, int ldy) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            // if (x[i + j*ldx] != y[i + j*ldy]) {
            if (abs(x[i + j*ldx] - y[i + j*ldy]) > 1e-10) {
                union {
                    double    d;
                    long long l;
                } tmp1, tmp2;
                tmp1.d = x[i + j*ldx];
                tmp2.d = y[i + j*ldy];
                printf("%.6f != %.6f\n %llx v.s. %llx\n", tmp1.d, tmp2.d, tmp1.l, tmp2.l);
                return 0;
            } 
        }
    }
    return 1;
}

double matrix_compare(int m, int n, double *a, int lda, double *b, int ldb)
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
        printf("#thread -> %d\n", t);
#endif
    }
    else {
        m = 1024;
    }
    n = k = m;

    int lda, ldb, ldc;
    lda = (m > 1000) ? m : 1000;
    ldb = (k > 1000) ? k : 1000;
    ldc = (m > 1000) ? m : 1000;

    double *a = (double *)memalign(32, sizeof(double) * lda*(k+1));
    double *b = (double *)memalign(32, sizeof(double) * ldb*n);
    double *c = (double *)memalign(32, sizeof(double) * ldc*n);
    double *d = (double *)memalign(32, sizeof(double) * ldc*n);

    struct timeval start, end;

    // init matrix
    rand_matrix(a, m, k, lda);
    rand_matrix(b, k, n, ldb);

    // clear destination matrix
    memset(c, 0, sizeof(double)*ldc*n);
    memset(d, 0, sizeof(double)*ldc*n);

    // compute the correct answer
    // gettimeofday(&start, NULL);
    // trivial_gemm(m, n, k, a, lda, b, ldb, c, ldc);
    // gettimeofday(&end, NULL);
    // printf("Trivial: %.6f sec\n", ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6);

    // compute the accelerated method
    gettimeofday(&start, NULL);
    my_gemm_fin(m, n, k, a, lda, b, ldb, d, ldc);
    gettimeofday(&end, NULL);
    printf("Improved: %.6f sec\n", ((end.tv_sec-start.tv_sec)*1.0e6 + end.tv_usec-start.tv_usec) / 1.0e6);

    // check the result
    if (!matrix_compare_bits(c, d, m, n, ldc, ldc)) {
        puts("ERROR");
    }
    else {
        puts("CORRECT");
    }
    printf("max elem-wise error = %g\n", matrix_compare(m, n, c, ldc, d, ldc));

    // print_matrix(a, m, k, lda);
    // print_matrix(b, k, n, ldb);
    // print_matrix(c, 8, 8, ldc);
    // print_matrix(d, 8, 8, ldc);

    free(a);
    free(b);
    free(c);
    free(d);
    return 0;
}
