#include <math.h>
// #include <mmintrin.h>
// #include <xmmintrin.h>  // SSE
// #include <pmmintrin.h>  // SSE2
// #include <emmintrin.h>  // SSE3
#include <immintrin.h>  

// column-major order matrices
#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
#define ikm 256
#define ikk 256

typedef union {
  __m256d v;
  double d[4];
} f4d;

void dot1x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
    /* compute C(0, 0:3) with loop unrolling. */

    register double c0_reg, c1_reg, c2_reg, c3_reg;
    register double ar_reg;
    int r;

    double *bp0 = &B(0,0);
    double *bp1 = &B(0,1);
    double *bp2 = &B(0,2);
    double *bp3 = &B(0,3);

    // initialize with zero.
    c0_reg = 0.0;
    c1_reg = 0.0;
    c2_reg = 0.0;
    c3_reg = 0.0;

    for (r = 0; r < k; r += 4) {
        // unroll the loop
        ar_reg = A(0, r);
        c0_reg += ar_reg * (*bp0);
        c1_reg += ar_reg * (*bp1);
        c2_reg += ar_reg * (*bp2);
        c3_reg += ar_reg * (*bp3);

        ar_reg = A(0, r+1);
        c0_reg += ar_reg * *(bp0+1);
        c1_reg += ar_reg * *(bp1+1);
        c2_reg += ar_reg * *(bp2+1);
        c3_reg += ar_reg * *(bp3+1);

        ar_reg = A(0, r+2);
        c0_reg += ar_reg * *(bp0+2);
        c1_reg += ar_reg * *(bp1+2);
        c2_reg += ar_reg * *(bp2+2);
        c3_reg += ar_reg * *(bp3+2);

        ar_reg = A(0, r+3);
        c0_reg += ar_reg * *(bp0+3);
        c1_reg += ar_reg * *(bp1+3);
        c2_reg += ar_reg * *(bp2+3);
        c3_reg += ar_reg * *(bp3+3);

        bp0 += 4;
        bp1 += 4;
        bp2 += 4;
        bp3 += 4;
    }

    // update C(0, 0:3)
    C(0,0) += c0_reg;
    C(0,1) += c1_reg;
    C(0,2) += c2_reg;
    C(0,3) += c3_reg;
}

void dot4x4(int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
    /* Compute C(0:3, 0:3) */

    // let compiler determine what should be used as registers.
    register double 
    c00_reg = 0.0, c10_reg = 0.0, c20_reg = 0.0, c30_reg = 0.0, 
    c01_reg = 0.0, c11_reg = 0.0, c21_reg = 0.0, c31_reg = 0.0,
    c02_reg = 0.0, c12_reg = 0.0, c22_reg = 0.0, c32_reg = 0.0,
    c03_reg = 0.0, c13_reg = 0.0, c23_reg = 0.0, c33_reg = 0.0;

    register double
    a0r_reg,
    a1r_reg,
    a2r_reg,
    a3r_reg;

    register double
    br0_reg,
    br1_reg,
    br2_reg,
    br3_reg;

    double *br0_ptr = &B(0,0);
    double *br1_ptr = &B(0,1);
    double *br2_ptr = &B(0,2);
    double *br3_ptr = &B(0,3);

    for (int r = 0; r < k; r++) {
        // read the r-th row of B
        br0_reg = *br0_ptr++;
        br1_reg = *br1_ptr++;
        br2_reg = *br2_ptr++;
        br3_reg = *br3_ptr++;

        // read the r-th col of A
        a0r_reg = A(0,r);
        a1r_reg = A(1,r);
        a2r_reg = A(2,r);
        a3r_reg = A(3,r);

        // update C
        c00_reg += a0r_reg * br0_reg;
        c01_reg += a0r_reg * br1_reg;
        c02_reg += a0r_reg * br2_reg;
        c03_reg += a0r_reg * br3_reg;

        c10_reg += a1r_reg * br0_reg;
        c11_reg += a1r_reg * br1_reg;
        c12_reg += a1r_reg * br2_reg;
        c13_reg += a1r_reg * br3_reg;
        
        c20_reg += a2r_reg * br0_reg;
        c21_reg += a2r_reg * br1_reg;
        c22_reg += a2r_reg * br2_reg;
        c23_reg += a2r_reg * br3_reg;

        c30_reg += a3r_reg * br0_reg;
        c31_reg += a3r_reg * br1_reg;
        c32_reg += a3r_reg * br2_reg;
        c33_reg += a3r_reg * br3_reg;
    }

    C(0,0) += c00_reg; C(1,0) += c10_reg; C(2,0) += c20_reg; C(3,0) += c30_reg;
    C(0,1) += c01_reg; C(1,1) += c11_reg; C(2,1) += c21_reg; C(3,1) += c31_reg;
    C(0,2) += c02_reg; C(1,2) += c12_reg; C(2,2) += c22_reg; C(3,2) += c32_reg;
    C(0,3) += c03_reg; C(1,3) += c13_reg; C(2,3) += c23_reg; C(3,3) += c33_reg;
}

void dot4x4_avx(int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
    /* Compute C(0:3, 0:3) */

    f4d c0_reg; // C(0,0) to C(3,0)
    f4d c1_reg; // ...
    f4d c2_reg;
    f4d c3_reg;

    // the r-th col of A, i.e., A(0,r) to A(3,r)
    f4d ar_reg;

    // duplicate B
    f4d b0_reg, b1_reg, b2_reg, b3_reg;
    
    double *br0_ptr = &B(0,0);
    double *br1_ptr = &B(0,1);
    double *br2_ptr = &B(0,2);
    double *br3_ptr = &B(0,3);

    // initialize with zeros
    c0_reg.v = _mm256_setzero_pd();
    c1_reg.v = _mm256_setzero_pd();
    c2_reg.v = _mm256_setzero_pd();
    c3_reg.v = _mm256_setzero_pd(); 

    for (int r = 0; r < k; r++) {
        ar_reg.v = _mm256_load_pd(&A(0, r));

        b0_reg.v = _mm256_broadcast_sd(br0_ptr++);
        b1_reg.v = _mm256_broadcast_sd(br1_ptr++);
        b2_reg.v = _mm256_broadcast_sd(br2_ptr++);
        b3_reg.v = _mm256_broadcast_sd(br3_ptr++);

        c0_reg.v = _mm256_add_pd(c0_reg.v,
                   _mm256_mul_pd(ar_reg.v, b0_reg.v));
        c1_reg.v = _mm256_add_pd(c1_reg.v,
                   _mm256_mul_pd(ar_reg.v, b1_reg.v));
        c2_reg.v = _mm256_add_pd(c2_reg.v,
                   _mm256_mul_pd(ar_reg.v, b2_reg.v));
        c3_reg.v = _mm256_add_pd(c3_reg.v,
                   _mm256_mul_pd(ar_reg.v, b3_reg.v));
    }
    
    C(0,0) += c0_reg.d[0]; C(1,0) += c0_reg.d[1]; C(2,0) += c0_reg.d[2]; C(3,0) += c0_reg.d[3]; 
    C(0,1) += c1_reg.d[0]; C(1,1) += c1_reg.d[1]; C(2,1) += c1_reg.d[2]; C(3,1) += c1_reg.d[3];
    C(0,2) += c2_reg.d[0]; C(1,2) += c2_reg.d[1]; C(2,2) += c2_reg.d[2]; C(3,2) += c2_reg.d[3];
    C(0,3) += c3_reg.d[0]; C(1,3) += c3_reg.d[1]; C(2,3) += c3_reg.d[2]; C(3,3) += c3_reg.d[3];
}

void dot4x4_pack(int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
    /* Compute C(0:3, 0:3) */

    // let compiler determine what should be used as registers.
    register double 
    c00_reg = 0.0, c10_reg = 0.0, c20_reg = 0.0, c30_reg = 0.0, 
    c01_reg = 0.0, c11_reg = 0.0, c21_reg = 0.0, c31_reg = 0.0,
    c02_reg = 0.0, c12_reg = 0.0, c22_reg = 0.0, c32_reg = 0.0,
    c03_reg = 0.0, c13_reg = 0.0, c23_reg = 0.0, c33_reg = 0.0;

    register double
    a0r_reg,
    a1r_reg,
    a2r_reg,
    a3r_reg;

    register double
    br0_reg,
    br1_reg,
    br2_reg,
    br3_reg;

    for (int r = 0; r < k; r++) {
        // read the r-th row of B
        br0_reg = b[0];
        br1_reg = b[1];
        br2_reg = b[2];
        br3_reg = b[3];

        // read the r-th col of A
        a0r_reg = a[0];
        a1r_reg = a[1];
        a2r_reg = a[2];
        a3r_reg = a[3];

        a += 4; b += 4;

        // update C
        c00_reg += a0r_reg * br0_reg;
        c01_reg += a0r_reg * br1_reg;
        c02_reg += a0r_reg * br2_reg;
        c03_reg += a0r_reg * br3_reg;

        c10_reg += a1r_reg * br0_reg;
        c11_reg += a1r_reg * br1_reg;
        c12_reg += a1r_reg * br2_reg;
        c13_reg += a1r_reg * br3_reg;
        
        c20_reg += a2r_reg * br0_reg;
        c21_reg += a2r_reg * br1_reg;
        c22_reg += a2r_reg * br2_reg;
        c23_reg += a2r_reg * br3_reg;

        c30_reg += a3r_reg * br0_reg;
        c31_reg += a3r_reg * br1_reg;
        c32_reg += a3r_reg * br2_reg;
        c33_reg += a3r_reg * br3_reg;
    }

    C(0,0) += c00_reg; C(1,0) += c10_reg; C(2,0) += c20_reg; C(3,0) += c30_reg;
    C(0,1) += c01_reg; C(1,1) += c11_reg; C(2,1) += c21_reg; C(3,1) += c31_reg;
    C(0,2) += c02_reg; C(1,2) += c12_reg; C(2,2) += c22_reg; C(3,2) += c32_reg;
    C(0,3) += c03_reg; C(1,3) += c13_reg; C(2,3) += c23_reg; C(3,3) += c33_reg;
}

void dot4x4_avx_pack(int k, double *a, int lda, double *b, int ldb, double *c, int ldc) {
    /* Compute C(0:3, 0:3) */

    f4d c0_reg, c1_reg, c2_reg, c3_reg; // C(0,0) to C(3,0)

    // the r-th col of A, i.e., A(0,r) to A(3,r)
    f4d ar_reg;

    // duplicate B
    f4d b0_reg, b1_reg, b2_reg, b3_reg;

    // initialize with zeros
    c0_reg.v = _mm256_setzero_pd();
    c1_reg.v = _mm256_setzero_pd();
    c2_reg.v = _mm256_setzero_pd();
    c3_reg.v = _mm256_setzero_pd(); 

    for (int r = 0; r < k; r++) {
        ar_reg.v = _mm256_load_pd(a);
        a += 4;

        b0_reg.v = _mm256_broadcast_sd(b++);
        b1_reg.v = _mm256_broadcast_sd(b++);
        b2_reg.v = _mm256_broadcast_sd(b++);
        b3_reg.v = _mm256_broadcast_sd(b++);

        c0_reg.v = _mm256_add_pd(c0_reg.v,
                   _mm256_mul_pd(ar_reg.v, b0_reg.v));
        c1_reg.v = _mm256_add_pd(c1_reg.v,
                   _mm256_mul_pd(ar_reg.v, b1_reg.v));
        c2_reg.v = _mm256_add_pd(c2_reg.v,
                   _mm256_mul_pd(ar_reg.v, b2_reg.v));
        c3_reg.v = _mm256_add_pd(c3_reg.v,
                   _mm256_mul_pd(ar_reg.v, b3_reg.v));
    }
    
    C(0,0) += c0_reg.d[0]; C(1,0) += c0_reg.d[1]; C(2,0) += c0_reg.d[2]; C(3,0) += c0_reg.d[3]; 
    C(0,1) += c1_reg.d[0]; C(1,1) += c1_reg.d[1]; C(2,1) += c1_reg.d[2]; C(3,1) += c1_reg.d[3];
    C(0,2) += c2_reg.d[0]; C(1,2) += c2_reg.d[1]; C(2,2) += c2_reg.d[2]; C(3,2) += c2_reg.d[3];
    C(0,3) += c3_reg.d[0]; C(1,3) += c3_reg.d[1]; C(2,3) += c3_reg.d[2]; C(3,3) += c3_reg.d[3];
}

void inner_kernel(int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc)
{
    for (int j = 0; j < n; j += 4) {
        for (int i = 0; i < m; i += 4) {
            dot4x4_avx(k, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void do_packA(int k, double *a, int lda, double *adest) {
    for (int r = 0; r < k; r++) {
        double *a0r_ptr = &A(0, r);

        *adest++ = *a0r_ptr;
        *adest++ = *(a0r_ptr+1);
        *adest++ = *(a0r_ptr+2);
        *adest++ = *(a0r_ptr+3);
    }
}

void do_packB(int k, double *b, int ldb, double *bdest) {
    double *br0_ptr = &B(0,0);
    double *br1_ptr = &B(0,1);
    double *br2_ptr = &B(0,2);
    double *br3_ptr = &B(0,3);

    for (int r = 0; r < k; r++) {
        *bdest++ = *br0_ptr++;
        *bdest++ = *br1_ptr++;
        *bdest++ = *br2_ptr++;
        *bdest++ = *br3_ptr++;
    }
}


void inner_kernel_packA(int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc)
{
    double packA[m*k];
    for (int j = 0; j < n; j += 4) {
        // do_packB(k, &B(0, j), ldb, &packB[j*k]);
        for (int i = 0; i < m; i += 4) {
            if (j == 0) do_packA(k, &A(i, 0), lda, &packA[i*k]);
            dot4x4(k, &packA[i*k], 4, &B(0, j), ldb, &C(i, j), ldc);
        }
    }
}

void inner_kernel_packAB(int m, int n, int k, double *a, int lda, 
                                       double *b, int ldb,
                                       double *c, int ldc)
{
    double *packA = (double*)memalign(32, sizeof(double) * (m*k));
    double *packB = (double*)memalign(32, sizeof(double) * (k*n));
    for (int j = 0; j < n; j += 4) {
        do_packB(k, &B(0, j), ldb, &packB[j*k]);
        for (int i = 0; i < m; i += 4) {
            if (j == 0) do_packA(k, &A(i, 0), lda, &packA[i*k]);
            dot4x4_avx_pack(k, &packA[i*k], 4, &packB[j*k], k, &C(i, j), ldc);
        }
    }
    free(packA);
    free(packB);
}