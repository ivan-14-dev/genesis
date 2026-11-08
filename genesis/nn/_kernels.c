/* GENESIS nn kernels: dense float64 matrix multiply, written for this repo.
 *
 * C (m x n) = op(A) @ op(B), op(A) is m x k, op(B) is k x n, all row-major.
 * ta: A is stored k x m (use its transpose); tb: B is stored n x k.
 * acc: add into C instead of overwriting it.
 *
 * Every C[i][j] is accumulated over p in increasing order with plain
 * multiply/add (built with -ffp-contract=off, no -ffast-math), so results are
 * deterministic and do not depend on the tiling below.
 */
#include <stddef.h>
#include <string.h>

#define GN_KB 64
#define GN_NB 256

void gn_matmul(const double *restrict A, const double *restrict B, double *restrict C,
               int m_, int k_, int n_, int ta, int tb, int acc)
{
    size_t m = (size_t)m_, k = (size_t)k_, n = (size_t)n_;
    size_t i, j, p, p0, pn, j0, jn;
    if (!acc)
        memset(C, 0, sizeof(double) * m * n);
    if (!tb) {
        /* C[i, j0:jn] += a_ip * B[p, j0:jn]: contiguous streams, B tile reused across rows. */
        for (j0 = 0; j0 < n; j0 += GN_NB) {
            jn = j0 + GN_NB < n ? j0 + GN_NB : n;
            for (p0 = 0; p0 < k; p0 += GN_KB) {
                pn = p0 + GN_KB < k ? p0 + GN_KB : k;
                for (i = 0; i < m; i++) {
                    double *c = C + i * n;
                    for (p = p0; p < pn; p++) {
                        const double a = ta ? A[p * m + i] : A[i * k + p];
                        const double *b = B + p * n;
                        for (j = j0; j < jn; j++)
                            c[j] += a * b[j];
                    }
                }
            }
        }
        return;
    }
    /* B stored n x k: dot products of op(A) rows with rows of B, four columns at a time. */
    for (i = 0; i < m; i++) {
        double *c = C + i * n;
        const size_t as = ta ? m : 1;
        const double *a = ta ? A + i : A + i * k;
        for (j = 0; j + 4 <= n; j += 4) {
            const double *b0 = B + j * k, *b1 = b0 + k, *b2 = b1 + k, *b3 = b2 + k;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (p = 0; p < k; p++) {
                const double av = a[p * as];
                s0 += av * b0[p];
                s1 += av * b1[p];
                s2 += av * b2[p];
                s3 += av * b3[p];
            }
            c[j] += s0;
            c[j + 1] += s1;
            c[j + 2] += s2;
            c[j + 3] += s3;
        }
        for (; j < n; j++) {
            const double *b0 = B + j * k;
            double s0 = 0.0;
            for (p = 0; p < k; p++)
                s0 += a[p * as] * b0[p];
            c[j] += s0;
        }
    }
}
