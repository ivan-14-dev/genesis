/* GENESIS nn kernels (float64), written for this repository. genesis/nn/backend.py holds a
 * pure-Python twin of every function below and self-tests this library against it before use.
 *
 * Determinism: built with -ffp-contract=off and without -ffast-math, every output element is
 * produced by one fixed sequence of IEEE operations, whatever the tiling, the SIMD width
 * (scalar / 2-wide / AVX 4-wide, picked at run time) or the number of threads:
 *   - gemm accumulates C[i][j] += op(A)[i][p] * op(B)[p][j] for p = 0, 1, ... directly onto C;
 *   - every other reduction (sums, maxima, column sums) runs in increasing index order;
 *   - threads only split the output of a product into whole rows or columns, never a sum.
 * Callers pass consistent sizes; id / target vectors are bounds-checked here as well.
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- gemm */
#define GN_MR 4               /* micro-tile rows */
#define GN_NR 8               /* micro-tile columns */
#define GN_KC 256             /* depth of a packed block */
#define GN_MC 64              /* rows of a packed A block */
#define GN_NC 512             /* columns of a packed B block */
#define GN_MAX_THREADS 8
#define GN_THREAD_WORK 2e6     /* m*n*k below which a product stays on one thread */
#define GN_CHUNKS 1           /* output chunks per thread, claimed dynamically */

typedef void (*gn_micro)(size_t kb, const double *a, const double *b, double *c, size_t ldc);

typedef struct {
    const double *a, *b;
    double *c;
    size_t k, lda, ldb, ldc;
    size_t i0, i1, j0, j1;    /* output block [i0,i1) x [j0,j1) owned by this job */
    int ta, tb;
    gn_micro micro;
} gn_job;

/* C[0:MR, 0:NR] += Ap^T Bp over kb packed steps (Ap: kb x MR, Bp: kb x NR). */
static void gn_micro_scalar(size_t kb, const double *a, const double *b, double *c, size_t ldc)
{
    size_t i, j, p;
    for (i = 0; i < GN_MR; i++)
        for (j = 0; j < GN_NR; j++) {
            double s = c[i * ldc + j];
            for (p = 0; p < kb; p++)
                s += a[p * GN_MR + i] * b[p * GN_NR + j];
            c[i * ldc + j] = s;
        }
}

#if defined(__GNUC__)
typedef double gn_v2 __attribute__((vector_size(16)));

/* Two 4x4 halves so the 8 accumulators fit in SSE2 / NEON registers. */
static void gn_micro_v2(size_t kb, const double *a, const double *b, double *c, size_t ldc)
{
    size_t h, p;
    for (h = 0; h < GN_NR; h += 4) {
        double *c0 = c + h, *c1 = c0 + ldc, *c2 = c1 + ldc, *c3 = c2 + ldc;
        const double *ap = a, *bp = b + h;
        gn_v2 x00, x01, x10, x11, x20, x21, x30, x31, b0, b1, s;
        memcpy(&x00, c0, 16); memcpy(&x01, c0 + 2, 16);
        memcpy(&x10, c1, 16); memcpy(&x11, c1 + 2, 16);
        memcpy(&x20, c2, 16); memcpy(&x21, c2 + 2, 16);
        memcpy(&x30, c3, 16); memcpy(&x31, c3 + 2, 16);
        for (p = 0; p < kb; p++, ap += GN_MR, bp += GN_NR) {
            memcpy(&b0, bp, 16);
            memcpy(&b1, bp + 2, 16);
            s = (gn_v2){ap[0], ap[0]}; x00 += s * b0; x01 += s * b1;
            s = (gn_v2){ap[1], ap[1]}; x10 += s * b0; x11 += s * b1;
            s = (gn_v2){ap[2], ap[2]}; x20 += s * b0; x21 += s * b1;
            s = (gn_v2){ap[3], ap[3]}; x30 += s * b0; x31 += s * b1;
        }
        memcpy(c0, &x00, 16); memcpy(c0 + 2, &x01, 16);
        memcpy(c1, &x10, 16); memcpy(c1 + 2, &x11, 16);
        memcpy(c2, &x20, 16); memcpy(c2 + 2, &x21, 16);
        memcpy(c3, &x30, 16); memcpy(c3 + 2, &x31, 16);
    }
}
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#define GN_HAVE_AVX 1
typedef double gn_v4 __attribute__((vector_size(32)));

/* Same arithmetic with 4-wide AVX vectors; only called if the CPU and OS support AVX. */
__attribute__((target("avx"))) static void gn_micro_avx(size_t kb, const double *a, const double *b,
                                                        double *c, size_t ldc)
{
    double *c0 = c, *c1 = c0 + ldc, *c2 = c1 + ldc, *c3 = c2 + ldc;
    gn_v4 x00, x01, x10, x11, x20, x21, x30, x31, b0, b1, s;
    size_t p;
    memcpy(&x00, c0, 32); memcpy(&x01, c0 + 4, 32);
    memcpy(&x10, c1, 32); memcpy(&x11, c1 + 4, 32);
    memcpy(&x20, c2, 32); memcpy(&x21, c2 + 4, 32);
    memcpy(&x30, c3, 32); memcpy(&x31, c3 + 4, 32);
    for (p = 0; p < kb; p++, a += GN_MR, b += GN_NR) {
        memcpy(&b0, b, 32);
        memcpy(&b1, b + 4, 32);
        s = (gn_v4){a[0], a[0], a[0], a[0]}; x00 += s * b0; x01 += s * b1;
        s = (gn_v4){a[1], a[1], a[1], a[1]}; x10 += s * b0; x11 += s * b1;
        s = (gn_v4){a[2], a[2], a[2], a[2]}; x20 += s * b0; x21 += s * b1;
        s = (gn_v4){a[3], a[3], a[3], a[3]}; x30 += s * b0; x31 += s * b1;
    }
    memcpy(c0, &x00, 32); memcpy(c0 + 4, &x01, 32);
    memcpy(c1, &x10, 32); memcpy(c1 + 4, &x11, 32);
    memcpy(c2, &x20, 32); memcpy(c2 + 4, &x21, 32);
    memcpy(c3, &x30, 32); memcpy(c3 + 4, &x31, 32);
}
#endif

static int gn_simd = 0;       /* 0 = best available, 1 = scalar, 2 = 2-wide, 3 = AVX */
static int gn_threads = 1;

static int gn_best_simd(void)
{
#ifdef GN_HAVE_AVX
    if (__builtin_cpu_supports("avx"))
        return 3;
#endif
#if defined(__GNUC__)
    return 2;
#else
    return 1;
#endif
}

/* Select the micro-kernel (0 = best); returns the level in use. Results do not depend on it. */
int gn_set_simd(int level)
{
    int best = gn_best_simd();
    gn_simd = level <= 0 || level > best ? 0 : level;
    return gn_simd ? gn_simd : best;
}

int gn_set_threads(int n)
{
    gn_threads = n < 1 ? 1 : n > GN_MAX_THREADS ? GN_MAX_THREADS : n;
    return gn_threads;
}

static gn_micro gn_pick_micro(void)
{
    switch (gn_simd ? gn_simd : gn_best_simd()) {
#ifdef GN_HAVE_AVX
    case 3: return gn_micro_avx;
#endif
#if defined(__GNUC__)
    case 2: return gn_micro_v2;
#endif
    default: return gn_micro_scalar;
    }
}

#define GN_A(J, i, p) ((J)->ta ? (J)->a[(p) * (J)->lda + (i)] : (J)->a[(i) * (J)->lda + (p)])
#define GN_B(J, p, j) ((J)->tb ? (J)->b[(j) * (J)->ldb + (p)] : (J)->b[(p) * (J)->ldb + (j)])

static size_t gn_min(size_t x, size_t y) { return x < y ? x : y; }

/* Pack op(A)[ic:ic+mb, pc:pc+kb] into MR-row panels (kb x MR each, zero padded). */
static void gn_pack_a(const gn_job *J, size_t ic, size_t mb, size_t pc, size_t kb, double *dst)
{
    size_t ir, p, q;
    for (ir = 0; ir < mb; ir += GN_MR, dst += GN_MR * kb) {
        const size_t h = gn_min(GN_MR, mb - ir);
        if (h < GN_MR)
            memset(dst, 0, sizeof(double) * GN_MR * kb);
        if (J->ta)
            for (p = 0; p < kb; p++) {
                const double *src = J->a + (pc + p) * J->lda + ic + ir;
                for (q = 0; q < h; q++)
                    dst[p * GN_MR + q] = src[q];
            }
        else
            for (q = 0; q < h; q++) {
                const double *src = J->a + (ic + ir + q) * J->lda + pc;
                for (p = 0; p < kb; p++)
                    dst[p * GN_MR + q] = src[p];
            }
    }
}

/* Pack op(B)[pc:pc+kb, jc:jc+nb] into NR-column panels (kb x NR each, zero padded). */
static void gn_pack_b(const gn_job *J, size_t pc, size_t kb, size_t jc, size_t nb, double *dst)
{
    size_t jr, p, q;
    for (jr = 0; jr < nb; jr += GN_NR, dst += GN_NR * kb) {
        const size_t w = gn_min(GN_NR, nb - jr);
        if (w < GN_NR)
            memset(dst, 0, sizeof(double) * GN_NR * kb);
        if (J->tb)
            for (q = 0; q < w; q++) {
                const double *src = J->b + (jc + jr + q) * J->ldb + pc;
                for (p = 0; p < kb; p++)
                    dst[p * GN_NR + q] = src[p];
            }
        else
            for (p = 0; p < kb; p++) {
                const double *src = J->b + (pc + p) * J->ldb + jc + jr;
                for (q = 0; q < w; q++)
                    dst[p * GN_NR + q] = src[q];
            }
    }
}

static void gn_job_naive(const gn_job *J)
{
    size_t i, j, p;
    for (i = J->i0; i < J->i1; i++)
        for (j = J->j0; j < J->j1; j++) {
            double s = J->c[i * J->ldc + j];
            for (p = 0; p < J->k; p++)
                s += GN_A(J, i, p) * GN_B(J, p, j);
            J->c[i * J->ldc + j] = s;
        }
}

static void gn_job_run(const gn_job *J)
{
    const size_t kc = gn_min(GN_KC, J->k);
    const size_t mc = gn_min(GN_MC, J->i1 - J->i0), nc = gn_min(GN_NC, J->j1 - J->j0);
    double *ap = malloc(sizeof(double) * kc * ((mc + GN_MR - 1) / GN_MR * GN_MR));
    double *bp = malloc(sizeof(double) * kc * ((nc + GN_NR - 1) / GN_NR * GN_NR));
    size_t jc, pc, ic, jr, ir, i, j;
    if (!ap || !bp) {         /* same per-element arithmetic, just slower */
        free(ap);
        free(bp);
        gn_job_naive(J);
        return;
    }
    for (jc = J->j0; jc < J->j1; jc += GN_NC) {
        const size_t nb = gn_min(GN_NC, J->j1 - jc);
        for (pc = 0; pc < J->k; pc += GN_KC) {
            const size_t kb = gn_min(GN_KC, J->k - pc);
            gn_pack_b(J, pc, kb, jc, nb, bp);
            for (ic = J->i0; ic < J->i1; ic += GN_MC) {
                const size_t mb = gn_min(GN_MC, J->i1 - ic);
                gn_pack_a(J, ic, mb, pc, kb, ap);
                for (jr = 0; jr < nb; jr += GN_NR)
                    for (ir = 0; ir < mb; ir += GN_MR) {
                        double *c = J->c + (ic + ir) * J->ldc + jc + jr;
                        const size_t h = gn_min(GN_MR, mb - ir), w = gn_min(GN_NR, nb - jr);
                        if (h == GN_MR && w == GN_NR) {
                            J->micro(kb, ap + ir * kb, bp + jr * kb, c, J->ldc);
                        } else {   /* edge tile: work on a padded copy */
                            double t[GN_MR * GN_NR] = {0.0};
                            for (i = 0; i < h; i++)
                                for (j = 0; j < w; j++)
                                    t[i * GN_NR + j] = c[i * J->ldc + j];
                            J->micro(kb, ap + ir * kb, bp + jr * kb, t, GN_NR);
                            for (i = 0; i < h; i++)
                                for (j = 0; j < w; j++)
                                    c[i * J->ldc + j] = t[i * GN_NR + j];
                        }
                    }
            }
        }
    }
    free(ap);
    free(bp);
}

/* Work queue of a split product: chunk i covers output columns (or rows) [lo, hi) with
 * boundaries rounded to whole micro-tiles. Workers claim chunks in any order, but a chunk's
 * arithmetic never depends on who runs it, so the result is the same for any thread count. */
typedef struct {
    gn_job base;
    size_t len, unit, chunks, next;
    int cols;
    pthread_mutex_t lock;
} gn_queue;

static int gn_claim(gn_queue *Q, gn_job *J)
{
    size_t i, lo, hi;
    pthread_mutex_lock(&Q->lock);
    i = Q->next < Q->chunks ? Q->next++ : Q->chunks;
    pthread_mutex_unlock(&Q->lock);
    if (i >= Q->chunks)
        return 0;
    lo = (Q->len * i / Q->chunks + Q->unit - 1) / Q->unit * Q->unit;
    hi = gn_min((Q->len * (i + 1) / Q->chunks + Q->unit - 1) / Q->unit * Q->unit, Q->len);
    *J = Q->base;
    if (Q->cols) { J->j0 = lo; J->j1 = hi; } else { J->i0 = lo; J->i1 = hi; }
    return 1;
}

static void *gn_worker(void *arg)
{
    gn_job J;
    while (gn_claim((gn_queue *)arg, &J))
        if (J.i0 < J.i1 && J.j0 < J.j1)
            gn_job_run(&J);
    return NULL;
}

/* C (m x n, row stride ldc) [+]= op(A) @ op(B); op(A)[i][p] = ta ? A[p*lda+i] : A[i*lda+p],
 * op(B)[p][j] = tb ? B[j*ldb+p] : B[p*ldb+j]. Large products are cut into whole column (or row)
 * chunks shared by up to gn_threads threads; the caller always works too, so a helper that
 * fails to start or is not scheduled only costs speed. */
static void gn_gemm(size_t m, size_t k, size_t n, const double *a, size_t lda, int ta,
                    const double *b, size_t ldb, int tb, double *c, size_t ldc, int acc)
{
    gn_queue Q;
    pthread_t tid[GN_MAX_THREADS];
    int started[GN_MAX_THREADS] = {0};
    size_t i, parts = (size_t)gn_threads, tiles;
    if (!acc)
        for (i = 0; i < m; i++)
            memset(c + i * ldc, 0, sizeof(double) * n);
    if (m == 0 || n == 0 || k == 0)
        return;
    {
        gn_job J = {a, b, c, k, lda, ldb, ldc, 0, m, 0, n, ta, tb, gn_pick_micro()};
        Q.base = J;
    }
    Q.cols = n >= m;
    Q.len = Q.cols ? n : m;
    Q.unit = Q.cols ? GN_NR : GN_MR;
    tiles = (Q.len + Q.unit - 1) / Q.unit;
    if (parts < 2 || tiles < 2 || (double)m * (double)n * (double)k < GN_THREAD_WORK
            || pthread_mutex_init(&Q.lock, NULL)) {
        gn_job_run(&Q.base);
        return;
    }
    parts = gn_min(parts, tiles);
    Q.chunks = gn_min(parts * GN_CHUNKS, tiles);
    Q.next = 0;
    for (i = 1; i < parts; i++)
        started[i] = pthread_create(&tid[i], NULL, gn_worker, &Q) == 0;
    gn_worker(&Q);
    for (i = 1; i < parts; i++)
        if (started[i])
            pthread_join(tid[i], NULL);
    pthread_mutex_destroy(&Q.lock);
}

/* C (m x n) = op(A) @ op(B) (+ C if acc); A is m x k (k x m if ta), B is k x n (n x k if tb). */
void gn_matmul(const double *A, const double *B, double *C, int m, int k, int n, int ta, int tb, int acc)
{
    gn_gemm((size_t)m, (size_t)k, (size_t)n, A, ta ? (size_t)m : (size_t)k, ta, B, tb ? (size_t)k : (size_t)n,
            tb, C, (size_t)n, acc);
}

/* ---------------------------------------------------------------- element-wise */
void gn_add(long n, const double *a, const double *b, double *out)   /* out may alias a or b */
{
    long i;
    for (i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

void gn_add_rows(long T, long D, const double *x, const double *bias, double *out)
{
    long t, j;
    for (t = 0; t < T; t++)
        for (j = 0; j < D; j++)
            out[t * D + j] = x[t * D + j] + bias[j];
}

void gn_colsum(long T, long D, const double *g, double *out)
{
    long t, j;
    memset(out, 0, sizeof(double) * (size_t)D);
    for (t = 0; t < T; t++)
        for (j = 0; j < D; j++)
            out[j] += g[t * D + j];
}

void gn_scale(long n, const double *a, double s, double *out)
{
    long i;
    for (i = 0; i < n; i++)
        out[i] = a[i] * s;
}

double gn_sumsq(long n, const double *a)
{
    double s = 0.0;
    long i;
    for (i = 0; i < n; i++)
        s += a[i] * a[i];
    return s;
}

/* GELU, tanh form: th = tanh(c (x + k x^3)), y = 0.5 x (1 + th). */
void gn_gelu_fwd(long n, double c, double k, const double *x, double *y, double *th)
{
    long i;
    for (i = 0; i < n; i++) {
        const double v = x[i], h = tanh(c * (v + k * v * v * v));
        th[i] = h;
        y[i] = 0.5 * v * (1.0 + h);
    }
}

void gn_gelu_bwd(long n, double c, double k, const double *g, const double *x, const double *th, double *dx)
{
    long i;
    for (i = 0; i < n; i++) {
        const double v = x[i], h = th[i];
        dx[i] = g[i] * (0.5 * (1.0 + h) + 0.5 * v * (1.0 - h * h) * c * (1.0 + 3.0 * k * v * v));
    }
}

/* ---------------------------------------------------------------- layer norm */
void gn_layernorm_fwd(long T, long D, const double *x, const double *gamma, const double *beta, double eps,
                      double *out, double *xhat, double *rstd)
{
    long t, j;
    for (t = 0; t < T; t++) {
        const double *row = x + t * D;
        double s = 0.0, q = 0.0, mu, r;
        for (j = 0; j < D; j++)
            s += row[j];
        mu = s / (double)D;
        for (j = 0; j < D; j++)
            q += (row[j] - mu) * (row[j] - mu);
        r = 1.0 / sqrt(q / (double)D + eps);
        for (j = 0; j < D; j++) {
            const double h = (row[j] - mu) * r;
            xhat[t * D + j] = h;
            out[t * D + j] = h * gamma[j] + beta[j];
        }
        rstd[t] = r;
    }
}

/* Any of dx / dgamma / dbeta may be NULL. */
void gn_layernorm_bwd(long T, long D, const double *g, const double *xhat, const double *rstd,
                      const double *gamma, double *dx, double *dgamma, double *dbeta)
{
    long t, j;
    if (dgamma) {
        memset(dgamma, 0, sizeof(double) * (size_t)D);
        for (t = 0; t < T; t++)
            for (j = 0; j < D; j++)
                dgamma[j] += g[t * D + j] * xhat[t * D + j];
    }
    if (dbeta)
        gn_colsum(T, D, g, dbeta);
    if (dx)
        for (t = 0; t < T; t++) {
            const double *gr = g + t * D, *xh = xhat + t * D, r = rstd[t];
            double s1 = 0.0, s2 = 0.0, m1, m2;
            for (j = 0; j < D; j++) {
                const double d = gr[j] * gamma[j];
                s1 += d;
                s2 += d * xh[j];
            }
            m1 = s1 / (double)D;
            m2 = s2 / (double)D;
            for (j = 0; j < D; j++)
                dx[t * D + j] = r * (gr[j] * gamma[j] - m1 - xh[j] * m2);
        }
}

/* ---------------------------------------------------------------- causal attention */
#define GN_AB 32   /* query rows per causal block: rows [i0, i1) only see keys [0, i1) */

/* qkv [T, 3D] = [Q | K | V] (heads contiguous in each); P [H, T, T] row-softmax of
 * scale * Q_h K_h^T over keys u <= t (zero above the diagonal); y [T, D] = concat_h P_h V_h. */
void gn_attn_fwd(long T_, long D_, long H_, double scale, const double *qkv, double *y, double *P)
{
    const size_t T = (size_t)T_, D = (size_t)D_, H = (size_t)H_, hd = D / H, D3 = 3 * D;
    size_t h, i0, t, u;
    memset(y, 0, sizeof(double) * T * D);
    for (h = 0; h < H; h++) {
        double *ph = P + h * T * T;
        memset(ph, 0, sizeof(double) * T * T);
        for (i0 = 0; i0 < T; i0 += GN_AB) {
            const size_t i1 = gn_min(T, i0 + GN_AB);
            gn_gemm(i1 - i0, hd, i1, qkv + i0 * D3 + h * hd, D3, 0, qkv + D + h * hd, D3, 1, ph + i0 * T, T, 1);
        }
        for (t = 0; t < T; t++) {
            double *row = ph + t * T, mx = row[0], z = 0.0;
            for (u = 1; u <= t; u++)
                if (row[u] > mx)
                    mx = row[u];
            for (u = 0; u <= t; u++) {
                row[u] = exp((row[u] - mx) * scale);
                z += row[u];
            }
            for (u = 0; u <= t; u++)
                row[u] = row[u] / z;
            for (u = t + 1; u < T; u++)
                row[u] = 0.0;
        }
        for (i0 = 0; i0 < T; i0 += GN_AB) {
            const size_t i1 = gn_min(T, i0 + GN_AB);
            gn_gemm(i1 - i0, i1, hd, ph + i0 * T, T, 0, qkv + 2 * D + h * hd, D3, 0, y + i0 * D + h * hd, D, 1);
        }
    }
}

/* dqkv [T, 3D] from dy [T, D]; ds is T*T scratch. */
void gn_attn_bwd(long T_, long D_, long H_, double scale, const double *qkv, const double *P, const double *dy,
                 double *dqkv, double *ds)
{
    const size_t T = (size_t)T_, D = (size_t)D_, H = (size_t)H_, hd = D / H, D3 = 3 * D;
    size_t h, i0, t, u;
    memset(dqkv, 0, sizeof(double) * T * D3);
    for (h = 0; h < H; h++) {
        const double *ph = P + h * T * T;
        memset(ds, 0, sizeof(double) * T * T);
        for (i0 = 0; i0 < T; i0 += GN_AB) {   /* dP = dY_h V_h^T */
            const size_t i1 = gn_min(T, i0 + GN_AB);
            gn_gemm(i1 - i0, hd, i1, dy + i0 * D + h * hd, D, 0, qkv + 2 * D + h * hd, D3, 1, ds + i0 * T, T, 1);
        }
        for (t = 0; t < T; t++) {             /* dS = P * (dP - <P, dP>) * scale */
            const double *p = ph + t * T;
            double *d = ds + t * T, c = 0.0;
            for (u = 0; u <= t; u++)
                c += p[u] * d[u];
            for (u = 0; u <= t; u++)
                d[u] = p[u] * (d[u] - c) * scale;
            for (u = t + 1; u < T; u++)
                d[u] = 0.0;
        }
        for (i0 = 0; i0 < T; i0 += GN_AB) {
            const size_t i1 = gn_min(T, i0 + GN_AB);
            /* dQ_h[i0:i1] = dS[i0:i1, :i1] K_h[:i1] */
            gn_gemm(i1 - i0, i1, hd, ds + i0 * T, T, 0, qkv + D + h * hd, D3, 0, dqkv + i0 * D3 + h * hd, D3, 1);
            /* dK_h[i0:i1] = dS[i0:, i0:i1]^T Q_h[i0:];  dV_h[i0:i1] = P_h[i0:, i0:i1]^T dY_h[i0:] */
            gn_gemm(i1 - i0, T - i0, hd, ds + i0 * T + i0, T, 1, qkv + i0 * D3 + h * hd, D3, 0,
                    dqkv + i0 * D3 + D + h * hd, D3, 1);
            gn_gemm(i1 - i0, T - i0, hd, ph + i0 * T + i0, T, 1, dy + i0 * D + h * hd, D, 0,
                    dqkv + i0 * D3 + 2 * D + h * hd, D3, 1);
        }
    }
}

/* ---------------------------------------------------------------- cross-entropy, embedding */
/* Row-wise softmax of logits [T, V] into probs; *total = sum_t (logsumexp_t - logit[t, y_t]). */
int gn_xent_fwd(long T, long V, const double *logits, const int *targets, double *probs, double *total)
{
    long t, u;
    double acc = 0.0;
    for (t = 0; t < T; t++)
        if (targets[t] < 0 || targets[t] >= V)
            return -1;
    for (t = 0; t < T; t++) {
        const double *row = logits + t * V;
        double *pr = probs + t * V, mx = row[0], z = 0.0;
        for (u = 1; u < V; u++)
            if (row[u] > mx)
                mx = row[u];
        for (u = 0; u < V; u++) {
            pr[u] = exp(row[u] - mx);
            z += pr[u];
        }
        for (u = 0; u < V; u++)
            pr[u] = pr[u] / z;
        acc += mx + log(z) - row[targets[t]];
    }
    *total = acc;
    return 0;
}

/* d = s * (probs - onehot(targets)). */
int gn_xent_bwd(long T, long V, const double *probs, const int *targets, double s, double *d)
{
    long t;
    for (t = 0; t < T; t++)
        if (targets[t] < 0 || targets[t] >= V)
            return -1;
    gn_scale(T * V, probs, s, d);
    for (t = 0; t < T; t++)
        d[t * V + targets[t]] -= s;
    return 0;
}

int gn_embed_fwd(long T, long D, long V, const double *w, const int *ids, double *out)
{
    long t;
    for (t = 0; t < T; t++)
        if (ids[t] < 0 || ids[t] >= V)
            return -1;
    for (t = 0; t < T; t++)
        memcpy(out + t * D, w + (long)ids[t] * D, sizeof(double) * (size_t)D);
    return 0;
}

/* gw [V, D] += scatter of g [T, D] rows by ids, in increasing t. */
int gn_embed_bwd(long T, long D, long V, const double *g, const int *ids, double *gw)
{
    long t, j;
    for (t = 0; t < T; t++)
        if (ids[t] < 0 || ids[t] >= V)
            return -1;
    for (t = 0; t < T; t++) {
        double *row = gw + (long)ids[t] * D;
        for (j = 0; j < D; j++)
            row[j] += g[t * D + j];
    }
    return 0;
}

/* ---------------------------------------------------------------- AdamW */
void gn_adamw(long n, double *w, const double *g, double *m, double *v, double b1, double c1, double b2,
              double c2, double step, double inv_bc2, double eps, double keep)
{
    long i;
    for (i = 0; i < n; i++) {
        const double mi = b1 * m[i] + c1 * g[i], vi = b2 * v[i] + c2 * g[i] * g[i];
        m[i] = mi;
        v[i] = vi;
        w[i] = w[i] * keep - step * mi / (sqrt(vi * inv_bc2) + eps);
    }
}
