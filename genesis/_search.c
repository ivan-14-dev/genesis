/* GENESIS program-search kernel (see genesis/search.py, which holds the pure-Python reference).
 *
 * Bottom-up enumeration of expressions over the innate primitives add/sub/mul/mod, with observational-
 * equivalence pruning, followed by scoring against the observed outputs. The semantics replicate
 * genesis.expressions.enumerate_programs and genesis.innate exactly:
 *   - add/sub/mul are exact integer operations, "undefined" outside [-1e18, 1e18];
 *   - mod is Python's remainder (sign of the divisor) by a constant > 1;
 *   - a candidate whose output on any observation is undefined is dropped;
 *   - candidates up to keep_all_upto nodes are all kept, larger ones only if their output signature is new;
 *   - enumeration order, commutativity and triviality pruning, candidate budget and counters are identical.
 * Deepening (only when nothing enumerated fits well) reaches larger expressions without enumerating them:
 * bias fits "e + r" and remainders "e % c" on the level two nodes below, and a meet-in-the-middle step that,
 * for each candidate A and operator, computes the output B must have (y - A, A - y, y + A, y / A) on a window
 * of observations and looks it up in a hash table of the enumerated signatures. When that lower level is not
 * enumerated either, an affine meet in the middle finds "(A op B) + c" by looking B up up to a constant shift
 * (window differences), c being fitted. Sizes grow by two (iterative deepening) until one explains the data.
 * Every hit is checked on all observations against the mismatch tolerance, so noisy observations inside one
 * window only cost a miss.
 * The kernel never allocates more than the caller-sized arena and only reads/writes caller buffers.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LIMIT 1000000000000000000LL
#define SAFE_ABS (1LL << 61) /* inputs are at most 2^61 (genesis.search._SAFE): differences fit in int64 */
enum { OP_ADD = 0, OP_SUB = 1, OP_MUL = 2, OP_MOD = 3 };

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t hash_row(const int64_t *row, int n) {
    uint64_t h = 0x9E3779B97F4A7C15ULL;
    for (int t = 0; t < n; t++) {
        h ^= (uint64_t)row[t] * 0xff51afd7ed558ccdULL;
        h = ((h << 31) | (h >> 33)) * 0xc4ceb9fe1a85ec53ULL;
    }
    return h ^ (h >> 29);
}

typedef struct {
    int32_t *slots;  /* node index + 1, 0 = empty */
    uint64_t mask;
} table_t;

static int table_find(const table_t *tb, const uint64_t *hashes, const int64_t *sigs, int n,
                      const int64_t *row, uint64_t h) {
    for (uint64_t i = h & tb->mask;; i = (i + 1) & tb->mask) {
        int32_t s = tb->slots[i];
        if (!s) return 0;
        if (hashes[s - 1] == h && memcmp(sigs + (int64_t)(s - 1) * n, row, sizeof(int64_t) * (size_t)n) == 0)
            return 1;
    }
}

static void table_insert(table_t *tb, int32_t node, uint64_t h) {
    uint64_t i = h & tb->mask;
    while (tb->slots[i]) i = (i + 1) & tb->mask;
    tb->slots[i] = node + 1;
}

static int trivial(int op, int a_const, int64_t a_val, int b_const, int64_t b_val, int same) {
    if (a_const && b_const) return 1;
    if ((op == OP_ADD || op == OP_SUB) && b_const && b_val == 0) return 1;
    if (op == OP_ADD && a_const && a_val == 0) return 1;
    if (op == OP_MUL && ((a_const && (a_val == 0 || a_val == 1)) || (b_const && (b_val == 0 || b_val == 1)))) return 1;
    if (op == OP_SUB && same) return 1;
    if (op == OP_MOD && !(b_const && b_val > 1)) return 1;
    return 0;
}

/* Returns 1 and fills out[] if defined on every observation, else 0. */
static int apply(int op, const int64_t *a, const int64_t *b, int n, int64_t *out) {
    int64_t r;
    switch (op) {
    case OP_ADD:
        for (int t = 0; t < n; t++) {
            if (__builtin_add_overflow(a[t], b[t], &r) || r < -LIMIT || r > LIMIT) return 0;
            out[t] = r;
        }
        return 1;
    case OP_SUB:
        for (int t = 0; t < n; t++) {
            if (__builtin_sub_overflow(a[t], b[t], &r) || r < -LIMIT || r > LIMIT) return 0;
            out[t] = r;
        }
        return 1;
    case OP_MUL:
        for (int t = 0; t < n; t++) {
            if (__builtin_mul_overflow(a[t], b[t], &r) || r < -LIMIT || r > LIMIT) return 0;
            out[t] = r;
        }
        return 1;
    case OP_MOD:
        for (int t = 0; t < n; t++) {
            int64_t m = b[t];
            if (m <= 0) return 0;
            r = a[t] % m;
            out[t] = r < 0 ? r + m : r;
        }
        return 1;
    }
    return 0;
}

static int combine(int op, int64_t a, int64_t b, int64_t *out) {
    int64_t r;
    switch (op) {
    case OP_ADD: if (__builtin_add_overflow(a, b, &r) || r < -LIMIT || r > LIMIT) return 0; break;
    case OP_SUB: if (__builtin_sub_overflow(a, b, &r) || r < -LIMIT || r > LIMIT) return 0; break;
    case OP_MUL: if (__builtin_mul_overflow(a, b, &r) || r < -LIMIT || r > LIMIT) return 0; break;
    case OP_MOD: if (b <= 0) return 0; r = a % b; if (r < 0) r += b; break;
    default: return 0;
    }
    *out = r;
    return 1;
}

/* Mismatches of op(row_l, row_r) against ys over all observations, stopping once above tol. */
static int verify(int op, const int64_t *l, const int64_t *r, const int64_t *ys, int n, int tol) {
    int bad = 0;
    for (int t = 0; t < n && bad <= tol; t++) {
        int64_t v;
        if (!combine(op, l[t], r[t], &v) || v != ys[t]) bad++;
    }
    return bad;
}

/* Mismatches of (op(row_l, row_r) + c) against ys, stopping once above tol. */
static int verify_affine(int op, const int64_t *l, const int64_t *r, int64_t c, const int64_t *ys, int n, int tol) {
    int bad = 0;
    for (int t = 0; t < n && bad <= tol; t++) {
        int64_t v, w;
        if (!combine(op, l[t], r[t], &v) || !combine(OP_ADD, v, c, &w) || w != ys[t]) bad++;
    }
    return bad;
}

#define CHAIN_MAX 64

/*
 * n            observations; ys[n] observed outputs
 * n_leaves     leaves (variables then constants), leaf_sigs[n_leaves * n], leaf_const[n_leaves] (0/1)
 * ops[n_ops]   operator codes in enumeration order
 * cap          arena capacity in nodes (>= n_leaves + 1); max_candidates is the candidate budget
 * node_*[cap]  out: node table (op = -1 for leaves; l/r = child node indices); sigs[cap * n] scratch arena
 * counts[max_size + 1], stats[4] = {evaluated, truncated, n_candidates, deepening ran}
 * match_*      out: nodes with <= tol mismatches, in enumeration order (capacity cap)
 * off_*        out: bias fits "node + r" (r != 0) with n - count(y - node == r) <= tol (capacity cap)
 * deepen_bad   deepen only if no match/offset has <= deepen_bad mismatches (-1: never)
 * deep_size    size of the deepened expressions; deep_max: iterative deepening up to this size (step 2)
 * win_rows[n_win * win_len] observation windows for the lookup
 * deep_*       out: kind -2 = bias (l + val), otherwise operator code with children l, r, plus val when
 *              val != 0 (affine: "(l op r) + val") (capacity cap)
 * Returns 0 on success, -1 on allocation failure.
 */
int gn_search(int n, const int64_t *ys, int n_leaves, const int64_t *leaf_sigs, const int32_t *leaf_const,
              int max_size, int keep_all_upto, int n_ops, const int32_t *ops, int64_t max_candidates,
              double max_seconds, int tol, int bias_max_size, int64_t cap,
              int32_t *node_op, int32_t *node_l, int32_t *node_r, int32_t *node_size, int64_t *sigs,
              int64_t *counts, int64_t *stats,
              int32_t *match_node, int32_t *match_bad, int64_t *n_match,
              int32_t *off_node, int32_t *off_bad, int64_t *off_val, int64_t *n_off,
              int deepen_bad, int deep_size, int deep_max, int n_win, int win_len, const int32_t *win_rows,
              int32_t *deep_kind, int32_t *deep_l, int32_t *deep_r, int32_t *deep_bad, int64_t *deep_val,
              int64_t *n_deep) {
    double t0 = now_s();
    int64_t evaluated = 0, ncand = 0;
    int truncated = 0;
    uint64_t tsize = 1;
    while (tsize < (uint64_t)cap * 2) tsize <<= 1;
    table_t tb;
    tb.slots = (int32_t *)calloc(tsize, sizeof(int32_t));
    tb.mask = tsize - 1;
    uint64_t *hashes = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)cap);
    int8_t *is_const = (int8_t *)calloc((size_t)cap, 1);
    int64_t lv_start[64], lv_end[64];
    if (!tb.slots || !hashes || !is_const || max_size >= 64 || n_leaves + 1 > cap) {
        free(tb.slots); free(hashes); free(is_const);
        return -1;
    }
    for (int s = 0; s <= max_size; s++) { lv_start[s] = lv_end[s] = 0; counts[s] = 0; }

    for (int k = 0; k < n_leaves; k++) {
        int64_t *row = sigs + ncand * n;
        memcpy(row, leaf_sigs + (int64_t)k * n, sizeof(int64_t) * (size_t)n);
        uint64_t h = hash_row(row, n);
        hashes[ncand] = h;
        if (!table_find(&tb, hashes, sigs, n, row, h)) table_insert(&tb, (int32_t)ncand, h);
        node_op[ncand] = -1; node_l[ncand] = k; node_r[ncand] = -1; node_size[ncand] = 1;
        is_const[ncand] = (int8_t)(leaf_const[k] != 0);
        ncand++;
    }
    counts[1] = n_leaves;
    lv_start[1] = 0; lv_end[1] = ncand;

    for (int size = 3; size <= max_size && !truncated; size += 2) {
        int64_t constructed = 0, start = ncand;
        for (int ls = 1; ls < size - 1 && !truncated; ls += 2) {
            int rs = size - 1 - ls;
            for (int pi = 0; pi < n_ops && !truncated; pi++) {
                int op = ops[pi];
                int comm = (op == OP_ADD || op == OP_MUL);
                if (comm && ls > rs) continue;
                if (op == OP_MOD && rs != 1) continue; /* divisor must be a constant leaf */
                for (int64_t i = lv_start[ls]; i < lv_end[ls] && !truncated; i++) {
                    int64_t j0 = (comm && ls == rs) ? lv_start[rs] + (i - lv_start[ls]) : lv_start[rs];
                    int a_const = is_const[i];
                    int64_t a_val = a_const ? sigs[i * n] : 0;
                    for (int64_t j = j0; j < lv_end[rs]; j++) {
                        int b_const = is_const[j];
                        int64_t b_val = b_const ? sigs[j * n] : 0;
                        if (trivial(op, a_const, a_val, b_const, b_val, i == j)) continue;
                        constructed++;
                        evaluated++;
                        int64_t *row = sigs + ncand * n;
                        if (!apply(op, sigs + i * n, sigs + j * n, n, row)) continue;
                        uint64_t h = hash_row(row, n);
                        int dup = table_find(&tb, hashes, sigs, n, row, h);
                        if (size > keep_all_upto && dup) continue;
                        hashes[ncand] = h;
                        if (!dup) table_insert(&tb, (int32_t)ncand, h);
                        node_op[ncand] = op; node_l[ncand] = (int32_t)i; node_r[ncand] = (int32_t)j;
                        node_size[ncand] = size;
                        ncand++;
                        if (ncand >= max_candidates || ncand >= cap ||
                            ((ncand & 255) == 0 && now_s() - t0 > max_seconds)) {
                            truncated = 1;
                            break;
                        }
                    }
                }
            }
        }
        counts[size] = constructed > 0 ? constructed : 1;
        lv_start[size] = start; lv_end[size] = ncand;
    }

    int64_t nm = 0, no = 0;
    for (int64_t k = 0; k < ncand; k++) {
        const int64_t *row = sigs + k * n;
        int bad = 0;
        for (int t = 0; t < n && bad <= tol; t++) bad += row[t] != ys[t];
        if (bad <= tol) { match_node[nm] = (int32_t)k; match_bad[nm] = bad; nm++; }
        if (n < 2 || is_const[k] || node_size[k] > bias_max_size) continue;
        /* Majority residual (Boyer-Moore): unique whenever it covers more than half of the observations. */
        int64_t cand = 0, votes = 0;
        if (tol == 0) {
            cand = ys[0] - row[0];
        } else {
            for (int t = 0; t < n; t++) {
                int64_t r = ys[t] - row[t];
                if (votes == 0) { cand = r; votes = 1; } else votes += (r == cand) ? 1 : -1;
            }
        }
        if (cand == 0) continue;
        int miss = 0;
        for (int t = 0; t < n && miss <= tol; t++) miss += (ys[t] - row[t]) != cand;
        if (miss <= tol) { off_node[no] = (int32_t)k; off_bad[no] = miss; off_val[no] = cand; no++; }
    }
    *n_match = nm;
    *n_off = no;
    stats[0] = evaluated; stats[1] = truncated; stats[2] = ncand; stats[3] = 0;
    *n_deep = 0;

    int best = n + 1;
    for (int64_t i = 0; i < nm; i++) if (match_bad[i] < best) best = match_bad[i];
    for (int64_t i = 0; i < no; i++) if (off_bad[i] < best) best = off_bad[i];
    int top = 1;
    for (int s = 1; s <= max_size; s += 2) if (lv_end[s] > lv_start[s]) top = s;
    int rc = 0;
    if (deepen_bad >= 0 && best > deepen_bad && deep_size > top && deep_size - 1 - top >= 1) {
        stats[3] = 1;
        int64_t nd = 0;
        int has_mod = 0;
        for (int pi = 0; pi < n_ops; pi++) has_mod |= ops[pi] == OP_MOD;
        uint64_t wsize = 1;
        while (wsize < (uint64_t)ncand * 2 + 2) wsize <<= 1;
        /* Two window indexes: exact window outputs (for A op B) and outputs up to a shift (for A op B + c). */
        int32_t *wslots = (int32_t *)malloc(sizeof(int32_t) * wsize);
        int32_t *dslots = (int32_t *)malloc(sizeof(int32_t) * wsize);
        int32_t *wnext = (int32_t *)malloc(sizeof(int32_t) * (size_t)(ncand + 1));
        int32_t *wtail = (int32_t *)malloc(sizeof(int32_t) * (size_t)(ncand + 1));
        int32_t *dnext = (int32_t *)malloc(sizeof(int32_t) * (size_t)(ncand + 1));
        int32_t *dtail = (int32_t *)malloc(sizeof(int32_t) * (size_t)(ncand + 1));
        uint64_t *whash = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(ncand + 1));
        uint64_t *dhash = (uint64_t *)malloc(sizeof(uint64_t) * (size_t)(ncand + 1));
        int64_t *need = (int64_t *)malloc(sizeof(int64_t) * (size_t)(win_len + 1));
        if (!wslots || !dslots || !wnext || !wtail || !dnext || !dtail || !whash || !dhash || !need) rc = -1;
        int timed_out = 0;
        /* Iterative deepening: deep_size, deep_size + 2, ... deep_max, stopping at the first size that explains
         * all but deepen_bad observations. */
        for (int ds = deep_size; rc == 0 && ds <= deep_max && nd < cap && !timed_out; ds += 2) {
            int64_t nd0 = nd;
            int last = ds - 2; /* level whose bias / mod fits have ds nodes */
            /* 1. Bias fits on the level just below ds. */
            if (last <= max_size && n >= 2) {
                for (int64_t k = lv_start[last]; k < lv_end[last] && nd < cap; k++) {
                    const int64_t *row = sigs + k * n;
                    int64_t cand = 0, votes = 0;
                    for (int t = 0; t < n; t++) {
                        int64_t r = ys[t] - row[t];
                        if (votes == 0) { cand = r; votes = 1; } else votes += (r == cand) ? 1 : -1;
                    }
                    if (tol == 0) cand = ys[0] - row[0];
                    if (cand == 0) continue;
                    int miss = 0;
                    for (int t = 0; t < n && miss <= tol; t++) miss += (ys[t] - row[t]) != cand;
                    if (miss <= tol) { deep_kind[nd] = -2; deep_l[nd] = (int32_t)k; deep_r[nd] = -1;
                                       deep_bad[nd] = miss; deep_val[nd] = cand; nd++; }
                }
                /* 2. Remainders of that level by a constant leaf. */
                for (int64_t k = lv_start[last]; has_mod && k < lv_end[last] && nd < cap; k++) {
                    for (int64_t c = lv_start[1]; c < lv_end[1] && nd < cap; c++) {
                        if (!is_const[c] || sigs[c * n] <= 1) continue;
                        int bad = verify(OP_MOD, sigs + k * n, sigs + c * n, ys, n, tol);
                        if (bad <= tol) { deep_kind[nd] = OP_MOD; deep_l[nd] = (int32_t)k; deep_r[nd] = (int32_t)c;
                                          deep_bad[nd] = bad; deep_val[nd] = 0; nd++; }
                    }
                }
            }
            /* When level ds - 2 is not enumerated, its bias fits come from an affine meet in the middle. */
            int affine = last > max_size;
            /* 3. Meet in the middle on each window: A op B (+ c) = y => B's window outputs follow from A's. */
            for (int w = 0; w < n_win && nd < cap && !timed_out; w++) {
                const int32_t *rows = win_rows + (int64_t)w * win_len;
                memset(wslots, 0, sizeof(int32_t) * wsize);
                if (affine) memset(dslots, 0, sizeof(int32_t) * wsize);
                for (int64_t k = 0; k < ncand; k++) {
                    const int64_t *S = sigs + k * n;
                    uint64_t h = 0x9E3779B97F4A7C15ULL;
                    for (int i = 0; i < win_len; i++) {
                        h ^= (uint64_t)S[rows[i]] * 0xff51afd7ed558ccdULL;
                        h = ((h << 31) | (h >> 33)) * 0xc4ceb9fe1a85ec53ULL;
                    }
                    h ^= h >> 29;
                    whash[k] = h;
                    wnext[k] = -1;
                    for (uint64_t i = h & (wsize - 1);; i = (i + 1) & (wsize - 1)) {
                        int32_t s = wslots[i];
                        if (!s) { wslots[i] = (int32_t)k + 1; wtail[k] = (int32_t)k; break; }
                        int64_t hd = s - 1;
                        if (whash[hd] != h) continue;
                        int eq = 1;
                        for (int q = 0; q < win_len && eq; q++) eq = sigs[hd * n + rows[q]] == S[rows[q]];
                        if (eq) { wnext[wtail[hd]] = (int32_t)k; wtail[hd] = (int32_t)k; break; }
                    }
                    if (!affine) continue;
                    h = 0x9E3779B97F4A7C15ULL;
                    for (int i = 1; i < win_len; i++) {
                        h ^= (uint64_t)(S[rows[i]] - S[rows[0]]) * 0xff51afd7ed558ccdULL;
                        h = ((h << 31) | (h >> 33)) * 0xc4ceb9fe1a85ec53ULL;
                    }
                    h ^= h >> 29;
                    dhash[k] = h;
                    dnext[k] = -1;
                    for (uint64_t i = h & (wsize - 1);; i = (i + 1) & (wsize - 1)) {
                        int32_t s = dslots[i];
                        if (!s) { dslots[i] = (int32_t)k + 1; dtail[k] = (int32_t)k; break; }
                        int64_t hd = s - 1;
                        if (dhash[hd] != h) continue;
                        const int64_t *H = sigs + hd * n;
                        int eq = 1;
                        for (int q = 1; q < win_len && eq; q++)
                            eq = H[rows[q]] - H[rows[0]] == S[rows[q]] - S[rows[0]];
                        if (eq) { dnext[dtail[hd]] = (int32_t)k; dtail[hd] = (int32_t)k; break; }
                    }
                }
                for (int64_t a = 0; a < ncand && nd < cap; a++) {
                    if ((a & 1023) == 0 && now_s() - t0 > max_seconds) { stats[1] = 1; timed_out = 1; break; }
                    int sa = node_size[a];
                    const int64_t *A = sigs + a * n;
                    int want = ds - 1 - sa;
                    /* (op, a on the left?) in a fixed order: add, sub(A - B), sub(B - A), mul */
                    for (int variant = 0; want >= 1 && want <= top && variant < 4 && nd < cap; variant++) {
                        int op = variant == 0 ? OP_ADD : variant == 3 ? OP_MUL : OP_SUB;
                        int a_left = variant != 2;
                        int allowed = 0;
                        for (int pi = 0; pi < n_ops; pi++) allowed |= ops[pi] == op;
                        /* A commutative pair is kept with its smaller side on the left (checked below), so a
                         * left side larger than any partner could be is never kept: skip the lookup. */
                        if (!allowed || ((op == OP_ADD || op == OP_MUL) && sa > want)) continue;
                        int ok = 1;
                        for (int i = 0; i < win_len && ok; i++) {
                            int64_t y = ys[rows[i]], x = A[rows[i]];
                            if (variant == 0) need[i] = y - x;
                            else if (variant == 1) need[i] = x - y;
                            else if (variant == 2) need[i] = y + x;
                            else if (x == 0 || y % x != 0) ok = 0;
                            else need[i] = y / x;
                        }
                        if (!ok) continue;
                        uint64_t h = 0x9E3779B97F4A7C15ULL;
                        for (int i = 0; i < win_len; i++) {
                            h ^= (uint64_t)need[i] * 0xff51afd7ed558ccdULL;
                            h = ((h << 31) | (h >> 33)) * 0xc4ceb9fe1a85ec53ULL;
                        }
                        h ^= h >> 29;
                        int64_t head = -1;
                        for (uint64_t i = h & (wsize - 1);; i = (i + 1) & (wsize - 1)) {
                            int32_t s = wslots[i];
                            if (!s) break;
                            int64_t hd = s - 1;
                            if (whash[hd] != h) continue;
                            int eq = 1;
                            for (int q = 0; q < win_len && eq; q++) eq = sigs[hd * n + rows[q]] == need[q];
                            if (eq) { head = hd; break; }
                        }
                        int steps = 0;
                        for (int64_t b = head; b >= 0 && steps < CHAIN_MAX && nd < cap; b = wnext[b], steps++) {
                            if (sa + node_size[b] + 1 != ds) continue;
                            int64_t l = a_left ? a : b, r = a_left ? b : a;
                            if ((op == OP_ADD || op == OP_MUL) &&
                                (node_size[l] > node_size[r] || (node_size[l] == node_size[r] && l > r))) continue;
                            if (trivial(op, is_const[l], is_const[l] ? sigs[l * n] : 0, is_const[r],
                                        is_const[r] ? sigs[r * n] : 0, l == r)) continue;
                            int bad = verify(op, sigs + l * n, sigs + r * n, ys, n, tol);
                            if (bad <= tol) { deep_kind[nd] = op; deep_l[nd] = (int32_t)l; deep_r[nd] = (int32_t)r;
                                              deep_bad[nd] = bad; deep_val[nd] = 0; nd++; }
                        }
                    }
                    int awant = ds - 3 - sa;
                    /* Affine: (A + B) + c, (A - B) + c, (B - A) + c with c != 0 fitted on the window. */
                    for (int variant = 0; affine && awant >= 1 && awant <= top && variant < 3 && nd < cap; variant++) {
                        int op = variant == 0 ? OP_ADD : OP_SUB;
                        int a_left = variant != 2;
                        int allowed = 0;
                        for (int pi = 0; pi < n_ops; pi++) allowed |= ops[pi] == op;
                        if (!allowed || (op == OP_ADD && sa > awant)) continue;
                        int ok = 1;
                        for (int i = 0; i < win_len && ok; i++) {
                            int64_t y = ys[rows[i]], x = A[rows[i]];
                            need[i] = variant == 0 ? y - x : variant == 1 ? x - y : y + x;
                            ok = need[i] >= -SAFE_ABS && need[i] <= SAFE_ABS;
                        }
                        if (!ok) continue;
                        uint64_t h = 0x9E3779B97F4A7C15ULL;
                        for (int i = 1; i < win_len; i++) {
                            h ^= (uint64_t)(need[i] - need[0]) * 0xff51afd7ed558ccdULL;
                            h = ((h << 31) | (h >> 33)) * 0xc4ceb9fe1a85ec53ULL;
                        }
                        h ^= h >> 29;
                        int64_t head = -1;
                        for (uint64_t i = h & (wsize - 1);; i = (i + 1) & (wsize - 1)) {
                            int32_t s = dslots[i];
                            if (!s) break;
                            int64_t hd = s - 1;
                            if (dhash[hd] != h) continue;
                            const int64_t *H = sigs + hd * n;
                            int eq = 1;
                            for (int q = 1; q < win_len && eq; q++) eq = H[rows[q]] - H[rows[0]] == need[q] - need[0];
                            if (eq) { head = hd; break; }
                        }
                        int steps = 0;
                        for (int64_t b = head; b >= 0 && steps < CHAIN_MAX && nd < cap; b = dnext[b], steps++) {
                            if (sa + node_size[b] + 3 != ds) continue;
                            int64_t l = a_left ? a : b, r = a_left ? b : a;
                            if (op == OP_ADD &&
                                (node_size[l] > node_size[r] || (node_size[l] == node_size[r] && l > r))) continue;
                            if (trivial(op, is_const[l], is_const[l] ? sigs[l * n] : 0, is_const[r],
                                        is_const[r] ? sigs[r * n] : 0, l == r)) continue;
                            int64_t b0 = sigs[b * n + rows[0]];
                            int64_t c = variant == 1 ? b0 - need[0] : need[0] - b0;
                            if (c == 0) continue;
                            int bad = verify_affine(op, sigs + l * n, sigs + r * n, c, ys, n, tol);
                            if (bad <= tol) { deep_kind[nd] = op; deep_l[nd] = (int32_t)l; deep_r[nd] = (int32_t)r;
                                              deep_bad[nd] = bad; deep_val[nd] = c; nd++; }
                        }
                    }
                }
            }
            int good = 0;
            for (int64_t k = nd0; k < nd && !good; k++) good = deep_bad[k] <= deepen_bad;
            if (good) break;
        }
        free(wslots); free(dslots); free(wnext); free(wtail); free(dnext); free(dtail); free(whash); free(dhash);
        free(need);
        *n_deep = nd;
    }
    free(tb.slots); free(hashes); free(is_const);
    return rc;
}

/* Procedural memory (genesis.induction._library_search): instantiate one remembered shape over every
 * assignment of the variables to its V slots and of the constants to its C slots, in itertools.product order
 * (V slots first, the last slot varying fastest). prog is the shape in post-order: a token >= 0 is a binary
 * operator, -1 the next V slot, -2 the next C slot. With fit_bias the shape is "add(X, C)" and prog is X: the
 * top-level constant is the majority residual y - X (unique because 2 * tol < n, so it is also the most
 * common residual), and an instance where X is undefined on some observation is skipped and not counted;
 * otherwise an undefined output is a mismatch. Instances with at most tol mismatches are reported with their
 * assignment index, their mismatch count and the fitted constant. Returns -1 on bad arguments. */
#define PROG_MAX 31
int gn_shapes(int n, const int64_t *ys, int n_vars, const int64_t *var_sigs, int n_consts,
              const int64_t *const_vals, int prog_len, const int32_t *prog, int n_vslots, int n_cslots,
              int fit_bias, int tol, int64_t total, int64_t *hit_idx, int32_t *hit_bad, int64_t *hit_r,
              int64_t *n_hits, int64_t *evaluated) {
    *n_hits = 0;
    *evaluated = 0;
    if (n <= 0 || prog_len <= 0 || prog_len > PROG_MAX || n_vslots < 0 || n_cslots < 0 ||
        n_vslots + n_cslots > PROG_MAX || (n_vslots && n_vars <= 0) || (n_cslots && n_consts <= 0) ||
        (fit_bias && 2 * tol >= n))
        return -1;
    int depth = 0, nv = 0, nc = 0;
    for (int k = 0; k < prog_len; k++) {
        int32_t tok = prog[k];
        if (tok == -1) { nv++; depth++; }
        else if (tok == -2) { nc++; depth++; }
        else if (tok >= OP_ADD && tok <= OP_MOD && depth >= 2) depth--;
        else return -1;
    }
    if (depth != 1 || nv != n_vslots || nc != n_cslots) return -1;
    int32_t vsel[PROG_MAX], csel[PROG_MAX];
    int64_t st[PROG_MAX + 1];
    int64_t *vals = fit_bias ? malloc(sizeof(int64_t) * (size_t)n) : NULL;
    if (fit_bias && !vals) return -1;
    for (int64_t idx = 0; idx < total; idx++) {
        int64_t rem = idx;
        for (int k = n_cslots - 1; k >= 0; k--) { csel[k] = (int32_t)(rem % n_consts); rem /= n_consts; }
        for (int k = n_vslots - 1; k >= 0; k--) { vsel[k] = (int32_t)(rem % n_vars); rem /= n_vars; }
        int bad = 0, skip = 0;
        for (int t = 0; t < n; t++) {
            int sp = 0, vi = 0, ci = 0, ok = 1;
            for (int k = 0; k < prog_len && ok; k++) {
                int32_t tok = prog[k];
                if (tok == -1) st[sp++] = var_sigs[(int64_t)vsel[vi++] * n + t];
                else if (tok == -2) st[sp++] = const_vals[csel[ci++]];
                else { sp--; ok = combine(tok, st[sp - 1], st[sp], &st[sp - 1]); }
            }
            if (fit_bias) {
                if (!ok) { skip = 1; break; }
                vals[t] = st[0];
            } else if (!ok || st[0] != ys[t]) {
                if (++bad > tol) break;
            }
        }
        if (skip) continue;
        ++*evaluated;
        int64_t r = 0;
        if (fit_bias) {
            /* Boyer-Moore majority of the residuals, then its exact mismatch count. */
            int64_t cnt = 0;
            for (int t = 0; t < n; t++) {
                int64_t res = ys[t] - vals[t];
                if (cnt == 0) { r = res; cnt = 1; }
                else cnt += res == r ? 1 : -1;
            }
            for (int t = 0; t < n && bad <= tol; t++)
                if (ys[t] - vals[t] != r || ys[t] < -LIMIT || ys[t] > LIMIT) bad++;
        }
        if (bad <= tol) {
            hit_idx[*n_hits] = idx;
            hit_bad[*n_hits] = bad;
            hit_r[*n_hits] = r;
            ++*n_hits;
        }
    }
    free(vals);
    return 0;
}
