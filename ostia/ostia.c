/*
 * Classical OSTIA (Oncina, Garcia, Vidal 1993) in C.
 *
 * Onward prefix-tree transducer, states considered in lexicographic order of
 * their prefixes (or shortlex with -s), each state merged into the first
 * earlier surviving state for which merge-and-fold succeeds.  A fold keeps the
 * state with the smaller rank, pushes the non-common output suffixes onward,
 * and fails if that would change the output of an already processed (red)
 * state or if final outputs conflict.  Failed merges are rolled back through
 * an undo log instead of copying the transducer.  The implementation was
 * cross-checked against an independent Python OSTIA: identical state counts
 * and test behaviour on the benchmark battery and on 300 sparse random
 * subsequential targets.
 *
 * Symbols are single bytes.  Input files contain one example per line:
 *   INPUT<TAB>OUTPUT
 * Usage: ostia [-s] TRAINFILE TESTFILE
 * Prints one JSON object: learned state count, test accuracy, undefined runs.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

typedef struct { const char *s; int len; } Str;

static int K;                 /* input alphabet size */
static int symidx[256];
static int N, cap;            /* number of states */
static int *tr;               /* tr[s*K+a] target or -1 */
static Str *out;              /* out[s*K+a] */
static Str *fin;
static char *findef, *alive;
static int *par, *parsym;
static int *rnk;
static const char **name;     /* prefix string of each tree state */
static int *namelen;
static int outer_rank;
static int shortlex = 0;

/* ---- arena for output strings, rolled back with failed merges ---- */
static char *arena;
static size_t arena_top, arena_size = (size_t)8 << 30;

static Str mkstr2(const char *a, int alen, const char *b, int blen) {
    if (alen + blen == 0) return (Str){"", 0};
    if (arena_top + alen + blen > arena_size) { fprintf(stderr, "arena full\n"); exit(2); }
    char *p = arena + arena_top;
    memcpy(p, a, alen);
    memcpy(p + alen, b, blen);
    arena_top += alen + blen;
    return (Str){p, alen + blen};
}

/* ---- undo log ---- */
enum { L_TR, L_OUT, L_FIN, L_FINDEF, L_ALIVE, L_PAR };
typedef struct { int kind, idx, i1, i2; Str sv; } LogEnt;
static LogEnt *logv;
static size_t logn, logcap;

static void logpush(LogEnt e) {
    if (logn == logcap) {
        logcap = logcap ? logcap * 2 : 1 << 16;
        logv = realloc(logv, logcap * sizeof *logv);
    }
    logv[logn++] = e;
}
static void set_tr(int s, int a, int t) { logpush((LogEnt){L_TR, s * K + a, tr[s * K + a], 0, {0}}); tr[s * K + a] = t; }
static void set_out(int s, int a, Str v) { logpush((LogEnt){L_OUT, s * K + a, 0, 0, out[s * K + a]}); out[s * K + a] = v; }
static void set_fin(int s, Str v) { logpush((LogEnt){L_FIN, s, 0, 0, fin[s]}); fin[s] = v; }
static void set_findef(int s, char v) { logpush((LogEnt){L_FINDEF, s, findef[s], 0, {0}}); findef[s] = v; }
static void set_alive(int s, char v) { logpush((LogEnt){L_ALIVE, s, alive[s], 0, {0}}); alive[s] = v; }
static void set_par(int s, int p, int a) { logpush((LogEnt){L_PAR, s, par[s], parsym[s], {0}}); par[s] = p; parsym[s] = a; }

static void undo(size_t mark) {
    while (logn > mark) {
        LogEnt e = logv[--logn];
        switch (e.kind) {
        case L_TR: tr[e.idx] = e.i1; break;
        case L_OUT: out[e.idx] = e.sv; break;
        case L_FIN: fin[e.idx] = e.sv; break;
        case L_FINDEF: findef[e.idx] = (char)e.i1; break;
        case L_ALIVE: alive[e.idx] = (char)e.i1; break;
        case L_PAR: par[e.idx] = e.i1; parsym[e.idx] = e.i2; break;
        }
    }
}

static int streq(Str a, Str b) { return a.len == b.len && memcmp(a.s, b.s, a.len) == 0; }
static int lcplen(Str a, Str b) {
    int n = a.len < b.len ? a.len : b.len, i = 0;
    while (i < n && a.s[i] == b.s[i]) i++;
    return i;
}

/* Prepend suffix to every output leaving state s (including its final output). */
static void pushback(int s, const char *suf, int len) {
    if (len == 0) return;
    for (int a = 0; a < K; a++)
        if (tr[s * K + a] >= 0) {
            Str o = out[s * K + a];
            set_out(s, a, mkstr2(suf, len, o.s, o.len));
        }
    if (findef[s]) set_fin(s, mkstr2(suf, len, fin[s].s, fin[s].len));
}

/* Fold state q into state p (p is kept). Returns 0 on failure. */
static int fold(int p, int q) {
    set_alive(q, 0);
    if (findef[q]) {
        if (findef[p]) { if (!streq(fin[p], fin[q])) return 0; }
        else { set_findef(p, 1); set_fin(p, fin[q]); }
    }
    for (int a = 0; a < K; a++) {
        int y = tr[q * K + a];
        if (y < 0) continue;
        int x = tr[p * K + a];
        Str w = out[q * K + a];
        if (x < 0) {
            set_tr(p, a, y); set_out(p, a, w); set_par(y, p, a);
            continue;
        }
        Str v = out[p * K + a];
        if (x == y) { if (!streq(v, w)) return 0; continue; }
        int u = lcplen(v, w);
        if (rnk[x] < outer_rank && v.len != u) return 0;
        if (rnk[y] < outer_rank && w.len != u) return 0;
        pushback(x, v.s + u, v.len - u);
        pushback(y, w.s + u, w.len - u);
        set_out(p, a, (Str){v.s, u});
        int keep = rnk[x] < rnk[y] ? x : y, other = keep == x ? y : x;
        if (keep != x) { set_tr(p, a, keep); set_par(keep, p, a); }
        if (!fold(keep, other)) return 0;
    }
    return 1;
}

static int try_merge(int r, int b) {
    int pp = par[b], ps = parsym[b];
    set_tr(pp, ps, r);
    return fold(r, b);
}

/* ---- data ---- */
typedef struct { char *in; int inlen; char *outs; int outlen; } Ex;

static Ex *readfile(const char *path, int *n) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); exit(1); }
    Ex *v = NULL; int nv = 0, cv = 0;
    char *line = NULL; size_t lc = 0; ssize_t len;
    while ((len = getline(&line, &lc, f)) >= 0) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        char *tab = strchr(line, '\t');
        if (!tab) { fprintf(stderr, "bad line: %s\n", line); exit(1); }
        *tab = 0;
        if (nv == cv) { cv = cv ? cv * 2 : 1024; v = realloc(v, cv * sizeof *v); }
        v[nv].in = strdup(line); v[nv].inlen = (int)strlen(line);
        v[nv].outs = strdup(tab + 1); v[nv].outlen = (int)strlen(tab + 1);
        nv++;
    }
    free(line); fclose(f);
    *n = nv;
    return v;
}

static int newstate(const char *nm, int nl) {
    if (N == cap) {
        cap = cap ? cap * 2 : 1024;
        tr = realloc(tr, (size_t)cap * K * sizeof *tr);
        out = realloc(out, (size_t)cap * K * sizeof *out);
        fin = realloc(fin, cap * sizeof *fin);
        findef = realloc(findef, cap);
        alive = realloc(alive, cap);
        par = realloc(par, cap * sizeof *par);
        parsym = realloc(parsym, cap * sizeof *parsym);
        name = realloc(name, cap * sizeof *name);
        namelen = realloc(namelen, cap * sizeof *namelen);
    }
    int s = N++;
    for (int a = 0; a < K; a++) { tr[s * K + a] = -1; out[s * K + a] = (Str){"", 0}; }
    fin[s] = (Str){"", 0}; findef[s] = 0; alive[s] = 1; par[s] = -1; parsym[s] = -1;
    name[s] = nm; namelen[s] = nl;
    return s;
}

/* Make the tree onward: return the common prefix removed from state s. */
static Str onward(int s) {
    Str pre = {0, -1};
    for (int a = 0; a < K; a++) {
        int c = tr[s * K + a];
        if (c < 0) continue;
        Str pc = onward(c);
        Str o = out[s * K + a];
        out[s * K + a] = mkstr2(o.s, o.len, pc.s, pc.len);
    }
    for (int a = 0; a < K; a++) {
        if (tr[s * K + a] < 0) continue;
        Str o = out[s * K + a];
        pre = pre.len < 0 ? o : (Str){pre.s, lcplen(pre, o)};
    }
    if (findef[s]) pre = pre.len < 0 ? fin[s] : (Str){pre.s, lcplen(pre, fin[s])};
    if (pre.len <= 0) return (Str){"", 0};
    Str keep = mkstr2(pre.s, pre.len, "", 0);
    for (int a = 0; a < K; a++)
        if (tr[s * K + a] >= 0) { out[s * K + a].s += pre.len; out[s * K + a].len -= pre.len; }
    if (findef[s]) { fin[s].s += pre.len; fin[s].len -= pre.len; }
    return keep;
}

static int cmp_state(const void *pa, const void *pb) {
    int a = *(const int *)pa, b = *(const int *)pb;
    if (shortlex && namelen[a] != namelen[b]) return namelen[a] - namelen[b];
    int n = namelen[a] < namelen[b] ? namelen[a] : namelen[b];
    int c = memcmp(name[a], name[b], n);
    if (c) return c;
    return namelen[a] - namelen[b];
}

int main(int argc, char **argv) {
    int ai = 1;
    if (ai < argc && strcmp(argv[ai], "-s") == 0) { shortlex = 1; ai++; }
    if (argc - ai != 2) { fprintf(stderr, "usage: ostia [-s] TRAIN TEST\n"); return 1; }
    int ntr, nte;
    Ex *train = readfile(argv[ai], &ntr), *test = readfile(argv[ai + 1], &nte);

    arena = mmap(NULL, arena_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (arena == MAP_FAILED) { perror("mmap"); return 1; }

    memset(symidx, -1, sizeof symidx);
    for (int i = 0; i < ntr; i++)
        for (int j = 0; j < train[i].inlen; j++) {
            unsigned char c = (unsigned char)train[i].in[j];
            if (symidx[c] < 0) symidx[c] = K++;
        }
    if (K == 0) K = 1;

    clock_t t0 = clock();
    int root = newstate("", 0);
    for (int i = 0; i < ntr; i++) {
        int s = root;
        for (int j = 0; j < train[i].inlen; j++) {
            int a = symidx[(unsigned char)train[i].in[j]];
            if (tr[s * K + a] < 0) {
                int t = newstate(train[i].in, j + 1);
                tr[s * K + a] = t; par[t] = s; parsym[t] = a;
            }
            s = tr[s * K + a];
        }
        Str o = {train[i].outs, train[i].outlen};
        if (findef[s] && !streq(fin[s], o)) { fprintf(stderr, "conflicting outputs\n"); return 1; }
        findef[s] = 1; fin[s] = o;
    }
    Str init = onward(root);

    int *order = malloc(N * sizeof *order);
    rnk = malloc(N * sizeof *rnk);
    for (int i = 0; i < N; i++) order[i] = i;
    qsort(order, N, sizeof *order, cmp_state);
    for (int i = 0; i < N; i++) rnk[order[i]] = i;

    long attempts = 0;
    for (int i = 1; i < N; i++) {
        int q = order[i];
        if (!alive[q]) continue;
        outer_rank = i;
        for (int j = 0; j < i; j++) {
            int p = order[j];
            if (!alive[p]) continue;
            size_t mark = logn, amark = arena_top;
            attempts++;
            if (try_merge(p, q)) { logn = 0; break; }
            undo(mark);
            arena_top = amark;
        }
    }
    double secs = (double)(clock() - t0) / CLOCKS_PER_SEC;

    int states = 0;
    for (int s = 0; s < N; s++) states += alive[s];

    /* replay check and test evaluation */
    int replay_bad = 0, correct = 0, undefined = 0;
    char *buf = malloc(1 << 20);
    for (int pass = 0; pass < 2; pass++) {
        Ex *v = pass ? test : train; int n = pass ? nte : ntr;
        for (int i = 0; i < n; i++) {
            int s = root, len = 0, ok = 1;
            memcpy(buf, init.s, init.len); len = init.len;
            for (int j = 0; j < v[i].inlen && ok; j++) {
                int c = symidx[(unsigned char)v[i].in[j]];
                if (c < 0 || tr[s * K + c] < 0) { ok = 0; break; }
                Str o = out[s * K + c];
                memcpy(buf + len, o.s, o.len); len += o.len;
                s = tr[s * K + c];
            }
            if (ok && !findef[s]) ok = 0;
            if (ok) { memcpy(buf + len, fin[s].s, fin[s].len); len += fin[s].len; }
            int match = ok && len == v[i].outlen && memcmp(buf, v[i].outs, len) == 0;
            if (pass == 0) replay_bad += !match;
            else { correct += match; undefined += !ok; }
        }
    }
    printf("{\"order\": \"%s\", \"tree_states\": %d, \"states\": %d, \"merge_attempts\": %ld, "
           "\"replay_failures\": %d, \"test_n\": %d, \"acc\": %.6f, \"undefined\": %d, \"learn_sec\": %.3f}\n",
           shortlex ? "shortlex" : "lex", N, states, attempts, replay_bad, nte,
           nte ? (double)correct / nte : 0.0, undefined, secs);
    return replay_bad != 0;
}
