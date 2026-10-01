// Oracle-assisted bimachine learner (JavaScript port of rpni_bimachine.py).
//
// Mirrors learn_bimachine_from_oracle step for step, with two differences:
//  * witness candidates are taken in shortlex order instead of a seeded shuffle
//    (the parity test runs the Python learner the same way);
//  * every step is recorded as an event with a snapshot of the hypothesis, so a
//    viewer can replay the run.
//
// test/parity.py checks that both learners produce the same machines, merges and
// oracle queries.
//
// Words are JavaScript strings of single-character symbols. An output is an
// array of chunks (strings), one per input symbol.

// ---------------------------------------------------------------------------
// DFA
// ---------------------------------------------------------------------------

export class DFA {
  constructor(start = 0, trans = new Map(), sink = null) {
    this.start = start;
    this.trans = trans; // Map<state, Map<symbol, state>>
    this.sink = sink;
  }

  clone() {
    const t = new Map();
    for (const [s, outs] of this.trans) t.set(s, new Map(outs));
    return new DFA(this.start, t, this.sink);
  }

  states() {
    const st = new Set([this.start]);
    for (const [s, outs] of this.trans) {
      st.add(s);
      for (const t of outs.values()) st.add(t);
    }
    if (this.sink !== null) st.add(this.sink);
    return [...st].sort((a, b) => a - b);
  }

  step(s, a) {
    const outs = this.trans.get(s);
    const t = outs ? outs.get(a) : undefined;
    if (t === undefined) return this.sink !== null ? this.sink : null;
    return t;
  }

  // q[0] = start, q[i+1] = delta(q[i], x[i]); -1 once undefined.
  runPrefix(x) {
    const q = [this.start];
    let cur = this.start;
    for (const ch of x) {
      const nxt = cur === -1 ? null : this.step(cur, ch);
      cur = nxt === null ? -1 : nxt;
      q.push(cur);
    }
    return q;
  }

  // Right-machine convention: p[n] = start, p[i] = delta(p[i+1], x[i]).
  runRight(x) {
    const n = x.length;
    const p = new Array(n + 1);
    p[n] = this.start;
    let cur = this.start;
    for (let i = n - 1; i >= 0; i--) {
      const nxt = cur === -1 ? null : this.step(cur, x[i]);
      cur = nxt === null ? -1 : nxt;
      p[i] = cur;
    }
    return p;
  }

  static fromPrefixTrie(words) {
    const trans = new Map();
    let nextId = 1;
    for (const w of words) {
      let cur = 0;
      for (const ch of w) {
        if (!trans.has(cur)) trans.set(cur, new Map());
        const outs = trans.get(cur);
        if (!outs.has(ch)) outs.set(ch, nextId++);
        cur = outs.get(ch);
      }
    }
    return new DFA(0, trans, null);
  }

  // Extend as a trie along x; returns the newly created states.
  addString(x) {
    const st = this.states();
    let nextId = st.length ? Math.max(...st) + 1 : 0;
    const created = [];
    let cur = this.start;
    for (const ch of x) {
      if (!this.trans.has(cur)) this.trans.set(cur, new Map());
      const outs = this.trans.get(cur);
      if (!outs.has(ch)) {
        outs.set(ch, nextId);
        created.push(nextId);
        nextId++;
      }
      cur = outs.get(ch);
    }
    return created;
  }

  isTotal(alpha) {
    return this.states().every((s) => alpha.every((a) => this.trans.get(s)?.has(a)));
  }

  // Add a sink only if needed; returns the sink id or null.
  completeWithSink(alpha) {
    if (this.isTotal(alpha)) return null;
    if (this.sink === null) {
      const st = this.states();
      this.sink = st.length ? Math.max(...st) + 1 : 0;
    }
    const sink = this.sink;
    if (!this.trans.has(sink)) this.trans.set(sink, new Map());
    for (const a of alpha) this.trans.get(sink).set(a, sink);
    for (const s of this.states()) {
      if (!this.trans.has(s)) this.trans.set(s, new Map());
      for (const a of alpha) if (!this.trans.get(s).has(a)) this.trans.get(s).set(a, sink);
    }
    return sink;
  }

  pruneUnreachable(alpha) {
    const reach = new Set([this.start]);
    const queue = [this.start];
    while (queue.length) {
      const s = queue.shift();
      const outs = this.trans.get(s);
      if (!outs) continue;
      for (const a of alpha) {
        const t = outs.get(a);
        if (t !== undefined && !reach.has(t)) {
          reach.add(t);
          queue.push(t);
        }
      }
    }
    const t2 = new Map();
    for (const [s, outs] of this.trans) {
      if (!reach.has(s)) continue;
      const o2 = new Map();
      for (const [a, t] of outs) if (reach.has(t)) o2.set(a, t);
      t2.set(s, o2);
    }
    this.trans = t2;
    if (this.sink !== null && !reach.has(this.sink)) this.sink = null;
  }

  toJSON() {
    const edges = [];
    for (const [s, outs] of this.trans) for (const [a, t] of outs) edges.push([s, a, t]);
    return { start: this.start, sink: this.sink, states: this.states(), edges };
  }
}

// Shortlex-least word reaching each state (BFS over the sorted alphabet).
export function shortestWitnesses(dfa, alpha) {
  const wit = new Map([[dfa.start, ""]]);
  const queue = [dfa.start];
  while (queue.length) {
    const s = queue.shift();
    for (const a of alpha) {
      const t = dfa.step(s, a);
      if (t === null) continue;
      if (!wit.has(t)) {
        wit.set(t, wit.get(s) + a);
        queue.push(t);
      }
    }
  }
  for (const s of dfa.states()) if (!wit.has(s)) wit.set(s, "");
  return wit;
}

const shortlexCmp = (u, v) => (u.length - v.length) || (u < v ? -1 : u > v ? 1 : 0);
const rev = (w) => [...w].reverse().join("");

// ---------------------------------------------------------------------------
// Bimachine hypothesis
// ---------------------------------------------------------------------------

const okey = (q, a, p) => `${q}|${a}|${p}`;
const unkey = (k) => {
  const [q, a, p] = k.split("|");
  return [Number(q), a, Number(p)];
};

// Provenance of an output entry, used only for display.
const SRC_RANK = { data: 0, witness: 1, completion: 2 };

export class Hypothesis {
  constructor(left, right, omega = new Map()) {
    this.left = left;
    this.right = right;
    this.omega = omega; // Map<"q|a|p", {out, src}>
  }

  clone() {
    const o = new Map();
    for (const [k, v] of this.omega) o.set(k, { ...v });
    return new Hypothesis(this.left.clone(), this.right.clone(), o);
  }

  get(q, a, p) {
    return this.omega.get(okey(q, a, p));
  }

  transduceChunks(x) {
    const q = this.left.runPrefix(x);
    const p = this.right.runRight(x);
    return [...x].map((a, i) => {
      const e = this.get(q[i], a, p[i + 1]);
      return e ? e.out : null;
    });
  }

  // With the identity fallback for undefined positions (as in Python's transduce).
  transduce(x) {
    return this.transduceChunks(x).map((c, i) => (c === null ? x[i] : c));
  }

  matches(x, y) {
    const c = this.transduceChunks(x);
    return c.every((v, i) => v === y[i]);
  }

  toJSON() {
    const entries = [];
    for (const [k, v] of this.omega) entries.push([...unkey(k), v.out, v.src]);
    return { left: this.left.toJSON(), right: this.right.toJSON(), omega: entries };
  }
}

function omegaFromData(left, right, data, src = "data") {
  const omega = new Map();
  for (const [x, y] of data) {
    const q = left.runPrefix(x);
    const p = right.runRight(x);
    for (let i = 0; i < x.length; i++) {
      const k = okey(q[i], x[i], p[i + 1]);
      if (omega.has(k) && omega.get(k).out !== y[i]) throw new Error(`inconsistent aligned data at ${k}`);
      omega.set(k, { out: y[i], src });
    }
  }
  return omega;
}

// ---------------------------------------------------------------------------
// Merge-and-fold
// ---------------------------------------------------------------------------

class UnionFind {
  constructor(items) {
    this.parent = new Map(items.map((x) => [x, x]));
    this.rank = new Map(items.map((x) => [x, 0]));
  }
  has(x) {
    return this.parent.has(x);
  }
  find(x) {
    let r = x;
    while (this.parent.get(r) !== r) r = this.parent.get(r);
    while (this.parent.get(x) !== r) {
      const nx = this.parent.get(x);
      this.parent.set(x, r);
      x = nx;
    }
    return r;
  }
  union(a, b) {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return false;
    if (this.rank.get(ra) < this.rank.get(rb)) [ra, rb] = [rb, ra];
    this.parent.set(rb, ra);
    if (this.rank.get(ra) === this.rank.get(rb)) this.rank.set(ra, this.rank.get(ra) + 1);
    return true;
  }
}

// One pass of determinism closure; returns the pairs that were unioned.
function foldClosure(uf, dfa, alpha) {
  const groups = new Map();
  for (const s of dfa.states()) {
    if (!uf.has(s)) continue;
    const r = uf.find(s);
    if (!groups.has(r)) groups.set(r, []);
    groups.get(r).push(s);
  }
  const folds = [];
  for (const members of groups.values()) {
    for (const a of alpha) {
      const targets = [];
      for (const s of members) {
        const t = dfa.trans.get(s)?.get(a);
        if (t === undefined || !uf.has(t)) continue;
        const rt = uf.find(t);
        if (!targets.includes(rt)) targets.push(rt);
      }
      targets.sort((x, y) => x - y);
      for (const other of targets.slice(1)) {
        if (uf.union(targets[0], other)) folds.push([targets[0], other, a]);
      }
    }
  }
  return folds;
}

// Try merging state `from` into `into` on one side ("L" or "R").
// Returns {cand, classes, folds} on success or {conflict, classes, folds} on an output clash.
export function tryMerge(hypo, side, into, from, alpha) {
  const dfa = side === "L" ? hypo.left : hypo.right;
  const live = dfa.states();
  const uf = new UnionFind(live);
  uf.union(into, from);
  const folds = [];
  for (;;) {
    const f = foldClosure(uf, dfa, alpha);
    folds.push(...f);
    if (!f.length) break;
  }
  const classMap = new Map();
  for (const s of live) {
    const r = uf.find(s);
    if (!classMap.has(r)) classMap.set(r, []);
    classMap.get(r).push(s);
  }
  const classes = [...classMap.values()].filter((c) => c.length > 1);

  // Output table under the merge; a clash means the merge is inconsistent.
  const newOmega = new Map();
  for (const [k, v] of hypo.omega) {
    const [q, a, p] = unkey(k);
    const nq = side === "L" ? (uf.has(q) ? uf.find(q) : null) : q;
    const np = side === "R" ? (uf.has(p) ? uf.find(p) : null) : p;
    if (nq === null || np === null) continue;
    const nk = okey(nq, a, np);
    const prev = newOmega.get(nk);
    if (prev && prev.out !== v.out) {
      return { conflict: { entry: [nq, a, np], outs: [prev.out, v.out], keys: [unkey(prev.from), [q, a, p]] }, classes, folds };
    }
    if (!prev || SRC_RANK[v.src] < SRC_RANK[prev.src]) newOmega.set(nk, { ...v, from: k });
  }
  for (const v of newOmega.values()) delete v.from;

  const newTrans = new Map();
  for (const [s, outs] of dfa.trans) {
    if (!uf.has(s)) continue;
    const rs = uf.find(s);
    if (!newTrans.has(rs)) newTrans.set(rs, new Map());
    for (const [a, t] of outs) if (uf.has(t)) newTrans.get(rs).set(a, uf.find(t));
  }
  const newDfa = new DFA(uf.find(dfa.start), newTrans, null);
  const cand =
    side === "L"
      ? new Hypothesis(newDfa, hypo.right.clone(), newOmega)
      : new Hypothesis(hypo.left.clone(), newDfa, newOmega);
  return { cand, classes, folds };
}

// Add paths for the witness strings and their oracle outputs as constraints.
function augment(hypo, examples) {
  const h = hypo.clone();
  h.left.sink = null;
  h.right.sink = null;
  const newL = [];
  const newR = [];
  for (const [x] of examples) {
    newL.push(...h.left.addString(x));
    newR.push(...h.right.addString(rev(x)));
  }
  for (const [x, y] of examples) {
    const q = h.left.runPrefix(x);
    const p = h.right.runRight(x);
    for (let i = 0; i < x.length; i++) {
      const k = okey(q[i], x[i], p[i + 1]);
      const prev = h.omega.get(k);
      if (prev && prev.out !== y[i]) {
        return { conflict: { witness: x, position: i, entry: [q[i], x[i], p[i + 1]], outs: [prev.out, y[i]], expected: y } };
      }
      if (!prev) h.omega.set(k, { out: y[i], src: "witness" });
    }
  }
  return { cand: h, newL, newR };
}

// Witness strings: wrap(w) for non-empty w with |w| <= maxLen, shortlex order,
// skipping training strings, at most k.
function witnessStrings(wrap, alpha, trainSet, k, maxLen) {
  const out = [];
  for (const w of shortlexWords(alpha, maxLen)) {
    if (out.length >= k) break;
    const x = wrap(w);
    if (!trainSet.has(x) && !out.includes(x)) out.push(x);
  }
  return out;
}

// Non-empty words of length <= maxLen in shortlex order, generated lazily so
// that only as many candidates as needed are ever built.
function* shortlexWords(alpha, maxLen) {
  for (let L = 1; L <= maxLen; L++) {
    const idx = new Array(L).fill(0);
    for (;;) {
      yield idx.map((i) => alpha[i]).join("");
      let pos = L - 1;
      while (pos >= 0 && idx[pos] === alpha.length - 1) idx[pos--] = 0;
      if (pos < 0) break;
      idx[pos]++;
    }
  }
}

// ---------------------------------------------------------------------------
// Completion and minimization
// ---------------------------------------------------------------------------

function fillOmega(hypo, oracle, alpha, emit) {
  const sinkL = hypo.left.completeWithSink(alpha);
  const sinkR = hypo.right.completeWithSink(alpha);
  const witL = shortestWitnesses(hypo.left, alpha);
  const witR = shortestWitnesses(hypo.right, alpha);
  const queries = [];
  for (const qL of hypo.left.states()) {
    const pref = witL.get(qL);
    for (const qR of hypo.right.states()) {
      const suf = rev(witR.get(qR));
      for (const a of alpha) {
        if (hypo.omega.has(okey(qL, a, qR))) continue;
        const x = pref + a + suf;
        const y = oracle.query(x);
        hypo.omega.set(okey(qL, a, qR), { out: y[pref.length], src: "completion" });
        queries.push({ entry: [qL, a, qR], x, pos: pref.length, y });
      }
    }
  }
  emit({
    kind: "complete",
    sinks: { L: sinkL, R: sinkR },
    queries,
    text: queries.length
      ? `Completion: made both automata total${sinkL !== null || sinkR !== null ? " (added a sink)" : ""} and filled ${queries.length} missing output ${queries.length === 1 ? "entry" : "entries"} with representative queries π(q)·a·σ(p).`
      : `Completion: every output entry ω(q,a,p) is already defined${sinkL !== null || sinkR !== null ? " (after adding a sink)" : ""}; no oracle queries needed.`,
  });
  return hypo;
}

// Partition refinement of one side given the other ("L" compares rows, "R" columns).
function minimizeSide(hypo, side, alpha) {
  const QL = hypo.left.states();
  const QR = hypo.right.states();
  const dfa = side === "L" ? hypo.left : hypo.right;
  const Q = side === "L" ? QL : QR;
  const sig0 = (s) =>
    side === "L"
      ? alpha.flatMap((a) => QR.map((p) => hypo.get(s, a, p).out))
      : QL.flatMap((q) => alpha.map((a) => hypo.get(q, a, s).out));

  const rounds = [];
  let part = new Map();
  let ids = new Map();
  for (const s of Q) {
    const k = JSON.stringify(sig0(s));
    if (!ids.has(k)) ids.set(k, ids.size);
    part.set(s, ids.get(k));
  }
  rounds.push(new Map(part));
  for (;;) {
    const np = new Map();
    ids = new Map();
    for (const s of Q) {
      const k = JSON.stringify([sig0(s), alpha.map((a) => part.get(dfa.step(s, a)))]);
      if (!ids.has(k)) ids.set(k, ids.size);
      np.set(s, ids.get(k));
    }
    const changed = Q.some((s) => np.get(s) !== part.get(s));
    if (!changed) break;
    part = np;
    rounds.push(new Map(part));
  }

  const trans = new Map();
  for (const s of Q) {
    const c = part.get(s);
    if (!trans.has(c)) trans.set(c, new Map());
    for (const a of alpha) trans.get(c).set(a, part.get(dfa.step(s, a)));
  }
  const quotient = new DFA(part.get(dfa.start), trans, null);
  quotient.completeWithSink(alpha);

  const omega = new Map();
  for (const [k, v] of hypo.omega) {
    const [q, a, p] = unkey(k);
    const nk = side === "L" ? okey(part.get(q), a, p) : okey(q, a, part.get(p));
    const prev = omega.get(nk);
    if (!prev || SRC_RANK[v.src] < SRC_RANK[prev.src]) omega.set(nk, { ...v });
  }
  const next =
    side === "L"
      ? new Hypothesis(quotient, hypo.right, omega)
      : new Hypothesis(hypo.left, quotient, omega);
  const signatures = Q.map((s) => ({ state: s, row: sig0(s), next: alpha.map((a) => dfa.step(s, a)) }));
  return { next, part, rounds, signatures };
}

function renumberDfa(dfa, alpha) {
  const order = [dfa.start];
  const seen = new Set(order);
  for (let i = 0; i < order.length; i++) {
    for (const a of alpha) {
      const t = dfa.step(order[i], a);
      if (t !== null && !seen.has(t)) {
        seen.add(t);
        order.push(t);
      }
    }
  }
  const map = new Map(order.map((s, i) => [s, i]));
  const trans = new Map();
  for (const [s, outs] of dfa.trans) {
    if (!map.has(s)) continue;
    const o2 = new Map();
    for (const [a, t] of outs) if (map.has(t)) o2.set(a, map.get(t));
    trans.set(map.get(s), o2);
  }
  const d = new DFA(0, trans, null);
  d.pruneUnreachable(alpha);
  return { dfa: d, map };
}

function renumber(hypo, alpha) {
  const L = renumberDfa(hypo.left, alpha);
  const R = renumberDfa(hypo.right, alpha);
  const omega = new Map();
  for (const [k, v] of hypo.omega) {
    const [q, a, p] = unkey(k);
    if (L.map.has(q) && R.map.has(p)) omega.set(okey(L.map.get(q), a, R.map.get(p)), { ...v });
  }
  return { hypo: new Hypothesis(L.dfa, R.dfa, omega), mapL: L.map, mapR: R.map };
}

// ---------------------------------------------------------------------------
// Oracle wrapper
// ---------------------------------------------------------------------------

// Memoizes a chunk function and records every distinct query in order.
export class Oracle {
  constructor(chunkFn) {
    this.chunkFn = chunkFn;
    this.cache = new Map();
    this.log = [];
  }
  query(x) {
    if (!this.cache.has(x)) {
      const y = this.chunkFn(x);
      if (!Array.isArray(y) || y.length !== x.length) {
        throw new Error(`oracle must return one chunk per input symbol for ${JSON.stringify(x)}`);
      }
      this.cache.set(x, y.map(String));
      this.log.push(x);
    }
    return this.cache.get(x);
  }
}

// ---------------------------------------------------------------------------
// The learner
// ---------------------------------------------------------------------------

export function learnBimachine({
  alphabet,
  train,           // array of input words
  oracle,          // Oracle instance
  kWitness = 6,
  contMaxLen = 2,
  headMaxLen = 2,
  maxRounds = 10,
  minIters = 3,
  maxEvents = 5000,
  record = true,   // false: keep only event kinds, without snapshots (fast; for tests)
}) {
  const alpha = [...alphabet].sort();
  const events = [];
  let hypo = null;
  let red = { L: [], R: [] };
  let blue = { L: [], R: [] };

  const snap = () => (hypo ? hypo.toJSON() : null);
  const emit = (ev) => {
    if (events.length >= maxEvents && ev.kind !== "stopped") throw new StepLimit();
    if (!record) {
      events.push({ kind: ev.kind });
      return;
    }
    events.push({
      ...ev,
      snapshot: ev.snapshot || snap(),
      red: { L: [...red.L], R: [...red.R] },
      blue: { L: [...blue.L], R: [...blue.R] },
      queryCount: oracle.log.length,
    });
  };

  const mergeRecord = [];
  const minSteps = [];
  try {
  // Training data and prefix trees.
  const data = train.map((x) => [x, oracle.query(x)]);
  const trainSet = new Set(train);
  const left = DFA.fromPrefixTrie(train);
  const right = DFA.fromPrefixTrie(train.map(rev));
  hypo = new Hypothesis(left, right, omegaFromData(left, right, data));
  emit({
    kind: "init",
    data,
    text: `Built the left prefix tree over the ${trainSet.size} distinct training inputs (${left.states().length} states) and the right prefix tree over their reversals (${right.states().length} states). Each aligned chunk fixes one output entry ω(q, a, p); ${hypo.omega.size} entries are known.`,
  });


  function pass(side) {
    const maxLen = side === "L" ? contMaxLen : headMaxLen;
    let changedAny = false;
    for (;;) {
      const dfa = () => (side === "L" ? hypo.left : hypo.right);
      const wit = shortestWitnesses(dfa(), alpha);
      // Representative as a string in reading order: prefix for L, suffix for R.
      const repr = (s) => (side === "L" ? wit.get(s) : rev(wit.get(s)));
      red[side] = [dfa().start];
      let mergedInRestart = false;
      for (;;) {
        const redSet = new Set(red[side]);
        const frontier = new Set();
        for (const r of red[side]) for (const t of (dfa().trans.get(r) || new Map()).values()) if (!redSet.has(t)) frontier.add(t);
        blue[side] = [...frontier].filter((u) => wit.has(u)).sort((u, v) => shortlexCmp(wit.get(u), wit.get(v)));
        if (!blue[side].length) break;
        const u = blue[side][0];
        const redList = [...red[side]].sort((x, y) => shortlexCmp(wit.get(x) ?? "", wit.get(y) ?? ""));
        emit({ kind: "pick", side, u, text: `${side === "L" ? "Left" : "Right"} pass: the next blue state is ${u} (representative "${repr(u) || "ε"}"). Try merging it into each red state, in order.` });

        let merged = false;
        for (const r of redList) {
          if (r === u) continue;
          const res = tryMerge(hypo, side, r, u, alpha);
          const where = `${u} → ${r}`;
          if (res.conflict) {
            const [q, a, p] = res.conflict.entry;
            emit({
              kind: "reject-fold", side, u, r, classes: res.classes, folds: res.folds, conflict: res.conflict,
              text: `Try ${where}: merge-and-fold identifies ${describeClasses(res.classes)}, but then ω(${q}, ${a}, ${p}) would need two different outputs ("${res.conflict.outs[0] || "ε"}" vs "${res.conflict.outs[1] || "ε"}"). Rejected.`,
            });
            continue;
          }
          const bad = data.find(([x, y]) => !res.cand.matches(x, y));
          if (bad) {
            emit({
              kind: "reject-data", side, u, r, classes: res.classes, folds: res.folds, example: bad,
              text: `Try ${where}: the merged hypothesis no longer reproduces training example "${bad[0]}". Rejected.`,
            });
            continue;
          }
          const tests = witnessStrings(
            (w) => (side === "L" ? wit.get(u) + w : w + rev(wit.get(u))),
            alpha, trainSet, kWitness, maxLen);
          const witnesses = tests.map((x) => [x, oracle.query(x)]);
          const aug = augment(res.cand, witnesses);
          if (aug.conflict) {
            const c = aug.conflict;
            emit({
              kind: "reject-witness", side, u, r, classes: res.classes, folds: res.folds, witnesses, conflict: c,
              text: `Try ${where}: consistent with the training data, so ask the oracle about ${witnesses.length} witness strings. For "${c.witness}" the oracle's chunk at position ${c.position + 1} is "${c.outs[1] || "ε"}", but the merged hypothesis already has "${c.outs[0] || "ε"}" there. Rejected.`,
            });
            continue;
          }
          const before = dfa().states().length;
          emit({
            kind: "accept", side, u, r, classes: res.classes, folds: res.folds, witnesses,
            text: `Try ${where}: merge-and-fold identifies ${describeClasses(res.classes)}; the result agrees with the training data and with all ${witnesses.length} witness answers from the oracle. Accepted.`,
          });
          hypo = aug.cand;
          mergeRecord.push([side, wit.get(u), wit.get(r)]);
          red[side] = [dfa().start];
          blue[side] = [];
          emit({
            kind: "merged", side, u, r, witnesses,
            newStates: { L: aug.newL, R: aug.newR },
            text: `After merging ${where}, the ${side === "L" ? "left" : "right"} automaton has ${dfa().states().length} states (it had ${before}). The witness strings and their answers were added as new paths and output entries (dashed), so later merges must respect them too. The pass restarts from the root.`,
          });
          changedAny = true;
          merged = true;
          mergedInRestart = true;
          break;
        }
        if (merged) break;
        red[side] = [...red[side], u];
        blue[side] = blue[side].filter((s) => s !== u);
        emit({ kind: "promote", side, u, text: `No red state accepts ${u}, so ${u} is promoted to red.` });
      }
      if (!mergedInRestart) break;
    }
    blue[side] = [];
    emit({ kind: "pass-end", side, text: `${side === "L" ? "Left" : "Right"} pass finished: no blue states remain.` });
    return changedAny;
  }

  for (let round = 1; round <= maxRounds; round++) {
    emit({ kind: "round", round, text: `Round ${round}: a left merge pass, then a right merge pass.` });
    const chL = pass("L");
    const chR = pass("R");
    if (!chL && !chR) {
      emit({ kind: "converged", round, text: `No merge was accepted in round ${round}, so merging stops.` });
      break;
    }
  }
  red = { L: [], R: [] };
  blue = { L: [], R: [] };

  // Completion and alternating minimization.
  hypo = fillOmega(hypo, oracle, alpha, emit);
  for (let it = 1; it <= minIters; it++) {
    for (const side of ["L", "R"]) {
      const before = (side === "L" ? hypo.left : hypo.right).states().length;
      const beforeSnap = hypo.toJSON();
      const res = minimizeSide(hypo, side, alpha);
      hypo = res.next;
      const after = (side === "L" ? hypo.left : hypo.right).states().length;
      minSteps.push([side, before, after]);
      // Later iterations are shown only when they change something.
      if (it === 1 || before !== after) emit({
        kind: "minimize", side, iteration: it, part: [...res.part], rounds: res.rounds.map((m) => [...m]), signatures: res.signatures,
        snapshot: beforeSnap,
        classes: blocksOf(res.part),
        before, after,
        text: `Minimize the ${side === "L" ? "left automaton: compare each state's output row" : "right automaton: compare each state's output column"} (over all ${side === "L" ? "right states" : "left states"} and symbols) together with where its transitions lead. ${before === after ? `All ${before} states are distinguishable; nothing merges.` : `${before} states collapse to ${after} (ringed states merge).`}`,
      });
      hypo = fillOmega(hypo, oracle, alpha, (ev) => {
        if (ev.queries.length) emit(ev);
      });
    }
  }

  const rn = renumber(hypo, alpha);
  hypo = rn.hypo;
  emit({
    kind: "done",
    text: `Done: renumber states breadth-first. The learned bimachine has ${hypo.left.states().length} left states, ${hypo.right.states().length} right states and ${hypo.omega.size} output entries, after ${oracle.log.length} distinct oracle queries.`,
  });
  } catch (err) {
    if (!(err instanceof StepLimit)) throw err;
    emit({
      kind: "stopped",
      text: `Stopped after ${maxEvents} steps. With this sample and witness budget the witness paths added by each merge keep creating new states to merge, so the run would take very long. Try fewer or longer training strings, or a smaller witness budget.`,
    });
    return { hypothesis: hypo, events, merges: mergeRecord, queries: [...oracle.log], minSteps, stopped: true };
  }

  return { hypothesis: hypo, events, merges: mergeRecord, queries: [...oracle.log], minSteps, stopped: false };
}

class StepLimit extends Error {}

function blocksOf(part) {
  const blocks = new Map();
  for (const [s, b] of part) {
    if (!blocks.has(b)) blocks.set(b, []);
    blocks.get(b).push(s);
  }
  return [...blocks.values()].filter((c) => c.length > 1);
}

function describeClasses(classes) {
  if (!classes.length) return "no states";
  return classes.map((c) => `{${c.join(", ")}}`).join(" and ");
}

// Canonical description of a learned machine (for parity tests).
export function canonical(hypo) {
  const dfaRows = (d) => d.states().map((s) => [s, [...(d.trans.get(s) || new Map())].sort()]);
  const omega = [...hypo.omega].map(([k, v]) => [...unkey(k), v.out]).sort((a, b) => (a[0] - b[0]) || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0) || (a[2] - b[2]));
  return { left: dfaRows(hypo.left), right: dfaRows(hypo.right), omega };
}
