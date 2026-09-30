#!/usr/bin/env python3
"""Classical OSTIA (Oncina, Garcia, Vidal 1993): a readable Python version.

This is a line-for-line counterpart of ostia.c and produces identical results;
the C version exists only for speed.

OSTIA learns a subsequential transducer from input/output string pairs:

1. Build the prefix tree of the training inputs. Each tree state is named by the
   input prefix that reaches it; the full output of an example is stored as the
   final output of the state its input reaches.
2. Make the tree *onward*: push the longest common prefix of everything that can
   still be emitted below a state up onto the edge entering it (and onto the
   initial output at the root).
3. Visit states in lexicographic order of their names (or length-first "shortlex"
   order). Try to merge each state q into every earlier surviving state p in turn,
   and keep the first merge that succeeds.

A merge redirects q's incoming edge to p and *folds* q into p: q's final output
and outgoing edges are moved to p. When both have an edge on the same symbol,
the two edge outputs are cut back to their longest common prefix, the cut-off
remainders are pushed onto the target states' outgoing outputs, and the two
targets are folded recursively. A merge fails if final outputs conflict, or if
a push would change the output of an already processed ("red") state, i.e. a
state earlier in the visiting order than q.

Failed merges are undone through a change log rather than by copying the
transducer, which is what makes the algorithm fast.

Input files contain one example per line: INPUT<TAB>OUTPUT (one character per
symbol). Usage: python3 ostia.py [-s] TRAINFILE TESTFILE
Prints one JSON object with the learned state count, test accuracy, and the
number of test inputs on which the learned transducer is undefined.
"""
import json
import sys
import time

_MISSING = object()


class Ostia:
    def __init__(self, examples, shortlex=False):
        self.shortlex = shortlex
        # Input symbols in order of first appearance (fixes the fold order).
        self.symbols = []
        for x, _ in examples:
            for a in x:
                if a not in self.symbols:
                    self.symbols.append(a)

        # Per-state tables, indexed by state id.
        self.trans = []      # trans[s][a] = target state
        self.out = []        # out[s][a]   = output string on that edge
        self.final = []      # final[s]    = final output, or absent (None)
        self.alive = []      # False once a state has been merged away
        self.parent = []     # (parent state, symbol) of the edge entering s
        self.name = []       # input prefix reaching s in the prefix tree
        self.log = []        # change log for undoing failed merges
        self.initial = ""    # output emitted before reading any input
        self._build_tree(examples)

    # -- construction -------------------------------------------------------

    def _new_state(self, name):
        self.trans.append({})
        self.out.append({})
        self.final.append(None)
        self.alive.append(True)
        self.parent.append(None)
        self.name.append(name)
        return len(self.name) - 1

    def _build_tree(self, examples):
        root = self._new_state("")
        for x, y in examples:
            s = root
            for i, a in enumerate(x):
                if a not in self.trans[s]:
                    t = self._new_state(x[: i + 1])
                    self.trans[s][a] = t
                    self.out[s][a] = ""
                    self.parent[t] = (s, a)
                s = self.trans[s][a]
            if self.final[s] is not None and self.final[s] != y:
                raise ValueError(f"input {x!r} has conflicting outputs")
            self.final[s] = y
        self.initial = self._make_onward(root)

    def _make_onward(self, s):
        """Push common output prefixes towards the root; return the prefix removed at s."""
        for a in self._edges(s):
            child = self.trans[s][a]
            self.out[s][a] += self._make_onward(child)
        pending = [self.out[s][a] for a in self._edges(s)]
        if self.final[s] is not None:
            pending.append(self.final[s])
        prefix = common_prefix(pending)
        if prefix:
            for a in self._edges(s):
                self.out[s][a] = self.out[s][a][len(prefix):]
            if self.final[s] is not None:
                self.final[s] = self.final[s][len(prefix):]
        return prefix

    def _edges(self, s):
        return [a for a in self.symbols if a in self.trans[s]]

    # -- logged updates -----------------------------------------------------

    def _set(self, table, key, value):
        self.log.append((table, key, table.get(key, _MISSING) if isinstance(table, dict) else table[key]))
        table[key] = value

    def _undo(self, mark):
        while len(self.log) > mark:
            table, key, old = self.log.pop()
            if old is _MISSING:
                del table[key]
            else:
                table[key] = old

    # -- merge and fold -----------------------------------------------------

    def _push_back(self, s, suffix):
        """Prepend suffix to every output leaving s, including its final output."""
        if not suffix:
            return
        for a in self._edges(s):
            self._set(self.out[s], a, suffix + self.out[s][a])
        if self.final[s] is not None:
            self._set(self.final, s, suffix + self.final[s])

    def _fold(self, p, q, red_limit):
        """Fold state q into state p. Returns False if the merge is inconsistent."""
        self._set(self.alive, q, False)
        if self.final[q] is not None:
            if self.final[p] is None:
                self._set(self.final, p, self.final[q])
            elif self.final[p] != self.final[q]:
                return False
        for a in self._edges(q):
            y, w = self.trans[q][a], self.out[q][a]
            if a not in self.trans[p]:
                self._set(self.trans[p], a, y)
                self._set(self.out[p], a, w)
                self._set(self.parent, y, (p, a))
                continue
            x, v = self.trans[p][a], self.out[p][a]
            if x == y:
                if v != w:
                    return False
                continue
            u = len(common_prefix([v, w]))
            # Outputs of already processed states may not change.
            if (self.rank[x] < red_limit and len(v) != u) or (self.rank[y] < red_limit and len(w) != u):
                return False
            self._push_back(x, v[u:])
            self._push_back(y, w[u:])
            self._set(self.out[p], a, v[:u])
            keep, other = (x, y) if self.rank[x] < self.rank[y] else (y, x)
            if keep != x:
                self._set(self.trans[p], a, keep)
                self._set(self.parent, keep, (p, a))
            if not self._fold(keep, other, red_limit):
                return False
        return True

    def _try_merge(self, p, q, red_limit):
        parent, a = self.parent[q]
        self._set(self.trans[parent], a, p)
        return self._fold(p, q, red_limit)

    # -- main loop ----------------------------------------------------------

    def learn(self):
        if self.shortlex:
            key = lambda s: (len(self.name[s]), self.name[s])
        else:
            key = lambda s: self.name[s]
        order = sorted(range(len(self.name)), key=key)
        self.rank = [0] * len(order)
        for i, s in enumerate(order):
            self.rank[s] = i

        self.merge_attempts = 0
        for i in range(1, len(order)):
            q = order[i]
            if not self.alive[q]:
                continue
            for p in order[:i]:
                if not self.alive[p]:
                    continue
                mark = len(self.log)
                self.merge_attempts += 1
                if self._try_merge(p, q, red_limit=i):
                    self.log.clear()
                    break
                self._undo(mark)
        return self

    # -- use ----------------------------------------------------------------

    def translate(self, x):
        """Output for input x, or None if the learned transducer is undefined on x."""
        s, output = 0, [self.initial]
        for a in x:
            if a not in self.trans[s]:
                return None
            output.append(self.out[s][a])
            s = self.trans[s][a]
        if self.final[s] is None:
            return None
        output.append(self.final[s])
        return "".join(output)

    @property
    def num_states(self):
        return sum(self.alive)


def common_prefix(strings):
    if not strings:
        return ""
    first = min(strings)
    last = max(strings)
    i = 0
    while i < len(first) and first[i] == last[i]:
        i += 1
    return first[:i]


def read_examples(path):
    examples = []
    with open(path) as fh:
        for line in fh:
            x, y = line.rstrip("\r\n").split("\t")
            examples.append((x, y))
    return examples


def main(argv):
    shortlex = len(argv) > 1 and argv[1] == "-s"
    args = argv[2:] if shortlex else argv[1:]
    if len(args) != 2:
        sys.exit("usage: ostia.py [-s] TRAINFILE TESTFILE")
    train, test = read_examples(args[0]), read_examples(args[1])

    t0 = time.process_time()
    model = Ostia(train, shortlex=shortlex).learn()
    seconds = time.process_time() - t0

    replay_failures = sum(model.translate(x) != y for x, y in train)
    results = [model.translate(x) for x, _ in test]
    correct = sum(r == y for r, (_, y) in zip(results, test))
    undefined = sum(r is None for r in results)
    print(json.dumps({
        "order": "shortlex" if shortlex else "lex",
        "tree_states": len(model.name),
        "states": model.num_states,
        "merge_attempts": model.merge_attempts,
        "replay_failures": replay_failures,
        "test_n": len(test),
        "acc": round(correct / len(test), 6) if test else 0.0,
        "undefined": undefined,
        "learn_sec": round(seconds, 3),
    }))
    return 1 if replay_failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
