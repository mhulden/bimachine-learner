#!/usr/bin/env python3
"""Parity test: the JS learner (jsdemo/learner.js) against rpni_bimachine.py.

The Python learner is run with witness candidates in shortlex order (no shuffle),
which is how the JS port orders them. For each configuration both learners get
the same training inputs; the test compares

  * the final (renumbered) bimachine,
  * the sequence of accepted merges, identified by the representatives of the
    merged states (state ids differ between the two implementations),
  * the set of distinct oracle queries,

and also checks that every JS oracle agrees with its Python counterpart on all
words up to length 6.

Run from the repository root:  python3 jsdemo/test/parity.py   (about a minute)
Requires Node.js.
"""
import itertools
import json
import os
import random
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import rpni_bimachine as rb  # noqa: E402


class NoShuffle(random.Random):
    def shuffle(self, x, *args, **kwargs):
        pass


rb.random = types.SimpleNamespace(Random=NoShuffle)


class Figure1:
    def chunk(self, x):
        return tuple(c.upper() if i > 0 and x[i - 1] == "a" and "a" in x[i + 1:] else c for i, c in enumerate(x))


ORACLES = {"figure1": (("a", "b"), Figure1())}
for key, spec in rb.build_test_registry().items():
    ORACLES[key] = (spec.alphabet, spec.oracle)


def run_python(cfg):
    alphabet, base = ORACLES[cfg["oracle"]]
    queries = set()

    class Counting:
        def chunk(self, x):
            queries.add("".join(x))
            return base.chunk(x)

    merges, last = [], {}
    tml, tmr, aug = rb.try_merge_left, rb.try_merge_right, rb.augment_hypothesis_with_examples

    def try_left(hypo, into, frm, alpha):
        last.update(side="L", wit=rb.shortest_witnesses(hypo.left, alpha), into=into, frm=frm)
        return tml(hypo, into, frm, alpha)

    def try_right(hypo, into, frm, alpha):
        last.update(side="R", wit=rb.shortest_witnesses(hypo.right, alpha), into=into, frm=frm)
        return tmr(hypo, into, frm, alpha)

    def augment(hypo, examples):
        out = aug(hypo, examples)
        if out is not None:
            w = last["wit"]
            merges.append([last["side"], "".join(w[last["frm"]]), "".join(w[last["into"]])])
        return out

    rb.try_merge_left, rb.try_merge_right, rb.augment_hypothesis_with_examples = try_left, try_right, augment
    try:
        oracle = rb.MemoizedChunkOracle(Counting())
        train = [tuple(w) for w in cfg["train"]]
        bm = rb.learn_bimachine_from_oracle(
            oracle, list(alphabet), rb.label_dataset(oracle, train), seed=0,
            k_witness=cfg["k"], cont_max_len=cfg["len"], head_max_len=cfg["len"], verbose=False,
        )
    finally:
        rb.try_merge_left, rb.try_merge_right, rb.augment_hypothesis_with_examples = tml, tmr, aug

    def rows(d):
        return [[s, sorted([a, t] for a, t in d.trans.get(s, {}).items())] for s in sorted(d.states())]

    omega = sorted([q, a, p, out] for (q, a, p), out in bm.omega.items())
    return {"machine": {"left": rows(bm.left), "right": rows(bm.right), "omega": omega},
            "merges": merges, "queries": sorted(queries)}


def make_configs():
    configs = []
    rng = random.Random(12345)

    def add(oracle, n, max_len, k, length, seed):
        alphabet = ORACLES[oracle][0]
        train = ["".join(x) for x in rb.make_random_inputs(alphabet, n=n, max_len=max_len, seed=seed)]
        configs.append({"id": len(configs), "oracle": oracle, "train": train, "k": k, "len": length})

    # The guided walkthrough itself (jsdemo/presets.js: WALKTHROUGH).
    configs.append({"id": 0, "oracle": "figure1", "train": ["bab", "bbbb", "aa", "baba", "abbab", "baa"], "k": 6, "len": 2})
    for seed in range(60):  # the walkthrough regime, including failures on tiny samples
        add("figure1", rng.randint(2, 10), rng.randint(3, 6), rng.choice([2, 4, 6, 14]), rng.choice([1, 2, 3]), seed)
    for key in ORACLES:
        if key == "figure1":
            continue
        # Small samples with large witness budgets can make the witness paths grow
        # for a very long time (in both implementations), so budgets stay modest here.
        for seed in range(6):
            add(key, rng.randint(4, 30), rng.randint(3, 7), rng.choice([4, 6, 10, 14]), 2, seed)
    return configs


def main():
    configs = make_configs()
    checks = []
    for key, (alphabet, _) in ORACLES.items():
        words = ["".join(w) for L in range(7) for w in itertools.product(alphabet, repeat=L)]
        checks.append({"oracle": key, "words": words})

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"configs": configs, "oracleChecks": checks}, fh)
        path = fh.name
    try:
        out = subprocess.run(["node", os.path.join(HERE, "run_js.mjs"), path], capture_output=True, text=True, check=True)
    finally:
        os.unlink(path)
    js = json.loads(out.stdout)

    failures = 0
    for chk in checks:
        _, oracle = ORACLES[chk["oracle"]]
        py = [list(oracle.chunk(tuple(w))) for w in chk["words"]]
        if py != js["oracleOutputs"][chk["oracle"]]:
            failures += 1
            print(f"oracle mismatch: {chk['oracle']}")

    for cfg, res in zip(configs, js["results"]):
        if res.get("stopped"):
            failures += 1
            print(f"config {cfg['id']}: JS run hit its step limit")
            continue
        py = run_python(cfg)
        for field in ("machine", "merges", "queries"):
            if py[field] != res[field]:
                failures += 1
                print(f"config {cfg['id']} ({cfg['oracle']}, n={len(cfg['train'])}, k={cfg['k']}, len={cfg['len']}): {field} differs")
                break

    print(f"{len(checks)} oracles, {len(configs)} learner configurations: "
          f"{'all identical' if failures == 0 else f'{failures} mismatches'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
