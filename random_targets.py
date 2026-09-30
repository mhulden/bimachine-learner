#!/usr/bin/env python3
"""Sample-only recovery on random target bimachines (paper Section 4.4).

Three families (ia = input alphabet size, oa = output alphabet size, c = max
chunk length), target sizes 2x2, 4x4, 6x6, 10x10, training sizes
n = 50, 100, 200, 400, and five random targets per point: 240 runs.

For each run:
  * draw a random total target bimachine and normalize it (minimization and
    breadth-first renumbering), rejecting targets whose normalized sides shrink;
  * draw n distinct training strings with lengths uniform in {0..6};
  * learn with the sample-only learner (no oracle queries);
  * evaluate on 700 test strings with lengths uniform in {0..10} that do not
    occur in the training sample (held-out agreement = exact chunk-sequence match);
  * record exact normalized recovery, the share of test strings that hit an
    undefined state or output entry (identity fallback), and, once per target,
    the share of output entries exercised by any string of length <= 6.

Writes one JSON line per run and prints the Table 3 summary plus the fallback
and coverage figures quoted in the text.

Usage: python3 random_targets.py [--workers 2] [--out results/random_targets.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import rpni_bimachine as rb

FAMILIES = [  # (label, input alphabet size, output alphabet size, max chunk length)
    ("ia=3, oa=2, c<=2", 3, 2, 2),
    ("ia=4, oa=2, c<=3", 4, 2, 3),
    ("ia=5, oa=4, c<=4", 5, 4, 4),
]
SIZES = [2, 4, 6, 10]
N_TRAIN = [50, 100, 200, 400]
INSTANCES = 5
TRAIN_MAX_LEN = 6
TEST_MAX_LEN = 10
N_TEST = 700
MAX_ROUNDS = 12
MAX_TARGET_ATTEMPTS = 200

# Seed layout: instance_seed = BASE_SEED + 100000*family + 10000*slot + instance.
# The size slots follow the original five-column grid (2, 4, 6, 8, 10); the 8x8
# column is not reported in the paper, so slot 3 is unused here.
BASE_SEED = 4100
SIZE_SLOT = {2: 0, 4: 1, 6: 2, 8: 3, 10: 4}

INPUT_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
OUTPUT_SYMBOLS = "01xyzuvw"  # disjoint from the input symbols
FALLBACK = "<?>"  # marker Bimachine.transduce_chunks uses for an undefined position


# --------------------------------------------------------------------------
# Random targets
# --------------------------------------------------------------------------

@dataclass
class BimachineOracle:
    bm: rb.Bimachine
    eps: str = ""

    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        return self.bm.transduce(x)


def random_total_dfa(num_states: int, alphabet: Sequence[str], rng: random.Random) -> rb.DFA:
    """Uniform random transitions; resample until every state is reachable."""
    while True:
        trans: Dict[int, Dict[str, int]] = {s: {} for s in range(num_states)}
        for s in range(num_states):
            for a in alphabet:
                trans[s][a] = rng.randrange(num_states)
        dfa = rb.DFA(start=0, trans=trans, sink=None)
        dfa.prune_unreachable(list(alphabet))
        if len(dfa.states()) == num_states:
            return dfa


def random_chunk(output_alphabet: Sequence[str], rng: random.Random, max_chunk_len: int) -> str:
    length = rng.randint(0, max_chunk_len)
    return "".join(rng.choice(output_alphabet) for _ in range(length))


def random_total_bimachine(input_alphabet, output_alphabet, n_left, n_right, max_chunk_len, seed) -> rb.Bimachine:
    rng = random.Random(seed)
    left = random_total_dfa(n_left, input_alphabet, rng)
    right = random_total_dfa(n_right, input_alphabet, rng)
    omega = {}
    for ql in sorted(left.states()):
        for a in sorted(input_alphabet):
            for qr in sorted(right.states()):
                omega[(ql, a, qr)] = random_chunk(output_alphabet, rng, max_chunk_len)
    return rb.Bimachine(left=left, right=right, omega=omega, eps="")


def normalize_bimachine(target: rb.Bimachine, alphabet: Sequence[str]) -> rb.Bimachine:
    oracle = rb.MemoizedChunkOracle(BimachineOracle(target, eps=target.eps))
    oracle.eps = target.eps
    bm = rb.minimize_bimachine(target, oracle, list(alphabet), iters=3)
    return rb.renumber_bimachine(bm, list(alphabet))


def draw_target(input_alphabet, output_alphabet, size, max_chunk_len, seed) -> rb.Bimachine:
    """Normalized random target whose normalized sides keep the requested size."""
    for attempt in range(MAX_TARGET_ATTEMPTS):
        raw = random_total_bimachine(input_alphabet, output_alphabet, size, size, max_chunk_len, seed + attempt)
        norm = normalize_bimachine(raw, input_alphabet)
        if len(norm.left.states()) >= size and len(norm.right.states()) >= size:
            return norm
    raise RuntimeError("failed to draw a target matching the normalized-size constraints")


def bimachine_signature(bm: rb.Bimachine, alphabet: Sequence[str]) -> Tuple:
    """Transition tables plus all defined output entries (for exact recovery)."""
    alpha = tuple(sorted(alphabet))

    def dfa_signature(dfa):
        states = tuple(sorted(dfa.states()))
        return (dfa.start, dfa.sink, tuple((s, tuple((a, dfa.step(s, a)) for a in alpha)) for s in states))

    omega = tuple(sorted((ql, a, qr, out) for (ql, a, qr), out in bm.omega.items() if a in alpha))
    return (dfa_signature(bm.left), dfa_signature(bm.right), omega)


def distinct_random_inputs(alphabet, n, max_len, seed) -> List[Tuple[str, ...]]:
    rng = random.Random(seed)
    seen, xs = set(), []
    while len(xs) < n:
        x = tuple(rng.choice(alphabet) for _ in range(rng.randint(0, max_len)))
        if x not in seen:
            seen.add(x)
            xs.append(x)
    return xs


def omega_coverage(target: rb.Bimachine, alphabet, max_len: int) -> float:
    """Share of the target's output entries exercised by some string of length <= max_len."""
    seen = set()
    for length in range(max_len + 1):
        for x in itertools.product(alphabet, repeat=length):
            q = target.left.run_prefix_states(x)
            p = target.right.run_right_states_after(x)
            for i, a in enumerate(x):
                seen.add((q[i], a, p[i + 1]))
    return len(seen & set(target.omega)) / len(target.omega)


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def run_one(job) -> dict:
    fam_idx, size, n_train, instance = job
    label, ia_size, oa_size, max_chunk = FAMILIES[fam_idx]
    seed = BASE_SEED + 100000 * fam_idx + 10000 * SIZE_SLOT[size] + instance
    ia = list(INPUT_SYMBOLS[:ia_size])
    oa = list(OUTPUT_SYMBOLS[:oa_size])

    target = draw_target(ia, oa, size, max_chunk, seed)
    oracle = rb.MemoizedChunkOracle(BimachineOracle(target, eps=target.eps))
    oracle.eps = target.eps

    pool = distinct_random_inputs(ia, n_train, TRAIN_MAX_LEN, seed + 777777)
    train = [(x, oracle.chunk(x)) for x in pool]
    learned = rb.learn_bimachine_from_examples(
        alphabet=ia, train=train, seed=seed, max_rounds=MAX_ROUNDS, min_iters=2, verbose=False,
    )
    learned = rb.renumber_bimachine(learned, ia)

    train_set = set(pool)
    rng = random.Random(seed + 888888)
    drawn = correct = fallback = correct_with_fallback = 0
    while drawn < N_TEST:
        x = tuple(rng.choice(ia) for _ in range(rng.randint(0, TEST_MAX_LEN)))
        if x in train_set:
            continue
        drawn += 1
        ok = oracle.chunk(x) == learned.transduce(x)
        used_fallback = FALLBACK in learned.transduce_chunks(x)
        correct += ok
        fallback += used_fallback
        correct_with_fallback += ok and used_fallback

    row = {
        "family": label, "size": f"{size}x{size}", "n_train": n_train, "instance": instance,
        "instance_seed": seed,
        "target_sizes": {"q_left": len(target.left.states()), "q_right": len(target.right.states()), "omega": len(target.omega)},
        "learned_sizes": {"q_left": len(learned.left.states()), "q_right": len(learned.right.states()), "omega": len(learned.omega)},
        "train_fit": sum(learned.matches_example(x, y) for x, y in train) / len(train),
        "heldout_agreement": correct / N_TEST,
        "exact_normalized_recovery": bimachine_signature(target, ia) == bimachine_signature(learned, ia),
        "fallback_share": fallback / N_TEST,
        "correct_with_fallback": correct_with_fallback,
    }
    if n_train == N_TRAIN[-1]:  # coverage depends only on the target
        row["omega_coverage_len_le_6"] = omega_coverage(target, ia, TRAIN_MAX_LEN)
    return row


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def summarize(rows: List[dict]) -> None:
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["family"], r["size"], r["n_train"])].append(r)

    def mean(key, fam, size, n):
        v = cells[(fam, size, n)]
        return sum(float(r[key]) for r in v) / len(v)

    print(f"{len(rows)} runs; train fit 1.0 in all: {all(r['train_fit'] == 1.0 for r in rows)}")
    print("\nHeld-out agreement / exact normalized recovery (Table 3 uses 4x4, 6x6, 10x10):")
    print(f"{'family':18s} {'size':6s} " + " ".join(f"{'n=' + str(n):>12s}" for n in N_TRAIN))
    for fam, *_ in FAMILIES:
        for size in SIZES:
            s = f"{size}x{size}"
            print(f"{fam:18s} {s:6s} " + " ".join(
                f"{mean('heldout_agreement', fam, s, n):.3f} / {mean('exact_normalized_recovery', fam, s, n):.1f}".rjust(12)
                for n in N_TRAIN))

    print("\nShare of test strings hitting the identity fallback at n=50:")
    for fam, *_ in FAMILIES:
        print(f"  {fam:18s} " + "  ".join(f"{size}x{size}: {mean('fallback_share', fam, f'{size}x{size}', 50):.3f}" for size in SIZES))
    print(f"Correct answers that used the fallback: {sum(r['correct_with_fallback'] for r in rows)}")

    print("\nShare of output entries exercised by strings of length <= 6 (per target):")
    for fam, *_ in FAMILIES:
        for size in SIZES:
            cov = [round(r["omega_coverage_len_le_6"], 3) for r in cells[(fam, f"{size}x{size}", N_TRAIN[-1])]]
            print(f"  {fam:18s} {size}x{size:<4d} {cov}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/random_targets.jsonl")
    args = ap.parse_args()

    jobs = [(f, size, n, i) for f in range(len(FAMILIES)) for size in SIZES for n in N_TRAIN for i in range(INSTANCES)]
    with mp.Pool(args.workers) as pool:
        rows = pool.map(run_one, jobs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    summarize(rows)


if __name__ == "__main__":
    main()
