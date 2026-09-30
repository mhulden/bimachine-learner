#!/usr/bin/env python3
"""Oracle-assisted benchmark battery (paper Section 4.1-4.2, Table 1).

For each of the ten tasks: 2,000 random training strings of length <= 12
(seed 0), witness budget k=80 with continuation/head length <= 3, at most 10
merge rounds, three alternating minimization iterations. Reports learned
|Q_L|, |Q_R|, |omega|, exact chunk accuracy on 500 test strings, and chunk-level
oracle agreement on 500 further random strings of length <= 25.

Usage: python3 run_battery.py [--out results/battery.jsonl]
"""
import argparse
import json
import os
import time

import rpni_bimachine as rb

# Registry key -> name used in the paper, in paper order.
TASKS = [
    ("swap_first_last_ab", "Swap first/last"),
    ("future_c_abc", "Future-c condition"),
    ("global_begin_end_b_ab", "Global begin/end"),
    ("local_cad_abcd", "Local cad"),
    ("global_or_local_abcd", "Global-or-local"),
    ("c_parity_left_even_right_odd_abc", "Parity left/right"),
    ("swap_ab_if_even_length_abc", "Even-length a<->b"),
    ("insert_a_between_cd_abcd", "Anchored insertion"),
    ("delete_a_even_else_delete_b_ab", "Conditional deletion"),
    ("delete_a_parity_by_last_symbol_ab", "Last-symbol parity deletion"),
]


def run_task(key, seed=0, n_train=2000, max_len_train=12, n_test=500, max_len_test=25):
    spec = rb.build_test_registry()[key]
    alphabet = list(spec.alphabet)
    oracle = rb.MemoizedChunkOracle(spec.oracle)
    train = rb.label_dataset(oracle, rb.make_random_inputs(spec.alphabet, n=n_train, max_len=max_len_train, seed=seed))
    test = rb.label_dataset(oracle, rb.make_random_inputs(spec.alphabet, n=n_test, max_len=max_len_test, seed=seed + 1337))
    t0 = time.time()
    bm = rb.learn_bimachine_from_oracle(
        oracle=oracle, alphabet=alphabet, train=train, seed=seed,
        max_rounds=10, k_witness=80, cont_max_len=3, head_max_len=3, verbose=False,
    )
    seconds = time.time() - t0
    return {
        "task": key,
        "q_left": len(bm.left.states()),
        "q_right": len(bm.right.states()),
        "omega": len(bm.omega),
        "test_chunk_accuracy": rb.exact_chunk_accuracy(bm, test),
        "oracle_agreement": rb.random_oracle_agreement(bm, oracle, alphabet, max_len=max_len_test, n=500, seed=seed + 2025),
        "learn_seconds": round(seconds, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/battery.jsonl")
    args = ap.parse_args()
    rows = []
    print(f"{'Task':30s} {'|Q_L|':>5s} {'|Q_R|':>5s} {'|omega|':>7s} {'test acc':>9s} {'agreement':>9s}")
    for key, name in TASKS:
        row = run_task(key)
        rows.append(row)
        print(f"{name:30s} {row['q_left']:5d} {row['q_right']:5d} {row['omega']:7d} "
              f"{row['test_chunk_accuracy']:9.3f} {row['oracle_agreement']:9.3f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
