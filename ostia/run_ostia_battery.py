#!/usr/bin/env python3
"""OSTIA baseline on the benchmark battery (paper Section 4.3, Table 2).

Training: the same 2,000 random strings (length <= 12, seed 0) as run_battery.py.
Test: the same 500 strings (length <= 25) used for oracle agreement there.
OSTIA sees whole (concatenated) outputs only; inputs on which the learned
transducer is undefined count as errors. The paper reports lexicographic order.

Build the learner first:  make -C ostia
Usage: python3 ostia/run_ostia_battery.py [--order lex|shortlex|both] [--out results/ostia_battery.jsonl]
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "ostia", "ostia")
sys.path.insert(0, ROOT)
import rpni_bimachine  # noqa: E402

TASKS = [
    "swap_first_last_ab",
    "future_c_abc",
    "global_begin_end_b_ab",
    "local_cad_abcd",
    "global_or_local_abcd",
    "c_parity_left_even_right_odd_abc",
    "swap_ab_if_even_length_abc",
    "insert_a_between_cd_abcd",
    "delete_a_even_else_delete_b_ab",
    "delete_a_parity_by_last_symbol_ab",
]


def write_examples(path, words, f):
    with open(path, "w") as fh:
        for x in words:
            fh.write("".join(x) + "\t" + f(x) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=["lex", "shortlex", "both"], default="lex")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--max_len_train", type=int, default=12)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--max_len_test", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ostia_battery.jsonl")
    args = ap.parse_args()

    m = rpni_bimachine
    registry = m.build_test_registry()
    orders = ["lex", "shortlex"] if args.order == "both" else [args.order]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for task in args.tasks:
            spec = registry[task]
            alphabet = spec.alphabet
            f = lambda x: "".join(spec.oracle.chunk(x))
            train = m.make_random_inputs(alphabet, n=args.n_train, max_len=args.max_len_train, seed=args.seed)
            rng = random.Random(args.seed + 2025)
            test = []
            for _ in range(args.n_test):
                length = rng.randint(0, args.max_len_test)
                test.append(tuple(rng.choice(alphabet) for _ in range(length)))
            trp, tep = os.path.join(tmp, "train.tsv"), os.path.join(tmp, "test.tsv")
            write_examples(trp, train, f)
            write_examples(tep, test, f)
            for order in orders:
                cmd = [BIN] + (["-s"] if order == "shortlex" else []) + [trp, tep]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise SystemExit(f"{task} {order}: {res.stderr or res.stdout}")
                row = {"task": task, "n_train": args.n_train, "max_len_train": args.max_len_train,
                       "seed": args.seed, **json.loads(res.stdout)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
