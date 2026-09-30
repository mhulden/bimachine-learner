# bimachine-learner

Code and data for the paper

> Mans Hulden and Michael Ginn. **Learning Bimachines from Aligned Examples.** ICGI 2026 (PMLR).

A bimachine computes a functional string transduction with a left-to-right DFA, a right-to-left DFA, and a local output table ω(q, a, p) that emits one output chunk per input position. This repository contains an RPNI-style learner that recovers all three from *aligned* examples, meaning input strings paired with one output chunk per input symbol (chunks may be empty or longer than one symbol, so deletions and anchored insertions are supported). It also contains the scripts that reproduce every table and the random-target figure in the paper.

## Contents

| File | What it is | Paper |
|---|---|---|
| `rpni_bimachine.py` | The learner (library and command-line tool) and the ten benchmark chunk oracles | §3, §4.1 |
| `run_battery.py` | Oracle-assisted benchmark battery | Table 1 |
| `ostia/ostia.c`, `ostia/run_ostia_battery.py` | OSTIA baseline in C and its driver | Table 2 |
| `random_targets.py` | Sample-only learning on random target bimachines | Table 3, §4.4 |
| `plot_random_targets.py` | Plot of the random-target results | Figure 3 |
| `results/` | Outputs of the scripts above, as used in the paper | |

## Requirements

- Python 3 (tested with 3.12). The learner and experiment scripts use only the standard library.
- `matplotlib`, for `plot_random_targets.py` only (`pip install -r requirements.txt`).
- A C compiler and `make`, for the OSTIA baseline.
- Optional: the `graphviz` Python package and the Graphviz `dot` binary, to render learned machines with `rpni_bimachine.py --view`.

## Reproducing the paper

All runs are deterministic (fixed seeds). Times are for one laptop-class core.

**Table 1: benchmark battery** (about 10 seconds)

```bash
python3 run_battery.py
```

It prints |Q_L|, |Q_R|, |ω|, test-set chunk accuracy and oracle agreement for each task, and writes `results/battery.jsonl`. Settings: 2,000 training strings of length ≤ 12 (seed 0), witness budget k = 80, continuation and head length ≤ 3, at most 10 merge rounds, three alternating minimization iterations. Agreement is exact chunk-sequence agreement on 500 random strings of length ≤ 25.

**Table 2: OSTIA baseline** (about 1 second)

```bash
make -C ostia
python3 ostia/run_ostia_battery.py
```

It writes `results/ostia_battery.jsonl`. OSTIA receives the same training and test inputs as Table 1, but only whole (concatenated) outputs, not chunks. States are merged in lexicographic order of their access strings; `--order shortlex` or `--order both` runs the length-first order as well. Inputs on which the learned transducer is undefined count as errors.

**Table 3, Figure 3 and the fallback/coverage figures in §4.4** (about 1 minute)

```bash
python3 random_targets.py            # 240 runs, writes results/random_targets.jsonl
python3 plot_random_targets.py       # writes results/random_targets.pdf and .png
```

`random_targets.py` prints held-out agreement / exact normalized recovery for every family, target size and training size; Table 3 is the 4×4, 6×6 and 10×10 rows. It also prints the share of test strings that hit an undefined state or output entry, and the share of each target's output entries reachable by strings of length ≤ 6. Both are quoted in §4.4. Use `--workers N` to change the number of processes (default 2).

The setup is three families (ia = input alphabet size, oa = output alphabet size, c = maximum chunk length), target sizes 2×2, 4×4, 6×6 and 10×10, and training sizes 50, 100, 200 and 400, with five random targets per point.
- **Targets:** random total bimachines, normalized by minimization and breadth-first renumbering, whose output symbols are disjoint from the input symbols.
- **Training data:** distinct random strings with lengths uniform in 0–6.
- **Learning:** sample-only, with no oracle queries.
- **Evaluation:** 700 random strings with lengths uniform in 0–10, excluding every training string.
- **Exact normalized recovery:** compares both transition tables and all defined output entries after normalization.

## Using the learner

Run one benchmark task with the paper's settings:

```bash
python3 rpni_bimachine.py --test future_c_abc
python3 rpni_bimachine.py --help        # all tasks and options
python3 rpni_bimachine.py --test swap_first_last_ab --view all --format pdf --out swap   # needs Graphviz
```

Learn from your own chunk oracle, i.e. any object with a `chunk(x)` method that returns one output chunk per input symbol:

```python
import rpni_bimachine as rb

class UpperAfterA:
    """Capitalize every symbol whose left neighbour is 'a'."""
    def chunk(self, x):
        return tuple(c.upper() if i > 0 and x[i - 1] == "a" else c for i, c in enumerate(x))

alphabet = ["a", "b"]
oracle = rb.MemoizedChunkOracle(UpperAfterA())
train = rb.label_dataset(oracle, rb.make_random_inputs(("a", "b"), n=200, max_len=8, seed=0))

# Oracle-assisted: witness validation and output-table completion query the oracle.
bm = rb.learn_bimachine_from_oracle(oracle, alphabet, train, k_witness=80, verbose=False)
rb.print_bimachine_tables(bm, alphabet)
print(bm.transduce(tuple("abaab")))     # ('a', 'B', 'a', 'A', 'B')

# Sample-only: learns from the aligned examples alone.
bm2 = rb.learn_bimachine_from_examples(alphabet, train, verbose=False)
```

In sample-only hypotheses, states or output entries may be undefined. `transduce` then emits the input symbol at that position (the identity fallback).

## Results

`results/` holds the outputs used in the paper:

- `battery.jsonl`: Table 1.
- `ostia_battery.jsonl`: Table 2.
- `random_targets.jsonl`: one line per random-target run. The fields are held-out agreement, exact recovery, fallback share, learned and target sizes, and, for n = 400, output-table coverage by strings of length ≤ 6.
- `random_targets.pdf` and `random_targets.png`: Figure 3.

Rerunning the scripts reproduces the same numbers and figure (PDF metadata such as creation dates will differ).

## License

MIT; see `LICENSE`.
