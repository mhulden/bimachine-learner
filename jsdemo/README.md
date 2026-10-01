# Interactive demo: learning a bimachine step by step

A browser version of the oracle-assisted learner that records every step so you can follow it: prefix trees, red/blue merge passes with merge-and-fold, witness queries to the oracle, output-table completion, and minimization.

## Running it

The page uses ES modules, so it must be served over HTTP rather than opened as a file:

```bash
cd jsdemo
python3 -m http.server
```

Then open <http://localhost:8000/>. There is no build step and nothing to install. The page loads no external resources and works offline.

## What it shows

- **Guided: Figure 1.** The learner recovers the bimachine of Figure 1 in the paper: capitalize a symbol if its left neighbour is `a` and another `a` follows. It uses six training inputs and a witness budget of k = 6 with witness length ≤ 2, and makes 26 distinct oracle queries in 63 steps. Every kind of step is explained the first time it occurs, and the final machine is Figure 1's, with states numbered from 0.
- **Try your own.** Pick the Figure 1 function or any of the paper's ten benchmark tasks, edit the training inputs, the witness budget or the oracle itself (a JavaScript function returning one output chunk per input symbol), and run the learner.

Each step shows:
- the left and right automata, with red (settled) and blue (candidate) states, rings around the states that merge-and-fold identifies, and dashed states added by witness strings;
- the output table ω(q, a, p), coloured by where each entry came from, with entries new in the step and conflicts outlined;
- the oracle's answers for the step: witness strings, completion queries, or the training data;
- during minimization, each state's output row or column and its block;
- a box to run any string through the current hypothesis and compare it with the oracle, including the input `aabbabaabb` from Figure 1.

Controls: the step buttons, the slider, ← / → and the space bar. `#step=N` in the URL opens the walkthrough at step N, and `#custom=TASK` (for example `#custom=local_cad_abcd`) opens a random sample for a task.

Small samples with a large witness budget can make the learner run for a very long time. Each accepted merge adds its witness strings as new paths, and those can keep producing new states to merge. The Python learner behaves the same way. The demo stops after 20,000 steps or 10 seconds of computing, whichever comes first, and lets you browse what happened up to that point.

## Relation to the Python learner

`learner.js` is a port of `learn_bimachine_from_oracle` in `../rpni_bimachine.py`. The only algorithmic difference is that witness candidates are taken in shortlex order instead of a seeded random order. For the walkthrough this makes no difference, because its budget covers every candidate. `test/parity.py` runs both learners on the same inputs, with the Python learner also using shortlex witness order. It checks that the final machines, the sequence of accepted merges and the set of oracle queries are identical. It covers 121 configurations: the walkthrough, 60 small Figure 1 samples, and six samples for each of the ten benchmark tasks. It also checks that the eleven JavaScript oracles agree with the Python ones on every word up to length 6. It needs Node.js and takes about two minutes:

```bash
python3 jsdemo/test/parity.py      # from the repository root
```

## Files

| File | Contents |
|---|---|
| `index.html`, `style.css`, `viewer.js` | The page and its rendering (automata layout, tables, narration) |
| `learner.js` | The learner, recording one event per step |
| `presets.js` | The Figure 1 oracle, the ten benchmark oracles, and the walkthrough settings |
| `test/parity.py`, `test/run_js.mjs` | Parity test against the Python learner |
