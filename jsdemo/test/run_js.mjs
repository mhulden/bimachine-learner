// Runs the JS learner on configurations from a JSON file (see parity.py).
// Usage: node run_js.mjs configs.json > results.json
import { readFileSync } from "node:fs";
import { learnBimachine, Oracle, canonical } from "../learner.js";
import { ORACLES, compileOracle } from "../presets.js";

const { configs, oracleChecks } = JSON.parse(readFileSync(process.argv[2], "utf8"));

const results = configs.map((c) => {
  const oracle = new Oracle(compileOracle(ORACLES[c.oracle].source));
  const res = learnBimachine({
    alphabet: ORACLES[c.oracle].alphabet,
    train: c.train,
    oracle,
    kWitness: c.k,
    contMaxLen: c.len,
    headMaxLen: c.len,
    record: false,
    maxEvents: 200000,
  });
  if (res.stopped) return { id: c.id, stopped: true };
  return { id: c.id, machine: canonical(res.hypothesis), merges: res.merges, queries: [...res.queries].sort() };
});

// Oracle outputs on every word up to the given length, for comparison with Python.
const oracleOutputs = {};
for (const { oracle, words } of oracleChecks) {
  const fn = compileOracle(ORACLES[oracle].source);
  oracleOutputs[oracle] = words.map((w) => fn(w).map(String));
}

process.stdout.write(JSON.stringify({ results, oracleOutputs }));
