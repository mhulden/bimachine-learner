import { learnBimachine, Oracle } from "./learner.js";
import { ORACLES, WALKTHROUGH, compileOracle } from "./presets.js";

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const RING_COLORS = ["#eb6834", "#1baf7a", "#4a3aa7", "#e8a800", "#e87ba4", "#008300", "#2a78d6", "#8a4b2a"];

let run = null;        // {events, alphabet, oracle, guided}
let step = 0;
let timer = null;

// ---------------------------------------------------------------------------
// Explanations shown the first time each kind of step appears
// ---------------------------------------------------------------------------

const GUIDE = {
  init: [
    "A bimachine has a <b>left automaton</b> that reads the input from left to right, a <b>right automaton</b> that reads it from right to left, and an <b>output table ω</b>. At each position, the left state (a summary of everything before the symbol) and the right state (a summary of everything after it) select the output chunk ω(q, a, p).",
    "The learner starts from prefix trees: every training prefix gets its own left state and every suffix its own right state. This reproduces the training data exactly but generalizes to nothing yet. Each aligned output chunk fixes one entry of the output table.",
  ],
  round: [
    "Learning generalizes by <b>merging states</b>, one side at a time, in the style of RPNI. <b>Red</b> states are settled; <b>blue</b> states are one transition away from a red state and are the candidates for merging. Initially only the start state is red.",
  ],
  pick: [
    "States are considered in shortlex order of their <em>representatives</em>, the shortest strings reaching them. For the right automaton the representative is a suffix of the input.",
  ],
  "reject-fold": [
    "Merging two states can force further merges: if both have an <code>a</code>-transition, the automaton stays deterministic only if those two targets merge as well. This is <b>merge-and-fold</b>; the coloured rings show the groups of states that end up identified.",
    "After folding, every output entry must still be unique. Here two entries that become the same cell ask for different chunks (outlined in the table), so the merge contradicts the data and is rejected.",
  ],
  "reject-data": [
    "The merge is consistent in the output table, but the merged automaton no longer produces the training example's chunks, so it is rejected.",
  ],
  "reject-witness": [
    "The training data cannot rule this merge out, so the learner asks the <b>oracle</b>. It builds <b>witness strings</b>: the blue state's representative followed by short continuations (left pass) or preceded by short heads (right pass). The oracle returns their aligned chunks, and the merged hypothesis must agree with all of them. One disagreement is enough to reject the merge.",
  ],
  accept: [
    "This merge survives both checks: the output table stays consistent, and the merged hypothesis agrees with every witness answer from the oracle.",
  ],
  merged: [
    "The witness strings and their answers are now part of the hypothesis: their paths are added to both automata (dashed states) and their chunks become output entries. Later merges must respect them as well. After every accepted merge the pass restarts from the root.",
  ],
  promote: [
    "If no red state can absorb a blue state, the blue state must differ from all of them, so it is <b>promoted</b> to red. Its successors become the new blue candidates.",
  ],
  converged: [
    "Merging stops after a round in which neither pass accepted a merge.",
  ],
  complete: [
    "<b>Completion.</b> Some output entries ω(q, a, p) may never have been observed. For each missing one the learner asks the oracle about the string π(q)·a·σ(p): the representative prefix of q, the symbol a, and the representative suffix of p. The chunk at the middle position is exactly ω(q, a, p). Transitions that are still missing go to a sink state.",
  ],
  minimize: [
    "<b>Minimization.</b> Two left states are interchangeable if they have the same output row (the same chunk for every symbol and every right state) and their transitions lead to interchangeable states; right states are compared by output columns in the same way. This is partition refinement, as in DFA minimization. The table below lists each state's row or column and its block.",
    "Here every state already has its own block. In the runs we have examined, including the paper's benchmark battery, merge-and-fold with witness validation reaches a machine that minimization cannot shrink further, so this step confirms minimality.",
  ],
  stopped: [
    "Each accepted merge adds its witness strings as new paths, and with a small sample and a large witness budget those paths can keep producing new states to merge. The steps so far can still be browsed.",
  ],
  done: [
    "The learner is finished. Use the box below the output table to run any string through the learned machine and compare with the oracle.",
  ],
};

const GUIDE_DONE_FIG1 =
  "This is exactly the bimachine of Figure 1 in the paper, with states numbered from 0: left state 1 records that the previous symbol was <code>a</code> (Figure 1's state 2), and right state 1 records that an <code>a</code> occurs further right (Figure 1's state 4). Try the paper's input <code>aabbabaabb</code> in the box below the output table.";

// ---------------------------------------------------------------------------
// Running the learner
// ---------------------------------------------------------------------------

function parseTrain(text) {
  return text.split(/[\s,]+/).filter(Boolean).map((w) => (w === "-" ? "" : w));
}

function startRun({ oracleKey, oracleSource, train, k, len, guided }) {
  stop();
  const spec = ORACLES[oracleKey];
  const alphabet = spec.alphabet;
  const fn = compileOracle(oracleSource);
  const bad = train.find((w) => [...w].some((c) => !alphabet.includes(c)));
  if (bad !== undefined) throw new Error(`"${bad}" uses a symbol outside the alphabet {${alphabet.join(", ")}}`);
  const oracle = new Oracle(fn);
  const res = learnBimachine({ alphabet, train, oracle, kWitness: k, contMaxLen: len, headMaxLen: len, maxEvents: 1500 });
  const seen = new Set();
  for (const ev of res.events) {
    ev.firstOfKind = !seen.has(ev.kind);
    seen.add(ev.kind);
  }
  run = { events: res.events, alphabet: [...alphabet].sort(), oracle, guided, fn };
  step = 0;
  $("scrub").max = String(run.events.length - 1);
  render();
}

// ---------------------------------------------------------------------------
// Graph layout and drawing
// ---------------------------------------------------------------------------

function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function layout(dfa, alphabet) {
  const trans = new Map();
  for (const [s, a, t] of dfa.edges) {
    if (!trans.has(s)) trans.set(s, new Map());
    trans.get(s).set(a, t);
  }
  const depth = new Map([[dfa.start, 0]]);
  const repr = new Map([[dfa.start, ""]]);
  const order = [dfa.start];
  for (let i = 0; i < order.length; i++) {
    const s = order[i];
    for (const a of alphabet) {
      const t = trans.get(s)?.get(a);
      if (t === undefined || depth.has(t)) continue;
      depth.set(t, depth.get(s) + 1);
      repr.set(t, repr.get(s) + a);
      order.push(t);
    }
  }
  const maxD = Math.max(0, ...depth.values());
  for (const s of dfa.states) {
    if (!depth.has(s)) {
      depth.set(s, maxD + 1);
      repr.set(s, "?");
      order.push(s);
    }
  }
  const layers = new Map();
  for (const s of order) {
    const d = depth.get(s);
    if (!layers.has(d)) layers.set(d, []);
    layers.get(d).push(s);
  }
  return { trans, depth, repr, layers };
}

function drawDFA(container, dfa, opts) {
  const { alphabet, mirror, red, blue, rings, fresh, marks } = opts;
  container.innerHTML = "";
  const L = layout(dfa, alphabet);
  const DX = 64, DY = 42, R = 13, PAD = 34, TOP = 30;
  const nLayers = Math.max(...L.layers.keys()) + 1;
  const maxRows = Math.max(...[...L.layers.values()].map((l) => l.length));
  const W = PAD * 2 + (nLayers - 1) * DX + 30;
  const H = PAD * 2 + (maxRows - 1) * DY + 10 + TOP;
  const pos = new Map();
  for (const [d, states] of L.layers) {
    states.forEach((s, i) => {
      const x = PAD + 15 + d * DX;
      const y = TOP + PAD + (i - (states.length - 1) / 2) * DY + ((maxRows - 1) * DY) / 2;
      pos.set(s, [mirror ? W - x : x, y]);
    });
  }
  const scale = W < 320 ? 1.5 : 1;  // draw small machines larger
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W * scale, height: H * scale, role: "img" });
  const defs = el("defs", {}, svg);
  const m = el("marker", { id: `arrow-${container.id}`, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" }, defs);
  el("path", { d: "M0,0 L10,5 L0,10 z", fill: "currentColor", class: "arrowhead" }, m);
  svg.style.color = getComputedStyle(document.documentElement).getPropertyValue("--edge");

  // Edges grouped by (source, target).
  const groups = new Map();
  for (const [s, a, t] of dfa.edges) {
    const k = `${s}>${t}`;
    if (!groups.has(k)) groups.set(k, { s, t, labels: [] });
    groups.get(k).labels.push(a);
  }
  const edgeLayer = el("g", {}, svg);
  for (const { s, t, labels } of groups.values()) {
    const [x1, y1] = pos.get(s);
    const [x2, y2] = pos.get(t);
    const label = labels.sort().join(",");
    let d, lx, ly;
    if (s === t) {
      d = `M ${x1 - 6} ${y1 - R + 2} C ${x1 - 16} ${y1 - R - 22}, ${x1 + 16} ${y1 - R - 22}, ${x1 + 6} ${y1 - R + 2}`;
      lx = x1; ly = y1 - R - 19;
    } else {
      const forward = L.depth.get(t) === L.depth.get(s) + 1 && !groups.has(`${t}>${s}`);
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
      const ux = dx / len, uy = dy / len;
      const sx = x1 + ux * R, sy = y1 + uy * R, ex = x2 - ux * R, ey = y2 - uy * R;
      if (forward) {
        d = `M ${sx} ${sy} L ${ex} ${ey}`;
        lx = (sx + ex) / 2; ly = (sy + ey) / 2 - 4;
      } else {
        const bend = groups.has(`${t}>${s}`) ? 18 : 26 + Math.min(40, Math.abs(L.depth.get(t) - L.depth.get(s)) * 6);
        const cx = (x1 + x2) / 2 - uy * bend, cy = (y1 + y2) / 2 + ux * bend;
        const a1 = Math.atan2(cy - y1, cx - x1), a2 = Math.atan2(cy - y2, cx - x2);
        d = `M ${x1 + R * Math.cos(a1)} ${y1 + R * Math.sin(a1)} Q ${cx} ${cy} ${x2 + R * Math.cos(a2)} ${y2 + R * Math.sin(a2)}`;
        lx = (x1 + x2) / 4 + cx / 2; ly = (y1 + y2) / 4 + cy / 2;
      }
    }
    el("path", { d, class: "edge", "marker-end": `url(#arrow-${container.id})` }, edgeLayer);
    const tx = el("text", { x: lx, y: ly, class: "elabel", "text-anchor": "middle" }, edgeLayer);
    tx.textContent = label;
  }

  // Start arrow.
  const [sx, sy] = pos.get(dfa.start);
  const dir = mirror ? 1 : -1;
  el("line", { x1: sx + dir * (R + 18), y1: sy, x2: sx + dir * (R + 2), y2: sy, class: "startarrow", "marker-end": `url(#arrow-${container.id})` }, edgeLayer);

  // Nodes.
  const ringOf = new Map();
  (rings || []).forEach((cls, i) => cls.forEach((s) => ringOf.set(s, RING_COLORS[i % RING_COLORS.length])));
  for (const s of dfa.states) {
    const [x, y] = pos.get(s);
    const classes = ["node"];
    if (red.includes(s)) classes.push("red");
    else if (blue.includes(s)) classes.push("blue");
    if (s === dfa.sink) classes.push("sink");
    if (fresh && fresh.has(s)) classes.push("fresh");
    const g = el("g", { class: classes.join(" ") }, svg);
    const title = el("title", {}, g);
    const rep = L.repr.get(s);
    title.textContent = `state ${s}; representative ${mirror ? "suffix" : "prefix"} "${mirror ? [...rep].reverse().join("") : rep}"${s === dfa.sink ? " (sink)" : ""}`;
    if (ringOf.has(s)) el("circle", { cx: x, cy: y, r: R + 4, class: "ring", stroke: ringOf.get(s) }, g);
    el("circle", { cx: x, cy: y, r: R, class: "body" }, g);
    const t = el("text", { x, y }, g);
    t.textContent = String(s);
    if (marks && marks.has(s)) {
      const tag = el("text", { x, y: y + R + 10, class: "tag", "text-anchor": "middle" }, g);
      tag.textContent = marks.get(s);
    }
  }
  container.appendChild(svg);
}

// ---------------------------------------------------------------------------
// Output table, oracle panel, minimization table, tester
// ---------------------------------------------------------------------------

const chunk = (c) => (c === "" ? "ε" : c);
const chunkSeq = (y, mark = -1) =>
  y.map((c, i) => (i === mark ? `<b>${esc(chunk(c))}</b>` : esc(chunk(c)))).join("·") || "(empty)";
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const word = (w) => (w === "" ? "ε" : w);

function omegaTable(snapshot, prevSnapshot, ev) {
  const QL = snapshot.left.states;
  const QR = snapshot.right.states;
  const alpha = run.alphabet;
  const cell = new Map(snapshot.omega.map(([q, a, p, out, src]) => [`${q}|${a}|${p}`, { out, src }]));
  const prev = new Set((prevSnapshot?.omega || []).map(([q, a, p, out]) => `${q}|${a}|${p}|${out}`));
  const conflict = new Set();
  if (ev.kind === "reject-fold" && ev.conflict?.keys) for (const [q, a, p] of ev.conflict.keys) conflict.add(`${q}|${a}|${p}`);
  // Only rows and columns with at least one defined entry are shown.
  const rows = QL.filter((q) => alpha.some((a) => QR.some((p) => cell.has(`${q}|${a}|${p}`))));
  const cols = alpha.map((a) => [a, QR.filter((p) => QL.some((q) => cell.has(`${q}|${a}|${p}`)))]).filter(([, ps]) => ps.length);
  const hiddenRows = QL.length - rows.length;
  const hiddenCols = alpha.length * QR.length - cols.reduce((n, [, ps]) => n + ps.length, 0);
  if (!rows.length) return `<p class="note">No output entries yet.</p>`;
  let html = "<table><thead><tr><th rowspan=2>q \\ a, p</th>";
  for (const [a, ps] of cols) html += `<th colspan=${ps.length}><b>${esc(a)}</b></th>`;
  html += "</tr><tr>";
  for (const [, ps] of cols) for (const p of ps) html += `<th>p=${p}</th>`;
  html += "</tr></thead><tbody>";
  for (const q of rows) {
    html += `<tr><th>${q}</th>`;
    for (const [a, ps] of cols) {
      for (const p of ps) {
        const k = `${q}|${a}|${p}`;
        const e = cell.get(k);
        if (!e) { html += "<td></td>"; continue; }
        const cls = [e.src];
        if (conflict.has(k)) cls.push("conflict");
        else if (prevSnapshot && e.src !== "data" && !prev.has(`${k}|${e.out}`)) cls.push("new");
        html += `<td class="${cls.join(" ")}" title="ω(${q}, ${esc(a)}, ${p}) = ${esc(chunk(e.out))} (${e.src})">${esc(chunk(e.out))}</td>`;
      }
    }
    html += "</tr>";
  }
  html += "</tbody></table>";
  if (hiddenRows || hiddenCols) {
    html += `<p class="note">Hidden: ${hiddenRows} left ${hiddenRows === 1 ? "state" : "states"} and ${hiddenCols} (a, p) ${hiddenCols === 1 ? "column" : "columns"} with no defined entry.</p>`;
  }
  return html;
}

function minimizeTable(ev) {
  const byState = new Map(ev.part);
  const blocks = [...new Set(ev.part.map(([, b]) => b))];
  const color = (b) => RING_COLORS[blocks.indexOf(b) % RING_COLORS.length];
  const side = ev.side === "L" ? "left" : "right";
  let html = `<h3>${ev.side === "L" ? "Left" : "Right"} states: output ${ev.side === "L" ? "row" : "column"}, successors, and block</h3>`;
  html += `<table><thead><tr><th>${side} state</th><th>${ev.side === "L" ? "row (a, then p)" : "column (q, then a)"}</th>`;
  for (const a of run.alphabet) html += `<th>—${esc(a)}→</th>`;
  html += "<th>block</th></tr></thead><tbody>";
  for (const { state, row, next } of ev.signatures) {
    const b = byState.get(state);
    html += `<tr><th>${state}</th><td>${row.map((c) => esc(chunk(c))).join(" ")}</td>`;
    for (const t of next) html += `<td>${t}</td>`;
    html += `<td style="color:${color(b)};font-weight:700">■ ${b}</td></tr>`;
  }
  return html + "</tbody></table>";
}

function oraclePanel(ev) {
  const parts = [];
  if (ev.kind === "init") {
    parts.push(`<p class="note">Training data: aligned examples (one output chunk per input symbol).</p><ul class="qlist">`);
    for (const [x, y] of ev.data) parts.push(`<li><span>${esc(word(x))}</span><span class="chunks">${chunkSeq(y)}</span></li>`);
    parts.push("</ul>");
  } else if (ev.witnesses) {
    const bad = ev.conflict?.witness;
    parts.push(`<p class="note">Witness queries for ${ev.side === "L" ? "left" : "right"} merge ${ev.u} → ${ev.r} (${ev.witnesses.length}):</p><ul class="qlist">`);
    for (const [x, y] of ev.witnesses) {
      const isBad = x === bad;
      parts.push(`<li class="${isBad ? "bad" : ""}"><span>${esc(x)}</span><span class="chunks">${chunkSeq(y, isBad ? ev.conflict.position : -1)}</span></li>`);
    }
    parts.push("</ul>");
    if (bad !== undefined) {
      parts.push(`<p class="note">At position ${ev.conflict.position + 1} of <code>${esc(bad)}</code> the oracle says <b>${esc(chunk(ev.conflict.outs[1]))}</b>; the merged hypothesis already has <b>${esc(chunk(ev.conflict.outs[0]))}</b>.</p>`);
    }
  } else if (ev.kind === "complete" && ev.queries.length) {
    parts.push(`<p class="note">Completion queries π(q)·a·σ(p); the chunk at the marked position fills ω(q, a, p):</p><ul class="qlist">`);
    for (const { entry, x, pos, y } of ev.queries) {
      parts.push(`<li><span>${esc(x)} <span class="muted">ω(${entry[0]},${esc(entry[1])},${entry[2]})</span></span><span class="chunks">${chunkSeq(y, pos)}</span></li>`);
    }
    parts.push("</ul>");
  } else if (ev.kind === "reject-data") {
    parts.push(`<p class="note">Training example that the merged hypothesis gets wrong:</p><ul class="qlist"><li class="bad"><span>${esc(word(ev.example[0]))}</span><span class="chunks">${chunkSeq(ev.example[1])}</span></li></ul>`);
  } else {
    parts.push(`<p class="note">No oracle queries in this step.</p>`);
  }
  return parts.join("");
}

function snapshotTransduce(snapshot, x) {
  const step = (dfa, s, a) => {
    for (const [u, b, t] of dfa.edges) if (u === s && b === a) return t;
    return dfa.sink ?? null;
  };
  const n = x.length;
  const q = [snapshot.left.start];
  for (let i = 0; i < n; i++) q.push(q[i] === null ? null : step(snapshot.left, q[i], x[i]));
  const p = new Array(n + 1);
  p[n] = snapshot.right.start;
  for (let i = n - 1; i >= 0; i--) p[i] = p[i + 1] === null ? null : step(snapshot.right, p[i + 1], x[i]);
  const cell = new Map(snapshot.omega.map(([a, b, c, out]) => [`${a}|${b}|${c}`, out]));
  const out = [...x].map((a, i) => (q[i] === null || p[i + 1] === null ? undefined : cell.get(`${q[i]}|${a}|${p[i + 1]}`)));
  return { q, p, out };
}

function renderTester(snapshot) {
  const x = $("t-input").value.trim();
  const box = $("t-out");
  if ([...x].some((c) => !run.alphabet.includes(c))) {
    box.innerHTML = `<p class="note">Use only the symbols {${run.alphabet.join(", ")}}.</p>`;
    return;
  }
  const { q, p, out } = snapshotTransduce(snapshot, x);
  let truth = null;
  try { truth = run.fn(x).map(String); } catch { truth = null; }
  const cellOrQ = (v) => (v === null || v === undefined ? "?" : v);
  let html = "<table><tbody>";
  html += `<tr><th>left state</th>${q.slice(0, x.length).map((v) => `<td class="l">${cellOrQ(v)}</td>`).join("")}</tr>`;
  html += `<tr><th>input</th>${[...x].map((c) => `<td><b>${esc(c)}</b></td>`).join("")}</tr>`;
  html += `<tr><th>right state</th>${p.slice(1).map((v) => `<td class="r">${cellOrQ(v)}</td>`).join("")}</tr>`;
  html += `<tr><th>output</th>${out.map((c, i) => `<td class="${truth && c !== truth[i] ? "bad" : ""}">${c === undefined ? "?" : esc(chunk(c))}</td>`).join("")}</tr>`;
  if (truth) html += `<tr><th>oracle</th>${truth.map((c) => `<td>${esc(chunk(c))}</td>`).join("")}</tr>`;
  html += "</tbody></table>";
  const ok = truth && out.every((c, i) => c === truth[i]);
  html += `<p class="note">${x.length === 0 ? "Empty input: empty output." : ok ? "Matches the oracle." : "? marks an undefined state or output entry; red outputs disagree with the oracle."} Each column shows the left state before the symbol and the right state for the suffix after it.</p>`;
  box.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Rendering a step
// ---------------------------------------------------------------------------

const PHASES = [
  ["Prefix trees", (e) => e.kind === "init"],
  ["Merging", (e) => ["round", "pick", "reject-fold", "reject-data", "reject-witness", "accept", "merged", "promote", "pass-end", "converged"].includes(e.kind)],
  ["Completion", (e) => e.kind === "complete"],
  ["Minimization", (e) => e.kind === "minimize"],
  ["Done", (e) => e.kind === "done" || e.kind === "stopped"],
];

function currentRound() {
  for (let i = step; i >= 0; i--) if (run.events[i].kind === "round") return run.events[i].round;
  return null;
}

function render() {
  if (!run) return;
  const ev = run.events[step];
  const prev = step > 0 ? run.events[step - 1] : null;
  $("scrub").value = String(step);
  $("counter").textContent = `Step ${step + 1} / ${run.events.length}`;
  $("b-play").textContent = timer ? "⏸ Pause" : "▶ Play";

  const phaseIdx = PHASES.findIndex(([, f]) => f(ev));
  const round = currentRound();
  $("phase").innerHTML = PHASES.map(([name], i) => `<span class="${i === phaseIdx ? "now" : ""}">${name}${name === "Merging" && round && i === phaseIdx ? ` · round ${round}${ev.side ? `, ${ev.side === "L" ? "left" : "right"} pass` : ""}` : ""}</span>`).join("");

  $("step-text").textContent = ev.text;
  const guide = $("guide");
  const notes = ev.firstOfKind ? [...(GUIDE[ev.kind] || [])] : [];
  if (ev.kind === "done" && run.guided) notes.push(GUIDE_DONE_FIG1);
  guide.hidden = notes.length === 0;
  guide.innerHTML = notes.map((n) => `<p>${n}</p>`).join("");

  const snap = ev.snapshot;
  const onSide = (side) => ev.side === side;
  const rings = (side) => (onSide(side) && ev.classes ? ev.classes : []);
  const marks = (side) => {
    const m = new Map();
    if (onSide(side) && ev.u !== undefined && ev.kind !== "merged") {
      m.set(ev.u, "u");
      if (ev.r !== undefined) m.set(ev.r, "r");
    }
    return m;
  };
  const fresh = (side) => new Set(ev.kind === "merged" ? ev.newStates[side] : []);
  for (const side of ["L", "R"]) {
    drawDFA($(side === "L" ? "g-left" : "g-right"), side === "L" ? snap.left : snap.right, {
      alphabet: run.alphabet,
      mirror: side === "R",
      red: ev.red[side],
      blue: ev.blue[side],
      rings: rings(side),
      fresh: fresh(side),
      marks: marks(side),
    });
  }

  $("omega").innerHTML = omegaTable(snap, prev?.snapshot, ev);
  $("minimize").innerHTML = ev.kind === "minimize" ? minimizeTable(ev) : "";
  $("oracle-panel").innerHTML = oraclePanel(ev);
  $("q-count").textContent = `· ${ev.queryCount} distinct ${ev.queryCount === 1 ? "query" : "queries"} so far`;
  renderTester(snap);
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function go(i) {
  if (!run) return;
  step = Math.max(0, Math.min(run.events.length - 1, i));
  if (run.guided) history.replaceState(null, "", `#step=${step + 1}`);
  render();
}
function stop() {
  if (timer) clearInterval(timer);
  timer = null;
  if (run) render();
}
function play() {
  if (!run) return;
  if (timer) return stop();
  if (step >= run.events.length - 1) step = 0;
  timer = setInterval(() => {
    if (step >= run.events.length - 1) return stop();
    go(step + 1);
  }, 1400);
  render();
}

$("b-first").onclick = () => { stop(); go(0); };
$("b-prev").onclick = () => { stop(); go(step - 1); };
$("b-next").onclick = () => { stop(); go(step + 1); };
$("b-last").onclick = () => { stop(); go(Infinity); };
$("b-play").onclick = play;
$("scrub").oninput = (e) => { stop(); go(Number(e.target.value)); };
$("t-input").oninput = () => run && renderTester(run.events[step].snapshot);
document.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
  if (e.key === "ArrowRight") { stop(); go(step + 1); }
  else if (e.key === "ArrowLeft") { stop(); go(step - 1); }
  else if (e.key === " ") { e.preventDefault(); play(); }
  else if (e.key === "Home") { stop(); go(0); }
  else if (e.key === "End") { stop(); go(Infinity); }
});

// Modes.
function showError(msg) {
  const box = $("c-error");
  box.hidden = !msg;
  box.textContent = msg || "";
}

function guided() {
  $("mode-guided").classList.add("active");
  $("mode-custom").classList.remove("active");
  $("custom").hidden = true;
  $("t-input").value = "aabbabaabb";
  const o = ORACLES[WALKTHROUGH.oracle];
  startRun({
    oracleKey: WALKTHROUGH.oracle, oracleSource: o.source, train: WALKTHROUGH.train,
    k: WALKTHROUGH.kWitness, len: WALKTHROUGH.contMaxLen, guided: true,
  });
}

function randomSample(alphabet, n = 6, maxLen = 5) {
  const words = new Set();
  let guard = 0;
  while (words.size < n && guard++ < 1000) {
    const len = 1 + Math.floor(Math.random() * maxLen);
    let w = "";
    for (let i = 0; i < len; i++) w += alphabet[Math.floor(Math.random() * alphabet.length)];
    words.add(w);
  }
  return [...words];
}

function loadPreset(key) {
  const o = ORACLES[key];
  $("c-oracle").value = o.source;
  $("c-train").value = key === WALKTHROUGH.oracle ? WALKTHROUGH.train.join(" ") : randomSample(o.alphabet).join(" ");
  $("t-input").value = randomSample(o.alphabet, 1, 8)[0];
}

function custom() {
  $("mode-custom").classList.add("active");
  $("mode-guided").classList.remove("active");
  $("custom").hidden = false;
  showError("");
}

function runCustom() {
  try {
    showError("");
    startRun({
      oracleKey: $("c-preset").value,
      oracleSource: $("c-oracle").value,
      train: parseTrain($("c-train").value),
      k: Number($("c-k").value),
      len: Number($("c-len").value),
      guided: false,
    });
  } catch (err) {
    showError(err.message || String(err));
  }
}

const sel = $("c-preset");
for (const [key, o] of Object.entries(ORACLES)) {
  const opt = document.createElement("option");
  opt.value = key;
  opt.textContent = `${o.title}  (Σ = {${o.alphabet.join(", ")}})`;
  sel.appendChild(opt);
}
sel.onchange = () => loadPreset(sel.value);
$("c-random").onclick = () => { $("c-train").value = randomSample(ORACLES[sel.value].alphabet).join(" "); };
$("c-run").onclick = runCustom;
$("mode-guided").onclick = guided;
$("mode-custom").onclick = custom;

// Links: #step=N opens the walkthrough at step N; #custom=TASK opens "Try your own"
// with that task and a random sample, and runs it.
const customLink = /custom=(\w+)/.exec(location.hash);
if (customLink && ORACLES[customLink[1]]) {
  sel.value = customLink[1];
  loadPreset(sel.value);
  custom();
  runCustom();
} else {
  loadPreset(sel.value);
  guided();
  const linked = /step=(\d+)/.exec(location.hash);
  if (linked) go(Number(linked[1]) - 1);
}
