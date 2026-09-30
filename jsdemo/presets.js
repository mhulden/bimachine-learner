// Chunk oracles for the demo: the Figure 1 function and the ten benchmark tasks
// of the paper (JavaScript versions of the classes in rpni_bimachine.py).
// Each oracle maps an input word (string) to an array with one chunk per symbol.

const chars = (x) => [...x];

export const ORACLES = {
  figure1: {
    title: "Figure 1: capitalize after a, if another a follows",
    alphabet: ["a", "b"],
    description: "Capitalize a symbol if its left neighbour is a and another a occurs somewhere to its right.",
    source: `(x) => [...x].map((c, i) =>
  i > 0 && x[i - 1] === "a" && x.slice(i + 1).includes("a")
    ? c.toUpperCase() : c)`,
  },
  swap_first_last_ab: {
    title: "Swap first/last",
    alphabet: ["a", "b"],
    description: "If the input has length at least 2, swap its first and last symbol; otherwise copy.",
    source: `(x) => {
  const y = [...x];
  if (y.length >= 2) [y[0], y[y.length - 1]] = [y[y.length - 1], y[0]];
  return y;
}`,
  },
  future_c_abc: {
    title: "Future-c condition",
    alphabet: ["a", "b", "c"],
    description: "Rewrite a→b if it is immediately preceded by b and some c occurs to its right.",
    source: `(x) => [...x].map((ch, i) =>
  ch === "a" && x[i - 1] === "b" && x.slice(i + 1).includes("c") ? "b" : ch)`,
  },
  global_begin_end_b_ab: {
    title: "Global begin/end",
    alphabet: ["a", "b"],
    description: "Rewrite a→b if the word begins and ends with b.",
    source: `(x) => {
  const g = x.length >= 1 && x[0] === "b" && x[x.length - 1] === "b";
  return [...x].map((ch) => (g && ch === "a" ? "b" : ch));
}`,
  },
  local_cad_abcd: {
    title: "Local cad",
    alphabet: ["a", "b", "c", "d"],
    description: "Rewrite a→b if it is immediately preceded by c and followed by d.",
    source: `(x) => [...x].map((ch, i) =>
  ch === "a" && x[i - 1] === "c" && x[i + 1] === "d" ? "b" : ch)`,
  },
  global_or_local_abcd: {
    title: "Global-or-local",
    alphabet: ["a", "b", "c", "d"],
    description: "Rewrite a→b if the word begins and ends with b, or a is in the context c_d.",
    source: `(x) => {
  const g = x.length >= 1 && x[0] === "b" && x[x.length - 1] === "b";
  return [...x].map((ch, i) =>
    ch === "a" && (g || (x[i - 1] === "c" && x[i + 1] === "d")) ? "b" : ch);
}`,
  },
  c_parity_left_even_right_odd_abc: {
    title: "Parity left/right",
    alphabet: ["a", "b", "c"],
    description: "Rewrite a→b if the number of c's to its left is even and to its right is odd.",
    source: `(x) => {
  const total = [...x].filter((c) => c === "c").length;
  let left = 0;
  return [...x].map((ch) => {
    let out = ch;
    const right = total - left - (ch === "c" ? 1 : 0);
    if (ch === "a" && left % 2 === 0 && right % 2 === 1) out = "b";
    if (ch === "c") left++;
    return out;
  });
}`,
  },
  swap_ab_if_even_length_abc: {
    title: "Even-length a↔b",
    alphabet: ["a", "b", "c"],
    description: "If the input has even length, rewrite a↔b at every position; otherwise copy.",
    source: `(x) => [...x].map((ch) =>
  x.length % 2 === 1 ? ch : ch === "a" ? "b" : ch === "b" ? "a" : ch)`,
  },
  insert_a_between_cd_abcd: {
    title: "Anchored insertion",
    alphabet: ["a", "b", "c", "d"],
    description: "Insert a between c and d by emitting the chunk ca on a c followed by d; copy everything else.",
    source: `(x) => [...x].map((ch, i) => (ch === "c" && x[i + 1] === "d" ? "ca" : ch))`,
  },
  delete_a_even_else_delete_b_ab: {
    title: "Conditional deletion",
    alphabet: ["a", "b"],
    description: "If the input has even length delete every a, otherwise delete every b (a deletion is an empty chunk).",
    source: `(x) => [...x].map((ch) =>
  x.length % 2 === 0 ? (ch === "a" ? "" : ch) : (ch === "b" ? "" : ch))`,
  },
  delete_a_parity_by_last_symbol_ab: {
    title: "Last-symbol parity deletion",
    alphabet: ["a", "b"],
    description: "Numbering positions from 1: delete a's at even positions if the last symbol is a, at odd positions if it is b.",
    source: `(x) => [...x].map((ch, i) => {
  if (ch !== "a") return ch;
  const pos = i + 1;
  const del = x[x.length - 1] === "a" ? pos % 2 === 0 : pos % 2 === 1;
  return del ? "" : ch;
})`,
  },
};

// Compile an oracle's source text into a chunk function.
export function compileOracle(source) {
  // eslint-disable-next-line no-new-func
  const fn = new Function(`"use strict"; return (${source});`)();
  if (typeof fn !== "function") throw new Error("the oracle must be a function (x) => chunks");
  return fn;
}

// Guided walkthrough: a fixed, small sample on which the learner recovers Figure 1.
export const WALKTHROUGH = {
  oracle: "figure1",
  train: ["bab", "bbbb", "aa", "baba", "abbab", "baa"],
  kWitness: 6,
  contMaxLen: 2,
  headMaxLen: 2,
};

export { chars };
