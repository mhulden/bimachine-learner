# learn_bimachine.py
# Complete code: learn a bimachine from an oracle using RPNI-style merges + minimization
# Test case: alphabet {a,b} - rewrite 'a' -> 'b' iff word begins with 'b' AND ends with 'b'

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Set, Iterable
from collections import defaultdict, deque
import itertools
import random
import shutil

@dataclass
class OracleSpec:
    name: str
    alphabet: Tuple[str, ...]
    oracle: object
    description: str


@dataclass
class MemoizedChunkOracle:
    """
    Generic chunk-oracle memoizer. The learner revisits many
    representative and witness strings.
    """

    base_oracle: object
    _cache: Dict[Tuple[str, ...], Tuple[str, ...]] = None

    def __post_init__(self):
        if self._cache is None:
            self._cache = {}

    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        if x not in self._cache:
            self._cache[x] = self.base_oracle.chunk(x)
        return self._cache[x]



class ChunkOracle_CParityLeftRight_ABC:
    """
    Σ={a,b,c}.
    Rewrite a->b iff:
      (# of c's strictly to the LEFT is even) AND
      (# of c's strictly to the RIGHT is odd).
    Otherwise copy.
    """
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        y = list(x)

        # prefix count of c's: pref_c[i] = #c in x[:i]
        pref_c = [0] * (n + 1)
        for i in range(n):
            pref_c[i + 1] = pref_c[i] + (1 if x[i] == "c" else 0)

        total_c = pref_c[n]

        for i, ch in enumerate(x):
            if ch != "a":
                continue
            left_c = pref_c[i]
            right_c = total_c - pref_c[i + 1]  # c's in x[i+1:]
            if (left_c % 2 == 0) and (right_c % 2 == 1):
                y[i] = "b"

        return tuple(y)

class ChunkOracle_SwapABIfEvenLength_ABC:
    """
    Σ={a,b,c}.
    If |x| is even: swap a<->b everywhere, keep c.
    If |x| is odd : copy unchanged.
    """
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        if n % 2 == 1:
            return x

        y = []
        for ch in x:
            if ch == "a":
                y.append("b")
            elif ch == "b":
                y.append("a")
            else:
                y.append(ch)  # c unchanged
        return tuple(y)


class ChunkOracle_SwapFirstLast_AB:
    """Σ={a,b}. If len>=2 swap first/last. Otherwise copy unchanged."""
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        if n < 2:
            return x
        y = list(x)
        y[0], y[-1] = y[-1], y[0]
        return tuple(y)

class ChunkOracle_FutureC_ABC:
    """Σ={a,b,c}. Rewrite a->b iff immediately preceded by b and there is some c somewhere to the right."""
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        y = list(x)
        # precompute suffix-has-c
        suf_has_c = [False]*(n+1)
        seen = False
        for i in range(n-1, -1, -1):
            if x[i] == "c":
                seen = True
            suf_has_c[i] = seen
        for i, ch in enumerate(x):
            if ch == "a":
                if i-1 >= 0 and x[i-1] == "b" and suf_has_c[i+1]:
                    y[i] = "b"
        return tuple(y)

class ChunkOracle_GlobalBeginEndB_AB:
    """Σ={a,b}. Rewrite a->b iff word begins with b and ends with b."""
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        global_cond = (n >= 1 and x[0] == "b" and x[-1] == "b")
        if not global_cond:
            return x
        return tuple(("b" if ch == "a" else ch) for ch in x)

class ChunkOracle_LocalCAD_ABCD:
    """Σ={a,b,c,d}. Rewrite a->b iff immediately preceded by c and followed by d."""
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        y = list(x)
        for i, ch in enumerate(x):
            if ch == "a":
                if i-1 >= 0 and i+1 < n and x[i-1] == "c" and x[i+1] == "d":
                    y[i] = "b"
        return tuple(y)

class ChunkOracle_GlobalOrLocal_ABCD:
    """
    Σ={a,b,c,d}. Rewrite a->b iff:
      (GLOBAL) word begins with b and ends with b
       OR
      (LOCAL)  context c a d
    """
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        global_cond = (n >= 1 and x[0] == "b" and x[-1] == "b")
        y = list(x)
        for i, ch in enumerate(x):
            if ch != "a":
                continue
            local_cond = (i-1 >= 0 and i+1 < n and x[i-1] == "c" and x[i+1] == "d")
            if global_cond or local_cond:
                y[i] = "b"
        return tuple(y)


class ChunkOracle_InsertA_BetweenCD_ABCD:
    """
    Σ={a,b,c,d}. Insert 'a' between c and d.
    Since we disallow unanchored ε-inputs, we represent the insertion as:
        ... c d ...   ->   ... (ca) d ...
    i.e., the output chunk aligned with input symbol 'c' becomes 'ca' when followed by 'd'.
    All other symbols are copied as singleton chunks.
    """
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        y: List[str] = []
        for i, ch in enumerate(x):
            if ch == "c" and i + 1 < n and x[i + 1] == "d":
                y.append("ca")
            else:
                y.append(ch)
        return tuple(y)


class ChunkOracle_DeleteAEvenElseDeleteB_AB:
    """
    Σ={a,b}. Deletion example (length-changing):
      - If |x| is even: delete all 'a' (emit ε for each 'a'), copy 'b'
      - If |x| is odd : delete all 'b' (emit ε for each 'b'), copy 'a'

    We model deletion as an empty output chunk '' aligned to the deleted input symbol.
    (This keeps the alignment length-preserving at the chunk level.)
    """
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        delete_a = (n % 2 == 0)
        y: List[str] = []
        for ch in x:
            if delete_a:
                y.append("" if ch == "a" else ch)
            else:
                y.append("" if ch == "b" else ch)
        return tuple(y)


class ChunkOracle_DeleteAParityByLastSymbol_AB:
    """
    Σ={a,b}. Deletion example with genuine right-context dependence:
      - If the last symbol is 'a': delete 'a' at even 1-based positions.
      - If the last symbol is 'b': delete 'a' at odd 1-based positions.
      - 'b' is always copied.

    Deletion is represented as an empty output chunk '' aligned to the deleted
    input symbol.
    """
    def chunk(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        n = len(x)
        if n == 0:
            return tuple()

        delete_even_a = (x[-1] == "a")
        y: List[str] = []
        for i, ch in enumerate(x, start=1):
            if ch != "a":
                y.append(ch)
                continue
            delete_here = (i % 2 == 0) if delete_even_a else (i % 2 == 1)
            y.append("" if delete_here else ch)
        return tuple(y)


def build_test_registry() -> Dict[str, OracleSpec]:
    tests = {}

    def add(name, alphabet, oracle, desc):
        tests[name] = OracleSpec(name=name, alphabet=alphabet, oracle=oracle, description=desc)

    add(
        "swap_first_last_ab",
        ("a","b"),
        ChunkOracle_SwapFirstLast_AB(),
        "len>=2 swap first/last; len<=1 copy unchanged"
    ) # yields |QL|=3, |QR|=3, |omega|=18: 

    add(
        "future_c_abc",
        ("a","b","c"),
        ChunkOracle_FutureC_ABC(),
        "a->b iff preceded by b and some c exists to the right"
    ) # yields: |QL|=2, |QR|=2, |omega|=12

    add(
        "global_begin_end_b_ab",
        ("a","b"),
        ChunkOracle_GlobalBeginEndB_AB(),
        "a->b iff word begins with b and ends with b"
    ) # yields: |QL|=3, |QR|=3, |omega|=18

    add(
        "local_cad_abcd",
        ("a","b","c","d"),
        ChunkOracle_LocalCAD_ABCD(),
        "a->b iff immediate context c _ d"
    ) # yields: |QL|=2, |QR|=2, |omega|=16

    add(
        "global_or_local_abcd",
        ("a","b","c","d"),
        ChunkOracle_GlobalOrLocal_ABCD(),
        "a->b iff (begins&ends with b) OR (context c a d)"
    ) # yields: |QL|=5, |QR|=5, |omega|=100
    add(
    "c_parity_left_even_right_odd_abc",
    ("a","b","c"),
    ChunkOracle_CParityLeftRight_ABC(),
    "a->b iff (#c left even) AND (#c right odd)"
    ) # yields: |QL|=2, |QR|=2, |omega|=12 

    add(
    "swap_ab_if_even_length_abc",
    ("a","b","c"),
    ChunkOracle_SwapABIfEvenLength_ABC(),
    "swap a<->b iff word length is even; else copy"
)


    
    add(
        "insert_a_between_cd_abcd",
        ("a","b","c","d"),
        ChunkOracle_InsertA_BetweenCD_ABCD(),
        "insert 'a' between c and d (modeled as chunk 'ca' on the c before d)"
    )

    add(
        "delete_a_even_else_delete_b_ab",
        ("a","b"),
        ChunkOracle_DeleteAEvenElseDeleteB_AB(),
        "if |x| even delete all a's else delete all b's (deletion as empty chunk '')"
    )

    add(
        "delete_a_parity_by_last_symbol_ab",
        ("a","b"),
        ChunkOracle_DeleteAParityByLastSymbol_AB(),
        "if last symbol is a delete a's at even positions else delete a's at odd positions"
    )

    return tests

def random_string(alphabet: Tuple[str, ...], max_len: int, rng: random.Random) -> Tuple[str, ...]:
    L = rng.randint(0, max_len)
    return tuple(rng.choice(alphabet) for _ in range(L))

def make_random_inputs(alphabet, n: int, max_len: int, seed: int):
    rng = random.Random(seed)
    xs = []
    for _ in range(n):
        xs.append(random_string(tuple(alphabet), max_len=max_len, rng=rng))
    return xs


def label_dataset(oracle, xs):
    return [(x, oracle.chunk(x)) for x in xs]


def make_dataset(oracle, alphabet, n: int, max_len: int, seed: int):
    xs = make_random_inputs(alphabet, n=n, max_len=max_len, seed=seed)
    return label_dataset(oracle, xs)



################################################################################
# CORE DATA STRUCTURES
################################################################################

@dataclass
class DFA:
    start: int
    trans: Dict[int, Dict[str, int]]
    sink: Optional[int] = None  # optional completion sink

    def states(self) -> Set[int]:
        st = {self.start}
        st.update(self.trans.keys())
        for s, outs in self.trans.items():
            for _, t in outs.items():
                st.add(t)
        if self.sink is not None:
            st.add(self.sink)
        return st

    def step(self, s: int, a: str) -> Optional[int]:
        nxt = self.trans.get(s, {}).get(a, None)
        if nxt is None and self.sink is not None:
            return self.sink
        return nxt

    def run_prefix_states(self, x: Tuple[str, ...]) -> List[int]:
        """
        q[0]=start, q[i+1]=δ(q[i],x[i])
        """
        q = [self.start]
        cur = self.start
        for ch in x:
            nxt = self.step(cur, ch)
            if nxt is None:
                q.append(-1)
                cur = -1
            else:
                q.append(nxt)
                cur = nxt
        return q

    def run_right_states_after(self, x: Tuple[str, ...]) -> List[int]:
        """
        Right-machine convention:
          p[n]=start
          p[i]=δ(p[i+1], x[i])  reading from right to left
        """
        n = len(x)
        p = [0] * (n + 1)
        p[n] = self.start
        cur = self.start
        for i in range(n - 1, -1, -1):
            nxt = self.step(cur, x[i])
            if nxt is None:
                p[i] = -1
                cur = -1
            else:
                p[i] = nxt
                cur = nxt
        return p

    @staticmethod
    def from_prefix_trie(strings: List[Tuple[str, ...]]) -> Tuple["DFA", Dict[Tuple[str, ...], int]]:
        """
        Build a prefix trie DFA.
        Returns (dfa, map_prefix_to_state).
        """
        trans: Dict[int, Dict[str, int]] = defaultdict(dict)
        prefix_state: Dict[Tuple[str, ...], int] = {}
        next_id = 0

        def new_state() -> int:
            nonlocal next_id
            sid = next_id
            next_id += 1
            return sid

        start = new_state()
        prefix_state[()] = start

        for s in strings:
            cur = start
            pref = []
            for ch in s:
                outs = trans[cur]
                if ch not in outs:
                    nxt = new_state()
                    outs[ch] = nxt
                cur = outs[ch]
                pref.append(ch)
                prefix_state[tuple(pref)] = cur

        return DFA(start=start, trans=dict(trans)), prefix_state

    def add_string(self, x: Tuple[str, ...]) -> None:
        """
        Extend DFA as a trie by adding missing transitions along x.
        (Used for active membership-query augmentation.)
        """
        st = self.states()
        next_id = max(st) + 1 if st else 0
        cur = self.start
        for ch in x:
            outs = self.trans.setdefault(cur, {})
            if ch not in outs:
                outs[ch] = next_id
                next_id += 1
            cur = outs[ch]

    def complete_with_sink(self, alphabet: List[str]) -> None:
        """
        Make the DFA total by adding a sink state ONLY IF needed.
        Idempotent and won't create an unreachable sink if the DFA is already total.
        """
        alpha = sorted(alphabet)
    
        # First check if we're already total (no sink needed).
        all_states = self.states()
        already_total = True
        for s in all_states:
            outs = self.trans.get(s, {})
            for a in alpha:
                if a not in outs:
                    already_total = False
                    break
            if not already_total:
                break
        if already_total:
            return
    
        # If we already have a sink, use it.
        if self.sink is not None:
            sink = self.sink
        else:
            st = self.states()
            sink = max(st) + 1 if st else 0
            self.sink = sink
    
        # Ensure sink exists and self-loops
        self.trans.setdefault(sink, {})
        for a in alpha:
            self.trans[sink][a] = sink
    
        # Fill missing transitions
        for s in (self.states() | {sink}):
            self.trans.setdefault(s, {})
            for a in alpha:
                if a not in self.trans[s]:
                    self.trans[s][a] = sink


    def prune_unreachable(self, alphabet: List[str]) -> None:
        """
        Remove unreachable states from the DFA (including an unreachable sink).
        """
        alpha = sorted(alphabet)
        start = self.start
    
        reachable = set([start])
        q = deque([start])
    
        while q:
            s = q.popleft()
            outs = self.trans.get(s, {})
            for a in alpha:
                t = outs.get(a, None)
                if t is None:
                    continue
                if t not in reachable:
                    reachable.add(t)
                    q.append(t)
    
        # Remove unreachable transitions
        self.trans = {s: outs for s, outs in self.trans.items() if s in reachable}
        for s in list(self.trans.keys()):
            self.trans[s] = {a: t for a, t in self.trans[s].items() if t in reachable}
    
        # Drop sink marker if it got removed
        if self.sink is not None and self.sink not in reachable:
            self.sink = None





@dataclass
class Bimachine:
    left: DFA
    right: DFA
    omega: Dict[Tuple[int, str, int], str]  # (qL, a, qR) -> output chunk
    eps: str = "0"

    def transduce_chunks(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        q = self.left.run_prefix_states(x)
        p = self.right.run_right_states_after(x)
        out = []
        for i, a in enumerate(x):
            key = (q[i], a, p[i + 1])
            out.append(self.omega.get(key, "<?>"))
        return tuple(out)

    def matches_example(self, x: Tuple[str, ...], y: Tuple[str, ...]) -> bool:
        return self.transduce_chunks(x) == y


    def transduce(self, x: Tuple[str, ...]) -> Tuple[str, ...]:
        """
        Produce output tuple y of same length as x, using omega(qL, a, qR).
        """
        n = len(x)
        if n == 0:
            return tuple()

        alpha_x = x  # tuple of symbols

        # Left states at each position i: state after consuming x[:i]
        qL_at = [None] * (n + 1)
        qL_at[0] = self.left.start
        for i in range(n):
            qL_at[i + 1] = self.left.step(qL_at[i], alpha_x[i])

        # Right states at each position i: state after consuming reversed suffix x[i+1:]
        qR_at = [None] * (n + 1)
        qR_at[n] = self.right.start
        for i in range(n - 1, -1, -1):
            # stepping on x[i] builds suffix-from-i, so qR_at[i] corresponds to suffix x[i:]
            qR_at[i] = self.right.step(qR_at[i + 1], alpha_x[i])

        # For output at position i, we want qL after prefix up to i-1 -> qL_at[i]
        # and qR for suffix after i -> qR_at[i+1]
        y = []
        for i, a in enumerate(alpha_x):
            qL = qL_at[i]
            qR = qR_at[i + 1]
            out = self.omega.get((qL, a, qR), None)
            if out is None:
                # fall back to identity if omega missing
                out = a
            y.append(out)

        return tuple(y)

################################################################################
# UTILS: tables and basic evaluation
################################################################################

def random_oracle_agreement(bm, oracle, alphabet, max_len=25, n=200, seed=0, **kwargs) -> float:
    """
    Compare bm vs oracle on n random strings of length up to max_len.
    Backwards-compatible: ignores extra kwargs and accepts alias names.
    """
    rng = random.Random(seed)

    # allow passing in a Random instance
    if "rng" in kwargs and kwargs["rng"] is not None:
        rng = kwargs["rng"]

    # allow common aliases if caller used different names
    if "length" in kwargs:
        max_len = kwargs["length"]
    if "maxlen" in kwargs:
        max_len = kwargs["maxlen"]
    if "max_len_x" in kwargs:
        max_len = kwargs["max_len_x"]
    if "L" in kwargs:
        max_len = kwargs["L"]

    ok = 0
    for _ in range(n):
        L = rng.randint(0, max_len)
        x = tuple(rng.choice(alphabet) for _ in range(L))
        y_or = oracle.chunk(x)
        y_bm = bm.transduce(x)
        if y_or == y_bm:
            ok += 1
    return ok / max(n, 1)


def exact_chunk_accuracy(bm, data, ys=None) -> float:
    """
    If called as exact_chunk_accuracy(bm, data): data is [(x,y), ...]
    If called as exact_chunk_accuracy(bm, xs, ys): data is xs, ys is targets
    """
    # Case 1: data is list of pairs
    if ys is None:
        pairs = data
    else:
        xs = data
        pairs = list(zip(xs, ys))

    correct = 0
    total = 0
    for x, y_true in pairs:
        y_pred = bm.transduce(x)
        if y_pred == y_true:
            correct += 1
        total += 1
    return correct / max(total, 1)

def print_dfa_table(dfa, alphabet, title="DFA transition table"):
    alpha = list(alphabet)
    states = sorted(dfa.states())

    print(title)
    header = ["state"] + alpha
    print(" " + "  ".join(f"{h:>7s}" for h in header))
    print("-" * (9 + 9 * len(alpha)))

    for s in states:
        row = [f"{s:>5d}"]
        for a in alpha:
            t = dfa.step(s, a)
            row.append(f"{t if t is not None else '-':>7}")
        print(" " + "  ".join(row))


def print_bimachine_tables(bm, alphabet):
    alpha = list(alphabet)

    print(f"\n|QL|={len(bm.left.states())}, |QR|={len(bm.right.states())}, |omega|={len(bm.omega)}\n")

    print_dfa_table(bm.left, alpha, title="LEFT DFA transition table")
    print()
    print_dfa_table(bm.right, alpha, title="RIGHT DFA transition table")
    print("\nOMEGA table")

    # Pretty-print omega grouped by (qL,a) over qR
    qLs = sorted(bm.left.states())
    qRs = sorted(bm.right.states())

    for qL in qLs:
        for a in alpha:
            # collect outputs by qR
            outs = []
            for qR in qRs:
                y = bm.omega.get((qL, a, qR), None)
                if y is None:
                    y = "?"
                elif y == "":
                    y = bm.eps
                outs.append(f"{qR}:{y}")
            # If all outputs same, compress
            if len(set(outs)) > 1:
                rhs = ", ".join(outs)
            else:
                rhs = outs[0]
            print(f"omega(qL={qL}, a='{a}', qR=*) -> {rhs}")



def format_transition_table(dfa: DFA, alphabet: List[str]) -> str:
    states = sorted(dfa.states())
    alpha = sorted(alphabet)
    lines = []
    header = ["state"] + alpha
    lines.append("  ".join(f"{h:>6}" for h in header))
    lines.append("-" * (8 * len(header)))
    for s in states:
        row = [f"{s:>6}"]
        for a in alpha:
            t = dfa.trans.get(s, {}).get(a, None)
            row.append(f"{t if t is not None else '-':>6}")
        lines.append("  ".join(row))
    return "\n".join(lines)


def format_omega_table(bm: Bimachine, alphabet: List[str]) -> str:
    alpha = sorted(alphabet)
    QL = sorted(bm.left.states())
    QR = sorted(bm.right.states())
    lines = []
    for qL in QL:
        for a in alpha:
            row = []
            any_def = False
            for qR in QR:
                val = bm.omega.get((qL, a, qR), None)
                if val is not None:
                    if val == "":
                        val = bm.eps
                    any_def = True
                    row.append(f"{qR}:{val}")
            if any_def:
                lines.append(f"omega(qL={qL}, a='{a}', qR=*) -> " + ", ".join(row))
    return "\n".join(lines) if lines else "(omega is empty)"


################################################################################
# SHORTEST WITNESSES (used in omega completion and merge tests)
################################################################################

def shortest_witnesses(dfa: DFA, alphabet: List[str]) -> Dict[int, Tuple[str, ...]]:
    """
    BFS from start to find a shortest word reaching each state.
    """
    alpha = sorted(alphabet)
    start = dfa.start
    wit: Dict[int, Tuple[str, ...]] = {start: ()}
    q = deque([start])

    while q:
        s = q.popleft()
        w = wit[s]
        for a in alpha:
            t = dfa.step(s, a)
            if t is None:
                continue
            if t not in wit:
                wit[t] = w + (a,)
                q.append(t)

    # If any unreachable states exist, still include them with empty witness
    for s in dfa.states():
        wit.setdefault(s, ())
    return wit


################################################################################
# UNION-FIND
################################################################################

class UnionFind:
    def __init__(self, items: Iterable[int]):
        self.parent: Dict[int, int] = {x: x for x in items}
        self.rank: Dict[int, int] = {x: 0 for x in items}

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


################################################################################
# BUILD INITIAL BIMACHINE FROM TRAINING DATA (PTAs) + omega constraints
################################################################################

def build_omega_from_data(left: DFA, right: DFA,
                          data: List[Tuple[Tuple[str, ...], Tuple[str, ...]]]) -> Dict[Tuple[int, str, int], str]:
    omega: Dict[Tuple[int, str, int], str] = {}
    for x, y in data:
        q = left.run_prefix_states(x)
        p = right.run_right_states_after(x)
        if len(x) != len(y):
            raise ValueError("Alignment mismatch: output must provide one chunk per input symbol (len(y)==len(x)).")
        for i, a in enumerate(x):
            key = (q[i], a, p[i + 1])
            if key in omega and omega[key] != y[i]:
                # inconsistent data (should not happen here)
                raise ValueError(f"Omega conflict on {key}: {omega[key]} vs {y[i]}")
            omega[key] = y[i]
    return omega


################################################################################
# MERGE+FOLD for LEFT and RIGHT
################################################################################

def _fold_closure_on_dfa(uf: UnionFind, dfa: DFA, alphabet: List[str]) -> bool:
    """
    Enforce determinism closure:
    for each representative and symbol, all transitions must go to same representative.
    Returns True if any union happened.
    """
    alpha = sorted(alphabet)
    changed = False

    # group by representative
    groups: Dict[int, List[int]] = defaultdict(list)
    for s in dfa.states():
        if s in uf.parent:
            groups[uf.find(s)].append(s)

    for r, members in groups.items():
        for a in alpha:
            targets = set()
            for s in members:
                t = dfa.trans.get(s, {}).get(a, None)
                if t is None or t not in uf.parent:
                    continue
                targets.add(uf.find(t))
            if len(targets) > 1:
                it = iter(targets)
                first = next(it)
                for other in it:
                    if uf.union(first, other):
                        changed = True
    return changed


def try_merge_left(hypo: Bimachine, merge_into: int, merge_from: int, alphabet: List[str]) -> Optional[Bimachine]:
    L, R, omega = hypo.left, hypo.right, hypo.omega
    live = sorted(L.states())
    if merge_into not in live or merge_from not in live:
        return None

    uf = UnionFind(live)
    uf.union(merge_into, merge_from)

    # closure: determinism + omega collision
    while True:
        # determinism closure
        ch = _fold_closure_on_dfa(uf, L, alphabet)

        # omega collision check + possible induced unions not needed here
        collapsed: Dict[Tuple[int, str, int], str] = {}
        for (q, a, p), out in omega.items():
            if q not in uf.parent:
                continue
            rq = uf.find(q)
            key = (rq, a, p)
            if key in collapsed and collapsed[key] != out:
                return None
            collapsed[key] = out

        if not ch:
            break

    # rebuild left DFA under reps
    new_trans: Dict[int, Dict[str, int]] = defaultdict(dict)
    for s, outs in L.trans.items():
        if s not in uf.parent:
            continue
        rs = uf.find(s)
        for a, t in outs.items():
            if t not in uf.parent:
                continue
            rt = uf.find(t)
            new_trans[rs][a] = rt

    new_left = DFA(start=uf.find(L.start), trans=dict(new_trans), sink=None)

    # rebuild omega keys on left side
    new_omega: Dict[Tuple[int, str, int], str] = {}
    for (q, a, p), out in omega.items():
        if q not in uf.parent:
            continue
        rq = uf.find(q)
        new_omega[(rq, a, p)] = out

    return Bimachine(left=new_left, right=R, omega=new_omega, eps=hypo.eps)


def try_merge_right(hypo: Bimachine, merge_into: int, merge_from: int, alphabet: List[str]) -> Optional[Bimachine]:
    L, R, omega = hypo.left, hypo.right, hypo.omega
    live = sorted(R.states())
    if merge_into not in live or merge_from not in live:
        return None

    uf = UnionFind(live)
    uf.union(merge_into, merge_from)

    while True:
        ch = _fold_closure_on_dfa(uf, R, alphabet)

        collapsed: Dict[Tuple[int, str, int], str] = {}
        for (q, a, p), out in omega.items():
            if p not in uf.parent:
                continue
            rp = uf.find(p)
            key = (q, a, rp)
            if key in collapsed and collapsed[key] != out:
                return None
            collapsed[key] = out

        if not ch:
            break

    new_trans: Dict[int, Dict[str, int]] = defaultdict(dict)
    for s, outs in R.trans.items():
        if s not in uf.parent:
            continue
        rs = uf.find(s)
        for a, t in outs.items():
            if t not in uf.parent:
                continue
            rt = uf.find(t)
            new_trans[rs][a] = rt

    new_right = DFA(start=uf.find(R.start), trans=dict(new_trans), sink=None)

    new_omega: Dict[Tuple[int, str, int], str] = {}
    for (q, a, p), out in omega.items():
        if p not in uf.parent:
            continue
        rp = uf.find(p)
        new_omega[(q, a, rp)] = out

    return Bimachine(left=L, right=new_right, omega=new_omega, eps=hypo.eps)


################################################################################
# ORACLE AUGMENTATION (membership queries): extend trie + add omega constraints
################################################################################

def augment_hypothesis_with_examples(
    hypo: Bimachine,
    examples: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
) -> Optional[Bimachine]:
    """
    Extend DFAs by adding paths for witness strings, then add omega constraints.
    Reject if omega conflicts.
    """
    # copy
    new_left = DFA(start=hypo.left.start, trans={s: dict(o) for s, o in hypo.left.trans.items()}, sink=None)
    new_right = DFA(start=hypo.right.start, trans={s: dict(o) for s, o in hypo.right.trans.items()}, sink=None)
    new_omega = dict(hypo.omega)

    # add paths (trie growth)
    for x, _ in examples:
        new_left.add_string(x)
        new_right.add_string(tuple(reversed(x)))

    # add omega constraints
    for x, y in examples:
        q = new_left.run_prefix_states(x)
        p = new_right.run_right_states_after(x)
        for i, a in enumerate(x):
            key = (q[i], a, p[i + 1])
            val = y[i]
            if key in new_omega and new_omega[key] != val:
                return None
            new_omega[key] = val

    return Bimachine(left=new_left, right=new_right, omega=new_omega, eps=hypo.eps)


################################################################################
# RPNI-style RED/BLUE order and WITNESS STRINGS
################################################################################

def red_blue_frontier(dfa: DFA, red: Set[int]) -> List[int]:
    blue = set()
    for r in red:
        for _, t in dfa.trans.get(r, {}).items():
            if t not in red:
                blue.add(t)
    return sorted(blue)


def all_strings_upto(alphabet: List[str], max_len: int) -> List[Tuple[str, ...]]:
    out = [()]
    for L in range(1, max_len + 1):
        out.extend(tuple(s) for s in itertools.product(alphabet, repeat=L))
    return out


def _witness_strings(
    wrap,
    alphabet: List[str],
    train_set: Set[Tuple[str, ...]],
    k: int,
    max_len: int,
    rng: random.Random,
) -> List[Tuple[str, ...]]:
    """
    Up to k strings wrap(w) for non-empty w with |w| <= max_len, excluding
    training strings, in a seeded random order.
    """
    alpha = sorted(alphabet)
    cands = []
    for L in range(1, max_len + 1):
        cands.extend(tuple(s) for s in itertools.product(alpha, repeat=L))
    rng.shuffle(cands)

    out: List[Tuple[str, ...]] = []
    seen: Set[Tuple[str, ...]] = set()
    for w in cands:
        if len(out) >= k:
            break
        x = wrap(w)
        if x not in train_set and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def witness_strings_for_left_merge(
    base_prefix: Tuple[str, ...],
    alphabet: List[str],
    train_set: Set[Tuple[str, ...]],
    k: int,
    cont_max_len: int,
    rng: random.Random,
) -> List[Tuple[str, ...]]:
    """Witness set W_L(u): strings pi(u) z with continuations 1 <= |z| <= cont_max_len."""
    return _witness_strings(lambda w: base_prefix + w, alphabet, train_set, k, cont_max_len, rng)


def witness_strings_for_right_merge(
    base_suffix: Tuple[str, ...],
    alphabet: List[str],
    train_set: Set[Tuple[str, ...]],
    k: int,
    head_max_len: int,
    rng: random.Random,
) -> List[Tuple[str, ...]]:
    """Witness set W_R(u): strings h sigma(u) with heads 1 <= |h| <= head_max_len."""
    return _witness_strings(lambda w: w + base_suffix, alphabet, train_set, k, head_max_len, rng)


################################################################################
# RPNI-style MERGE PASSES with RESTART after each accepted merge
################################################################################

def rpni_pass_left(
    hypo: Bimachine,
    oracle: ChunkOracle,
    data: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    alphabet: List[str],
    k_witness: int,
    cont_max_len: int,
    rng: random.Random,
    verbose: bool = True,
) -> Tuple[Bimachine, bool]:
    train_set = set(x for x, _ in data)
    changed_any = False

    while True:
        red: Set[int] = {hypo.left.start}
        wit = shortest_witnesses(hypo.left, alphabet)

        merged_in_restart = False

        while True:
            blue = red_blue_frontier(hypo.left, red)
            blue = [u for u in blue if u in wit]
            if not blue:
                break

            blue.sort(key=lambda s: (len(wit[s]), wit[s]))
            u = blue[0]
            u_prefix = wit[u]

            merged = False
            red_list = sorted(list(red), key=lambda s: (len(wit.get(s, ())), wit.get(s, ())))
            for r in red_list:
                if r == u:
                    continue
                cand = try_merge_left(hypo, r, u, alphabet)
                if cand is None:
                    continue
                if not all(cand.matches_example(x, y) for x, y in data):
                    continue

                tests = witness_strings_for_left_merge(u_prefix, alphabet, train_set, k_witness, cont_max_len, rng)
                witness_examples = [(x, oracle.chunk(x)) for x in tests]
                cand2 = augment_hypothesis_with_examples(cand, witness_examples)
                if cand2 is None:
                    continue

                if verbose:
                    print(f"[LEFT] merge {u} -> {r}  (|QL| {len(hypo.left.states())} -> {len(cand2.left.states())})")
                hypo = cand2
                changed_any = True
                merged = True
                merged_in_restart = True
                break

            if merged:
                break
            else:
                red.add(u)
                if verbose:
                    print(f"[LEFT] promote {u} (prefix={''.join(u_prefix)})")

        if not merged_in_restart:
            break

    return hypo, changed_any


def rpni_pass_right(
    hypo: Bimachine,
    oracle: ChunkOracle,
    data: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    alphabet: List[str],
    k_witness: int,
    head_max_len: int,
    rng: random.Random,
    verbose: bool = True,
) -> Tuple[Bimachine, bool]:
    train_set = set(x for x, _ in data)
    changed_any = False

    while True:
        red: Set[int] = {hypo.right.start}
        wit_rev = shortest_witnesses(hypo.right, alphabet)

        def state_to_suffix(st: int) -> Tuple[str, ...]:
            # witness word is in R-machine reading direction; suffix forward is reversed
            return tuple(reversed(wit_rev[st]))

        merged_in_restart = False

        while True:
            blue = red_blue_frontier(hypo.right, red)
            blue = [u for u in blue if u in wit_rev]
            if not blue:
                break

            blue.sort(key=lambda s: (len(wit_rev[s]), wit_rev[s]))
            u = blue[0]
            u_suffix = state_to_suffix(u)

            merged = False
            red_list = sorted(list(red), key=lambda s: (len(wit_rev.get(s, ())), wit_rev.get(s, ())))
            for r in red_list:
                if r == u:
                    continue
                cand = try_merge_right(hypo, r, u, alphabet)
                if cand is None:
                    continue
                if not all(cand.matches_example(x, y) for x, y in data):
                    continue

                tests = witness_strings_for_right_merge(u_suffix, alphabet, train_set, k_witness, head_max_len, rng)
                witness_examples = [(x, oracle.chunk(x)) for x in tests]
                cand2 = augment_hypothesis_with_examples(cand, witness_examples)
                if cand2 is None:
                    continue

                if verbose:
                    print(f"[RIGHT] merge {u} -> {r} (|QR| {len(hypo.right.states())} -> {len(cand2.right.states())})")
                hypo = cand2
                changed_any = True
                merged = True
                merged_in_restart = True
                break

            if merged:
                break
            else:
                red.add(u)
                if verbose:
                    print(f"[RIGHT] promote {u} (suffix={''.join(u_suffix)})")

        if not merged_in_restart:
            break

    return hypo, changed_any


################################################################################
# SAMPLE-ONLY MERGE PASSES / MINIMIZATION (no oracle queries)
################################################################################

def rpni_pass_left_examples(
    hypo: Bimachine,
    data: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    alphabet: List[str],
    verbose: bool = True,
) -> Tuple[Bimachine, bool]:
    changed_any = False

    while True:
        red: Set[int] = {hypo.left.start}
        wit = shortest_witnesses(hypo.left, alphabet)
        merged_in_restart = False

        while True:
            blue = red_blue_frontier(hypo.left, red)
            blue = [u for u in blue if u in wit]
            if not blue:
                break

            blue.sort(key=lambda s: (len(wit[s]), wit[s]))
            u = blue[0]
            merged = False

            red_list = sorted(list(red), key=lambda s: (len(wit.get(s, ())), wit.get(s, ())))
            for r in red_list:
                if r == u:
                    continue
                cand = try_merge_left(hypo, r, u, alphabet)
                if cand is None:
                    continue
                if not all(cand.matches_example(x, y) for x, y in data):
                    continue
                if verbose:
                    print(f"[LEFT sample] merge {u} -> {r} (|QL| {len(hypo.left.states())} -> {len(cand.left.states())})")
                hypo = cand
                changed_any = True
                merged = True
                merged_in_restart = True
                break

            if merged:
                break
            red.add(u)
            if verbose:
                print(f"[LEFT sample] promote {u} (prefix={''.join(wit[u])})")

        if not merged_in_restart:
            break

    return hypo, changed_any


def rpni_pass_right_examples(
    hypo: Bimachine,
    data: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    alphabet: List[str],
    verbose: bool = True,
) -> Tuple[Bimachine, bool]:
    changed_any = False

    while True:
        red: Set[int] = {hypo.right.start}
        wit_rev = shortest_witnesses(hypo.right, alphabet)

        def state_to_suffix(st: int) -> Tuple[str, ...]:
            return tuple(reversed(wit_rev[st]))

        merged_in_restart = False

        while True:
            blue = red_blue_frontier(hypo.right, red)
            blue = [u for u in blue if u in wit_rev]
            if not blue:
                break

            blue.sort(key=lambda s: (len(wit_rev[s]), wit_rev[s]))
            u = blue[0]
            merged = False
            red_list = sorted(list(red), key=lambda s: (len(wit_rev.get(s, ())), wit_rev.get(s, ())))

            for r in red_list:
                if r == u:
                    continue
                cand = try_merge_right(hypo, r, u, alphabet)
                if cand is None:
                    continue
                if not all(cand.matches_example(x, y) for x, y in data):
                    continue
                if verbose:
                    print(f"[RIGHT sample] merge {u} -> {r} (|QR| {len(hypo.right.states())} -> {len(cand.right.states())})")
                hypo = cand
                changed_any = True
                merged = True
                merged_in_restart = True
                break

            if merged:
                break
            red.add(u)
            if verbose:
                print(f"[RIGHT sample] promote {u} (suffix={''.join(state_to_suffix(u))})")

        if not merged_in_restart:
            break

    return hypo, changed_any


def sample_complete_omega_from_data(
    bm: Bimachine,
    data: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    alphabet: List[str],
) -> Bimachine:
    """
    Fill additional omega entries only when the representative string
    prefix(qL) + a + suffix(qR) is itself present in the labeled sample.
    No oracle calls are made.
    """
    alpha = sorted(alphabet)
    sample_map = {x: y for x, y in data}
    witL = shortest_witnesses(bm.left, alpha)
    witR_rev = shortest_witnesses(bm.right, alpha)

    def suffix_for_qR(qR: int) -> Tuple[str, ...]:
        return tuple(reversed(witR_rev[qR]))

    new_omega = dict(bm.omega)
    for qL in sorted(bm.left.states()):
        pref = witL[qL]
        for qR in sorted(bm.right.states()):
            suf = suffix_for_qR(qR)
            for a in alpha:
                key = (qL, a, qR)
                if key in new_omega:
                    continue
                x = pref + (a,) + suf
                y = sample_map.get(x)
                if y is None:
                    continue
                i = len(pref)
                if i < len(y):
                    new_omega[key] = y[i]

    return Bimachine(left=bm.left, right=bm.right, omega=new_omega, eps=bm.eps)


def minimize_left_given_right_partial(bm: Bimachine, alphabet: List[str]) -> Tuple[DFA, Dict[int, int]]:
    alpha = sorted(alphabet)
    QL = sorted(bm.left.states())
    QR = sorted(bm.right.states())
    missing = object()

    def omega_row(qL: int) -> Tuple[object, ...]:
        return tuple(bm.omega.get((qL, a, qR), missing) for a in alpha for qR in QR)

    part: Dict[int, int] = {}
    sig2id: Dict[Tuple, int] = {}
    for q in QL:
        sig = omega_row(q)
        sig2id.setdefault(sig, len(sig2id))
        part[q] = sig2id[sig]

    changed = True
    while changed:
        changed = False
        new_part: Dict[int, int] = {}
        sig2id = {}
        for q in QL:
            trans_sig = tuple(part.get(bm.left.step(q, a), -1) for a in alpha)
            sig = (omega_row(q), trans_sig)
            sig2id.setdefault(sig, len(sig2id))
            new_part[q] = sig2id[sig]
        if any(new_part[q] != part[q] for q in QL):
            part = new_part
            changed = True

    new_trans: Dict[int, Dict[str, int]] = defaultdict(dict)
    for q in QL:
        cq = part[q]
        for a in alpha:
            t = bm.left.step(q, a)
            if t is not None:
                new_trans[cq][a] = part[t]

    new_left = DFA(start=part[bm.left.start], trans=dict(new_trans), sink=None)
    return new_left, part


def minimize_right_given_left_partial(bm: Bimachine, alphabet: List[str]) -> Tuple[DFA, Dict[int, int]]:
    alpha = sorted(alphabet)
    QL = sorted(bm.left.states())
    QR = sorted(bm.right.states())
    missing = object()

    def omega_col(qR: int) -> Tuple[object, ...]:
        return tuple(bm.omega.get((qL, a, qR), missing) for qL in QL for a in alpha)

    part: Dict[int, int] = {}
    sig2id: Dict[Tuple, int] = {}
    for q in QR:
        sig = omega_col(q)
        sig2id.setdefault(sig, len(sig2id))
        part[q] = sig2id[sig]

    changed = True
    while changed:
        changed = False
        new_part: Dict[int, int] = {}
        sig2id = {}
        for q in QR:
            trans_sig = tuple(part.get(bm.right.step(q, a), -1) for a in alpha)
            sig = (omega_col(q), trans_sig)
            sig2id.setdefault(sig, len(sig2id))
            new_part[q] = sig2id[sig]
        if any(new_part[q] != part[q] for q in QR):
            part = new_part
            changed = True

    new_trans: Dict[int, Dict[str, int]] = defaultdict(dict)
    for q in QR:
        cq = part[q]
        for a in alpha:
            t = bm.right.step(q, a)
            if t is not None:
                new_trans[cq][a] = part[t]

    new_right = DFA(start=part[bm.right.start], trans=dict(new_trans), sink=None)
    return new_right, part


def minimize_bimachine_from_examples(
    bm: Bimachine,
    data: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    alphabet: List[str],
    iters: int = 2,
) -> Bimachine:
    bm = sample_complete_omega_from_data(bm, data, alphabet)

    for _ in range(iters):
        new_left, mapL = minimize_left_given_right_partial(bm, alphabet)
        omega1: Dict[Tuple[int, str, int], str] = {}
        for (qL, a, qR), out in bm.omega.items():
            omega1[(mapL[qL], a, qR)] = out
        bm = Bimachine(left=new_left, right=bm.right, omega=omega1, eps=bm.eps)

        bm = sample_complete_omega_from_data(bm, data, alphabet)

        new_right, mapR = minimize_right_given_left_partial(bm, alphabet)
        omega2: Dict[Tuple[int, str, int], str] = {}
        for (qL, a, qR), out in bm.omega.items():
            omega2[(qL, a, mapR[qR])] = out
        bm = Bimachine(left=bm.left, right=new_right, omega=omega2, eps=bm.eps)

        bm = sample_complete_omega_from_data(bm, data, alphabet)

    return bm


def learn_bimachine_from_examples(
    alphabet: List[str],
    train: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    seed: int = 0,
    max_rounds: int = 10,
    min_iters: int = 2,
    verbose: bool = True,
) -> Bimachine:
    train_strings = [x for x, _ in train]
    left_pta, _ = DFA.from_prefix_trie(train_strings)
    right_pta, _ = DFA.from_prefix_trie([tuple(reversed(x)) for x, _ in train])

    omega0 = build_omega_from_data(left_pta, right_pta, train)
    hypo = Bimachine(left=left_pta, right=right_pta, omega=omega0, eps="")

    for r in range(1, max_rounds + 1):
        if verbose:
            print("=" * 80)
            print(f"SAMPLE-ONLY ROUND {r}")
        changed = False

        hypo, chL = rpni_pass_left_examples(hypo, train, alphabet, verbose=verbose)
        changed |= chL

        hypo, chR = rpni_pass_right_examples(hypo, train, alphabet, verbose=verbose)
        changed |= chR

        if verbose:
            print(f"After sample-only round {r}: |QL|={len(hypo.left.states())}, |QR|={len(hypo.right.states())}, |omega|={len(hypo.omega)}")

        if not changed:
            if verbose:
                print("No sample-only merges accepted; stopping.")
            break

    hypo = minimize_bimachine_from_examples(hypo, train, alphabet, iters=min_iters)
    hypo = renumber_bimachine(hypo, alphabet)
    return hypo


################################################################################
# BIMACHINE MINIMIZATION (the key for begin/end predicates)
################################################################################

def fill_omega_all_triples(bm: Bimachine, oracle: ChunkOracle, alphabet: List[str]) -> Bimachine:
    """
    Define omega for every (qL,a,qR) by constructing a string:
      x = prefix(qL) + a + suffix(qR)
    and reading oracle chunk at that position.
    """
    alpha = sorted(alphabet)
    bm.left.complete_with_sink(alpha)
    bm.right.complete_with_sink(alpha)

    witL = shortest_witnesses(bm.left, alpha)
    witR_rev = shortest_witnesses(bm.right, alpha)

    def suffix_for_qR(qR: int) -> Tuple[str, ...]:
        return tuple(reversed(witR_rev[qR]))

    new_omega = dict(bm.omega)
    QL = sorted(bm.left.states())
    QR = sorted(bm.right.states())

    for qL in QL:
        pref = witL[qL]
        for qR in QR:
            suf = suffix_for_qR(qR)
            for a in alpha:
                key = (qL, a, qR)
                if key in new_omega:
                    continue
                x = pref + (a,) + suf
                y = oracle.chunk(x)
                i = len(pref)
                new_omega[key] = y[i]

    return Bimachine(left=bm.left, right=bm.right, omega=new_omega, eps=bm.eps)


def minimize_left_given_right(bm: Bimachine, alphabet: List[str]) -> Tuple[DFA, Dict[int, int]]:
    alpha = sorted(alphabet)
    QL = sorted(bm.left.states())
    QR = sorted(bm.right.states())

    def omega_row(qL: int) -> Tuple[str, ...]:
        return tuple(bm.omega[(qL, a, qR)] for a in alpha for qR in QR)

    # initial partition by omega rows
    part: Dict[int, int] = {}
    sig2id: Dict[Tuple, int] = {}
    for q in QL:
        sig = omega_row(q)
        sig2id.setdefault(sig, len(sig2id))
        part[q] = sig2id[sig]

    changed = True
    while changed:
        changed = False
        new_part: Dict[int, int] = {}
        sig2id = {}
        for q in QL:
            trans_sig = tuple(part[bm.left.step(q, a)] for a in alpha)
            sig = (omega_row(q), trans_sig)
            sig2id.setdefault(sig, len(sig2id))
            new_part[q] = sig2id[sig]
        if any(new_part[q] != part[q] for q in QL):
            part = new_part
            changed = True

    # build quotient DFA
    new_trans: Dict[int, Dict[str, int]] = defaultdict(dict)
    for q in QL:
        cq = part[q]
        for a in alpha:
            t = bm.left.step(q, a)
            new_trans[cq][a] = part[t]

    new_left = DFA(start=part[bm.left.start], trans=dict(new_trans), sink=None)
    new_left.complete_with_sink(alpha)
    return new_left, part


def minimize_right_given_left(bm: Bimachine, alphabet: List[str]) -> Tuple[DFA, Dict[int, int]]:
    alpha = sorted(alphabet)
    QL = sorted(bm.left.states())
    QR = sorted(bm.right.states())

    def omega_col(qR: int) -> Tuple[str, ...]:
        return tuple(bm.omega[(qL, a, qR)] for qL in QL for a in alpha)

    part: Dict[int, int] = {}
    sig2id: Dict[Tuple, int] = {}
    for q in QR:
        sig = omega_col(q)
        sig2id.setdefault(sig, len(sig2id))
        part[q] = sig2id[sig]

    changed = True
    while changed:
        changed = False
        new_part: Dict[int, int] = {}
        sig2id = {}
        for q in QR:
            trans_sig = tuple(part[bm.right.step(q, a)] for a in alpha)
            sig = (omega_col(q), trans_sig)
            sig2id.setdefault(sig, len(sig2id))
            new_part[q] = sig2id[sig]
        if any(new_part[q] != part[q] for q in QR):
            part = new_part
            changed = True

    new_trans: Dict[int, Dict[str, int]] = defaultdict(dict)
    for q in QR:
        cq = part[q]
        for a in alpha:
            t = bm.right.step(q, a)
            new_trans[cq][a] = part[t]

    new_right = DFA(start=part[bm.right.start], trans=dict(new_trans), sink=None)
    new_right.complete_with_sink(alpha)
    return new_right, part


def minimize_bimachine(bm: Bimachine, oracle: ChunkOracle, alphabet: List[str], iters: int = 3) -> Bimachine:
    """
    Iteratively:
      complete omega
      minimize left
      minimize right
      renormalize omega
    """
    alpha = sorted(alphabet)
    bm = fill_omega_all_triples(bm, oracle, alpha)

    for _ in range(iters):
        new_left, mapL = minimize_left_given_right(bm, alpha)
        omega1: Dict[Tuple[int, str, int], str] = {}
        for (qL, a, qR), out in bm.omega.items():
            omega1[(mapL[qL], a, qR)] = out
        bm = Bimachine(left=new_left, right=bm.right, omega=omega1, eps=bm.eps)

        bm = fill_omega_all_triples(bm, oracle, alpha)

        new_right, mapR = minimize_right_given_left(bm, alpha)
        omega2: Dict[Tuple[int, str, int], str] = {}
        for (qL, a, qR), out in bm.omega.items():
            omega2[(qL, a, mapR[qR])] = out
        bm = Bimachine(left=bm.left, right=new_right, omega=omega2, eps=bm.eps)

        bm = fill_omega_all_triples(bm, oracle, alpha)

    return bm


################################################################################
# RENNUMBER START STATES TO 0 (for nicer printing)
################################################################################

def renumber_dfa_bfs(dfa: DFA, alphabet: List[str]) -> Tuple[DFA, Dict[int, int]]:
    alpha = sorted(alphabet)
    start = dfa.start

    order = []
    seen = set([start])
    q = deque([start])

    while q:
        s = q.popleft()
        order.append(s)
        for a in alpha:
            t = dfa.step(s, a)
            if t is None:
                continue
            if t not in seen:
                seen.add(t)
                q.append(t)

    old2new = {old: i for i, old in enumerate(order)}

    new_trans: Dict[int, Dict[str, int]] = defaultdict(dict)
    for s_old, outs in dfa.trans.items():
        if s_old not in old2new:
            continue
        s_new = old2new[s_old]
        for a, t_old in outs.items():
            if t_old not in old2new:
                continue
            new_trans[s_new][a] = old2new[t_old]

    new_dfa = DFA(start=0, trans=dict(new_trans), sink=None)
    
    # Optional: if you want total DFAs for internal work, do it elsewhere.
    # Here we prefer minimal printable machines:
    new_dfa.prune_unreachable(alpha)
    
    return new_dfa, old2new



def renumber_bimachine(bm: Bimachine, alphabet: List[str]) -> Bimachine:
    alpha = sorted(alphabet)
    L, mapL = renumber_dfa_bfs(bm.left, alpha)
    R, mapR = renumber_dfa_bfs(bm.right, alpha)

    new_omega: Dict[Tuple[int, str, int], str] = {}
    for (qL, a, qR), out in bm.omega.items():
        if qL in mapL and qR in mapR:
            new_omega[(mapL[qL], a, mapR[qR])] = out
    return Bimachine(left=L, right=R, omega=new_omega, eps=bm.eps)


################################################################################
# LEARNING PIPELINE
################################################################################



def learn_bimachine_from_oracle(
    oracle: ChunkOracle,
    alphabet: List[str],
    train: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    seed: int = 0,
    max_rounds: int = 10,
    k_witness: int = 40,
    cont_max_len: int = 3,
    head_max_len: int = 3,
    verbose: bool = True,
) -> Bimachine:
    rng = random.Random(seed)

    # initial PTAs
    train_strings = [x for x, _ in train]
    left_pta, _ = DFA.from_prefix_trie(train_strings)
    right_pta, _ = DFA.from_prefix_trie([tuple(reversed(x)) for x, _ in train])

    omega0 = build_omega_from_data(left_pta, right_pta, train)
    hypo = Bimachine(left=left_pta, right=right_pta, omega=omega0, eps=getattr(oracle, "eps", "0"))

    # merge rounds
    for r in range(1, max_rounds + 1):
        if verbose:
            print("=" * 80)
            print(f"ROUND {r}")
        changed = False

        hypo, chL = rpni_pass_left(hypo, oracle, train, alphabet,
                                  k_witness=k_witness, cont_max_len=cont_max_len,
                                  rng=rng, verbose=verbose)
        changed |= chL

        hypo, chR = rpni_pass_right(hypo, oracle, train, alphabet,
                                   k_witness=k_witness, head_max_len=head_max_len,
                                   rng=rng, verbose=verbose)
        changed |= chR

        if verbose:
            agree = random_oracle_agreement(hypo, oracle, alphabet, length=25, n=200, rng=rng)
            print(f"After round {r}: |QL|={len(hypo.left.states())}, |QR|={len(hypo.right.states())}, |omega|={len(hypo.omega)}")
            print(f"Random oracle agreement (len=25, n=200): {agree:.3f}")

        if not changed:
            if verbose:
                print("No merges accepted; stopping.")
            break

    # --- crucial finishing step: minimize the bimachine ---
    hypo = minimize_bimachine(hypo, oracle, alphabet, iters=3)
    hypo = renumber_bimachine(hypo, alphabet)
    return hypo


################################################################################
# OPTIONAL: Graphviz view (like pyfoma, but minimal)
################################################################################

def check_graphviz_installed() -> bool:
    return shutil.which("dot") is not None


def view_dfa(dfa: DFA, alphabet: List[str], title: str = "DFA"):
    import graphviz
    if not check_graphviz_installed():
        raise EnvironmentError("Graphviz 'dot' not found. Install graphviz (e.g. brew install graphviz).")

    alpha = sorted(alphabet)
    g = graphviz.Digraph(title, graph_attr={"rankdir": "LR"})
    g.attr("node", shape="circle")

    for s in sorted(dfa.states()):
        if s == dfa.start:
            g.node(str(s), style="bold")
        else:
            g.node(str(s))

    for s in sorted(dfa.states()):
        grouped = defaultdict(list)
        for a in alpha:
            t = dfa.step(s, a)
            grouped[t].append(a)
        for t, labs in grouped.items():
            g.edge(str(s), str(t), label=",".join(sorted(labs)))
    return g


def _check_graphviz_available():
    """
    Returns (ok, msg). ok=True if python graphviz package and the 'dot' executable are available.
    """
    try:
        import graphviz  # noqa: F401
    except Exception as e:
        return False, f"Python package 'graphviz' not installed ({e}). Try: pip install graphviz"

    import shutil
    if shutil.which("dot") is None:
        return False, "Graphviz 'dot' executable not found. Install Graphviz system package (e.g. brew install graphviz)."

    return True, ""


def dfa_to_graphviz(dfa, alphabet, title="DFA", show_sink=True):
    """
    Return graphviz.Digraph for a DFA.
    - start state is bold
    - sink state (if dfa.sink) is dashed
    - nodes are circles
    """
    ok, msg = _check_graphviz_available()
    if not ok:
        raise EnvironmentError(msg)

    import graphviz

    alpha = list(alphabet)
    g = graphviz.Digraph(title, graph_attr={"rankdir": "LR", "label": title, "labelloc": "t"})

    states = sorted(dfa.states())

    # Nodes
    for s in states:
        attrs = {"shape": "circle"}
        if s == dfa.start:
            attrs["style"] = "bold"
            attrs["penwidth"] = "2"
        if show_sink and getattr(dfa, "sink", None) is not None and s == dfa.sink:
            attrs["style"] = (attrs.get("style", "") + ",dashed").strip(",")
        g.node(str(s), **attrs)

    # Edges (group targets by label set like PyFoma does)
    for s in states:
        grouped = {}  # target -> [symbols]
        for a in alpha:
            t = dfa.step(s, a)
            if t is None:
                continue
            grouped.setdefault(t, []).append(a)

        for t, labels in grouped.items():
            lab = ",".join(labels)
            g.edge(str(s), str(t), label=lab)

    return g


def bimachine_to_graphviz(bm, alphabet, title="Bimachine"):
    """
    Combined Graphviz figure: Left DFA cluster + Right DFA cluster.
    We do NOT draw omega edges (too large); instead add a note.
    """
    ok, msg = _check_graphviz_available()
    if not ok:
        raise EnvironmentError(msg)

    import graphviz

    alpha = list(alphabet)
    g = graphviz.Digraph(title, graph_attr={"rankdir": "LR", "label": title, "labelloc": "t"})

    # Left cluster
    with g.subgraph(name="cluster_left") as c:
        c.attr(label="Left DFA", color="gray")
        gL = dfa_to_graphviz(bm.left, alpha, title="Left DFA")
        # graphviz doesn't let us directly merge Digraphs, so we recreate in cluster:
        for s in sorted(bm.left.states()):
            attrs = {"shape": "circle"}
            if s == bm.left.start:
                attrs["style"] = "bold"
                attrs["penwidth"] = "2"
            if getattr(bm.left, "sink", None) is not None and s == bm.left.sink:
                attrs["style"] = (attrs.get("style", "") + ",dashed").strip(",")
            c.node(f"L{s}", label=str(s), **attrs)

        for s in sorted(bm.left.states()):
            grouped = {}
            for a in alpha:
                t = bm.left.step(s, a)
                if t is None:
                    continue
                grouped.setdefault(t, []).append(a)
            for t, labels in grouped.items():
                c.edge(f"L{s}", f"L{t}", label=",".join(labels))

    # Right cluster
    with g.subgraph(name="cluster_right") as c:
        c.attr(label="Right DFA", color="gray")
        for s in sorted(bm.right.states()):
            attrs = {"shape": "circle"}
            if s == bm.right.start:
                attrs["style"] = "bold"
                attrs["penwidth"] = "2"
            if getattr(bm.right, "sink", None) is not None and s == bm.right.sink:
                attrs["style"] = (attrs.get("style", "") + ",dashed").strip(",")
            c.node(f"R{s}", label=str(s), **attrs)

        for s in sorted(bm.right.states()):
            grouped = {}
            for a in alpha:
                t = bm.right.step(s, a)
                if t is None:
                    continue
                grouped.setdefault(t, []).append(a)
            for t, labels in grouped.items():
                c.edge(f"R{s}", f"R{t}", label=",".join(labels))

    # Omega note node
    omega_note = f"|QL|={len(bm.left.states())}, |QR|={len(bm.right.states())}, |omega|={len(bm.omega)}"
    g.node("OMEGA_NOTE", label=f"ω table stored separately\\n{omega_note}", shape="note")

    # Link note visually to both clusters (dashed)
    g.edge("OMEGA_NOTE", "L0", style="dashed")
    g.edge("OMEGA_NOTE", "R0", style="dashed")

    return g


def render_graph(g, outpath_no_ext, fmt="svg", view=False):
    """
    Render graphviz.Digraph to file.
    outpath_no_ext: e.g. "graphs/out_left" -> writes "graphs/out_left.svg"
    fmt: svg|png|pdf
    view: if True, graphviz may attempt to open the file (depends on OS)
    """
    ok, msg = _check_graphviz_available()
    if not ok:
        raise EnvironmentError(msg)

    # graphviz.render expects filename without extension; format controls extension
    g.format = fmt
    return g.render(filename=outpath_no_ext, cleanup=True, view=view)



################################################################################
# MAIN DEMO
################################################################################

import argparse

def main():
    tests = build_test_registry()

    parser = argparse.ArgumentParser(
        description="Learn a bimachine from aligned examples and a chunk oracle (paper Table 1 settings by default)."
    )
    parser.add_argument("--test", type=str, default="global_or_local_abcd",
                        choices=sorted(tests.keys()),
                        help="Which oracle/transduction to learn")
    parser.add_argument("--n_train", type=int, default=2000)
    parser.add_argument("--n_test", type=int, default=500)
    parser.add_argument("--max_len_train", type=int, default=12)
    parser.add_argument("--max_len_test", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)

    # learner knobs
    parser.add_argument("--max_rounds", type=int, default=10)
    parser.add_argument("--k_witness", type=int, default=80)
    parser.add_argument("--cont_max_len", type=int, default=3)
    parser.add_argument("--head_max_len", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--view", type=str, default="none", choices=["none", "left", "right", "bimachine", "all"], help="Render Graphviz diagrams to files")
    parser.add_argument("--format", type=str, default="svg", choices=["svg", "png", "pdf"], help="Graphviz output format")
    parser.add_argument("--out", type=str, default="bimachine", help="Output prefix for graph files (no extension)")
    parser.add_argument("--open", action="store_true", help="Try to open the rendered file (Graphviz view=True)")

    args = parser.parse_args()

    spec = tests[args.test]
    alphabet = list(spec.alphabet)
    oracle = MemoizedChunkOracle(spec.oracle)

    print("=" * 80)
    print(f"TEST: {spec.name}")
    print(spec.description)
    print(f"Alphabet: {alphabet}")
    print("=" * 80)

    train_xs = make_random_inputs(spec.alphabet, n=args.n_train, max_len=args.max_len_train, seed=args.seed)
    test_xs = make_random_inputs(spec.alphabet, n=args.n_test, max_len=args.max_len_test, seed=args.seed + 1337)
    train = label_dataset(oracle, train_xs)
    test = label_dataset(oracle, test_xs)

    bm = learn_bimachine_from_oracle(
        oracle=oracle,
        alphabet=alphabet,
        train=train,
        seed=args.seed,
        max_rounds=args.max_rounds,
        k_witness=args.k_witness,
        cont_max_len=args.cont_max_len,
        head_max_len=args.head_max_len,
        verbose=args.verbose,
    )

    print("\n" + "=" * 80)
    print("FINAL BIMACHINE (after minimization & renumbering)")
    print_bimachine_tables(bm, alphabet)

    print("\n" + "=" * 80)
    print(f"Test-set exact chunk accuracy: {exact_chunk_accuracy(bm, test):.3f}")

    if args.view != "none":
        try:
            if args.view in ("left", "all"):
                gL = dfa_to_graphviz(bm.left, alphabet, title=f"{spec.name} — Left DFA")
                render_graph(gL, f"{args.out}_left", fmt=args.format, view=args.open)
                print(f"[graphviz] wrote {args.out}_left.{args.format}")
            if args.view in ("right", "all"):
                gR = dfa_to_graphviz(bm.right, alphabet, title=f"{spec.name} — Right DFA")
                render_graph(gR, f"{args.out}_right", fmt=args.format, view=args.open)
                print(f"[graphviz] wrote {args.out}_right.{args.format}")
            if args.view in ("bimachine", "all"):
                gB = bimachine_to_graphviz(bm, alphabet, title=f"{spec.name} — Bimachine")
                render_graph(gB, f"{args.out}_bimachine", fmt=args.format, view=args.open)
                print(f"[graphviz] wrote {args.out}_bimachine.{args.format}")
        except EnvironmentError as e:
            print(f"[graphviz] disabled: {e}")

    agree = random_oracle_agreement(bm, oracle, alphabet, max_len=args.max_len_test, n=500, seed=args.seed + 2025)
    print(f"Random oracle agreement (len={args.max_len_test}, n=500): {agree:.3f}")


def list_tests():
    tests = build_test_registry()
    for k in sorted(tests):
        print(f"{k:22s}  Σ={{{','.join(tests[k].alphabet)}}}  - {tests[k].description}")



if __name__ == "__main__":
    main()
