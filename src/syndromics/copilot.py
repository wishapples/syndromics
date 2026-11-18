"""
Plain-English -> quantum circuit.

Built-in recipes cover the circuits people actually run on today's hardware (and teach in
class): Bell pairs, GHZ states, Bernstein-Vazirani, Grover search, QFT, teleportation,
random bits, and a small QAOA MaxCut. Each recipe returns the circuit, a short explanation
for students, and what the ideal output should be.

Prompts the recipes do not cover can be sent to a language model if ANTHROPIC_API_KEY is set
(the model must answer in OpenQASM 2, which is parsed, never executed as code).

Honesty rule: requests for large industrial optimisation ("routes for 50 trucks") are answered
truthfully: today's quantum computers do not beat classical solvers on these, so we say so and
offer a small teaching-size version instead.
"""
from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np

MAX_QUBITS = 40


@dataclass
class Recipe:
    name: str
    circuit: object
    explanation: str
    expected: str
    notes: list = field(default_factory=list)


_NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def _numbers(text):
    out = [int(x) for x in re.findall(r"\b\d+\b", text)]
    out += [_NUM_WORDS[w] for w in re.findall(r"[a-z]+", text) if w in _NUM_WORDS]
    return out


def _qubit_count(text, default):
    m = re.search(r"(\d+|\b(?:" + "|".join(_NUM_WORDS) + r")\b)[\s-]*(?:[a-z]+[\s-]+)?(?:qubits?|qbits?|bits?|nodes?|cities|trucks?|vertices)", text)
    if m:
        v = m.group(1)
        return int(v) if v.isdigit() else _NUM_WORDS[v]
    return default


def _bitstring(text):
    m = re.search(r"\b([01]{2,})\b", text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- recipes
def bell():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2, 2, name="bell")
    qc.h(0); qc.cx(0, 1); qc.measure([0, 1], [0, 1])
    return Recipe("Bell pair", qc,
                  "A Hadamard puts qubit 0 in superposition; a CNOT copies that into qubit 1, "
                  "so the two qubits are entangled: measured, they always agree.",
                  "00 half the time, 11 half the time, never 01 or 10.")


def ghz(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n, name=f"ghz{n}")
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return Recipe(f"GHZ state on {n} qubits", qc,
                  f"A chain of CNOTs spreads one superposition across all {n} qubits. It is the "
                  "standard stress test for a chip: one bad qubit or gate anywhere breaks it.",
                  f"All zeros or all ones ({'0'*n} / {'1'*n}), 50/50.")


def bernstein_vazirani(secret):
    from qiskit import QuantumCircuit
    n = len(secret)
    qc = QuantumCircuit(n + 1, n, name=f"bv_{secret}")
    qc.x(n); qc.h(range(n + 1))
    for i, b in enumerate(reversed(secret)):
        if b == "1":
            qc.cx(i, n)
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return Recipe(f"Bernstein-Vazirani, secret {secret}", qc,
                  "A hidden bit string is stored in an 'oracle'. Classically you need one query "
                  f"per bit ({n} queries); this circuit finds the whole string with one query.",
                  f"Always {secret}.")


def grover(marked):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import MCMTGate, ZGate
    n = len(marked)
    qc = QuantumCircuit(n, n, name=f"grover_{marked}")
    qc.h(range(n))
    iters = max(1, int(math.floor(math.pi / 4 * math.sqrt(2 ** n))))

    def mcz():
        if n == 1:
            qc.z(0)
        elif n == 2:
            qc.cz(0, 1)
        else:
            qc.append(MCMTGate(ZGate(), n - 1, 1), list(range(n)))

    for _ in range(iters):
        for i, b in enumerate(reversed(marked)):
            if b == "0":
                qc.x(i)
        mcz()
        for i, b in enumerate(reversed(marked)):
            if b == "0":
                qc.x(i)
        qc.h(range(n)); qc.x(range(n)); mcz(); qc.x(range(n)); qc.h(range(n))
    qc.measure(range(n), range(n))
    p = math.sin((2 * iters + 1) * math.asin(1 / math.sqrt(2 ** n))) ** 2
    return Recipe(f"Grover search for {marked}", qc,
                  f"Searches {2**n} possibilities for the marked one using {iters} "
                  f"iteration(s) of 'mark, then reflect about the average'. A classical search "
                  f"needs about {2**(n-1)} tries on average.",
                  f"{marked} with probability {p:.0%} on a perfect machine.")


def qft(n, value=None):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate
    value = (2 ** n // 3) if value is None else value % (2 ** n)
    qc = QuantumCircuit(n, n, name=f"qft{n}")
    for i in range(n):
        if (value >> i) & 1:
            qc.x(i)
    qc.append(QFTGate(n), range(n))
    qc.append(QFTGate(n).inverse(), range(n))
    qc.measure(range(n), range(n))
    return Recipe(f"QFT round trip on {n} qubits", qc,
                  f"Loads the number {value}, applies the quantum Fourier transform (the core of "
                  "Shor's algorithm and phase estimation), then undoes it. A perfect machine gives "
                  "the number back, so any difference is hardware error.",
                  f"Always {value:0{n}b}.")


def teleport():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(3, 1, name="teleport")
    qc.ry(1.1, 0)                       # the state to send
    qc.h(1); qc.cx(1, 2)                # shared entangled pair
    qc.cx(0, 1); qc.h(0)                # sender's Bell measurement ...
    qc.cx(1, 2); qc.cz(0, 2)            # ... with corrections applied coherently
    qc.ry(-1.1, 2)                      # undo the state at the receiver: should give 0
    qc.measure(2, 0)
    return Recipe("Quantum teleportation", qc,
                  "Moves the state of qubit 0 onto qubit 2 using one shared entangled pair. "
                  "Corrections are applied with gates (deferred measurement), so it runs on any "
                  "chip. We rotate the received state back, so a perfect run always reads 0.",
                  "Always 0.")


def random_bits(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n, n, name=f"qrng{n}")
    qc.h(range(n)); qc.measure(range(n), range(n))
    return Recipe(f"{n} quantum random bits", qc,
                  "Each qubit is put in an equal superposition and measured, giving a fair coin "
                  "from quantum mechanics. (Real randomness services also test and whiten the bits.)",
                  "Every bit string equally likely.")


def qaoa_maxcut(n, seed=1):
    """p=1 QAOA for MaxCut on a ring with chords; angles tuned on a noiseless simulation."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    edges = [(i, (i + 1) % n) for i in range(n)]
    for _ in range(max(0, n // 2 - 1)):
        a, b = rng.choice(n, 2, replace=False)
        if (a, b) not in edges and (b, a) not in edges and abs(a - b) % n not in (1, n - 1):
            edges.append((int(a), int(b)))

    def build(g, b, meas=True):
        qc = QuantumCircuit(n, n if meas else 0, name=f"qaoa_maxcut{n}")
        qc.h(range(n))
        for a, c in edges:
            qc.rzz(2 * g, a, c)
        qc.rx(2 * b, range(n))
        if meas:
            qc.measure(range(n), range(n))
        return qc

    from qiskit.quantum_info import Statevector
    def cut(x):
        return sum(((x >> a) & 1) != ((x >> c) & 1) for a, c in edges)
    cuts = np.array([cut(x) for x in range(2 ** n)])
    best = (-1, 0, 0)
    for g in np.linspace(0.05, np.pi / 2, 16):
        for b in np.linspace(0.05, np.pi / 2, 16):
            p = Statevector(build(g, b, meas=False)).probabilities()
            e = float(p @ cuts)
            if e > best[0]:
                best = (e, g, b)
    return Recipe(f"QAOA MaxCut on {n} nodes", build(best[1], best[2]),
                  f"Splits a {n}-node network ({len(edges)} links) into two groups cutting as "
                  "many links as possible, the same maths as many scheduling and routing "
                  "problems. One QAOA layer; angles tuned on a classical simulator.",
                  f"Average cut {best[0]:.2f} links on a perfect machine (best possible "
                  f"{cuts.max()}).",
                  notes=[f"edges: {edges}"])


# ---------------------------------------------------------------- the copilot
_INDUSTRIAL = re.compile(r"\b(route|routes|routing|truck|trucks|delivery|deliveries|logistic|"
                         r"portfolio|schedule|scheduling|supply chain|fleet|vehicle|warehouse|"
                         r"timetable|optimi[sz]e)\b")

HONEST_NOTE = (
    "Honest note: for real problems of this size, today's quantum computers are slower and "
    "less accurate than classical solvers (for example Google OR-Tools, Gurobi, or a good "
    "heuristic). Published quantum routing experiments solve around a dozen locations. Use a "
    "classical solver for the business problem. Below is a teaching-size quantum version of "
    "the same maths, so you can see how it works and how a real chip's noise affects it.")


def generate(prompt: str, allow_llm: bool = True) -> Recipe:
    t = prompt.lower().strip()
    notes = []
    if _INDUSTRIAL.search(t):
        asked = _qubit_count(t, 0)
        n = min(max(asked, 4), 8) if asked else 6
        r = qaoa_maxcut(n)
        r.notes.insert(0, HONEST_NOTE + (f" (You asked about {asked}; this demo uses {n} nodes.)"
                                         if asked and asked != n else ""))
        return r
    if "bell" in t or ("entangle" in t and _qubit_count(t, 2) == 2):
        return bell()
    if any(w in t for w in ("ghz", "cat state", "entangle", "entangled")):
        n = _qubit_count(t, 5)
        return ghz(_limit(n, notes))
    if any(w in t for w in ("bernstein", "vazirani", "secret", "hidden string", "hidden bit")):
        s = _bitstring(t) or "1011"
        return bernstein_vazirani(s[:MAX_QUBITS - 1])
    if any(w in t for w in ("grover", "search", "find the", "marked")):
        s = _bitstring(t) or "101"
        if len(s) > 5:
            notes.append(f"Grover on {len(s)} qubits is far too deep for today's chips; using the last 5 bits.")
            s = s[-5:]
        r = grover(s); r.notes += notes
        return r
    if any(w in t for w in ("fourier", "qft")):
        n = _qubit_count(t, 4)
        r = qft(min(n, 8)); r.notes += notes
        return r
    if "teleport" in t:
        return teleport()
    if "random" in t or "coin" in t or "dice" in t:
        n = _qubit_count(t, 8)
        return random_bits(_limit(n, notes))
    if "maxcut" in t or "max cut" in t or "qaoa" in t:
        n = _qubit_count(t, 6)
        return qaoa_maxcut(min(max(n, 3), 10))
    if allow_llm and os.environ.get("ANTHROPIC_API_KEY"):
        return llm_generate(prompt)
    raise ValueError(
        "I don't have a recipe for that yet. Try: 'Bell pair', 'GHZ state on 12 qubits', "
        "'Bernstein-Vazirani with secret 10110', 'Grover search for 101', 'QFT on 5 qubits', "
        "'teleport a qubit', '16 random bits', 'MaxCut on 6 nodes'. "
        "(Set ANTHROPIC_API_KEY to let a language model handle other prompts.)")


def _limit(n, notes):
    if n > MAX_QUBITS:
        notes.append(f"Capped at {MAX_QUBITS} qubits.")
        return MAX_QUBITS
    return max(2, n)


def llm_generate(prompt: str, model: str = None) -> Recipe:
    """Ask a language model for OpenQASM 2.0; parse it (never execute code). Untested offline."""
    from qiskit import qasm2
    model = model or os.environ.get("SYNDROMICS_LLM_MODEL", "claude-sonnet-5-5")
    system = ("You write quantum circuits. Reply with JSON only: {\"qasm\": <OpenQASM 2.0 "
              "string using qelib1.inc, with measurements>, \"explanation\": <2 sentences for a "
              "student>, \"expected\": <ideal output in one sentence>}. At most 20 qubits. If the "
              "request is an industrial optimisation problem, say in the explanation that "
              "classical solvers are better today.")
    body = json.dumps({"model": model, "max_tokens": 2000, "system": system,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={
        "content-type": "application/json", "anthropic-version": "2023-06-01",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"]})
    with urllib.request.urlopen(req, timeout=60) as r:
        text = "".join(b.get("text", "") for b in json.load(r)["content"])
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0))
    qc = qasm2.loads(d["qasm"], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    if qc.num_qubits > MAX_QUBITS:
        raise ValueError("model produced too many qubits")
    return Recipe("LLM-generated circuit", qc, d.get("explanation", ""), d.get("expected", ""),
                  notes=["Generated by a language model: check it before trusting the result."])
