"""Noise-aware placement and the plain-English copilot."""
import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")
fake = pytest.importorskip("qiskit_ibm_runtime.fake_provider")

from syndromics import copilot, ibm, placement as pl


@pytest.fixture(scope="module")
def backend():
    return fake.FakeKingston()


@pytest.fixture(scope="module")
def lines(backend):
    return pl.cover_with_lines(backend)


def _drift(backend):
    cz = {(142, 143): 0.06}
    for p in ibm._gate_errors(backend):
        if 104 in p:
            cz[p] = 0.04
    return {"readout": {103: 0.15, 105: 0.09}, "sq": {104: 0.01}, "cz": cz}


def test_recipes_match_their_ideal_output():
    cases = {"Bernstein-Vazirani secret 10110": "10110", "QFT on 5 qubits": "01010",
             "teleport a qubit": "0"}
    for prompt, want in cases.items():
        r = copilot.generate(prompt, allow_llm=False)
        ideal = pl.ideal_distribution(r.circuit)
        assert ideal.get(want, 0) > 0.99, prompt
    g = pl.ideal_distribution(copilot.generate("GHZ state on 6 qubits", allow_llm=False).circuit)
    assert set(g) == {"000000", "111111"}
    assert copilot.generate("give me 6 random bits", allow_llm=False).circuit.num_qubits == 6


def test_industrial_prompt_gets_honest_note():
    r = copilot.generate("optimize delivery routes for 50 trucks", allow_llm=False)
    assert "classical solvers" in r.notes[0] and r.circuit.num_qubits <= 8


def test_unknown_prompt_without_llm_is_refused():
    with pytest.raises(ValueError):
        copilot.generate("write me a poem", allow_llm=False)


def test_no_false_alarms_without_drift(backend, lines):
    h = pl.diagnose_chip(backend, pl.SimulatedChip(backend).executor, lines=lines, log=lambda s: None)
    assert h.flagged == {}


def test_finds_planted_drift_and_labels_readout(backend, lines):
    chip = pl.SimulatedChip(backend, _drift(backend))
    h = pl.diagnose_chip(backend, chip.executor, lines=lines, log=lambda s: None)
    for k in ("q103", "q105", "q103-q104", "q104-q105", "q142-q143"):
        assert k in h.flagged, k
    assert h.flagged["q103"].startswith("readout")
    assert abs(h.readout[103] - 0.15) < 0.01
    assert pl.ChipHealth.from_json(h.to_json()).flagged == h.flagged


def test_placement_beats_default_under_drift(backend, lines):
    chip = pl.SimulatedChip(backend, _drift(backend))
    h = pl.diagnose_chip(backend, chip.executor, lines=lines, log=lambda s: None)
    r = copilot.generate("Grover search for 101", allow_llm=False)
    ideal = pl.ideal_distribution(r.circuit)
    d = pl.place(r.circuit, backend, None)
    s = pl.place(r.circuit, backend, h)
    f_d, f_s = pl.fidelity(chip.run(d.circuit), ideal), pl.fidelity(chip.run(s.circuit), ideal)
    assert f_s > f_d + 0.2
    assert not s.touched_flags
