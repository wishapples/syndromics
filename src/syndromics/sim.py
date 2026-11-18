"""
Simulated quantum chip with hidden, device-specific noise.

The standard decoder (PyMatching / MWPM) is given only the *nominal* noise model
(uniform error rate p on every qubit) - which is what a hardware team uses before
detailed characterisation. The real chip additionally has:
  * a few "bad" qubits with much higher error rates (fabrication defects, TLS defects)
  * correlated crosstalk: two neighbouring data qubits flipping together
  * a biased readout on some measurement qubits
Diagnostics never see this model; they must find the defects from syndrome data alone.
"""
import numpy as np
import stim


def nominal_circuit(d: int, rounds: int, p: float, basis: str = "z") -> stim.Circuit:
    """Uniform circuit-level noise surface-code memory experiment (what the vendor assumes)."""
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis.lower()}",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def _qubit_roles(circ: stim.Circuit):
    coords = circ.get_final_qubit_coordinates()
    data = sorted(q for q, (x, y) in coords.items() if int(x) % 2 == 1 and int(y) % 2 == 1)
    meas = sorted(q for q in coords if q not in data)
    return coords, data, meas


def real_chip_circuit(d: int, rounds: int, p: float, seed: int = 7, basis: str = "z",
                      n_bad: int = 2, bad_factor: float = 6.0,
                      n_xtalk: int = 3, xtalk_p_factor: float = 3.0,
                      n_bad_readout: int = 2, readout_factor: float = 5.0):
    """Same circuit plus hidden device-specific noise. Returns (circuit, description)."""
    base = nominal_circuit(d, rounds, p, basis).flattened()
    coords, data, meas = _qubit_roles(base)
    rng = np.random.default_rng(seed)

    bad = sorted(rng.choice(data, size=n_bad, replace=False).tolist())
    # neighbouring data-qubit pairs (diagonal distance 2 on the lattice)
    pairs = [(a, b) for i, a in enumerate(data) for b in data[i + 1:]
             if abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) == 2]
    xtalk = [pairs[i] for i in rng.choice(len(pairs), size=min(n_xtalk, len(pairs)), replace=False)]
    bad_ro = sorted(rng.choice(meas, size=n_bad_readout, replace=False).tolist())
    data_set = set(data)

    out = stim.Circuit()
    for inst in base:
        name = inst.name
        tg = [t.value for t in inst.targets_copy()]
        if name == "DEPOLARIZE1" and set(tg) == data_set:
            # start of a QEC round: inject hidden data-qubit noise
            out.append(inst)
            out.append("DEPOLARIZE1", bad, (p * (bad_factor - 1),))
            for a, b in xtalk:
                out.append("CORRELATED_ERROR", [stim.target_x(a), stim.target_x(b)], (p * xtalk_p_factor,))
            continue
        if name == "X_ERROR" and inst.gate_args_copy() and set(tg) <= set(meas) and len(tg) > 0:
            # pre-measurement flips on ancillas: make some readouts worse
            out.append(inst)
            hit = [q for q in tg if q in bad_ro]
            if hit:
                out.append("X_ERROR", hit, (p * (readout_factor - 1),))
            continue
        out.append(inst)

    desc = {"bad_data_qubits": bad, "crosstalk_pairs": xtalk, "bad_readout_qubits": bad_ro,
            "p": p, "d": d, "rounds": rounds}
    return out, desc


def sample(circ: stim.Circuit, shots: int, seed: int | None = None):
    sampler = circ.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    return det.astype(np.uint8), obs[:, 0].astype(np.uint8)
