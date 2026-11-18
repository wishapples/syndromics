# Running Syndromics on real hardware data

## A. Google Willow (no account needed)

1. Go to https://doi.org/10.5281/zenodo.13273331 and download
   `google_105Q_surface_code_d3_d5_d7.zip` (11.6 GB). The record also has larger files;
   you only need this one.
2. Unzip it, for example into `~/data/willow/`. Check that a path like
   `~/data/willow/.../d3_at_q6_7/Z/r10/detection_events.b8` exists.
3. Install and run:

   ```bash
   pip install "syndromics[google] @ git+https://github.com/syndromics/syndromics.git"
   syndromics google ~/data/willow --rounds 10 --limit 4      # quick first look
   syndromics google ~/data/willow --rounds 10                # everything
   ```

4. Results go to `willow_results/`:
   * `summary.json`: per-experiment outliers, held-out decoding (nominal vs calibrated) and
     repeatability between independent experiments
   * `components.csv`: every fitted component (qubit, readout, gate, crosstalk) per experiment

What counts as success: the calibrated model decodes held-out shots better than Google's
nominal SI1000 model in most experiments, and qubits flagged in one experiment are flagged again
in another one that covers them. Note the numbers either way; an honest negative result is still
worth publishing.

Expect roughly 10-60 seconds per experiment on a laptop for distance 3 at ~10 rounds, and longer
for distance 5 and 7. Add `--no-crosstalk` to go faster.

## B. IBM Quantum (free Open Plan)

1. Create a free account at https://quantum.cloud.ibm.com and copy your API key.
2. Save it once:

   ```bash
   pip install "syndromics[ibm] @ git+https://github.com/syndromics/syndromics.git"
   python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_cloud', token='YOUR_KEY', set_as_default=True)"
   ```

   (If your account uses a different channel or needs an instance CRN, follow IBM's
   current setup guide; this changes occasionally.)

3. Rehearse on a local simulation of the device first. This costs nothing:

   ```bash
   syndromics ibm --backend ibm_kingston --fake
   ```

4. Run on hardware:

   ```bash
   syndromics ibm --backend ibm_kingston --n-data 5 --rounds 10 --shots 60000
   ```

   Check the QPU time used on your IBM dashboard after the first run; the free plan allows
   10 minutes per 28 days. If time is tight, use `--shots 20000`.

5. Results go to `ibm_results/`: raw measurements (`measurements.npy`, so you never need to
   re-run), and `comparison.json` with Syndromics' per-qubit estimates against IBM's published
   calibration.

What counts as success: a high correlation between Syndromics' readout estimates and IBM's
published readout errors (the simulation gives r = 0.99). Real hardware will be lower because
calibration data is hours old and real noise includes leakage and drift. Disagreements between
the two are interesting in themselves: they may be qubits that changed since IBM's last calibration.
