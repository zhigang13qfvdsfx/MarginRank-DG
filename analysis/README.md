# Analysis scripts

The numbered files preserve the final MarginRank-DG v4 workflow:

- `35` prepares the expanded development domains.
- `36` runs nested leave-one-domain-out model selection.
- `37` evaluates the fixed selected configuration.
- `38` runs selected-gene missingness stress tests.
- `39` freezes the final v4 model and specification.
- `40` audits fresh-cohort eligibility before risk scoring.
- `41` performs the one-time PACA-CA validation.
- `42` runs post-freeze ablation and invariance checks.
- `43` records the GSE77435 cross-platform technical attempt.
- `44` runs the locked GSE77435 extreme-survival stress test.
- `45-46` provide single-cell and ranked biological context.
- `47-48` prepare and evaluate the PRINCE exploratory boundary test.
- `49` verifies the packaged software release.
- `50-53` generate and audit manuscript figures.
- `55` performs the post-freeze anti-overfitting audit.

## Expected local layout

Scripts resolve paths from the repository root and expect:

```text
data/processed/     locally prepared cohort matrices (not distributed)
config/             frozen specifications and governance records
results/            generated and committed aggregate result tables
src/transsurv/      installable scoring code and frozen v4 model
models/predecessors frozen comparator used by prespecified analyses
```

## Installation

For scoring, install from the public GitHub archive using the [main README](../README.md#install-from-github).
To run the analysis scripts locally, clone this repository and run the following
from its root in a Python 3.12 virtual environment:

```bash
python -m pip install -e ".[analysis]"
```

Analyses expect the prepared cohort inputs described above. The scoring package
includes the frozen model and can be installed with no cohort files.

The one-time external scripts are preserved for auditability. Do not rerun them
as a route to model tuning, cutoff selection, recalibration, or endpoint
switching. The committed model and external gate reports are immutable release
artifacts.

