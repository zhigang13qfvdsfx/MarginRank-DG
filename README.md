# MarginRank-DG

The released Python package, `transsurv-pdac` version 0.5.0, provides the frozen
MarginRank-DG relative-risk score for research in primary or resectable
pancreatic ductal adenocarcinoma.

## Install from GitHub

Use Python 3.12 in a fresh virtual environment. Package metadata pins the
runtime dependencies recorded for the frozen estimator.

```sh
python -m pip install "https://github.com/zhigang13qfvdsfx/MarginRank-DG/archive/refs/heads/main.zip"
transsurv-pdac --version
transsurv-pdac model-info
```

## Score expression data

```sh
transsurv-pdac score expression.tsv risk.tsv
```

Use one sample per row, a sample identifier in the first column and human gene
symbols in the remaining column headers. The scorer computes a continuous
relative-risk score and coverage fields. Optional within-batch percentiles and
tiers require at least 40 samples with non-zero score dispersion. The model
package is for research use.

The installed model file `transsurv_pdac_v4_marginrank_locked.joblib` has SHA-256
`4ae839a9326ffd4fe5e8f6413fc832688b4090bbfb52f9958404b76e28ab0b41`.

See [the analysis guide](analysis/README.md) for installation with analysis
dependencies and the expected layout for rebuilding the reported analyses.
