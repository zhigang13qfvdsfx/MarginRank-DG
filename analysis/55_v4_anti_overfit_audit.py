from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"
N_PERMUTATIONS = 2000
SEED = 550400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(concordance_index_censored(event.astype(bool), time.astype(float), risk.astype(float))[0])


def main() -> None:
    inner_rankings = pd.read_csv(RESULTS / "36_v4_marginrank_inner_rankings.csv")
    selected = inner_rankings.loc[inner_rankings["selected"].astype(bool), ["outer_test", "candidate", "mean_uno"]].rename(
        columns={"mean_uno": "inner_mean_uno"}
    )
    outer = pd.read_csv(RESULTS / "36_v4_marginrank_outer_metrics.csv")
    outer = outer[["cohort", "selected_candidate", "uno_c"]].rename(
        columns={"cohort": "outer_test", "selected_candidate": "candidate", "uno_c": "outer_uno"}
    )
    nested = selected.merge(outer, on=["outer_test", "candidate"], validate="one_to_one")
    nested["inner_minus_outer"] = nested["inner_mean_uno"] - nested["outer_uno"]
    nested = nested.sort_values("outer_test")
    nested.to_csv(RESULTS / "55_v4_nested_inner_outer_gap.csv", index=False)

    paca = pd.read_csv(RESULTS / "41_v4_paca_ca_predictions.csv")
    time = paca["OS_TIME"].to_numpy(float)
    event = paca["OS_EVENT"].to_numpy(bool)
    risk = paca["v4_risk"].to_numpy(float)
    observed = harrell_c(time, event, risk)
    rng = np.random.default_rng(SEED)
    null = np.asarray([harrell_c(time, event, rng.permutation(risk)) for _ in range(N_PERMUTATIONS)], dtype=float)
    pd.DataFrame({"permutation": np.arange(N_PERMUTATIONS), "harrell_c": null}).to_csv(
        RESULTS / "55_v4_paca_ca_permutation_null.csv", index=False
    )
    p_perm = float((1 + np.sum(null >= observed)) / (N_PERMUTATIONS + 1))

    audit = {
        "status": "POSTFREEZE_ANTI_OVERFIT_AUDIT_COMPLETE",
        "interpretation": "descriptive diagnostic; it did not change the frozen model",
        "nested_selection": {
            "outer_domains": int(len(nested)),
            "mean_inner_uno": float(nested["inner_mean_uno"].mean()),
            "mean_outer_uno": float(nested["outer_uno"].mean()),
            "mean_inner_minus_outer": float(nested["inner_minus_outer"].mean()),
            "median_inner_minus_outer": float(nested["inner_minus_outer"].median()),
            "maximum_absolute_gap": float(nested["inner_minus_outer"].abs().max()),
            "outer_domains_above_chance": int((nested["outer_uno"] > 0.5).sum()),
            "outer_domain_minimum": float(nested["outer_uno"].min()),
        },
        "paca_ca_fixed_risk_permutation": {
            "n": int(len(paca)),
            "observed_harrell_c": observed,
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_q025": float(np.quantile(null, 0.025)),
            "null_q975": float(np.quantile(null, 0.975)),
            "permutations": N_PERMUTATIONS,
            "seed": SEED,
            "p_ge_observed": p_perm,
        },
        "frozen_model_unchanged": True,
        "model_sha256": sha256(MODEL),
        "descriptive_conditions": {
            "nested_mean_absolute_gap_le_0_10": bool(nested["inner_minus_outer"].abs().mean() <= 0.10),
            "nested_outer_mean_uno_ge_0_60": bool(nested["outer_uno"].mean() >= 0.60),
            "all_outer_domains_above_chance": bool((nested["outer_uno"] > 0.5).all()),
            "paca_ca_fixed_risk_above_permutation_null": bool(observed > np.quantile(null, 0.975)),
        },
    }
    audit["descriptive_status"] = "PASS" if all(audit["descriptive_conditions"].values()) else "REVIEW"
    (RESULTS / "55_v4_anti_overfit_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
