from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transsurv.metrics import harrell_c, uno_c


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
LOCK = CONFIG / "V4_MARGINRANK_SELECTION.lock"


def load_v4_module():
    path = Path(__file__).with_name("36_v4_marginrank_nested.py")
    spec = importlib.util.spec_from_file_location("v4_marginrank_core", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mean_jaccard(sets: list[set[str]]) -> float:
    values = []
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            values.append(len(left & right) / len(left | right))
    return float(np.mean(values))


def main() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "LOCKED_AFTER_DYNAMIC_NESTED_GATE_PASS_BEFORE_FIXED_CONFIGURATION_SELECTION":
        raise RuntimeError("Unexpected fixed-selection lock")
    dynamic = json.loads((RESULTS / "36_v4_marginrank_gate.json").read_text(encoding="utf-8"))
    if dynamic["gate"] != "PASS":
        raise RuntimeError("Dynamic nested gate did not pass")
    v4 = load_v4_module()
    domains, genes = v4.load_domains()
    z_table = v4.precompute_domain_z(domains, genes)
    rows = []
    for fold, test_name in enumerate(v4.PRIMARY, start=1):
        train_names = [name for name in v4.PRIMARY if name != test_name]
        train_time = np.concatenate([domains[name].time for name in train_names])
        train_event = np.concatenate([domains[name].event for name in train_names])
        cache: dict = {}
        for candidate in v4.CANDIDATES:
            try:
                risk, _ = v4.fit_candidate(
                    candidate, train_names, test_name, domains, z_table, cache
                )
                h = harrell_c(domains[test_name].time, domains[test_name].event, risk)
                u = uno_c(
                    train_time,
                    train_event,
                    domains[test_name].time,
                    domains[test_name].event,
                    risk,
                )
                error = ""
            except Exception as exc:
                h, u, error = np.nan, np.nan, repr(exc)
            rows.append(
                {
                    "cohort": test_name,
                    "candidate": candidate.name,
                    "harrell_c": h,
                    "uno_c": u,
                    "error": error,
                }
            )
        print(f"[{fold}/{len(v4.PRIMARY)}] fixed-selection fold complete: {test_name}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "37_v4_fixed_candidate_loso_metrics.csv", index=False)
    ranking = v4.selection_ranking(
        [
            {
                "candidate": row.candidate,
                "harrell_c": row.harrell_c,
                "uno_c": row.uno_c,
            }
            for row in metrics.itertuples(index=False)
        ],
        len(v4.PRIMARY),
    )
    selected_name = str(ranking.iloc[0]["candidate"])
    ranking["selected"] = ranking["candidate"] == selected_name
    ranking.to_csv(RESULTS / "37_v4_fixed_candidate_ranking.csv", index=False)
    candidate_by_name = {candidate.name: candidate for candidate in v4.CANDIDATES}
    selected = candidate_by_name[selected_name]

    selected_rows = []
    prediction_rows = []
    gene_sets = []
    continuity_pass = True
    for test_name in v4.PRIMARY:
        train_names = [name for name in v4.PRIMARY if name != test_name]
        risk, definition = v4.fit_candidate(
            selected, train_names, test_name, domains, z_table, {}
        )
        gene_set = v4.gene_set_from_definition(definition)
        gene_sets.append(gene_set)
        distinct = int(pd.Series(np.round(risk, 12)).nunique())
        required_distinct = min(20, int(np.ceil(len(risk) / 3)))
        continuity_pass &= distinct >= required_distinct
        row = metrics[
            (metrics["cohort"] == test_name) & (metrics["candidate"] == selected_name)
        ].iloc[0]
        selected_rows.append(
            {
                "cohort": test_name,
                "harrell_c": float(row["harrell_c"]),
                "uno_c": float(row["uno_c"]),
                "n": int(len(risk)),
                "events": int(domains[test_name].event.sum()),
                "distinct_risks": distinct,
                "required_distinct_risks": required_distinct,
                "continuity_pass": distinct >= required_distinct,
                "selected_genes": len(gene_set),
                "selected_margins": len(definition["pairs"]),
            }
        )
        prediction_rows.extend(
            {
                "sample_id": sample_id,
                "cohort": test_name,
                "time": float(time),
                "event": int(event),
                "risk": float(value),
            }
            for sample_id, time, event, value in zip(
                domains[test_name].x.index,
                domains[test_name].time,
                domains[test_name].event,
                risk,
            )
        )
    selected_frame = pd.DataFrame(selected_rows)
    baseline = pd.read_csv(RESULTS / "30_v3_2_selected_loco.csv")[
        ["cohort", "uno_c"]
    ].rename(columns={"uno_c": "v2_uno_c"})
    selected_frame = selected_frame.merge(baseline, on="cohort", how="left")
    selected_frame["delta_uno_v2"] = selected_frame["uno_c"] - selected_frame["v2_uno_c"]
    selected_frame.to_csv(RESULTS / "37_v4_selected_loso_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        RESULTS / "37_v4_selected_loso_predictions.csv", index=False
    )

    gate = lock["fixed_configuration_gate"]
    mean_uno = float(selected_frame["uno_c"].mean())
    median_uno = float(selected_frame["uno_c"].median())
    min_uno = float(selected_frame["uno_c"].min())
    positive = int((selected_frame["uno_c"] > 0.5).sum())
    gain = float(selected_frame["delta_uno_v2"].mean())
    jaccard = mean_jaccard(gene_sets)
    conditions = {
        "mean_uno": mean_uno >= gate["mean_uno_c"],
        "median_uno": median_uno >= gate["median_uno_c"],
        "minimum_uno": min_uno >= gate["minimum_uno_c"],
        "positive_domains": positive >= gate["minimum_positive_domains"],
        "mean_gain_over_v2": gain >= gate["mean_uno_gain_over_locked_v2"],
        "feature_jaccard": jaccard >= gate["leave_one_domain_feature_jaccard"],
        "risk_continuity": bool(continuity_pass),
    }
    summary = {
        "status": "V4_FIXED_CONFIGURATION_SELECTION_COMPLETE",
        "gate": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "selected_candidate": selected_name,
        "mean_harrell_c": float(selected_frame["harrell_c"].mean()),
        "mean_uno_c": mean_uno,
        "median_uno_c": median_uno,
        "minimum_uno_c": min_uno,
        "positive_domains": positive,
        "mean_uno_gain_over_v2": gain,
        "leave_one_domain_feature_jaccard": jaccard,
        "continuity_pass": bool(continuity_pass),
        "gse224564_used": False,
        "fresh_candidate_scores_generated": False,
    }
    (RESULTS / "37_v4_fixed_selection_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (CONFIG / "v4_selected_configuration.json").write_text(
        json.dumps(
            {
                "status": "SELECTED_NOT_YET_FROZEN",
                "candidate": selected_name,
                "feature": selected.feature.__dict__,
                "learner": selected.learner,
                "alpha": selected.alpha,
                "include_auxiliary": selected.include_auxiliary,
                "selection_gate": summary["gate"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
