from __future__ import annotations

import json
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from transsurv.learners import PooledSurvivalLearner
from transsurv.metrics import harrell_c, uno_c
from transsurv.model import DomainData
from transsurv.screening import _cox_score_z


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
MIGRATION = CONFIG / "V4_DEVELOPMENT_MIGRATION.lock"

PRIMARY = [
    "TCGA_PAAD",
    "CPTAC_PDACC",
    "GSE62452",
    "GSE85916",
    "E-MTAB-6134",
    "GSE183795",
    "GSE57495",
    "GSE79668",
    "PACA-AU",
    "GSE71729",
]
AUXILIARY = "GSE205154_MET"
CLINICAL = {
    "TCGA_PAAD": "tcga_paad_clinical.csv",
    "CPTAC_PDACC": "cptac_pdacc_clinical.csv",
    "GSE62452": "gse62452_clinical.csv",
    "GSE85916": "gse85916_clinical.csv",
    "E-MTAB-6134": "emtab6134_clinical.csv",
    "GSE183795": "gse183795_v3_development_clinical.csv",
    "GSE57495": "gse57495_external_clinical.csv",
    "GSE79668": "gse79668_external_clinical.csv",
    "PACA-AU": "paca_au_v2_confirmation_clinical.csv",
    "GSE71729": "gse71729_v2_confirmation_clinical.csv",
    AUXILIARY: "gse205154_met_v4_clinical.csv",
}
HOUSEKEEPING = [
    "ACTB",
    "B2M",
    "GAPDH",
    "GUSB",
    "HPRT1",
    "PGK1",
    "PPIA",
    "RPL13A",
    "RPLP0",
    "TBP",
]


@dataclass(frozen=True)
class FeatureSpec:
    kind: str
    n_genes: int
    n_pairs: int = 0

    @property
    def name(self) -> str:
        return f"{self.kind}_g{self.n_genes}_p{self.n_pairs}"


@dataclass(frozen=True)
class Candidate:
    feature: FeatureSpec
    learner: str
    alpha: float
    include_auxiliary: bool

    @property
    def name(self) -> str:
        aux = "aux" if self.include_auxiliary else "primary"
        return f"{self.feature.name}__{self.learner}_a{self.alpha:g}__{aux}"


FEATURE_SPECS = [
    FeatureSpec("rank", 20),
    FeatureSpec("rank", 40),
    FeatureSpec("rank", 80),
    FeatureSpec("anchor", 20),
    FeatureSpec("anchor", 40),
    FeatureSpec("anchor", 80),
    FeatureSpec("contrast", 20, 20),
    FeatureSpec("contrast", 40, 40),
    FeatureSpec("hybrid", 20, 20),
    FeatureSpec("hybrid", 40, 40),
]
CANDIDATES = [
    Candidate(feature, "pooled_ridge", alpha, include_auxiliary=False)
    for feature in FEATURE_SPECS
    for alpha in (0.1, 1.0, 10.0)
]
CANDIDATES += [
    Candidate(feature, "pooled_ridge", 1.0, include_auxiliary=True)
    for feature in FEATURE_SPECS
]
CANDIDATES += [
    Candidate(feature, "expert_median", alpha, include_auxiliary=aux)
    for feature in [
        FeatureSpec("rank", 40),
        FeatureSpec("anchor", 40),
        FeatureSpec("contrast", 40, 40),
        FeatureSpec("hybrid", 40, 40),
    ]
    for alpha in (1.0, 10.0)
    for aux in (False, True)
]


def rank_path(domain: str) -> Path:
    stem = domain.lower().replace("-", "_")
    return DATA / f"{stem}_v4_single_sample_rank.parquet"


def ids_from_clinical(frame: pd.DataFrame) -> list[str]:
    for column in ("SAMPLE_ID", "PATIENT_ID"):
        if column in frame:
            return frame[column].astype(str).tolist()
    raise RuntimeError("Clinical data lack an identifier")


def load_domains() -> tuple[dict[str, DomainData], list[str]]:
    domains = {}
    genes = None
    for name in [*PRIMARY, AUXILIARY]:
        clinical = pd.read_csv(DATA / CLINICAL[name])
        ids = ids_from_clinical(clinical)
        rank = pd.read_parquet(rank_path(name))
        rank.index = rank.index.astype(str)
        rank = rank.reindex(ids)
        if genes is None:
            genes = list(rank.columns.astype(str))
        elif list(rank.columns.astype(str)) != genes:
            raise RuntimeError(f"v4 gene order mismatch in {name}")
        domains[name] = DomainData(
            x=rank,
            time=pd.to_numeric(clinical["OS_TIME"], errors="coerce").to_numpy(float),
            event=pd.to_numeric(clinical["OS_EVENT"], errors="coerce").to_numpy(bool),
        )
    assert genes is not None
    return domains, genes


def precompute_domain_z(domains: dict[str, DomainData], genes: list[str]) -> pd.DataFrame:
    rows = []
    for name, domain in domains.items():
        frame = domain.x[genes].astype(float)
        center = frame.mean(axis=0)
        scale = frame.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        standardized = ((frame - center) / scale).fillna(0.0).to_numpy(float)
        rows.append(
            pd.Series(
                _cox_score_z(standardized, domain.time, domain.event),
                index=genes,
                name=name,
            )
        )
    return pd.DataFrame(rows)


def screening_table(train_names: list[str], z_table: pd.DataFrame) -> pd.DataFrame:
    matrix = z_table.loc[train_names].to_numpy(float)
    meta_z = matrix.sum(axis=0) / np.sqrt(len(train_names))
    dispersion = matrix.std(axis=0, ddof=0)
    positive_fraction = (matrix > 0).mean(axis=0)
    negative_fraction = (matrix < 0).mean(axis=0)
    directional_consistency = np.maximum(positive_fraction, negative_fraction)
    stability = (
        np.abs(meta_z)
        * (0.25 + 0.75 * directional_consistency)
        / (1.0 + dispersion)
    )
    return pd.DataFrame(
        {
            "gene": z_table.columns,
            "meta_z": meta_z,
            "dispersion": dispersion,
            "directional_consistency": directional_consistency,
            "stability": stability,
        }
    ).set_index("gene")


def balanced_genes(table: pd.DataFrame, n_genes: int) -> list[str]:
    half = n_genes // 2
    positive = (
        table[table["meta_z"] > 0]
        .sort_values(["stability", "directional_consistency"], ascending=False)
        .head(half)
        .index.tolist()
    )
    negative = (
        table[table["meta_z"] < 0]
        .sort_values(["stability", "directional_consistency"], ascending=False)
        .head(half)
        .index.tolist()
    )
    selected = positive + negative
    if len(selected) < n_genes:
        fill = table.drop(index=selected).sort_values("stability", ascending=False)
        selected += fill.head(n_genes - len(selected)).index.tolist()
    return selected


def contrast_pairs(
    table: pd.DataFrame,
    genes: list[str],
    n_pairs: int,
    train_names: list[str],
    domains: dict[str, DomainData],
) -> list[tuple[str, str]]:
    positive = [gene for gene in genes if table.loc[gene, "meta_z"] > 0]
    negative = [gene for gene in genes if table.loc[gene, "meta_z"] < 0]
    candidates = []
    for left in positive:
        for right in negative:
            domain_means = []
            domain_sds = []
            for name in train_names:
                values = (
                    domains[name].x[left].to_numpy(float)
                    - domains[name].x[right].to_numpy(float)
                )
                domain_means.append(float(np.mean(values)))
                domain_sds.append(float(np.std(values, ddof=0)))
            drift = float(np.std(domain_means, ddof=0))
            informativeness = float(np.median(domain_sds))
            score = (
                (table.loc[left, "stability"] + table.loc[right, "stability"])
                * informativeness
                / (1.0 + 5.0 * drift)
            )
            candidates.append((float(score), left, right))
    candidates.sort(reverse=True)
    selected = []
    endpoint_count: Counter[str] = Counter()
    endpoint_cap = max(3, int(np.ceil(n_pairs / max(len(genes), 1) * 4)))
    for _, left, right in candidates:
        if endpoint_count[left] >= endpoint_cap or endpoint_count[right] >= endpoint_cap:
            continue
        selected.append((left, right))
        endpoint_count[left] += 1
        endpoint_count[right] += 1
        if len(selected) == n_pairs:
            break
    if len(selected) < n_pairs:
        for _, left, right in candidates:
            if (left, right) in selected:
                continue
            selected.append((left, right))
            if len(selected) == n_pairs:
                break
    return selected


def feature_definition(
    train_names: list[str],
    spec: FeatureSpec,
    domains: dict[str, DomainData],
    z_table: pd.DataFrame,
) -> dict:
    table = screening_table(train_names, z_table)
    genes = balanced_genes(table, spec.n_genes)
    anchors = [gene for gene in HOUSEKEEPING if gene in table.index]
    if len(anchors) < 6:
        raise RuntimeError("Fewer than six locked housekeeping anchors in v4 universe")
    pairs = []
    if spec.kind in {"contrast", "hybrid"}:
        pairs = contrast_pairs(table, genes, spec.n_pairs, train_names, domains)
    return {"kind": spec.kind, "genes": genes, "anchors": anchors, "pairs": pairs}


def transform_features(rank: pd.DataFrame, definition: dict) -> pd.DataFrame:
    output = pd.DataFrame(index=rank.index)
    kind = definition["kind"]
    genes = definition["genes"]
    if kind == "rank":
        for gene in genes:
            output[f"rank::{gene}"] = rank[gene].to_numpy(float)
    if kind in {"anchor", "hybrid"}:
        anchor_center = rank[definition["anchors"]].median(axis=1).to_numpy(float)
        for gene in genes:
            output[f"anchor::{gene}"] = rank[gene].to_numpy(float) - anchor_center
    if kind in {"contrast", "hybrid"}:
        for left, right in definition["pairs"]:
            output[f"margin::{left}|{right}"] = (
                rank[left].to_numpy(float) - rank[right].to_numpy(float)
            )
    return output


class ExpertMedian:
    def __init__(self, alpha: float):
        self.alpha = alpha

    def fit(self, domains: dict[str, DomainData]) -> "ExpertMedian":
        self.experts = []
        for domain in domains.values():
            model = PooledSurvivalLearner("coxph", {"alpha": self.alpha}).fit(
                {"expert": domain}
            )
            training_prediction = model.predict(domain.x)
            location = float(np.mean(training_prediction))
            scale = float(np.std(training_prediction, ddof=0)) or 1.0
            self.experts.append((model, location, scale))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        values = np.vstack(
            [(model.predict(x) - location) / scale for model, location, scale in self.experts]
        )
        return np.median(values, axis=0)


def fit_candidate(
    candidate: Candidate,
    train_names: list[str],
    test_name: str,
    domains: dict[str, DomainData],
    z_table: pd.DataFrame,
    split_cache: dict | None = None,
) -> tuple[np.ndarray, dict]:
    feature_train_names = list(train_names)
    if candidate.include_auxiliary:
        feature_train_names.append(AUXILIARY)
    cache_key = (tuple(feature_train_names), candidate.feature)
    if split_cache is None:
        split_cache = {}
    if cache_key not in split_cache:
        definition = feature_definition(
            feature_train_names, candidate.feature, domains, z_table
        )
        split_cache[cache_key] = {"definition": definition, "domains": {}}
    entry = split_cache[cache_key]
    definition = entry["definition"]
    transformed = entry["domains"]
    for name in [*feature_train_names, test_name]:
        if name not in transformed:
            transformed[name] = DomainData(
                x=transform_features(domains[name].x, definition),
                time=domains[name].time,
                event=domains[name].event,
            )
    train = {name: transformed[name] for name in feature_train_names}
    test = transformed[test_name]
    if candidate.learner == "pooled_ridge":
        model = PooledSurvivalLearner(
            "coxph", {"alpha": candidate.alpha}
        ).fit(train)
    elif candidate.learner == "expert_median":
        model = ExpertMedian(candidate.alpha).fit(train)
    else:
        raise ValueError(candidate.learner)
    return model.predict(test.x), definition


def selection_ranking(rows: list[dict], expected_folds: int) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    ranking = (
        frame.groupby("candidate")
        .agg(
            mean_uno=("uno_c", "mean"),
            worst_uno=("uno_c", "min"),
            sd_uno=("uno_c", "std"),
            mean_harrell=("harrell_c", "mean"),
            valid_folds=("uno_c", "count"),
        )
        .reset_index()
    )
    ranking = ranking[ranking["valid_folds"] == expected_folds].copy()
    ranking["selection_score"] = (
        ranking["mean_uno"]
        + 0.5 * (ranking["worst_uno"] - 0.5)
        - 0.25 * ranking["sd_uno"].fillna(0.0)
    )
    return ranking.sort_values(
        ["selection_score", "mean_uno", "candidate"],
        ascending=[False, False, True],
    )


def gene_set_from_definition(definition: dict) -> set[str]:
    genes = set(definition["genes"])
    for left, right in definition["pairs"]:
        genes.update([left, right])
    return genes


def mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    values = []
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            values.append(len(left & right) / len(left | right))
    return float(np.mean(values))


def main() -> None:
    warnings.filterwarnings("ignore")
    migration = json.loads(MIGRATION.read_text(encoding="utf-8"))
    if migration["status"] != "LOCKED_BEFORE_V4_MODEL_DEVELOPMENT":
        raise RuntimeError("Unexpected v4 migration lock")
    domains, genes = load_domains()
    if "GSE224564" in domains:
        raise RuntimeError("Consumed v3.3 confirmation leaked into v4")
    z_table = precompute_domain_z(domains, genes)
    candidate_by_name = {candidate.name: candidate for candidate in CANDIDATES}
    all_inner_rows = []
    all_rankings = []
    outer_rows = []
    prediction_rows = []
    selected_feature_rows = []
    selected_gene_sets = []

    for outer_index, outer_test in enumerate(PRIMARY, start=1):
        outer_train = [name for name in PRIMARY if name != outer_test]
        print(f"[{outer_index}/{len(PRIMARY)}] outer domain {outer_test}", flush=True)
        inner_rows = []
        for inner_index, inner_test in enumerate(outer_train, start=1):
            inner_train = [name for name in outer_train if name != inner_test]
            train_time = np.concatenate([domains[name].time for name in inner_train])
            train_event = np.concatenate([domains[name].event for name in inner_train])
            split_cache: dict = {}
            for candidate in CANDIDATES:
                try:
                    risk, _ = fit_candidate(
                        candidate,
                        inner_train,
                        inner_test,
                        domains,
                        z_table,
                        split_cache,
                    )
                    h = harrell_c(
                        domains[inner_test].time, domains[inner_test].event, risk
                    )
                    u = uno_c(
                        train_time,
                        train_event,
                        domains[inner_test].time,
                        domains[inner_test].event,
                        risk,
                    )
                    error = ""
                except Exception as exc:
                    h, u, error = np.nan, np.nan, repr(exc)
                inner_rows.append(
                    {
                        "outer_test": outer_test,
                        "inner_test": inner_test,
                        "candidate": candidate.name,
                        "harrell_c": h,
                        "uno_c": u,
                        "error": error,
                    }
                )
            print(
                f"  inner {inner_index}/{len(outer_train)} complete: {inner_test}",
                flush=True,
            )
        ranking = selection_ranking(inner_rows, len(outer_train))
        if ranking.empty:
            raise RuntimeError(f"No valid v4 candidate for outer domain {outer_test}")
        selected_name = str(ranking.iloc[0]["candidate"])
        selected = candidate_by_name[selected_name]
        outer_cache: dict = {}
        risk, definition = fit_candidate(
            selected, outer_train, outer_test, domains, z_table, outer_cache
        )
        train_time = np.concatenate([domains[name].time for name in outer_train])
        train_event = np.concatenate([domains[name].event for name in outer_train])
        outer_h = harrell_c(domains[outer_test].time, domains[outer_test].event, risk)
        outer_u = uno_c(
            train_time,
            train_event,
            domains[outer_test].time,
            domains[outer_test].event,
            risk,
        )
        outer_rows.append(
            {
                "cohort": outer_test,
                "selected_candidate": selected_name,
                "harrell_c": outer_h,
                "uno_c": outer_u,
                "n": len(domains[outer_test].time),
                "events": int(domains[outer_test].event.sum()),
                "n_features": len(transform_features(domains[outer_test].x, definition).columns),
            }
        )
        prediction_rows.extend(
            {
                "sample_id": sample_id,
                "cohort": outer_test,
                "time": float(time),
                "event": int(event),
                "risk": float(value),
                "selected_candidate": selected_name,
            }
            for sample_id, time, event, value in zip(
                domains[outer_test].x.index,
                domains[outer_test].time,
                domains[outer_test].event,
                risk,
            )
        )
        selected_gene_sets.append(gene_set_from_definition(definition))
        selected_feature_rows.extend(
            {
                "outer_test": outer_test,
                "selected_candidate": selected_name,
                "feature_type": "gene",
                "feature": gene,
            }
            for gene in definition["genes"]
        )
        selected_feature_rows.extend(
            {
                "outer_test": outer_test,
                "selected_candidate": selected_name,
                "feature_type": "margin",
                "feature": f"{left}|{right}",
            }
            for left, right in definition["pairs"]
        )
        ranking.insert(0, "outer_test", outer_test)
        ranking["selected"] = ranking["candidate"] == selected_name
        all_rankings.append(ranking)
        all_inner_rows.extend(inner_rows)
        pd.DataFrame(all_inner_rows).to_csv(
            RESULTS / "36_v4_marginrank_inner_progress.csv", index=False
        )

    outer = pd.DataFrame(outer_rows)
    baseline = pd.read_csv(RESULTS / "30_v3_2_selected_loco.csv")
    baseline = baseline[["cohort", "harrell_c", "uno_c"]].rename(
        columns={"harrell_c": "v2_harrell_c", "uno_c": "v2_uno_c"}
    )
    outer = outer.merge(baseline, on="cohort", how="left")
    outer["delta_uno_v2"] = outer["uno_c"] - outer["v2_uno_c"]
    outer["delta_harrell_v2"] = outer["harrell_c"] - outer["v2_harrell_c"]
    outer.to_csv(RESULTS / "36_v4_marginrank_outer_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        RESULTS / "36_v4_marginrank_outer_predictions.csv", index=False
    )
    pd.DataFrame(selected_feature_rows).to_csv(
        RESULTS / "36_v4_marginrank_selected_features.csv", index=False
    )
    pd.DataFrame(all_inner_rows).to_csv(
        RESULTS / "36_v4_marginrank_inner_metrics.csv", index=False
    )
    pd.concat(all_rankings, ignore_index=True).to_csv(
        RESULTS / "36_v4_marginrank_inner_rankings.csv", index=False
    )

    gate_spec = migration["v4_primary_gate"]
    mean_uno = float(outer["uno_c"].mean())
    median_uno = float(outer["uno_c"].median())
    minimum_uno = float(outer["uno_c"].min())
    positive_domains = int((outer["uno_c"] > 0.5).sum())
    mean_gain = float(outer["delta_uno_v2"].mean())
    stability = mean_pairwise_jaccard(selected_gene_sets)
    conditions = {
        "mean_uno": mean_uno >= gate_spec["mean_uno_c"],
        "median_uno": median_uno >= gate_spec["median_uno_c"],
        "minimum_uno": minimum_uno >= gate_spec["minimum_uno_c"],
        "positive_domains": positive_domains >= gate_spec["minimum_positive_domains_of_10"],
        "catastrophic_floor": minimum_uno >= gate_spec["catastrophic_domain_floor"],
        "mean_gain_over_v2": mean_gain >= gate_spec["required_mean_uno_gain_over_locked_v2_on_same_domains"],
        "feature_stability": stability >= gate_spec["required_pair_or_feature_selection_stability"],
    }
    summary = {
        "status": "V4_MARGINRANK_NESTED_GATE_COMPLETE",
        "gate": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "mean_harrell_c": float(outer["harrell_c"].mean()),
        "mean_uno_c": mean_uno,
        "median_uno_c": median_uno,
        "minimum_uno_c": minimum_uno,
        "positive_domains_of_10": positive_domains,
        "mean_uno_gain_over_v2": mean_gain,
        "mean_selected_gene_jaccard": stability,
        "selected_candidate_counts": outer["selected_candidate"].value_counts().to_dict(),
        "candidate_count": len(CANDIDATES),
        "outer_domains": len(PRIMARY),
        "auxiliary_domain_available_only_when_selected": AUXILIARY,
        "gse224564_used": False,
        "fresh_candidate_scores_generated": False,
    }
    (RESULTS / "36_v4_marginrank_gate.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
