from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.ticker import NullLocator
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from figure_style import (  # noqa: E402
    PALETTE,
    arrow,
    box,
    clean_axes,
    configure_style,
    off_axes,
    panel_title,
    save_figure,
    stable_jitter,
    write_manifest,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODEL_PATH = ROOT / "src" / "transsurv" / "models" / "transsurv_pdac_v4_marginrank_locked.joblib"

MODEL_COLORS = {
    "v4_risk": PALETTE["navy"],
    "v4_marginrank": PALETTE["navy"],
    "v2_risk": PALETTE["blue"],
    "v2_predecessor": PALETTE["blue"],
    "legacy_risk": PALETTE["grey"],
    "legacy_six_gene": PALETTE["grey"],
    "clinical_oof_risk": PALETTE["orange"],
    "combined_oof_risk": PALETTE["teal"],
}

MODEL_LABELS = {
    "v4_risk": "MarginRank-DG",
    "v4_marginrank": "MarginRank-DG",
    "v2_risk": "Frozen v2",
    "v2_predecessor": "Frozen v2",
    "legacy_risk": "Legacy score",
    "legacy_six_gene": "Legacy score",
    "clinical_oof_risk": "Clinical OOF",
    "combined_oof_risk": "Combined OOF",
}

COHORT_LABELS = {
    "TCGA_PAAD": "TCGA",
    "CPTAC_PDACC": "CPTAC",
    "GSE62452": "GSE62452",
    "GSE85916": "GSE85916",
    "E-MTAB-6134": "E-MTAB-6134",
    "GSE183795": "GSE183795",
    "GSE57495": "GSE57495",
    "GSE79668": "GSE79668",
    "PACA-AU": "PACA-AU",
    "GSE71729": "GSE71729",
    "GSE205154_MET": "GSE205154",
}


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def mosaic(height_ratios: tuple[float, float, float] = (1, 1, 1)):
    fig, axes = plt.subplot_mosaic(
        [["a", "b"], ["c", "d"], ["e", "f"]],
        figsize=(7.2, 8.65),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.55, "wspace": 0.42},
    )
    return fig, axes


def format_p(value: float) -> str:
    if value < 0.001:
        return f"P={value:.1e}"
    return f"P={value:.3f}"


def bootstrap_interval(values: pd.Series) -> tuple[float, float, float]:
    clean = values.dropna().to_numpy(float)
    return (
        float(np.quantile(clean, 0.025)),
        float(np.median(clean)),
        float(np.quantile(clean, 0.975)),
    )


def build_figure1() -> dict:
    audit = read_csv("35_v4_development_data_audit.csv")
    fig = plt.figure(figsize=(7.2, 8.65))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=(1.02, 1.05, 0.93),
        hspace=0.55,
        wspace=0.42,
    )
    ax = {
        "a": fig.add_subplot(gs[0, 0]),
        "b": fig.add_subplot(gs[0, 1]),
        "c": fig.add_subplot(gs[1, 0]),
        "d": fig.add_subplot(gs[1, 1]),
        "e": fig.add_subplot(gs[2, :]),
    }

    off_axes(ax["a"])
    panel_title(ax["a"], "a", "Evidence architecture")
    stages = [
        (0.10, "DEVELOP", "10 domains\nn=1,181", PALETTE["navy"]),
        (0.36, "SELECT", "56 candidates\n11,681 genes", PALETTE["teal"]),
        (0.63, "FREEZE", "40 genes\n6 anchors", PALETTE["orange"]),
        (0.90, "VALIDATE", "PACA-CA once\nn=168", PALETTE["navy"]),
    ]
    ax["a"].plot([0.10, 0.90], [0.56, 0.56], color=PALETTE["light"], lw=2.2, zorder=1)
    for (x, label, description, color), x_next in zip(stages[:-1], [stage[0] for stage in stages[1:]]):
        arrow(ax["a"], (x + 0.045, 0.56), (x_next - 0.045, 0.56), color=color)
    for x, label, description, color in stages:
        ax["a"].scatter(x, 0.56, s=470, color=color, edgecolor=PALETTE["white"], linewidth=1.2, zorder=3)
        ax["a"].text(x, 0.86, label, ha="center", va="center", fontsize=7.9, weight="bold")
        ax["a"].text(x, 0.22, description, ha="center", va="center", fontsize=7.0)
    ax["a"].text(0.50, 0.03, "External outcomes opened only after freeze.", ha="center", fontsize=7.1)

    data = audit.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))
    role_color = [
        PALETTE["purple"] if role.startswith("auxiliary") else PALETTE["navy"]
        for role in data["role"]
    ]
    ax["b"].barh(y, data["n"], color=PALETTE["pale_blue"], height=0.66)
    ax["b"].barh(y, data["events"], color=role_color, height=0.66)
    for yi, n, events in zip(y, data["n"], data["events"]):
        ax["b"].text(n + 5, yi, f"{events}/{n}", va="center", fontsize=7.4)
    ax["b"].set_yticks(y, [COHORT_LABELS.get(x, x) for x in data["domain"]])
    ax["b"].set_xlabel("Patients (dark segment = events)")
    ax["b"].set_xlim(0, 325)
    clean_axes(ax["b"], "x")
    panel_title(ax["b"], "b", "Development evidence burden")

    roles = [
        ("Primary development", [1, 1, 0, 0]),
        ("Auxiliary domain", [1, 0, 0, 0]),
        ("PACA-CA", [0, 0, 1, 0]),
        ("GSE77435", [0, 0, 0, 1]),
        ("PRINCE", [0, 0, 0, 1]),
    ]
    matrix = np.array([row for _, row in roles], dtype=float)
    cmap = plt.matplotlib.colors.ListedColormap([PALETTE["white"], PALETTE["navy"]])
    ax["c"].pcolormesh(
        np.arange(matrix.shape[1] + 1) - 0.5,
        np.arange(matrix.shape[0] + 1) - 0.5,
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        shading="flat",
        edgecolors=PALETTE["white"],
        linewidth=0.6,
    )
    ax["c"].set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax["c"].set_xticks(range(4), ["Fit", "Select", "Validate", "Stress"])
    ax["c"].set_yticks(range(len(roles)), [name for name, _ in roles])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax["c"].text(j, i, "active" if matrix[i, j] else "-", ha="center", va="center", fontsize=7.6, color=PALETTE["white"] if matrix[i, j] else PALETTE["grey"])
    ax["c"].tick_params(length=0)
    for spine in ax["c"].spines.values():
        spine.set_visible(False)
    panel_title(ax["c"], "c", "Non-overlapping cohort roles")

    assay = pd.DataFrame(
        {
            "role": ["Primary development", "Auxiliary", "Primary validation", "Small stress", "Boundary stress"],
            "RNA-seq": [429, 70, 168, 0, 63],
            "Microarray": [752, 0, 0, 24, 0],
        }
    )
    yy = np.arange(len(assay))[::-1]
    ax["d"].barh(yy, assay["RNA-seq"], color=PALETTE["blue"], label="RNA-seq")
    ax["d"].barh(yy, assay["Microarray"], left=assay["RNA-seq"], color=PALETTE["orange"], label="Microarray")
    totals = assay["RNA-seq"] + assay["Microarray"]
    for yi, total in zip(yy, totals):
        ax["d"].text(total + max(totals) * 0.02, yi, f"n={int(total)}", va="center", fontsize=7.7)
    ax["d"].set_yticks(yy, assay["role"])
    ax["d"].set_xlabel("Patients")
    ax["d"].legend(frameon=False, loc="lower right")
    clean_axes(ax["d"], "x")
    panel_title(ax["d"], "d", "Assay diversity across roles")

    gate_rows = [
        ("Nested development", 1, "PASS"),
        ("Missingness", 1, "PASS"),
        ("Invariance", 1, "PASS"),
        ("PACA-CA", 1, "PASS"),
        ("GSE77435", 0, "supportive / NS"),
        ("PRINCE", -1, "boundary fail"),
        ("Software release", 1, "PASS"),
    ]
    gy = np.arange(len(gate_rows))[::-1]
    color_map = {1: PALETTE["teal"], 0: PALETTE["orange"], -1: PALETTE["coral"]}
    for yi, (name, status, label) in zip(gy, gate_rows):
        ax["e"].plot([-1, 1], [yi, yi], color=PALETTE["light"], lw=0.7)
        ax["e"].scatter(status, yi, s=72, color=color_map[status], edgecolor="white", linewidth=0.8, zorder=3)
        ax["e"].text(status + 0.10, yi, label, va="center", fontsize=7.5)
    ax["e"].set_yticks(gy, [row[0] for row in gate_rows])
    ax["e"].set_xticks([-1, 0, 1], ["Boundary", "Uncertain", "Passed"])
    ax["e"].set_xlim(-1.25, 1.45)
    ax["e"].tick_params(axis="y", length=0)
    clean_axes(ax["e"])
    panel_title(ax["e"], "e", "Prespecified result gates")

    return save_figure(
        fig,
        FIGURES,
        "Figure1_study_architecture",
        "Study architecture and evidence governance",
        ["Evidence architecture", "Development evidence burden", "Non-overlapping cohort roles", "Assay diversity across roles", "Prespecified result gates"],
        ["minimalist flow", "stacked bar", "role matrix", "stacked bar", "status dot plot"],
    )


def candidate_family(candidate: str) -> str:
    return candidate.split("_g", 1)[0]


def build_figure2() -> dict:
    selected = read_csv("37_v4_selected_loso_metrics.csv")
    ranking = read_csv("37_v4_fixed_candidate_ranking.csv")
    ablation = read_csv("42_v4_architecture_ablation.csv")
    all_metrics = read_csv("37_v4_fixed_candidate_loso_metrics.csv")
    fig, ax = mosaic((0.95, 1.05, 1.0))

    off_axes(ax["a"])
    panel_title(ax["a"], "a", "MarginRank-DG mechanism")
    stages = [
        (0.12, "1", "RANK", "within-sample\npercentiles", PALETTE["navy"]),
        (0.50, "2", "MARGIN", "6 anchors /\n40 margins", PALETTE["teal"]),
        (0.88, "3", "RIDGE", "raw relative\nrisk", PALETTE["purple"]),
    ]
    ax["a"].plot([0.15, 0.85], [0.52, 0.52], color=PALETTE["light"], lw=2.2, zorder=1)
    for (x, number, label, description, color), x_next in zip(stages[:-1], [stage[0] for stage in stages[1:]]):
        arrow(ax["a"], (x + 0.075, 0.52), (x_next - 0.075, 0.52), color=color)
    for x, number, label, description, color in stages:
        ax["a"].scatter(x, 0.52, s=560, color=color, edgecolor=PALETTE["white"], linewidth=1.2, zorder=3)
        ax["a"].text(x, 0.52, number, ha="center", va="center", fontsize=10.0, color=PALETTE["white"], weight="bold")
        ax["a"].text(x, 0.82, label, ha="center", va="center", fontsize=8.0, weight="bold")
        ax["a"].text(x, 0.24, description, ha="center", va="center", fontsize=7.2)
    ax["a"].text(0.50, 0.04, "No cohort labels or batch normalization at inference.", ha="center", fontsize=7.1, weight="bold")

    off_axes(ax["b"])
    panel_title(ax["b"], "b", "Nested leave-one-domain-out design")
    names = [COHORT_LABELS[x] for x in selected["cohort"]]
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    center = (0.50, 0.50)
    ax["b"].add_patch(Circle(center, 0.34, fill=False, edgecolor=PALETTE["light"], linewidth=1.0, linestyle="--"))
    ax["b"].add_patch(Circle(center, 0.18, fill=False, edgecolor=PALETTE["purple"], linewidth=1.4))
    ax["b"].text(center[0], center[1] + 0.04, "9 domains", ha="center", va="center", fontsize=8.2, weight="bold")
    ax["b"].text(center[0], center[1] - 0.07, "train + inner\nLOSO select", ha="center", va="center", fontsize=7.2)
    for i, (name, angle) in enumerate(zip(names, angles)):
        x = center[0] + 0.39 * np.cos(angle)
        y = center[1] + 0.34 * np.sin(angle)
        color = PALETTE["navy"] if i == 0 else PALETTE["blue"]
        ax["b"].scatter(x, y, s=95 if i == 0 else 56, color=color, edgecolor=PALETTE["white"], linewidth=0.8, zorder=3)
        ha = "left" if np.cos(angle) > 0.2 else "right" if np.cos(angle) < -0.2 else "center"
        va = "bottom" if np.sin(angle) > 0.35 else "top" if np.sin(angle) < -0.35 else "center"
        ax["b"].text(x + 0.02 * np.sign(np.cos(angle)), y + 0.02 * np.sign(np.sin(angle)), name, ha=ha, va=va, fontsize=6.7)
    ax["b"].text(0.50, 0.05, "Outer holdout repeats across all 10 domains.", ha="center", fontsize=7.1)

    ordered = selected.sort_values("uno_c")
    y = np.arange(len(ordered))
    ax["c"].hlines(y, ordered["v2_uno_c"], ordered["uno_c"], color=PALETTE["light"], lw=2.0)
    ax["c"].scatter(ordered["v2_uno_c"], y, color=PALETTE["blue"], s=28, label="Frozen v2", zorder=3)
    ax["c"].scatter(ordered["uno_c"], y, color=PALETTE["navy"], s=34, label="MarginRank-DG", zorder=3)
    ax["c"].axvline(0.5, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["c"].set_yticks(y, [COHORT_LABELS[x] for x in ordered["cohort"]])
    ax["c"].set_xlabel("Uno C-index")
    ax["c"].legend(frameon=False, loc="lower right")
    clean_axes(ax["c"], "x")
    panel_title(ax["c"], "c", "Cross-domain discrimination")

    family_colors = {"anchor": PALETTE["navy"], "rank": PALETTE["blue"], "contrast": PALETTE["orange"], "hybrid": PALETTE["purple"]}
    ranking = ranking.copy()
    ranking["family"] = ranking["candidate"].map(candidate_family)
    for family, group in ranking.groupby("family"):
        ax["d"].scatter(group["mean_uno"], group["worst_uno"], s=22 + 180 * (0.05 - group["sd_uno"]).clip(lower=0), alpha=0.72, color=family_colors.get(family, PALETTE["grey"]), label=family)
    winner = ranking.loc[ranking["selected"]].iloc[0]
    ax["d"].scatter(winner["mean_uno"], winner["worst_uno"], s=90, facecolor=PALETTE["navy"], edgecolor=PALETTE["dark"], linewidth=1.0, zorder=5)
    ax["d"].annotate("selected", (winner["mean_uno"], winner["worst_uno"]), xytext=(-42, -18), textcoords="offset points", arrowprops={"arrowstyle": "-", "color": PALETTE["mid"]}, fontsize=8.0)
    ax["d"].set_xlabel("Mean Uno C-index")
    ax["d"].set_ylabel("Worst-domain Uno C-index")
    ax["d"].legend(frameon=False, ncol=2, loc="lower right")
    clean_axes(ax["d"], "both")
    panel_title(ax["d"], "d", "Candidate trade-off landscape")

    role_labels = {
        "selected": "Selected",
        "without_anchor": "No anchor",
        "without_auxiliary_domain": "No auxiliary",
        "twenty_genes": "20 genes",
        "eighty_genes": "80 genes",
        "expert_ensemble": "Expert ensemble",
        "hybrid_margin": "Hybrid",
    }
    plot = ablation.copy()
    plot["label"] = plot["role"].map(role_labels)
    plot = plot.sort_values("worst_uno")
    y = np.arange(len(plot))
    ax["e"].hlines(y, plot["worst_uno"], plot["mean_uno"], color=PALETTE["light"], lw=2.0)
    colors = [PALETTE["navy"] if role == "selected" else PALETTE["grey"] for role in plot["role"]]
    ax["e"].scatter(plot["worst_uno"], y, color=colors, marker="D", s=28, label="Worst domain")
    ax["e"].scatter(plot["mean_uno"], y, color=colors, marker="o", s=32, label="Mean")
    ax["e"].set_yticks(y, plot["label"])
    ax["e"].set_xlabel("Uno C-index")
    ax["e"].legend(frameon=False, loc="lower right")
    clean_axes(ax["e"], "x")
    panel_title(ax["e"], "e", "Architecture ablation")

    selected_name = "anchor_g40_p0__pooled_ridge_a1__aux"
    comparators = {
        "No anchor": "rank_g40_p0__pooled_ridge_a1__aux",
        "No auxiliary": "anchor_g40_p0__pooled_ridge_a1__primary",
        "20 genes": "anchor_g20_p0__pooled_ridge_a1__aux",
        "80 genes": "anchor_g80_p0__pooled_ridge_a1__aux",
        "Expert": "anchor_g40_p0__expert_median_a10__aux",
    }
    pivot = all_metrics.pivot(index="cohort", columns="candidate", values="uno_c")
    rows = []
    for label, candidate in comparators.items():
        for cohort in pivot.index:
            rows.append({"comparison": label, "delta": pivot.loc[cohort, selected_name] - pivot.loc[cohort, candidate], "cohort": cohort})
    deltas = pd.DataFrame(rows)
    order = list(comparators)
    sns.violinplot(data=deltas, x="comparison", y="delta", order=order, color=PALETTE["pale_teal"], inner=None, linewidth=0, cut=0, ax=ax["f"])
    for j, label in enumerate(order):
        values = deltas.loc[deltas["comparison"] == label, "delta"].to_numpy()
        ax["f"].scatter(j + np.array(stable_jitter(len(values), 0.11)), values, s=17, color=PALETTE["teal"], alpha=0.85, edgecolor="white", linewidth=0.35)
        ax["f"].plot([j - 0.15, j + 0.15], [np.median(values)] * 2, color=PALETTE["dark"], lw=1.3)
    ax["f"].axhline(0, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["f"].set_xlabel("")
    ax["f"].set_ylabel("Selected minus comparator Uno C")
    ax["f"].tick_params(axis="x", rotation=25)
    clean_axes(ax["f"], "y")
    panel_title(ax["f"], "f", "Paired domain effects")

    return save_figure(
        fig,
        FIGURES,
        "Figure2_model_development",
        "MarginRank-DG development and selection",
        ["MarginRank-DG mechanism", "Nested leave-one-domain-out design", "Cross-domain discrimination", "Candidate trade-off landscape", "Architecture ablation", "Paired domain effects"],
        ["minimalist pipeline", "nested-circle schematic", "dumbbell plot", "bubble scatter", "paired dot plot", "violin and strip plot"],
    )


def build_figure3() -> dict:
    coefficients = read_csv("39_v4_final_feature_coefficients.csv")
    selected_features = read_csv("36_v4_marginrank_selected_features.csv")
    predictions = read_csv("37_v4_selected_loso_predictions.csv")
    full_rank = read_csv("46_v4_full_gene_survival_rank.csv")
    fig, ax = mosaic((1.12, 0.90, 0.98))

    top = coefficients.nlargest(16, "absolute_coefficient").sort_values("coefficient")
    y = np.arange(len(top))
    colors = [PALETTE["coral"] if value > 0 else PALETTE["teal"] for value in top["coefficient"]]
    ax["a"].hlines(y, 0, top["coefficient"], color=colors, lw=1.7)
    ax["a"].scatter(top["coefficient"], y, color=colors, s=35, edgecolor="white", linewidth=0.5, zorder=3)
    ax["a"].axvline(0, color=PALETTE["grey"], lw=0.7)
    ax["a"].set_yticks(y, top["gene"])
    ax["a"].set_xlabel("Frozen ridge coefficient")
    clean_axes(ax["a"], "x")
    panel_title(ax["a"], "a", "Largest frozen coefficients")

    neg = coefficients.loc[coefficients["coefficient"] < 0, "coefficient"]
    pos = coefficients.loc[coefficients["coefficient"] >= 0, "coefficient"]
    bins = np.linspace(coefficients["coefficient"].min() - 0.01, coefficients["coefficient"].max() + 0.01, 13)
    ax["b"].hist(neg, bins=bins, color=PALETTE["teal"], alpha=0.82, label="Negative")
    ax["b"].hist(pos, bins=bins, color=PALETTE["coral"], alpha=0.82, label="Positive")
    ax["b"].plot(coefficients["coefficient"], np.full(len(coefficients), -0.25), "|", color=PALETTE["dark"], markersize=6)
    ax["b"].axvline(0, color=PALETTE["grey"], lw=0.8)
    ax["b"].set_xlabel("Coefficient")
    ax["b"].set_ylabel("Genes")
    ax["b"].legend(frameon=False)
    clean_axes(ax["b"], "y")
    panel_title(ax["b"], "b", "All 40 model coefficients")

    gene_features = selected_features.loc[selected_features["feature_type"] == "gene"].copy()
    frequency = gene_features.groupby("feature")["outer_test"].nunique().sort_values(ascending=False).head(18).sort_values()
    ax["c"].barh(np.arange(len(frequency)), frequency.values, color=PALETTE["purple"], height=0.68)
    ax["c"].set_yticks(np.arange(len(frequency)), frequency.index)
    ax["c"].set_xlim(0, 10.5)
    ax["c"].set_xlabel("Outer folds selecting gene (of 10)")
    clean_axes(ax["c"], "x")
    panel_title(ax["c"], "c", "Nested selection frequency")

    pred = predictions.copy()
    pred["cohort_label"] = pred["cohort"].map(COHORT_LABELS)
    order = [COHORT_LABELS[x] for x in predictions.groupby("cohort")["risk"].median().sort_values().index]
    sns.violinplot(data=pred, x="risk", y="cohort_label", order=order, color=PALETTE["pale_blue"], inner=None, cut=0, linewidth=0, ax=ax["d"])
    sns.boxplot(data=pred, x="risk", y="cohort_label", order=order, width=0.18, color=PALETTE["navy"], showfliers=False, boxprops={"alpha": 0.8}, whiskerprops={"linewidth": 0.7}, ax=ax["d"])
    ax["d"].axvline(0, color=PALETTE["grey"], lw=0.7, ls="--")
    ax["d"].set_xlabel("LOSO raw risk")
    ax["d"].set_ylabel("")
    clean_axes(ax["d"], "x")
    panel_title(ax["d"], "d", "Continuous risk across domains")

    merged = coefficients.merge(full_rank[["gene", "meta_z", "stability", "directional_consistency"]], on="gene", suffixes=("", "_rank"))
    ax["e"].scatter(merged["meta_z"], merged["coefficient"], s=22 + 35 * merged["stability"] / merged["stability"].max(), color=[PALETTE["coral"] if x > 0 else PALETTE["teal"] for x in merged["coefficient"]], alpha=0.80, edgecolor="white", linewidth=0.4)
    ax["e"].axhline(0, color=PALETTE["grey"], lw=0.7)
    ax["e"].axvline(0, color=PALETTE["grey"], lw=0.7)
    label_offsets = {"IL1RAP": (3, 3), "UBA6": (3, 3), "MIA3": (3, -2), "RALGPS1": (3, 7)}
    for gene, offset in label_offsets.items():
        row = merged.loc[merged["gene"] == gene].iloc[0]
        ax["e"].annotate(gene, (row["meta_z"], row["coefficient"]), xytext=offset, textcoords="offset points", fontsize=7.0)
    ax["e"].set_xlabel("Cross-domain meta-z")
    ax["e"].set_ylabel("Frozen coefficient")
    clean_axes(ax["e"], "both")
    panel_title(ax["e"], "e", "Marginal and multivariable effects")

    heat_genes = coefficients.nlargest(10, "absolute_coefficient")["gene"].tolist()
    z_cols = [column for column in full_rank.columns if column.startswith("z_") and not column.endswith("MET")]
    heat = full_rank.set_index("gene").loc[heat_genes, z_cols]
    vmax = float(np.quantile(np.abs(heat.to_numpy()), 0.95))
    image = ax["f"].pcolormesh(
        np.arange(heat.shape[1] + 1) - 0.5,
        np.arange(heat.shape[0] + 1) - 0.5,
        heat.to_numpy(),
        cmap=sns.diverging_palette(220, 18, as_cmap=True),
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
        edgecolors=PALETTE["white"],
        linewidth=0.25,
    )
    ax["f"].set_ylim(heat.shape[0] - 0.5, -0.5)
    ax["f"].set_yticks(range(len(heat)), heat.index)
    ax["f"].set_xticks(range(len(z_cols)), [COHORT_LABELS.get(x[2:], x[2:]) for x in z_cols], rotation=55, ha="right")
    ax["f"].tick_params(length=0)
    for spine in ax["f"].spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax["f"], fraction=0.035, pad=0.02)
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
    cbar.set_label("Cox z-score", fontsize=8.0)
    panel_title(ax["f"], "f", "Gene effects across ten domains")

    return save_figure(
        fig,
        FIGURES,
        "Figure3_frozen_model_anatomy",
        "Frozen model anatomy and stability",
        ["Largest frozen coefficients", "All 40 model coefficients", "Nested selection frequency", "Continuous risk across domains", "Marginal and multivariable effects", "Gene effects across ten domains"],
        ["lollipop plot", "histogram and rug", "horizontal bar", "violin and box plot", "bubble scatter", "heatmap"],
    )


def build_figure4() -> dict:
    summary = read_csv("38_v4_missingness_summary.csv")
    replicates = read_csv("38_v4_missingness_replicates.csv")
    nested = read_csv("55_v4_nested_inner_outer_gap.csv")
    permutation = read_csv("55_v4_paca_ca_permutation_null.csv")
    audit = read_json("55_v4_anti_overfit_audit.json")
    fig, ax = mosaic((0.92, 1.04, 1.04))

    levels = summary["level"] * 100
    ax["a"].plot(levels, summary["mean_risk_correlation"], marker="o", color=PALETTE["navy"], label="Mean")
    ax["a"].plot(levels, summary["minimum_cohort_mean_correlation"], marker="D", color=PALETTE["orange"], label="Worst cohort mean")
    ax["a"].fill_between(levels, summary["minimum_cohort_mean_correlation"], summary["mean_risk_correlation"], color=PALETTE["pale_blue"], alpha=0.8)
    ax["a"].set_xticks(levels, [f"{int(x)}%" for x in levels])
    ax["a"].set_ylim(0.90, 1.005)
    ax["a"].set_xlabel("Input genes randomly masked")
    ax["a"].set_ylabel("Risk-score correlation")
    ax["a"].legend(frameon=False, loc="lower left")
    clean_axes(ax["a"], "y")
    panel_title(ax["a"], "a", "Risk stability under missingness")

    ax["b"].plot(levels, summary["mean_uno_drop"], marker="o", color=PALETTE["navy"], label="Mean loss")
    ax["b"].plot(levels, summary["maximum_cohort_mean_uno_drop"], marker="D", color=PALETTE["coral"], label="Worst cohort mean")
    ax["b"].plot(levels, summary["q95_uno_drop"], marker="s", color=PALETTE["orange"], label="95th percentile")
    ax["b"].axhline(0, color=PALETTE["grey"], lw=0.7)
    ax["b"].set_xticks(levels, [f"{int(x)}%" for x in levels])
    ax["b"].set_xlabel("Input genes randomly masked")
    ax["b"].set_ylabel("Uno C-index loss")
    ax["b"].legend(frameon=False, loc="upper left")
    clean_axes(ax["b"], "y")
    panel_title(ax["b"], "b", "Discrimination loss under missingness")

    rep = replicates.copy()
    rep["level_label"] = (rep["level"] * 100).round().astype(int).astype(str) + "%"
    order = ["5%", "10%", "20%"]
    sns.violinplot(data=rep, x="level_label", y="uno_drop", order=order, palette=[PALETTE["pale_teal"], PALETTE["pale_blue"], PALETTE["pale_orange"]], inner=None, linewidth=0, cut=0, ax=ax["c"])
    sns.boxplot(data=rep, x="level_label", y="uno_drop", order=order, width=0.20, color=PALETTE["navy"], showfliers=False, ax=ax["c"])
    ax["c"].axhline(0, color=PALETTE["grey"], lw=0.7, ls="--")
    ax["c"].set_xlabel("Genes masked")
    ax["c"].set_ylabel("Uno C-index loss across 2,000 runs")
    clean_axes(ax["c"], "y")
    panel_title(ax["c"], "c", "Replicate-level loss distributions")

    lo = float(min(nested["inner_mean_uno"].min(), nested["outer_uno"].min()) - 0.01)
    hi = float(max(nested["inner_mean_uno"].max(), nested["outer_uno"].max()) + 0.01)
    ax["d"].plot([lo, hi], [lo, hi], color=PALETTE["grey"], lw=0.9, ls="--")
    ax["d"].scatter(nested["inner_mean_uno"], nested["outer_uno"], s=38, color=PALETTE["navy"], alpha=0.86, edgecolor="white", linewidth=0.45)
    ax["d"].text(0.04, 0.94, f"mean gap = {audit['nested_selection']['mean_inner_minus_outer']:.3f}\nmax |gap| = {audit['nested_selection']['maximum_absolute_gap']:.3f}", transform=ax["d"].transAxes, fontsize=7.5, va="top")
    ax["d"].set_xlabel("Inner selected-model Uno C")
    ax["d"].set_ylabel("Outer held-out Uno C")
    ax["d"].set_xlim(lo, hi)
    ax["d"].set_ylim(lo, hi)
    clean_axes(ax["d"], "both")
    panel_title(ax["d"], "d", "Nested inner-to-outer performance")

    values = permutation["harrell_c"].to_numpy(float)
    ax["e"].hist(values, bins=32, color=PALETTE["pale_teal"], edgecolor=PALETTE["white"], linewidth=0.35)
    observed = audit["paca_ca_fixed_risk_permutation"]["observed_harrell_c"]
    q975 = audit["paca_ca_fixed_risk_permutation"]["null_q975"]
    ax["e"].axvline(q975, color=PALETTE["grey"], lw=1.0, ls="--")
    ax["e"].axvline(observed, color=PALETTE["coral"], lw=1.7)
    ax["e"].text(observed + 0.003, ax["e"].get_ylim()[1] * 0.88, f"observed = {observed:.3f}", color=PALETTE["coral"], fontsize=7.5, va="top")
    ax["e"].text(q975 - 0.003, ax["e"].get_ylim()[1] * 0.70, f"null 97.5% = {q975:.3f}", color=PALETTE["mid"], fontsize=7.2, ha="right")
    ax["e"].set_xlabel("Harrell C under fixed-risk permutation")
    ax["e"].set_ylabel("Permutations (n=2,000)")
    clean_axes(ax["e"], "y")
    panel_title(ax["e"], "e", "PACA-CA fixed-risk permutation null")

    conditions = [
        ("Nested selection", True),
        ("10/10 outer domains > chance", audit["descriptive_conditions"]["all_outer_domains_above_chance"]),
        ("Mean gap <= 0.10", audit["descriptive_conditions"]["nested_mean_absolute_gap_le_0_10"]),
        ("Permutation null", audit["descriptive_conditions"]["paca_ca_fixed_risk_above_permutation_null"]),
        ("Missingness gates", True),
        ("Technical invariance (S3d)", True),
    ]
    gy = np.arange(len(conditions))[::-1]
    for yi, (label, passed) in zip(gy, conditions):
        ax["f"].plot([0, 1], [yi, yi], color=PALETTE["light"], lw=0.8)
        ax["f"].scatter(1 if passed else 0, yi, s=72, color=PALETTE["teal"] if passed else PALETTE["coral"], edgecolor=PALETTE["white"], linewidth=0.8, zorder=3)
        ax["f"].text(1.06, yi, "PASS" if passed else "REVIEW", va="center", fontsize=7.2)
    ax["f"].set_yticks([])
    for yi, (label, _) in zip(gy, conditions):
        ax["f"].text(0.02, yi, label, va="center", ha="left", fontsize=6.8, color=PALETTE["dark"])
    ax["f"].set_xticks([0, 1], ["Not met", "Met"])
    ax["f"].set_xlim(-0.18, 1.35)
    ax["f"].tick_params(axis="y", length=0)
    clean_axes(ax["f"])
    panel_title(ax["f"], "f", "Frozen-model stress audit")

    return save_figure(
        fig,
        FIGURES,
        "Figure4_technical_robustness",
        "Missingness and anti-overfitting stress tests",
        ["Risk stability under missingness", "Discrimination loss under missingness", "Replicate-level loss distributions", "Nested inner-to-outer performance", "PACA-CA fixed-risk permutation null", "Frozen-model stress audit"],
        ["line and ribbon", "multi-line plot", "violin and box plot", "identity scatter", "null histogram", "status dot plot"],
    )


def build_figure5() -> dict:
    predictions = read_csv("41_v4_paca_ca_predictions.csv")
    km = read_csv("41_v4_paca_ca_km_curves.csv")
    metrics = read_csv("41_v4_paca_ca_metrics.csv")
    time_auc = read_csv("41_v4_paca_ca_time_auc.csv")
    bootstrap = read_csv("41_v4_paca_ca_bootstrap.csv")
    gate = read_json("41_v4_paca_ca_validation_gate.json")
    fig, ax = mosaic((0.86, 1.12, 1.02))

    off_axes(ax["a"])
    panel_title(ax["a"], "a", "Untouched validation flow")
    # A one-time eligibility lock is a sequence, not a process map.  Circles and
    # a single baseline make the temporal order explicit without repeating the
    # box language used in the study-architecture panels.
    ax["a"].plot([0.14, 0.50, 0.86], [0.56, 0.56, 0.56], color=PALETTE["mid"], lw=1.5, zorder=1)
    stages = [
        (0.14, "SCREEN", "243 donors", PALETTE["blue"]),
        (0.50, "LOCK", "168 eligible", PALETTE["teal"]),
        (0.86, "SCORE", "136 deaths", PALETTE["orange"]),
    ]
    for step, (x, heading, detail, color) in enumerate(stages, start=1):
        ax["a"].add_patch(Circle((x, 0.56), 0.075, facecolor="white", edgecolor=color, lw=2.0, zorder=2))
        ax["a"].text(x, 0.56, str(step), ha="center", va="center", fontsize=10.0, weight="bold", color=color)
        ax["a"].text(x, 0.76, heading, ha="center", va="center", fontsize=7.1, weight="bold", color=color)
        ax["a"].text(x, 0.36, detail, ha="center", va="center", fontsize=8.2, color=PALETTE["dark"])
    ax["a"].text(0.50, 0.17, "Primary untreated PACA-CA · exclusions frozen before any risk score", ha="center", fontsize=7.7, weight="bold", color=PALETTE["dark"])

    for event, label, color in [(1, "Death", PALETTE["coral"]), (0, "Censored", PALETTE["blue"])]:
        values = predictions.loc[predictions["OS_EVENT"] == event, "v4_risk"]
        sns.kdeplot(values, ax=ax["b"], color=color, fill=True, alpha=0.20, linewidth=1.4, label=f"{label} (n={len(values)})")
        ax["b"].plot(values, np.full(len(values), -0.015 if event else -0.030), "|", color=color, markersize=4.5, alpha=0.65)
    ax["b"].axvline(predictions["v4_risk"].median(), color=PALETTE["dark"], lw=0.8, ls="--")
    ax["b"].set_xlabel("Frozen raw risk")
    ax["b"].set_ylabel("Density")
    ax["b"].legend(frameon=False)
    clean_axes(ax["b"], "y")
    panel_title(ax["b"], "b", "Outcome-stratified score distribution")

    tier_colors = {"low": PALETTE["teal"], "middle": PALETTE["orange"], "high": PALETTE["coral"]}
    tier_labels = {"low": "Low", "middle": "Middle", "high": "High"}
    for tier in ["low", "middle", "high"]:
        curve = km[km["risk_tier"] == tier]
        ax["c"].step(curve["time_months"], curve["survival"], where="post", color=tier_colors[tier], label=f"{tier_labels[tier]} (n={int(curve['n'].iloc[0])})")
    times = [0, 12, 24, 36, 48, 60]
    y_rows = {"low": -0.07, "middle": -0.13, "high": -0.19}
    ax["c"].text(-11, -0.13, "At risk", ha="right", va="center", fontsize=7.3)
    for tier in ["low", "middle", "high"]:
        subset = predictions[predictions["risk_tier"] == tier]
        for time in times:
            count = int((subset["OS_TIME"] >= time).sum())
            ax["c"].text(time, y_rows[tier], str(count), ha="center", va="center", fontsize=6.9, color=tier_colors[tier])
    ax["c"].set_xlim(0, 60)
    ax["c"].set_ylim(-0.23, 1.03)
    ax["c"].set_yticks(np.linspace(0, 1, 6))
    ax["c"].set_xticks(times)
    ax["c"].set_xlabel("Overall survival (months)")
    ax["c"].set_ylabel("Survival probability")
    ax["c"].legend(frameon=False, loc="upper right")
    clean_axes(ax["c"])
    panel_title(ax["c"], "c", "Risk-tertile survival separation")

    forest = metrics[["model", "hr_per_sd", "hr_ci_low", "hr_ci_high", "hr_p"]].copy()
    adjusted = gate["adjusted"]
    forest = pd.concat([forest, pd.DataFrame([{"model": "v4_adjusted", "hr_per_sd": adjusted["hr_per_sd"], "hr_ci_low": adjusted["hr_ci_low"], "hr_ci_high": adjusted["hr_ci_high"], "hr_p": adjusted["hr_p"]}])], ignore_index=True)
    order = ["legacy_risk", "clinical_oof_risk", "v2_risk", "v4_risk", "v4_adjusted", "combined_oof_risk"]
    forest = forest.set_index("model").loc[order].reset_index()
    labels = ["Legacy score", "Clinical OOF", "Frozen v2", "MarginRank-DG", "Adjusted v4", "Combined OOF"]
    yy = np.arange(len(forest))
    colors = [MODEL_COLORS.get(x, PALETTE["navy"]) if x != "v4_adjusted" else PALETTE["purple"] for x in forest["model"]]
    for i, row in forest.iterrows():
        ax["d"].plot([row.hr_ci_low, row.hr_ci_high], [i, i], color=colors[i], lw=1.5)
        ax["d"].scatter(row.hr_per_sd, i, s=38, color=colors[i], edgecolor="white", linewidth=0.5, zorder=3)
        ax["d"].text(2.75, i, format_p(float(row.hr_p)), va="center", ha="right", fontsize=7.2)
    ax["d"].axvline(1, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["d"].set_yticks(yy, labels)
    ax["d"].set_xlim(0.75, 2.85)
    ax["d"].set_xlabel("Hazard ratio per SD (95% CI)")
    clean_axes(ax["d"], "x")
    panel_title(ax["d"], "d", "External prognostic association")

    model_order = ["legacy_risk", "clinical_oof_risk", "v2_risk", "v4_risk", "combined_oof_risk"]
    for model in model_order:
        values = time_auc[time_auc["model"] == model]
        ax["e"].plot(values["time_months"], values["auc"], marker="o", color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax["e"].axhline(0.5, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["e"].set_xticks([12, 24, 36])
    ax["e"].set_ylim(0.48, 0.79)
    ax["e"].set_xlabel("Time horizon (months)")
    ax["e"].set_ylabel("Time-dependent AUC")
    ax["e"].legend(frameon=False, ncol=2, loc="upper left")
    clean_axes(ax["e"], "both")
    panel_title(ax["e"], "e", "Time-dependent discrimination")

    distributions = pd.DataFrame({"Harrell C": bootstrap["v4_risk_harrell_c"], "Uno C": bootstrap["v4_risk_uno_c"]}).melt(var_name="metric", value_name="value")
    sns.violinplot(data=distributions, x="metric", y="value", palette=[PALETTE["pale_blue"], PALETTE["pale_teal"]], inner=None, cut=0, linewidth=0, ax=ax["f"])
    for j, metric in enumerate(["Harrell C", "Uno C"]):
        values = distributions.loc[distributions["metric"] == metric, "value"]
        lo, med, hi = bootstrap_interval(values)
        ax["f"].plot([j, j], [lo, hi], color=PALETTE["dark"], lw=1.7)
        ax["f"].scatter(j, med, s=38, color=PALETTE["navy"], edgecolor="white", linewidth=0.5, zorder=3)
        ax["f"].text(j, hi + 0.008, f"{lo:.3f}-{hi:.3f}", ha="center", fontsize=7.4)
    ax["f"].axhline(0.5, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["f"].set_ylim(0.48, 0.70)
    ax["f"].set_xlabel("")
    ax["f"].set_ylabel("Bootstrap C-index")
    clean_axes(ax["f"], "y")
    panel_title(ax["f"], "f", "Frozen-score bootstrap uncertainty")

    return save_figure(
        fig,
        FIGURES,
        "Figure5_paca_ca_validation",
        "Untouched PACA-CA validation",
        ["Untouched validation flow", "Outcome-stratified score distribution", "Risk-tertile survival separation", "External prognostic association", "Time-dependent discrimination", "Frozen-score bootstrap uncertainty"],
        ["minimalist timeline", "density and rug", "Kaplan-Meier plot", "forest plot", "multi-line plot", "violin and interval plot"],
    )


def build_figure6() -> dict:
    predictions = read_csv("41_v4_paca_ca_predictions.csv")
    bootstrap = read_csv("41_v4_paca_ca_bootstrap.csv")
    subgroups = read_csv("41_v4_paca_ca_subgroups.csv")
    gate = read_json("41_v4_paca_ca_validation_gate.json")
    fig, ax = mosaic((0.94, 1.03, 1.03))

    model_cols = ["legacy_risk", "clinical_oof_risk", "v2_risk", "v4_risk", "combined_oof_risk"]
    ci_rows = []
    for model in model_cols:
        lo, med, hi = bootstrap_interval(bootstrap[f"{model}_uno_c"])
        ci_rows.append((model, lo, med, hi))
    yy = np.arange(len(ci_rows))
    for i, (model, lo, med, hi) in enumerate(ci_rows):
        ax["a"].plot([lo, hi], [i, i], color=MODEL_COLORS[model], lw=1.8)
        ax["a"].scatter(med, i, s=40, color=MODEL_COLORS[model], edgecolor="white", linewidth=0.5, zorder=3)
    ax["a"].axvline(0.5, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["a"].set_yticks(yy, [MODEL_LABELS[x[0]] for x in ci_rows])
    ax["a"].set_xlabel("Bootstrap Uno C-index (95% interval)")
    clean_axes(ax["a"], "x")
    panel_title(ax["a"], "a", "Model discrimination in PACA-CA")

    delta_map = {
        "v4 vs v2": "delta_uno_v4_vs_v2",
        "v4 vs legacy": "delta_uno_v4_vs_legacy",
        "Combined vs clinical": "delta_uno_combined_vs_clinical",
    }
    delta_long = pd.concat([pd.DataFrame({"comparison": label, "delta": bootstrap[column]}) for label, column in delta_map.items()], ignore_index=True)
    order = list(delta_map)
    sns.violinplot(data=delta_long, x="comparison", y="delta", order=order, palette=[PALETTE["pale_blue"], PALETTE["pale_purple"], PALETTE["pale_teal"]], inner=None, cut=0, linewidth=0, ax=ax["b"])
    for j, label in enumerate(order):
        values = delta_long.loc[delta_long["comparison"] == label, "delta"]
        lo, med, hi = bootstrap_interval(values)
        ax["b"].plot([j, j], [lo, hi], color=PALETTE["dark"], lw=1.4)
        ax["b"].scatter(j, med, s=34, color=[PALETTE["blue"], PALETTE["purple"], PALETTE["teal"]][j], edgecolor="white", linewidth=0.4, zorder=3)
    ax["b"].axhline(0, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["b"].set_xlabel("")
    ax["b"].set_ylabel("Paired bootstrap Uno C difference")
    ax["b"].tick_params(axis="x", rotation=20)
    clean_axes(ax["b"], "y")
    panel_title(ax["b"], "b", "Incremental discrimination")

    rho, rho_p = stats.spearmanr(predictions["clinical_oof_risk"], predictions["v4_risk"])
    colors = np.where(predictions["OS_EVENT"].to_numpy() == 1, PALETTE["coral"], PALETTE["blue"])
    ax["c"].scatter(predictions["clinical_oof_risk"], predictions["v4_risk"], s=22, color=colors, alpha=0.68, edgecolor="white", linewidth=0.3)
    ax["c"].text(0.04, 0.94, f"Spearman rho={rho:.2f}\n{format_p(rho_p)}", transform=ax["c"].transAxes, va="top", fontsize=7.8)
    ax["c"].set_xlabel("Cross-fitted clinical risk")
    ax["c"].set_ylabel("Frozen transcriptomic risk")
    clean_axes(ax["c"], "both")
    panel_title(ax["c"], "c", "Complementary clinical information")

    label_map = {
        "age_lt65": "Age <65",
        "age_ge65": "Age >=65",
        "female": "Female",
        "male": "Male",
        "stage_early_IA_IB_IIA": "Stage IA-IB-IIA",
        "stage_advanced_IIB_III_IV": "Stage IIB-III-IV",
        "grade_well_or_moderate": "Grade well/moderate",
        "grade_poor_or_undifferentiated": "Grade poor/undiff.",
        "specimen_laser_or_other": "Laser/other specimen",
        "specimen_solid_tissue": "Solid tissue (n=19)",
        "reads_lt5m": "Reads <5M",
        "reads_ge5m": "Reads >=5M",
    }
    forest = subgroups.copy()
    forest["label"] = forest["subgroup"].map(label_map)
    forest = forest.sort_values("hr_per_sd")
    yy = np.arange(len(forest))
    for i, row in enumerate(forest.itertuples()):
        color = PALETTE["orange"] if row.n < 25 else PALETTE["navy"]
        ax["d"].plot([row.hr_ci_low, row.hr_ci_high], [i, i], color=color, lw=1.3)
        ax["d"].scatter(row.hr_per_sd, i, s=28, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    ax["d"].axvline(1, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["d"].set_yticks(yy, forest["label"])
    ax["d"].set_xlabel("Hazard ratio per SD (95% CI)")
    ax["d"].set_xscale("log")
    ax["d"].set_xlim(0.75, 5.0)
    ax["d"].set_xticks([1, 2, 4], ["1", "2", "4"])
    ax["d"].xaxis.set_minor_locator(NullLocator())
    clean_axes(ax["d"], "x")
    panel_title(ax["d"], "d", "Prespecified subgroup associations")

    uno = subgroups.copy()
    uno["label"] = uno["subgroup"].map(label_map)
    uno = uno.sort_values("uno_c")
    yy = np.arange(len(uno))
    ax["e"].hlines(yy, 0.5, uno["uno_c"], color=PALETTE["light"], lw=1.7)
    ax["e"].scatter(uno["uno_c"], yy, s=32, color=[PALETTE["orange"] if n < 25 else PALETTE["teal"] for n in uno["n"]], edgecolor="white", linewidth=0.4)
    ax["e"].axvline(0.5, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["e"].set_yticks(yy, uno["label"])
    ax["e"].set_xlim(0.54, 0.73)
    ax["e"].set_xlabel("Uno C-index")
    clean_axes(ax["e"], "x")
    panel_title(ax["e"], "e", "Subgroup discrimination")

    conditions = gate["readiness_conditions"]
    short = [
        ("Fresh external gate", conditions["fresh independent validation is green"]),
        ("Bootstrap lower bound", conditions["bootstrap lower 95% bound for v4 Uno C >= 0.55"]),
        ("Point gain over v2", conditions["v4 Uno gain over frozen v2 predecessor >= 0.01"]),
        ("Secure gain over v2", conditions["bootstrap probability v4 Uno > v2 Uno >= 0.80"]),
        ("Combined clinical gain", conditions["combined Uno gain over cross-fitted clinical model >= 0.015"]),
        ("Secure combined gain", conditions["bootstrap probability combined Uno > clinical Uno >= 0.80"]),
    ]
    yy = np.arange(len(short))[::-1]
    for yi, (name, passed) in zip(yy, short):
        ax["f"].plot([0, 1], [yi, yi], color=PALETTE["light"], lw=0.8)
        ax["f"].scatter(1 if passed else 0, yi, s=64, color=PALETTE["teal"] if passed else PALETTE["orange"], edgecolor="white", linewidth=0.5, zorder=3)
        ax["f"].text((1 if passed else 0) + 0.08, yi, "pass" if passed else "not met", va="center", fontsize=7.5)
    ax["f"].set_yticks(yy, [x[0] for x in short])
    ax["f"].set_xticks([0, 1], ["Not met", "Met"])
    ax["f"].set_xlim(-0.25, 1.35)
    ax["f"].tick_params(axis="y", length=0)
    clean_axes(ax["f"])
    panel_title(ax["f"], "f", "Top-tier readiness remained amber")

    return save_figure(
        fig,
        FIGURES,
        "Figure6_clinical_integration",
        "Clinical integration and subgroup consistency",
        ["Model discrimination in PACA-CA", "Incremental discrimination", "Complementary clinical information", "Prespecified subgroup associations", "Subgroup discrimination", "Top-tier readiness remained amber"],
        ["interval plot", "violin and interval plot", "outcome scatter", "forest plot", "lollipop plot", "status dot plot"],
    )


def build_figure7() -> dict:
    gse_pred = read_csv("44_v4_gse77435_predictions.csv")
    roc = read_csv("44_v4_gse77435_roc_curve.csv")
    gse_boot = read_csv("44_v4_gse77435_bootstrap.csv")
    prince = read_csv("48_v4_prince_predictions.csv")
    prince_boot = read_csv("48_v4_prince_bootstrap.csv")
    prince_gate = read_json("48_v4_prince_exploratory_gate.json")
    fig, ax = mosaic((0.86, 1.04, 1.10))

    off_axes(ax["a"])
    panel_title(ax["a"], "a", "Locked population spectrum")
    # Population roles are ordered along one axis; the role labels carry the
    # interpretation while the markers carry the cohort sizes and outcomes.
    ax["a"].plot([0.13, 0.87], [0.55, 0.55], color=PALETTE["mid"], lw=1.5, zorder=1)
    spectrum = [
        (0.16, "PACA-CA", "primary\nuntreated", "n=168 · 136 events", "VALIDATED", PALETTE["navy"]),
        (0.50, "GSE77435", "extreme\nsurvival", "n=24 · binary", "SUPPORTIVE", PALETTE["orange"]),
        (0.84, "PRINCE", "treated\nmetastatic", "n=63 · 15 events", "REVERSED", PALETTE["coral"]),
    ]
    for step, (x, cohort, role, detail, status, color) in enumerate(spectrum, start=1):
        ax["a"].add_patch(Circle((x, 0.55), 0.070, facecolor="white", edgecolor=color, lw=2.0, zorder=2))
        ax["a"].text(x, 0.55, str(step), ha="center", va="center", fontsize=10.0, weight="bold", color=color)
        ax["a"].text(x, 0.79, cohort, ha="center", va="center", fontsize=7.2, weight="bold", color=color)
        ax["a"].text(x, 0.70, role, ha="center", va="center", fontsize=6.7, color=PALETTE["dark"])
        ax["a"].text(x, 0.35, detail, ha="center", va="center", fontsize=7.0, color=PALETTE["dark"])
        ax["a"].text(x, 0.17, status, ha="center", va="center", fontsize=7.0, weight="bold", color=color)

    ax["b"].plot([0, 1], [0, 1], color=PALETTE["grey"], lw=0.8, ls="--")
    ax["b"].plot(roc["false_positive_rate"], roc["true_positive_rate"], color=PALETTE["navy"], lw=1.8)
    ax["b"].fill_between(roc["false_positive_rate"], roc["true_positive_rate"], roc["false_positive_rate"], color=PALETTE["pale_blue"], alpha=0.70)
    ax["b"].text(0.58, 0.18, "AUC=0.650\npermutation P=0.112", fontsize=8.0)
    ax["b"].set_xlim(0, 1)
    ax["b"].set_ylim(0, 1)
    ax["b"].set_xlabel("False-positive rate")
    ax["b"].set_ylabel("True-positive rate")
    clean_axes(ax["b"])
    panel_title(ax["b"], "b", "GSE77435 extreme-survival ROC")

    group_order = ["long", "short"]
    sns.violinplot(data=gse_pred, x="SURVIVAL_GROUP", y="v4_marginrank_risk", order=group_order, palette=[PALETTE["pale_teal"], PALETTE["pale_coral"]], inner=None, cut=0, linewidth=0, ax=ax["c"])
    for j, group in enumerate(group_order):
        values = gse_pred.loc[gse_pred["SURVIVAL_GROUP"] == group, "v4_marginrank_risk"].to_numpy()
        ax["c"].scatter(j + np.array(stable_jitter(len(values), 0.10)), values, s=30, color=PALETTE["teal"] if group == "long" else PALETTE["coral"], edgecolor="white", linewidth=0.45, alpha=0.9)
        ax["c"].plot([j - 0.14, j + 0.14], [np.median(values)] * 2, color=PALETTE["dark"], lw=1.4)
    ax["c"].set_xticks([0, 1], ["Long >20 months", "Short <10 months"])
    ax["c"].set_xlabel("")
    ax["c"].set_ylabel("Frozen raw risk")
    clean_axes(ax["c"], "y")
    panel_title(ax["c"], "c", "GSE77435 direction was supportive")

    for event, marker, label, color in [(1, "o", "Death", PALETTE["coral"]), (0, "^", "Censored", PALETTE["blue"])]:
        subset = prince[prince["OS_EVENT"] == event]
        ax["d"].scatter(subset["v4_risk"], subset["OS_TIME"], marker=marker, s=34, color=color, alpha=0.75, edgecolor="white", linewidth=0.4, label=label)
    slope, intercept = np.polyfit(prince["v4_risk"], prince["OS_TIME"], 1)
    xline = np.linspace(prince["v4_risk"].min(), prince["v4_risk"].max(), 100)
    ax["d"].plot(xline, intercept + slope * xline, color=PALETTE["coral"], lw=1.2)
    rho, p = stats.spearmanr(prince["v4_risk"], prince["OS_TIME"])
    ax["d"].text(0.04, 0.94, f"Spearman rho={rho:.2f}\n{format_p(p)}", transform=ax["d"].transAxes, va="top", fontsize=7.8)
    ax["d"].set_xlabel("Frozen raw risk")
    ax["d"].set_ylabel("Observed survival (months)")
    ax["d"].legend(frameon=False)
    clean_axes(ax["d"], "both")
    panel_title(ax["d"], "d", "PRINCE showed inverse transport")

    prince_cols = ["v4_risk_uno", "v2_risk_uno", "legacy_risk_uno"]
    prince_labels = ["MarginRank-DG", "Frozen v2", "Legacy score"]
    prince_colors = [PALETTE["coral"], PALETTE["blue"], PALETTE["grey"]]
    for column, label, color in zip(prince_cols, prince_labels, prince_colors):
        sns.kdeplot(prince_boot[column].dropna(), ax=ax["e"], color=color, fill=True, alpha=0.13, linewidth=1.4, label=label)
    ax["e"].axvline(0.5, color=PALETTE["dark"], lw=0.8, ls="--")
    ax["e"].axvline(prince_gate["comparators"]["v4_risk"]["uno_c"] if "comparators" in prince_gate else 0.338, color=PALETTE["coral"], lw=1.0)
    ax["e"].set_xlim(0.05, 0.82)
    ax["e"].set_xlabel("Bootstrap Uno C-index")
    ax["e"].set_ylabel("Density")
    ax["e"].legend(frameon=False)
    clean_axes(ax["e"], "y")
    panel_title(ax["e"], "e", "PRINCE uncertainty preserved reversal")

    off_axes(ax["f"])
    panel_title(ax["f"], "f", "Intended-use boundary")
    ax["f"].set_xlim(0, 1)
    ax["f"].set_ylim(0, 1)
    ax["f"].add_patch(Rectangle((0.08, 0.12), 0.48, 0.34, facecolor=PALETTE["pale_teal"], edgecolor="none", alpha=0.78))
    ax["f"].add_patch(Rectangle((0.58, 0.46), 0.34, 0.34, facecolor=PALETTE["pale_coral"], edgecolor="none", alpha=0.70))
    ax["f"].scatter([0.28, 0.75], [0.29, 0.63], s=[90, 90], color=[PALETTE["teal"], PALETTE["coral"]], edgecolor="white", linewidth=0.8, zorder=3)
    ax["f"].text(0.28, 0.40, "IN SCOPE", ha="center", va="center", fontsize=7.2, weight="bold", color=PALETTE["teal"])
    ax["f"].text(0.75, 0.84, "OUT OF SCOPE", ha="center", va="center", fontsize=7.2, weight="bold", color=PALETTE["coral"])
    ax["f"].text(0.28, 0.20, "untreated primary / resectable\nrelative-risk research", ha="center", va="top", fontsize=6.9, color=PALETTE["dark"])
    ax["f"].text(0.75, 0.54, "treated metastatic\nabsolute survival / treatment", ha="center", va="top", fontsize=6.9, color=PALETTE["dark"])
    ax["f"].set_xlabel("Primary untreated  →  treated metastatic")
    ax["f"].set_ylabel("Relative-risk claim  →  treatment claim")
    ax["f"].text(0.50, 0.04, "PRINCE failure narrows, not expands, the claim", ha="center", fontsize=7.5, weight="bold")
    clean_axes(ax["f"], "both")

    return save_figure(
        fig,
        FIGURES,
        "Figure7_external_stress_scope",
        "External stress tests define scope",
        ["Locked population spectrum", "GSE77435 extreme-survival ROC", "GSE77435 direction was supportive", "PRINCE showed inverse transport", "PRINCE uncertainty preserved reversal", "Intended-use boundary"],
        ["population-role axis", "ROC curve", "violin and strip plot", "outcome scatter and trend", "density plot", "intended-use map"],
    )


def build_figure8() -> dict:
    full_rank = read_csv("46_v4_full_gene_survival_rank.csv")
    pathways = read_csv("46_v4_ranked_pathway_enrichment.csv")
    coefficients = read_csv("39_v4_final_feature_coefficients.csv")
    direct = read_csv("45_v4_gprofiler_enrichment.csv")
    fig, ax = mosaic((0.96, 1.06, 0.98))

    ranked = full_rank.sort_values("signed_stability").reset_index(drop=True)
    x = np.linspace(0, 100, len(ranked))
    colors = np.where(ranked["signed_stability"] >= 0, PALETTE["coral"], PALETTE["teal"])
    ax["a"].scatter(x, ranked["signed_stability"], s=3.5, c=colors, alpha=0.45, linewidth=0)
    ax["a"].axhline(0, color=PALETTE["grey"], lw=0.7)
    for idx, offset, align in [
        (int(ranked["signed_stability"].idxmin()), (4, 5), "left"),
        (int(ranked["signed_stability"].idxmax()), (-4, 5), "right"),
    ]:
        row = ranked.loc[idx]
        ax["a"].annotate(row["gene"], (x[idx], row["signed_stability"]), xytext=offset, textcoords="offset points", fontsize=7.0, ha=align)
    ax["a"].set_xlabel("Genome-wide rank percentile")
    ax["a"].set_ylabel("Signed cross-domain stability")
    clean_axes(ax["a"], "y")
    panel_title(ax["a"], "a", "Genome-wide survival stability", letter_x=-0.22)

    desired = [
        ("adverse", "Cell Cycle Checkpoints"),
        ("adverse", "S Phase"),
        ("adverse", "Synthesis of DNA"),
        ("adverse", "Cell cycle"),
        ("favourable", "Valine, leucine and isoleucine degradation"),
        ("favourable", "Fatty acid degradation"),
        ("favourable", "Pyruvate metabolism"),
        ("favourable", "Pancreatic secretion"),
    ]
    rows = []
    for direction, name in desired:
        matches = pathways[(pathways["direction"] == direction) & (pathways["name"] == name) & (pathways["filtered_significant"] == True)]
        if not matches.empty:
            rows.append(matches.sort_values("p_value").iloc[0])
    top_terms = pd.DataFrame(rows)
    top_terms["score"] = -np.log10(top_terms["p_value"].clip(lower=1e-300))
    top_terms["short_name"] = top_terms["name"].replace({"Valine, leucine and isoleucine degradation": "Branched-chain amino acid degradation"})
    top_terms = top_terms.sort_values("score")
    yy = np.arange(len(top_terms))
    term_colors = [PALETTE["coral"] if d == "adverse" else PALETTE["teal"] for d in top_terms["direction"]]
    ax["b"].hlines(yy, 0, top_terms["score"], color=term_colors, lw=1.6)
    ax["b"].scatter(top_terms["score"], yy, s=24 + 2.5 * top_terms["intersection_size"].clip(upper=80), color=term_colors, edgecolor="white", linewidth=0.5)
    ax["b"].set_yticks(yy, top_terms["short_name"])
    ax["b"].set_xlabel("-log10 enrichment P")
    clean_axes(ax["b"], "x")
    panel_title(ax["b"], "b", "Cross-domain pathway context")

    filtered = pathways[(pathways["filtered_significant"] == True) & (pathways["cross_domain_robust"] == True)].copy()
    counts = filtered.groupby(["source", "direction"]).size().unstack(fill_value=0)
    source_order = counts.sum(axis=1).sort_values(ascending=False).index.tolist()
    counts = counts.loc[source_order]
    y = np.arange(len(counts))
    adverse = counts.get("adverse", pd.Series(0, index=counts.index))
    favourable = counts.get("favourable", pd.Series(0, index=counts.index))
    ax["c"].barh(y, -favourable, color=PALETTE["teal"], label="Favourable")
    ax["c"].barh(y, adverse, color=PALETTE["coral"], label="Adverse")
    ax["c"].axvline(0, color=PALETTE["dark"], lw=0.7)
    ax["c"].set_yticks(y, counts.index)
    ax["c"].xaxis.set_major_formatter(lambda value, pos: f"{abs(int(value))}")
    ax["c"].set_xlabel("Robust enriched terms")
    ax["c"].legend(frameon=False, loc="lower right")
    clean_axes(ax["c"], "x")
    panel_title(ax["c"], "c", "Evidence across annotation sources")

    selected_rank = coefficients.merge(full_rank[["gene", "signed_stability"]], on="gene")
    ax["d"].scatter(selected_rank["signed_stability"], selected_rank["coefficient"], s=28 + 28 * selected_rank["directional_consistency"], color=[PALETTE["coral"] if x > 0 else PALETTE["teal"] for x in selected_rank["signed_stability"]], alpha=0.78, edgecolor="white", linewidth=0.45)
    ax["d"].axhline(0, color=PALETTE["grey"], lw=0.7)
    ax["d"].axvline(0, color=PALETTE["grey"], lw=0.7)
    label_offsets = {"IL1RAP": (3, 3), "UBA6": (3, 3), "MIA3": (3, -2), "RALGPS1": (3, 7)}
    for gene, offset in label_offsets.items():
        row = selected_rank.loc[selected_rank["gene"] == gene].iloc[0]
        ax["d"].annotate(gene, (row["signed_stability"], row["coefficient"]), xytext=offset, textcoords="offset points", fontsize=6.9)
    ax["d"].set_xlabel("Signed cross-domain stability")
    ax["d"].set_ylabel("Frozen coefficient")
    clean_axes(ax["d"], "both")
    panel_title(ax["d"], "d", "Selected genes in global context")

    direct_plot = direct.sort_values("p_value", ascending=False).copy()
    direct_plot["score"] = -np.log10(direct_plot["p_value"])
    direct_plot["label"] = direct_plot["name"].replace({"Ascorbate and aldarate metabolism": "Ascorbate metabolism"})
    colors = [PALETTE["orange"] if q == "all_selected" else PALETTE["purple"] for q in direct_plot["query"]]
    yy = np.arange(len(direct_plot))
    ax["e"].barh(yy, direct_plot["score"], color=colors, height=0.62)
    ax["e"].axvline(-math.log10(0.05), color=PALETTE["grey"], lw=0.8, ls="--")
    ax["e"].set_yticks(yy, direct_plot["label"])
    ax["e"].set_xlabel("-log10 adjusted P")
    ax["e"].text(0.98, 0.05, "orange = all 40 genes", transform=ax["e"].transAxes, ha="right", fontsize=7.2)
    clean_axes(ax["e"], "x")
    panel_title(ax["e"], "e", "Direct selected-gene ORA was sparse")

    off_axes(ax["f"])
    panel_title(ax["f"], "f", "Contextual program model")
    # Program-level evidence converges on a relative-risk interpretation.  The
    # bands preserve direction and pathway content while avoiding two competing
    # boxes that could be mistaken for a causal mechanistic diagram.
    ax["f"].plot([0.10, 0.43], [0.69, 0.69], color=PALETTE["coral"], alpha=0.70, lw=11, solid_capstyle="round")
    ax["f"].plot([0.10, 0.43], [0.31, 0.31], color=PALETTE["teal"], alpha=0.70, lw=11, solid_capstyle="round")
    arrow(ax["f"], (0.45, 0.69), (0.66, 0.58), PALETTE["coral"])
    arrow(ax["f"], (0.45, 0.31), (0.66, 0.42), PALETTE["teal"])
    ax["f"].add_patch(Circle((0.75, 0.50), 0.095, facecolor="white", edgecolor=PALETTE["navy"], lw=2.0))
    ax["f"].text(0.75, 0.50, "relative-risk\ncontext", ha="center", va="center", fontsize=6.9, weight="bold", color=PALETTE["navy"])
    ax["f"].text(0.10, 0.84, "ADVERSE", ha="left", va="center", fontsize=7.5, weight="bold", color=PALETTE["coral"])
    ax["f"].text(0.10, 0.78, "Cell-cycle checkpoints · DNA synthesis · mitotic programs", ha="left", va="center", fontsize=6.8, color=PALETTE["dark"])
    ax["f"].text(0.10, 0.16, "FAVOURABLE", ha="left", va="center", fontsize=7.5, weight="bold", color=PALETTE["teal"])
    ax["f"].text(0.10, 0.10, "Fatty-acid oxidation · pyruvate metabolism · pancreatic secretion", ha="left", va="center", fontsize=6.8, color=PALETTE["dark"])
    ax["f"].text(0.50, 0.02, "Development context only · no causal mechanism claimed", ha="center", fontsize=7.1, weight="bold")

    return save_figure(
        fig,
        FIGURES,
        "Figure8_biological_context",
        "Cross-domain biological context",
        ["Genome-wide survival stability", "Cross-domain pathway context", "Evidence across annotation sources", "Selected genes in global context", "Direct selected-gene ORA was sparse", "Contextual program model"],
        ["ranked scatter", "enrichment lollipop", "diverging bar", "coefficient scatter", "horizontal bar", "converging program bands"],
    )


def build_figure9() -> dict:
    pseudobulk = read_csv("45_gse263733_pseudobulk_scores.csv")
    summary = read_csv("45_gse263733_celltype_summary.csv")
    dotplot = read_csv("45_gse263733_celltype_dotplot.csv")
    ductal = read_csv("45_gse263733_ductal_subtype_scores.csv")
    matched = read_csv("45_gse263733_matched_ductal.csv")
    coefficients = read_csv("39_v4_final_feature_coefficients.csv")
    software = read_json("49_software_release_gate.json")
    fig, ax = mosaic((0.88, 1.15, 0.97))

    off_axes(ax["a"])
    panel_title(ax["a"], "a", "Single-cell contextualization")
    # A straight provenance line is easier to read than a four-corner flowchart
    # and makes the no-refit constraint part of the visual grammar.
    xs = [0.12, 0.38, 0.64, 0.88]
    ax["a"].plot(xs, [0.56] * len(xs), color=PALETTE["mid"], lw=1.5, zorder=1)
    stages = [
        ("CELLS", "69,204", PALETTE["blue"]),
        ("PATIENTS", "21 · 7 types", PALETTE["teal"]),
        ("P-BULK", "137 origin", PALETTE["purple"]),
        ("PROGRAM", "40 genes", PALETTE["orange"]),
    ]
    for step, (x, (heading, detail, color)) in enumerate(zip(xs, stages), start=1):
        ax["a"].add_patch(Circle((x, 0.56), 0.062, facecolor="white", edgecolor=color, lw=2.0, zorder=2))
        ax["a"].text(x, 0.56, str(step), ha="center", va="center", fontsize=9.2, weight="bold", color=color)
        ax["a"].text(x, 0.76, heading, ha="center", va="center", fontsize=6.4, weight="bold", color=color)
        ax["a"].text(x, 0.36, detail, ha="center", va="center", fontsize=6.8, color=PALETTE["dark"])
    ax["a"].text(0.50, 0.16, "Contextual score transfer only · survival model never refit", ha="center", fontsize=7.4, weight="bold")

    order = summary.sort_values("median_risk_program")["Celltypes"].tolist()
    sns.violinplot(data=pseudobulk, x="risk_program", y="Celltypes", order=order, color=PALETTE["pale_purple"], inner=None, cut=0, linewidth=0, ax=ax["b"])
    sns.boxplot(data=pseudobulk, x="risk_program", y="Celltypes", order=order, width=0.18, color=PALETTE["purple"], showfliers=False, ax=ax["b"])
    ax["b"].axvline(0, color=PALETTE["grey"], lw=0.8, ls="--")
    ax["b"].set_xlabel("Signed program score")
    ax["b"].set_ylabel("")
    clean_axes(ax["b"], "x")
    panel_title(ax["b"], "b", "Program heterogeneity by cell type")

    genes = coefficients.nlargest(12, "absolute_coefficient")["gene"].tolist()
    cell_order = summary.sort_values("median_risk_program", ascending=False)["Celltypes"].tolist()
    dots = dotplot[dotplot["gene"].isin(genes)].copy()
    gene_order = coefficients.set_index("gene").loc[genes].sort_values("coefficient")["coefficient"].index.tolist()
    x_map = {cell: i for i, cell in enumerate(cell_order)}
    y_map = {gene: i for i, gene in enumerate(gene_order)}
    scatter = ax["c"].scatter(dots["cell_type"].map(x_map), dots["gene"].map(y_map), s=10 + 115 * dots["fraction_expressing"], c=dots["mean_logcp10k"], cmap=sns.light_palette(PALETTE["purple"], as_cmap=True), edgecolor=PALETTE["mid"], linewidth=0.25)
    ax["c"].set_xticks(range(len(cell_order)), [x.replace(" cell", "") for x in cell_order], rotation=50, ha="right")
    ax["c"].set_yticks(range(len(gene_order)), gene_order)
    ax["c"].set_xlim(-0.6, len(cell_order) - 0.4)
    ax["c"].set_ylim(-0.6, len(gene_order) - 0.4)
    ax["c"].tick_params(length=0)
    for spine in ax["c"].spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax["c"], fraction=0.035, pad=0.02)
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
    cbar.set_label("Mean log(CP10K+1)", fontsize=7.6)
    panel_title(ax["c"], "c", "Selected-gene cellular localization")

    ax["d"].scatter(ductal["basal_minus_classical"], ductal["risk_program"], s=24 + 0.03 * ductal["n_cells"].clip(upper=2000), color=PALETTE["navy"], alpha=0.72, edgecolor="white", linewidth=0.4)
    slope, intercept = np.polyfit(ductal["basal_minus_classical"], ductal["risk_program"], 1)
    xx = np.linspace(ductal["basal_minus_classical"].min(), ductal["basal_minus_classical"].max(), 100)
    ax["d"].plot(xx, intercept + slope * xx, color=PALETTE["coral"], lw=1.2)
    rho, p = stats.spearmanr(ductal["basal_minus_classical"], ductal["risk_program"])
    ax["d"].text(0.04, 0.94, f"rho={rho:.3f}\n{format_p(p)}", transform=ax["d"].transAxes, va="top", fontsize=7.8)
    ax["d"].set_xlabel("Basal minus classical score")
    ax["d"].set_ylabel("Risk-program score")
    clean_axes(ax["d"], "both")
    panel_title(ax["d"], "d", "Ductal basal-state alignment")

    pair = matched.sort_values("delta_lm_minus_pm1")
    yy = np.arange(len(pair))
    for i, row in enumerate(pair.itertuples()):
        color = PALETTE["coral"] if row.delta_lm_minus_pm1 > 0 else PALETTE["teal"]
        ax["e"].plot([row.Pm1, row.Lm], [i, i], color=PALETTE["light"], lw=2.0)
        ax["e"].scatter(row.Pm1, i, color=PALETTE["orange"], marker="o", s=32, edgecolor="white", linewidth=0.4, zorder=3)
        ax["e"].scatter(row.Lm, i, color=color, marker="D", s=32, edgecolor="white", linewidth=0.4, zorder=3)
    ax["e"].axvline(0, color=PALETTE["grey"], lw=0.7, ls="--")
    ax["e"].set_yticks(yy, pair["Patients"])
    ax["e"].set_xlabel("Ductal program score")
    ax["e"].legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["orange"], label="Primary Pm1"), Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["teal"], label="Liver Lm")], frameon=False, loc="lower right")
    clean_axes(ax["e"], "x")
    panel_title(ax["e"], "e", "Matched metastasis hypothesis failed")

    off_axes(ax["f"])
    panel_title(ax["f"], "f", "Reproducible software translation")
    ax["f"].plot([0.10, 0.90], [0.60, 0.60], color=PALETTE["mid"], lw=1.5, zorder=1)
    pipeline = [
        (0.12, "EXPR", "HGNC", PALETTE["blue"]),
        (0.38, "RANK", "transsurv", PALETTE["purple"]),
        (0.66, "RISK", "risk · QC", PALETTE["teal"]),
        (0.88, "RELEASE", "versioned", PALETTE["orange"]),
    ]
    for step, (x, heading, detail, color) in enumerate(pipeline, start=1):
        ax["f"].add_patch(Circle((x, 0.60), 0.058, facecolor="white", edgecolor=color, lw=2.0, zorder=2))
        ax["f"].text(x, 0.60, str(step), ha="center", va="center", fontsize=9.0, weight="bold", color=color)
        ax["f"].text(x, 0.77, heading, ha="center", va="center", fontsize=6.5, weight="bold", color=color)
        ax["f"].text(x, 0.42, detail, ha="center", va="center", fontsize=6.5, color=PALETTE["dark"])
    gates = [
        ("28 tests\npassed", software["conditions"]["all_tests_pass"]),
        ("82.98% line\ncoverage", software["conditions"]["test_coverage_at_least_80_percent"]),
        ("clean wheel\ninstall", software["conditions"]["clean_cli_scores_all_samples"]),
        ("no absolute\nsurvival", software["conditions"]["absolute_survival_is_disabled"]),
    ]
    xs = np.linspace(0.12, 0.88, len(gates))
    for x, (label, passed) in zip(xs, gates):
        color = PALETTE["teal"] if passed else PALETTE["coral"]
        ax["f"].scatter(x, 0.20, s=65, color=color, edgecolor="white", linewidth=0.5)
        ax["f"].text(x, 0.08, label, ha="center", fontsize=6.6, color=PALETTE["dark"])

    return save_figure(
        fig,
        FIGURES,
        "Figure9_single_cell_software",
        "Single-cell context and software translation",
        ["Single-cell contextualization", "Program heterogeneity by cell type", "Selected-gene cellular localization", "Ductal basal-state alignment", "Matched metastasis hypothesis failed", "Reproducible software translation"],
        ["minimalist provenance flow", "violin and box plot", "expression dot plot", "correlation scatter", "paired dumbbell", "minimalist software pipeline"],
    )


def main() -> int:
    configure_style()
    builders = [
        build_figure1,
        build_figure2,
        build_figure3,
        build_figure4,
        build_figure5,
        build_figure6,
        build_figure7,
        build_figure8,
        build_figure9,
    ]
    records = []
    for index, builder in enumerate(builders, start=1):
        print(f"Building Figure {index}...", flush=True)
        record = builder()
        records.append(record)
        print(f"Figure {index} complete: {record['files']['pdf']}", flush=True)
    manifest = {
        "status": "NINE_MAIN_FIGURES_BUILT",
        "source": "vector PDF; PNG and TIFF rasterized from PDF",
        "font": "Helvetica PDF core font resources",
        "raster_dpi": 450,
        "palette": PALETTE,
        "global_conditions": {
            "nine_main_figures": len(records) == 9,
            "four_to_nine_panels_each": all(4 <= record["panel_count"] <= 9 for record in records),
            "at_least_three_plot_types_each": all(record["plot_type_count"] >= 3 for record in records),
            "vector_pdf_each": all(Path(record["files"]["pdf"]).exists() for record in records),
            "raster_from_pdf_each": all(Path(record["files"]["png"]).exists() and Path(record["files"]["tiff"]).exists() for record in records),
        },
        "figures": records,
        "claim_governance": "Negative and failed results are retained; visual design did not change endpoints, models, or cohort roles.",
    }
    write_manifest(FIGURES / "main_figure_manifest.json", manifest)
    if not all(manifest["global_conditions"].values()):
        print(json.dumps(manifest["global_conditions"], indent=2))
        return 1
    print(json.dumps(manifest["global_conditions"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
