from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import auc as roc_auc
from sklearn.metrics import roc_curve


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
}

MODEL_COLORS = {
    "MarginRank-DG": PALETTE["navy"],
    "Frozen v2": PALETTE["blue"],
    "Legacy score": PALETTE["grey"],
    "Clinical OOF": PALETTE["orange"],
    "Combined OOF": PALETTE["teal"],
}


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def read_json(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def mosaic():
    return plt.subplot_mosaic(
        [["a", "b"], ["c", "d"]],
        figsize=(7.2, 6.45),
        gridspec_kw={"hspace": 0.56, "wspace": 0.48},
    )


def violin_with_points(ax, groups, labels, colors, ylabel, reference=None):
    positions = np.arange(len(groups))
    vp = ax.violinplot(groups, positions=positions, widths=0.72, showextrema=False)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.23)
    for x, (values, color) in enumerate(zip(groups, colors)):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.vlines(x, q1, q3, color=color, lw=5.0, zorder=3)
        ax.scatter(x, med, s=29, color=PALETTE["white"], edgecolor=color, linewidth=1.2, zorder=4)
        if len(values) <= 60:
            jitter = np.asarray(stable_jitter(len(values), 0.12))
            ax.scatter(x + jitter, values, s=7, color=color, alpha=0.35, linewidth=0, zorder=2)
    if reference is not None:
        ax.axhline(reference, color=PALETTE["mid"], lw=0.9, ls="--")
    ax.set_xticks(positions, labels)
    ax.set_ylabel(ylabel)
    clean_axes(ax, "y")


def forest(ax, frame, label_col, point_col, low_col, high_col, color, xlabel, reference=1.0):
    data = frame.reset_index(drop=True)
    y = np.arange(len(data))[::-1]
    ax.axvline(reference, color=PALETTE["mid"], lw=0.9, ls="--")
    x = data[point_col].to_numpy(float)
    low = data[low_col].to_numpy(float)
    high = data[high_col].to_numpy(float)
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([x - low, high - x]),
        fmt="o",
        color=color,
        ecolor=color,
        elinewidth=1.1,
        capsize=2.2,
        markersize=4.8,
    )
    ax.set_yticks(y, data[label_col])
    ax.set_xlabel(xlabel)
    clean_axes(ax, "x")


def short_candidate(name: str) -> str:
    tokens = name.split("__")
    left = tokens[0].replace("contrast", "pair").replace("hybrid", "hyb")
    learner = tokens[1].replace("pooled_ridge", "ridge").replace("expert_median", "median") if len(tokens) > 1 else ""
    suffix = tokens[2] if len(tokens) > 2 else ""
    return f"{left} | {learner} | {suffix}"


def short_pathway(name: str) -> str:
    if "Endometrium; stromal fibroblasts" in name:
        level = "medium" if "Medium" in name else "low"
        return f"Endometrium fibroblasts ({level})"
    if "Colon; fibroblasts" in name:
        return "Colon fibroblasts (high)"
    if name == "Valine, leucine and isoleucine degradation":
        return "BCAA degradation"
    if name == "Leucine isoleucine and valine metabolism":
        return "BCAA metabolism"
    if name == "Mitotic G1 phase and G1/S transition":
        return "Mitotic G1 and G1/S transition"
    return name[:34]


def build_s1() -> dict:
    fig, ax = mosaic()

    v1 = read_csv("03_real_loso_fixed_metrics.csv")
    v1 = v1[v1["model"] == "rank_transsurv"]
    gates = [
        ("v1 rank-pair", float(v1["uno_c"].mean()), float(v1["uno_c"].min()), "FAIL"),
        ("v2.1 BatchScale", read_json("18_v2_1_development_gate.json")["mean_uno_c"], read_json("18_v2_1_development_gate.json")["minimum_uno_c"], "PASS"),
        ("v3.0 CalibraGuard", read_json("26_v3_development_gate.json")["mean_uno_c"], read_json("26_v3_development_gate.json")["minimum_uno_c"], "FAIL"),
        ("v3.2 ten-domain", read_json("30_v3_2_development_gate.json")["mean_uno_c"], read_json("30_v3_2_development_gate.json")["minimum_uno_c"], "FAIL"),
        ("v4 MarginRank-DG", read_json("37_v4_fixed_selection_gate.json")["mean_uno_c"], read_json("37_v4_fixed_selection_gate.json")["minimum_uno_c"], "PASS"),
    ]
    names, means, worsts, status = zip(*gates)
    y = np.arange(len(names))[::-1]
    for yi, mean, worst, gate_status in zip(y, means, worsts, status):
        color = PALETTE["teal"] if gate_status == "PASS" else PALETTE["coral"]
        ax["a"].plot([worst, mean], [yi, yi], color=PALETTE["light"], lw=3.0)
        ax["a"].scatter(worst, yi, marker="|", s=75, color=color, linewidth=1.5)
        ax["a"].scatter(mean, yi, s=34, color=color, edgecolor=PALETTE["white"], linewidth=0.7, zorder=3)
        ax["a"].text(mean + 0.002, yi, f"{mean:.3f}", va="center", fontsize=7.2)
    ax["a"].set_yticks(y, names)
    ax["a"].set_xlabel("Development Uno C (bar = worst to mean)")
    ax["a"].set_xlim(0.54, 0.65)
    ax["a"].axvline(0.60, color=PALETTE["mid"], ls="--", lw=0.9)
    clean_axes(ax["a"], "x")
    panel_title(ax["a"], "a", "Algorithmic progression")

    cal_rows = []
    for cohort, slope in read_json("14_external_gate.json")["calibration_slopes"].items():
        cal_rows.append(("v1 external", cohort, slope))
    for cohort, slope in read_json("24_v2_confirmation_gate.json")["calibration_slopes"].items():
        cal_rows.append(("v2 confirmation", cohort, slope))
    v3_cal = read_csv("26_v3_nested_outer_calibration.csv")
    for row in v3_cal.itertuples():
        cal_rows.append(("v3 development", COHORT_LABELS.get(row.cohort, row.cohort), row.point_calibration_slope))
    cal = pd.DataFrame(cal_rows, columns=["cycle", "cohort", "slope"])
    order = ["v1 external", "v2 confirmation", "v3 development"]
    colors = [PALETTE["grey"], PALETTE["blue"], PALETTE["purple"]]
    ax["b"].axhspan(0.70, 1.30, color=PALETTE["pale_teal"], zorder=0)
    ax["b"].axhline(1.0, color=PALETTE["teal"], lw=0.9)
    for x, (cycle, color) in enumerate(zip(order, colors)):
        values = cal.loc[cal["cycle"] == cycle, "slope"].to_numpy(float)
        jitter = np.asarray(stable_jitter(len(values), 0.12))
        ax["b"].scatter(x + jitter, values, s=27, color=color, alpha=0.85, edgecolor=PALETTE["white"], linewidth=0.5)
        ax["b"].scatter(x, np.median(values), marker="_", s=170, color=PALETTE["dark"], linewidth=1.4)
    ax["b"].set_xticks(range(len(order)), ["v1\nexternal", "v2\nconfirmation", "v3\ndevelopment"])
    ax["b"].set_ylabel("Calibration slope")
    ax["b"].set_ylim(0.25, 1.65)
    clean_axes(ax["b"], "y")
    panel_title(ax["b"], "b", "Calibration did not transport")

    v33 = read_json("33_v3_3_confirmation_gate.json")
    paca = read_json("41_v4_paca_ca_validation_gate.json")
    gse = read_json("44_v4_gse77435_extreme_survival_gate.json")
    prince = read_json("48_v4_prince_exploratory_gate.json")
    paca_ci = paca["bootstrap"]
    rows = [
        ("GSE224564 v3.3", v33["primary"]["uno_c"], v33["bootstrap"]["risk_uno_c_ci_low"], v33["bootstrap"]["risk_uno_c_ci_high"], "Uno C", "FAIL"),
        ("PACA-CA v4", paca["primary"]["uno_c"], paca_ci["v4_risk_uno_c_ci_low"], paca_ci["v4_risk_uno_c_ci_high"], "Uno C", "PASS"),
        ("GSE77435 v4", gse["primary"]["auc"], gse["v4_auc_ci_95"][0], gse["v4_auc_ci_95"][1], "binary AUC", "NS"),
        ("PRINCE v4", prince["primary"]["uno_c"], prince["bootstrap"]["uno_ci_95"][0], prince["bootstrap"]["uno_ci_95"][1], "Uno C", "FAIL"),
    ]
    val = pd.DataFrame(rows, columns=["label", "point", "low", "high", "metric", "status"])
    yy = np.arange(len(val))[::-1]
    ax["c"].axvline(0.5, color=PALETTE["mid"], ls="--", lw=0.9)
    for yi, row in zip(yy, val.itertuples()):
        color = PALETTE["teal"] if row.status == "PASS" else PALETTE["orange"] if row.status == "NS" else PALETTE["coral"]
        marker = "s" if row.metric == "binary AUC" else "o"
        ax["c"].errorbar(row.point, yi, xerr=[[row.point - row.low], [row.high - row.point]], fmt=marker, color=color, ecolor=color, capsize=2.4, markersize=5.0)
        ax["c"].text(row.high + 0.012, yi, row.status, va="center", fontsize=7.2, color=PALETTE["dark"])
    ax["c"].set_yticks(yy, val["label"])
    ax["c"].set_xlabel("Discrimination (95% bootstrap CI)")
    ax["c"].set_xlim(0.08, 0.92)
    clean_axes(ax["c"], "x")
    panel_title(ax["c"], "c", "One-time external results")

    off_axes(ax["d"])
    panel_title(ax["d"], "d", "Outcome-access decision log")
    stages = [
        ("v1-v2", "2 external cohorts\ncalibration failed", PALETTE["coral"]),
        ("v3.3", "GSE224564\nconsumed; red", PALETTE["coral"]),
        ("v4", "PACA-CA\nprimary; green", PALETTE["teal"]),
        ("stress", "GSE77435 NS\nPRINCE reversed", PALETTE["orange"]),
    ]
    xs = np.linspace(0.12, 0.88, len(stages))
    ax["d"].plot([xs[0], xs[-1]], [0.55, 0.55], color=PALETTE["light"], lw=2.2)
    for i, ((stage, text, color), x) in enumerate(zip(stages, xs)):
        ax["d"].scatter(x, 0.55, s=92, color=color, edgecolor=PALETTE["white"], linewidth=0.9, zorder=3)
        ax["d"].text(x, 0.73 if i % 2 == 0 else 0.35, stage, ha="center", va="center", weight="bold", fontsize=8.0)
        ax["d"].text(x, 0.86 if i % 2 == 0 else 0.20, text, ha="center", va="center", fontsize=7.3)
    ax["d"].text(0.50, 0.02, "No consumed cohort was returned to tuning", ha="center", fontsize=8.2, weight="bold")

    return save_figure(
        fig,
        FIGURES,
        "SupplementaryFigureS1_algorithm_audit",
        "Algorithmic progression and failure audit",
        ["Algorithmic progression", "Calibration did not transport", "One-time external results", "Outcome-access decision log"],
        ["lollipop", "strip plot", "forest plot", "timeline"],
    )


def build_s2() -> dict:
    fig, ax = mosaic()
    sim = read_csv("01e_simulation_strict_results.csv")
    per_seed = sim.groupby(["seed", "model"], as_index=False)["harrell_c"].mean()
    model_order = ["mean_all", "strict_stable_mean", "transsurv"]
    labels = ["All-gene\nmean", "Stable\nmean", "TransSurv\nsimulation"]
    groups = [per_seed.loc[per_seed["model"] == model, "harrell_c"].to_numpy() for model in model_order]
    violin_with_points(ax["a"], groups, labels, [PALETTE["grey"], PALETTE["blue"], PALETTE["navy"]], "Mean held-out Harrell C", 0.5)
    panel_title(ax["a"], "a", "Fresh simulation replicates")

    pivot = sim.pivot_table(index=["seed", "held_out"], columns="model", values="harrell_c").reset_index()
    gains = [
        pivot["transsurv"] - pivot["mean_all"],
        pivot["transsurv"] - pivot["strict_stable_mean"],
    ]
    colors = [PALETTE["navy"], PALETTE["purple"]]
    for values, label, color in zip(gains, ["vs all-gene mean", "vs stable mean"], colors):
        x = np.sort(values.to_numpy(float))
        y = np.arange(1, len(x) + 1) / len(x)
        ax["b"].plot(x, y, color=color, label=label)
        ax["b"].scatter(np.median(x), 0.5, s=28, color=color, edgecolor=PALETTE["white"], linewidth=0.6, zorder=3)
    ax["b"].axvline(0, color=PALETTE["mid"], ls="--", lw=0.9)
    ax["b"].set_xlabel("Held-out C-index gain")
    ax["b"].set_ylabel("Empirical cumulative fraction")
    ax["b"].legend(frameon=False, loc="lower right")
    clean_axes(ax["b"], "both")
    panel_title(ax["b"], "b", "Paired simulation gains")

    real = read_csv("03_real_loso_fixed_metrics.csv")
    model_map = {
        "rank_strict_mean": "Rank mean",
        "rank_transsurv": "Rank TransSurv",
        "cohort_z_strict_mean": "Z-score mean",
        "cohort_z_transsurv": "Z-score TransSurv",
        "legacy_locked": "Legacy",
    }
    cohort_order = ["TCGA_PAAD", "CPTAC_PDACC", "GSE62452", "GSE85916"]
    real["model_label"] = real["model"].map(model_map)
    mat = real.pivot(index="model_label", columns="held_out", values="uno_c").reindex(index=list(model_map.values()), columns=cohort_order)
    xx = np.arange(mat.shape[1] + 1) - 0.5
    yy = np.arange(mat.shape[0] + 1) - 0.5
    mesh = ax["c"].pcolormesh(xx, yy, mat.to_numpy(), cmap="Blues", vmin=0.52, vmax=0.68, shading="flat", edgecolors=PALETTE["white"], linewidth=0.8)
    ax["c"].set_ylim(mat.shape[0] - 0.5, -0.5)
    ax["c"].set_xticks(range(len(cohort_order)), [COHORT_LABELS[x] for x in cohort_order], rotation=28, ha="right")
    ax["c"].set_yticks(range(len(mat.index)), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iloc[i, j]
            ax["c"].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.0, color=PALETTE["white"] if value > 0.61 else PALETTE["dark"])
    ax["c"].tick_params(length=0)
    for spine in ax["c"].spines.values():
        spine.set_visible(False)
    panel_title(ax["c"], "c", "Early real-data LOSO benchmark")

    outer = read_csv("37_v4_selected_loso_metrics.csv").sort_values("uno_c")
    y = np.arange(len(outer))
    ax["d"].axvline(0.5, color=PALETTE["mid"], ls="--", lw=0.9)
    ax["d"].plot(outer["uno_c"], y, color=PALETTE["light"], lw=1.0)
    ax["d"].scatter(outer["uno_c"], y, s=31, color=PALETTE["navy"], label="Uno C")
    ax["d"].scatter(outer["harrell_c"], y, s=29, marker="s", facecolor=PALETTE["white"], edgecolor=PALETTE["orange"], label="Harrell C")
    ax["d"].set_yticks(y, [COHORT_LABELS.get(x, x) for x in outer["cohort"]])
    ax["d"].set_xlabel("Nested leave-domain-out concordance")
    ax["d"].set_xlim(0.48, 0.72)
    ax["d"].legend(frameon=False, loc="lower right")
    clean_axes(ax["d"], "x")
    panel_title(ax["d"], "d", "v4 ten-domain generalization")

    return save_figure(
        fig,
        FIGURES,
        "SupplementaryFigureS2_simulation_nested",
        "Simulation and nested validation",
        ["Fresh simulation replicates", "Paired simulation gains", "Early real-data LOSO benchmark", "v4 ten-domain generalization"],
        ["violin plot", "ECDF", "heatmap", "paired dot plot"],
    )


def build_s3() -> dict:
    fig, ax = mosaic()
    candidates = read_csv("37_v4_fixed_candidate_ranking.csv")
    selected = candidates["selected"].astype(bool)
    ax["a"].scatter(candidates.loc[~selected, "worst_uno"], candidates.loc[~selected, "mean_uno"], c=candidates.loc[~selected, "sd_uno"], cmap="Purples_r", s=28, alpha=0.72, edgecolor=PALETTE["white"], linewidth=0.4)
    row = candidates.loc[selected].iloc[0]
    ax["a"].scatter(row["worst_uno"], row["mean_uno"], s=82, marker="*", color=PALETTE["orange"], edgecolor=PALETTE["dark"], linewidth=0.6, zorder=4)
    ax["a"].annotate("locked v4", (row["worst_uno"], row["mean_uno"]), xytext=(-39, 13), textcoords="offset points", fontsize=7.5, arrowprops={"arrowstyle": "-", "color": PALETTE["mid"], "lw": 0.7})
    ax["a"].set_xlabel("Worst-domain Uno C")
    ax["a"].set_ylabel("Mean Uno C")
    clean_axes(ax["a"], "both")
    panel_title(ax["a"], "a", "Fixed candidate landscape")

    missing = read_csv("38_v4_missingness_replicates.csv")
    levels = sorted(missing["level"].unique())
    groups = [missing.loc[missing["level"] == level, "uno_drop"].to_numpy() for level in levels]
    violin_with_points(ax["b"], groups, [f"{int(x * 100)}%" for x in levels], [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]], "Uno C loss", 0.0)
    ax["b"].set_xlabel("Randomly masked model inputs")
    ax["b"].set_ylim(-0.12, 0.13)
    panel_title(ax["b"], "b", "Missing-gene stress")

    logo = read_csv("38_v4_leave_one_gene_out.csv")
    gene = logo.groupby("missing_gene", as_index=False).agg(mean_drop=("uno_drop", "mean"), max_drop=("uno_drop", "max"), mean_corr=("risk_correlation", "mean"))
    gene = gene.sort_values("mean_drop", ascending=False).head(15).sort_values("mean_drop")
    y = np.arange(len(gene))
    ax["c"].hlines(y, 0, gene["mean_drop"], color=PALETTE["light"], lw=2.2)
    ax["c"].scatter(gene["mean_drop"], y, s=28, color=PALETTE["coral"])
    ax["c"].scatter(gene["max_drop"], y, marker="|", s=70, color=PALETTE["navy"], linewidth=1.2)
    ax["c"].axvline(0, color=PALETTE["mid"], lw=0.8)
    ax["c"].set_yticks(y, gene["missing_gene"])
    ax["c"].set_xlabel("Uno C loss (circle mean; tick maximum)")
    clean_axes(ax["c"], "x")
    panel_title(ax["c"], "c", "Single-gene ablation")

    affine = read_csv("42_v4_affine_rank_invariance.csv")
    batch = read_csv("42_v4_batch_composition_invariance.csv")
    inv = [
        ("Positive affine", affine["difference"].abs().max()),
        ("Sample scored alone", batch["alone_difference"].abs().max()),
        ("Mixed cohort batch", batch["mixed_batch_difference"].abs().max()),
        ("Column permutation", batch["permuted_columns_difference"].abs().max()),
    ]
    inv_df = pd.DataFrame(inv, columns=["condition", "difference"])
    inv_df["plot"] = np.maximum(inv_df["difference"], 1e-18)
    y = np.arange(len(inv_df))[::-1]
    ax["d"].barh(y, inv_df["plot"], color=[PALETTE["purple"], PALETTE["blue"], PALETTE["teal"], PALETTE["orange"]], height=0.56)
    for yi, value in zip(y, inv_df["difference"]):
        ax["d"].text(max(value, 1e-18) * 1.25, yi, f"{value:.1e}", va="center", fontsize=7.2)
    ax["d"].set_yticks(y, inv_df["condition"])
    ax["d"].set_xscale("log")
    ax["d"].set_xlim(5e-19, 3e-15)
    ax["d"].set_xticks([1e-18, 1e-17, 1e-16, 1e-15], ["1e-18", "1e-17", "1e-16", "1e-15"])
    ax["d"].minorticks_off()
    ax["d"].set_xlabel("Maximum absolute risk-score difference")
    clean_axes(ax["d"], "x")
    panel_title(ax["d"], "d", "Rank-score invariance")

    return save_figure(
        fig,
        FIGURES,
        "SupplementaryFigureS3_selection_robustness",
        "Selection and technical robustness",
        ["Fixed candidate landscape", "Missing-gene stress", "Single-gene ablation", "Rank-score invariance"],
        ["scatter plot", "violin plot", "lollipop plot", "log bar plot"],
    )


def build_s4() -> dict:
    fig, ax = mosaic()
    auc = read_csv("41_v4_paca_ca_time_auc.csv")
    label_map = {
        "v4_risk": "MarginRank-DG",
        "v2_risk": "Frozen v2",
        "legacy_risk": "Legacy score",
        "clinical_oof_risk": "Clinical OOF",
        "combined_oof_risk": "Combined OOF",
    }
    for model, frame in auc.groupby("model"):
        label = label_map[model]
        ax["a"].plot(frame["time_months"], frame["auc"], marker="o", color=MODEL_COLORS[label], label=label)
    ax["a"].axhline(0.5, color=PALETTE["mid"], ls="--", lw=0.9)
    ax["a"].set_xticks([12, 24, 36])
    ax["a"].set_xlabel("Time (months)")
    ax["a"].set_ylabel("Time-dependent AUC")
    ax["a"].set_ylim(0.49, 0.78)
    ax["a"].legend(frameon=False, ncol=2, loc="lower center", fontsize=7.0)
    clean_axes(ax["a"], "both")
    panel_title(ax["a"], "a", "PACA-CA time-dependent discrimination")

    boot = read_csv("41_v4_paca_ca_bootstrap.csv")
    cols = ["delta_uno_v4_vs_v2", "delta_uno_v4_vs_legacy", "delta_uno_combined_vs_clinical", "delta_uno_combined_vs_v4"]
    labels = ["v4-v2", "v4-legacy", "combo-clinical", "combo-v4"]
    colors = [PALETTE["blue"], PALETTE["grey"], PALETTE["teal"], PALETTE["orange"]]
    violin_with_points(ax["b"], [boot[x].dropna().to_numpy() for x in cols], labels, colors, "Paired bootstrap Uno C difference", 0.0)
    ax["b"].tick_params(axis="x", labelrotation=18, labelsize=7.2)
    ax["b"].set_ylim(-0.09, 0.14)
    panel_title(ax["b"], "b", "PACA-CA paired uncertainty")

    subgroup = read_csv("41_v4_paca_ca_subgroups.csv").copy()
    rename = {
        "age_lt65": "Age <65",
        "age_ge65": "Age 65+",
        "female": "Female",
        "male": "Male",
        "stage_early_IA_IB_IIA": "Stage IA-IIA",
        "stage_advanced_IIB_III_IV": "Stage IIB-IV",
        "grade_well_or_moderate": "Grade well/moderate",
        "grade_poor_or_undifferentiated": "Grade poor/undiff.",
        "specimen_laser_or_other": "Laser/other specimen",
        "specimen_solid_tissue": "Solid tissue",
        "reads_lt5m": "Reads <5M",
        "reads_ge5m": "Reads 5M+",
    }
    subgroup["label"] = [f"{rename.get(s, s)} (n={n})" for s, n in zip(subgroup["subgroup"], subgroup["n"])]
    forest(ax["c"], subgroup, "label", "hr_per_sd", "hr_ci_low", "hr_ci_high", PALETTE["navy"], "Hazard ratio per SD")
    ax["c"].set_xscale("log")
    ax["c"].set_xlim(0.65, 5.1)
    ax["c"].set_xticks([0.7, 1.0, 2.0, 4.0], ["0.7", "1", "2", "4"])
    ax["c"].minorticks_off()
    panel_title(ax["c"], "c", "PACA-CA subgroup consistency")

    pred = read_csv("41_v4_paca_ca_predictions.csv")
    event_colors = np.where(pred["OS_EVENT"].to_numpy(int) == 1, PALETTE["coral"], PALETTE["blue"])
    ax["d"].scatter(pred["clinical_oof_risk"], pred["v4_risk"], s=18, c=event_colors, alpha=0.62, edgecolor=PALETTE["white"], linewidth=0.3)
    slope, intercept, r_value, _, _ = stats.linregress(pred["clinical_oof_risk"], pred["v4_risk"])
    xx = np.linspace(pred["clinical_oof_risk"].min(), pred["clinical_oof_risk"].max(), 100)
    ax["d"].plot(xx, intercept + slope * xx, color=PALETTE["dark"], lw=1.1)
    rho = stats.spearmanr(pred["clinical_oof_risk"], pred["v4_risk"]).statistic
    ax["d"].text(0.04, 0.93, f"Spearman rho={rho:.2f}", transform=ax["d"].transAxes, va="top", fontsize=8.0)
    ax["d"].set_xlabel("Clinical OOF risk")
    ax["d"].set_ylabel("MarginRank-DG risk")
    ax["d"].legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["coral"], label="Event", markersize=5), Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["blue"], label="Censored", markersize=5)], frameon=False, loc="lower right")
    clean_axes(ax["d"], "both")
    panel_title(ax["d"], "d", "Molecular-clinical complementarity")

    return save_figure(
        fig,
        FIGURES,
        "SupplementaryFigureS4_paca_ca_extended",
        "Extended PACA-CA validation",
        ["PACA-CA time-dependent discrimination", "PACA-CA paired uncertainty", "PACA-CA subgroup consistency", "Molecular-clinical complementarity"],
        ["line plot", "violin plot", "forest plot", "scatter plot"],
    )


def build_s5() -> dict:
    fig, ax = mosaic()
    gpred = read_csv("44_v4_gse77435_predictions.csv")
    label_map = {"v4_marginrank_risk": "MarginRank-DG", "v2_predecessor_risk": "Frozen v2", "legacy_six_gene_risk": "Legacy score"}
    for model, label in label_map.items():
        fpr, tpr, _ = roc_curve(gpred["SHORT_SURVIVAL_LABEL"], gpred[model])
        value = roc_auc(fpr, tpr)
        ax["a"].plot(fpr, tpr, color=MODEL_COLORS.get(label, PALETTE["grey"]), label=f"{label} ({value:.2f})")
    ax["a"].plot([0, 1], [0, 1], color=PALETTE["mid"], ls="--", lw=0.9)
    ax["a"].set_xlabel("False-positive rate")
    ax["a"].set_ylabel("True-positive rate")
    ax["a"].set_xlim(0, 1)
    ax["a"].set_ylim(0, 1)
    ax["a"].legend(frameon=False, loc="lower right")
    clean_axes(ax["a"], "both")
    panel_title(ax["a"], "a", "GSE77435 extreme-survival ROC")

    gboot = read_csv("44_v4_gse77435_bootstrap.csv")
    perm = read_csv("44_v4_gse77435_permutation.csv")
    cols = ["v4_marginrank_auc", "v2_predecessor_auc", "legacy_six_gene_auc"]
    labels = ["MarginRank-DG", "Frozen v2", "Legacy"]
    colors = [PALETTE["navy"], PALETTE["blue"], PALETTE["grey"]]
    violin_with_points(ax["b"], [gboot[c].dropna().to_numpy() for c in cols], labels, colors, "Bootstrap binary AUC", 0.5)
    null_q = np.quantile(perm["auc"].dropna(), [0.025, 0.975])
    ax["b"].axhspan(null_q[0], null_q[1], color=PALETTE["pale_orange"], alpha=0.75, zorder=0)
    ax["b"].set_ylim(0.05, 1.02)
    panel_title(ax["b"], "b", "GSE77435 uncertainty and null range")

    prince = read_csv("48_v4_prince_subgroups.csv").copy()
    names = {
        "female": "Female",
        "male": "Male",
        "stage_iv": "Stage IV",
        "liver_biopsy": "Liver biopsy",
        "other_or_unknown_biopsy": "Other/unknown biopsy",
    }
    prince["label"] = [f"{names.get(s, s)} (n={n})" for s, n in zip(prince["subgroup"], prince["n"])]
    forest(ax["c"], prince, "label", "hr_per_sd", "hr_ci_low", "hr_ci_high", PALETTE["coral"], "Hazard ratio per SD")
    ax["c"].set_xscale("log")
    ax["c"].set_xlim(0.11, 2.6)
    ax["c"].set_xticks([0.2, 0.5, 1.0, 2.0], ["0.2", "0.5", "1", "2"])
    ax["c"].minorticks_off()
    panel_title(ax["c"], "c", "PRINCE subgroup reversal")

    pboot = read_csv("48_v4_prince_bootstrap.csv")
    pcols = ["v4_risk_uno", "v2_risk_uno", "legacy_risk_uno"]
    plabels = ["MarginRank-DG", "Frozen v2", "Legacy"]
    violin_with_points(ax["d"], [pboot[c].dropna().to_numpy() for c in pcols], plabels, colors, "Bootstrap Uno C", 0.5)
    ax["d"].set_ylim(0.02, 0.98)
    panel_title(ax["d"], "d", "PRINCE discrimination distributions")

    return save_figure(
        fig,
        FIGURES,
        "SupplementaryFigureS5_external_boundary",
        "External stress tests and scope boundary",
        ["GSE77435 extreme-survival ROC", "GSE77435 uncertainty and null range", "PRINCE subgroup reversal", "PRINCE discrimination distributions"],
        ["ROC curve", "violin plot", "forest plot", "violin plot"],
    )


def build_s6() -> dict:
    fig, ax = mosaic()
    pathway = read_csv("46_v4_ranked_pathway_enrichment.csv")
    pathway = pathway[(pathway["filtered_significant"] == True) & (pathway["cross_domain_robust"] == True)].copy()  # noqa: E712
    adverse = pathway[pathway["direction"] == "adverse"].nsmallest(8, "p_value")
    favourable = pathway[pathway["direction"] == "favourable"].nsmallest(8, "p_value")
    selected = pd.concat([adverse, favourable], ignore_index=True)
    selected["score"] = -np.log10(selected["p_value"].clip(lower=1e-300))
    selected["label"] = selected["name"].map(short_pathway)
    selected = selected.sort_values(["direction", "score"], ascending=[True, True]).reset_index(drop=True)
    y = np.arange(len(selected))
    pcolors = [PALETTE["coral"] if x == "adverse" else PALETTE["teal"] for x in selected["direction"]]
    sizes = 18 + 2.0 * np.sqrt(selected["intersection_size"].clip(lower=1))
    ax["a"].scatter(selected["score"], y, s=sizes, c=pcolors, alpha=0.85, edgecolor=PALETTE["white"], linewidth=0.5)
    ax["a"].set_yticks(y, selected["label"])
    ax["a"].set_xlabel("-log10 enrichment P")
    ax["a"].legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["coral"], label="Adverse", markersize=6), Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["teal"], label="Favourable", markersize=6)], frameon=False, loc="lower right")
    clean_axes(ax["a"], "x")
    panel_title(ax["a"], "a", "Cross-domain pathway context")

    dots = read_csv("45_gse263733_celltype_dotplot.csv")
    coeff = dots[["gene", "coefficient"]].drop_duplicates().assign(abscoef=lambda x: x["coefficient"].abs()).nlargest(18, "abscoef")
    gene_order = coeff.sort_values("coefficient")["gene"].tolist()
    cell_order = ["Ductal cell", "Myeloid", "Fibroblast", "Endothelial cell", "Acinar cell", "T cell", "B cell"]
    subset = dots[dots["gene"].isin(gene_order) & dots["cell_type"].isin(cell_order)].copy()
    xmap = {cell: i for i, cell in enumerate(cell_order)}
    ymap = {gene: i for i, gene in enumerate(gene_order)}
    sc = ax["b"].scatter(
        subset["cell_type"].map(xmap),
        subset["gene"].map(ymap),
        s=12 + subset["fraction_expressing"] * 90,
        c=subset["mean_logcp10k"],
        cmap="Blues",
        vmin=0,
        vmax=float(subset["mean_logcp10k"].quantile(0.97)),
        edgecolor=PALETTE["white"],
        linewidth=0.25,
    )
    ax["b"].set_xticks(range(len(cell_order)), [x.replace(" cell", "") for x in cell_order], rotation=35, ha="right")
    ax["b"].set_yticks(range(len(gene_order)), gene_order)
    ax["b"].tick_params(length=0)
    for spine in ax["b"].spines.values():
        spine.set_visible(False)
    panel_title(ax["b"], "b", "Single-cell expression of locked genes")

    pb = read_csv("45_gse263733_pseudobulk_scores.csv")
    order = read_csv("45_gse263733_celltype_summary.csv").sort_values("median_risk_program", ascending=False)["Celltypes"].tolist()
    groups = [pb.loc[pb["Celltypes"] == c, "risk_program"].dropna().to_numpy() for c in order]
    vp = ax["c"].violinplot(groups, positions=np.arange(len(order)), widths=0.75, showextrema=False)
    for body, color in zip(vp["bodies"], [PALETTE["navy"], PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["purple"], PALETTE["grey"], PALETTE["coral"]]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
    for x, (values, color) in enumerate(zip(groups, [PALETTE["navy"], PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["purple"], PALETTE["grey"], PALETTE["coral"]])):
        ax["c"].scatter(x, np.median(values), s=28, color=color, edgecolor=PALETTE["white"], linewidth=0.6, zorder=3)
    ax["c"].axhline(0, color=PALETTE["mid"], lw=0.8)
    ax["c"].set_xticks(range(len(order)), [x.replace(" cell", "") for x in order], rotation=35, ha="right")
    ax["c"].set_ylabel("Pseudobulk program score")
    clean_axes(ax["c"], "y")
    panel_title(ax["c"], "c", "Cell-type risk-program heterogeneity")

    gate = read_json("49_software_release_gate.json")
    off_axes(ax["d"])
    panel_title(ax["d"], "d", "Reproducible software release")
    steps = [
        ((0.03, 0.60), "Input\nCSV / TSV\n2 orientations", PALETTE["pale_blue"], PALETTE["blue"]),
        ((0.58, 0.60), "Locked model\nhash verified\n40 + 6 genes", PALETTE["pale_orange"], PALETTE["orange"]),
        ((0.58, 0.20), f"Quality\n{gate['tests']['collected']} tests\n{gate['tests']['line_coverage'] * 100:.1f}% coverage", PALETTE["pale_teal"], PALETTE["teal"]),
        ((0.03, 0.20), f"Output\n{gate['clean_install_smoke']['samples']} clean scores\n0 survival values", PALETTE["pale_purple"], PALETTE["purple"]),
    ]
    for xy, text, face, edge in steps:
        box(ax["d"], xy, 0.39, 0.24, text, face, edge, fontsize=7.0)
    arrow(ax["d"], (0.43, 0.72), (0.57, 0.72))
    arrow(ax["d"], (0.78, 0.59), (0.78, 0.45))
    arrow(ax["d"], (0.57, 0.32), (0.43, 0.32))
    ax["d"].text(0.50, 0.04, "Research-only boundary in package and model card", ha="center", fontsize=7.9)

    return save_figure(
        fig,
        FIGURES,
        "SupplementaryFigureS6_biology_software",
        "Biological context and software reproducibility",
        ["Cross-domain pathway context", "Single-cell expression of locked genes", "Cell-type risk-program heterogeneity", "Reproducible software release"],
        ["bubble plot", "dot plot", "violin plot", "workflow"],
    )


def main() -> None:
    configure_style()
    builders = [build_s1, build_s2, build_s3, build_s4, build_s5, build_s6]
    figures = [builder() for builder in builders]
    manifest = {
        "status": "SUPPLEMENTARY_FIGURES_COMPLETE",
        "figure_count": len(figures),
        "style": {
            "font": "Helvetica",
            "panel_titles": "single centered black title",
            "panel_borders": "none",
            "palette": "low-saturation coordinated Nature-style palette",
            "pdf_source": "matplotlib vector PDF",
            "rasterization": "PNG and TIFF generated from PDF at 450 dpi",
        },
        "figures": figures,
    }
    write_manifest(FIGURES / "supplementary_figure_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "figure_count": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
