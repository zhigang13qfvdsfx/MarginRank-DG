"""Command-line interface for TransSurv-PDAC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .release import load_default_model, load_model_card


def _separator(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return ","
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"
    raise ValueError("Input must be .csv, .tsv, or .txt")


def _normalise_gene(value: object) -> str:
    return str(value).strip().upper()


def read_expression(
    path: str | Path,
    orientation: str = "samples-by-genes",
    identifier_column: str = "0",
) -> pd.DataFrame:
    """Read and validate a delimited expression matrix.

    The identifier column is specified by zero-based position or exact name.
    For samples-by-genes input it contains sample identifiers; for transposed
    input it contains gene symbols.
    """
    input_path = Path(path)
    frame = pd.read_csv(input_path, sep=_separator(input_path), low_memory=False)
    if frame.empty:
        raise ValueError("Input expression matrix is empty")
    if identifier_column.isdigit():
        position = int(identifier_column)
        if position >= len(frame.columns):
            raise ValueError("Identifier-column position is outside the matrix")
        identifier = frame.columns[position]
    else:
        if identifier_column not in frame.columns:
            raise ValueError(f"Identifier column not found: {identifier_column}")
        identifier = identifier_column
    if frame[identifier].isna().any():
        raise ValueError("Identifier column contains missing values")
    if orientation == "samples-by-genes":
        sample_ids = frame.pop(identifier).astype(str).str.strip()
        if sample_ids.duplicated().any():
            raise ValueError("Sample identifiers must be unique")
        frame.index = sample_ids
        frame.columns = [_normalise_gene(column) for column in frame.columns]
    else:
        gene_ids = frame.pop(identifier).map(_normalise_gene)
        frame.index = gene_ids
        frame.columns = frame.columns.astype(str).str.strip()
        frame = frame.T
        frame.index = frame.index.astype(str)
        frame.columns = [_normalise_gene(column) for column in frame.columns]
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if numeric.columns.duplicated().any():
        numeric = numeric.T.groupby(level=0, sort=False).median().T
    if numeric.shape[1] < 2:
        raise ValueError("At least two gene columns are required")
    if (numeric.notna().sum(axis=1) < 2).any():
        bad = numeric.index[numeric.notna().sum(axis=1) < 2].tolist()[:5]
        raise ValueError(f"Samples with fewer than two numeric genes: {bad}")
    return numeric


def _score(args: argparse.Namespace) -> int:
    expression = read_expression(
        args.input,
        orientation=args.orientation,
        identifier_column=args.identifier_column,
    )
    model = load_default_model()
    output = model.predict_from_expression(expression)
    output.index.name = "sample_id"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sep = _separator(output_path)
    output.to_csv(output_path, sep=sep, index=True, na_rep="NA")

    degraded = int(output["input_degraded"].sum())
    reference_warning = int(output["reference_coverage_warning"].sum())
    relative_available = bool(output["relative_outputs_available"].all())
    summary = {
        "samples": int(len(output)),
        "degraded_samples": degraded,
        "reference_coverage_warning_samples": reference_warning,
        "batch_relative_outputs_available": relative_available,
        "absolute_survival_outputs": False,
        "output": str(output_path.resolve()),
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    if degraded or reference_warning:
        print(
            "WARNING: low-coverage scores are exploratory and should not be interpreted.",
            file=sys.stderr,
        )
    if not relative_available:
        print(
            "NOTE: fewer than 40 samples (or zero risk dispersion); percentiles and tiers are unavailable.",
            file=sys.stderr,
        )
    return 0


def _model_info(args: argparse.Namespace) -> int:
    card = load_model_card()
    payload = json.dumps(card, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transsurv-pdac",
        description=(
            "Frozen relative-risk scoring for untreated primary/resectable PDAC "
            "transcriptomes (research use only)."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="score an expression matrix")
    score.add_argument("input", help="input .csv/.tsv/.txt expression matrix")
    score.add_argument("output", help="output .csv/.tsv/.txt score file")
    score.add_argument(
        "--orientation",
        choices=("samples-by-genes", "genes-by-samples"),
        default="samples-by-genes",
    )
    score.add_argument(
        "--identifier-column",
        default="0",
        help="zero-based position or exact name of the sample/gene identifier column",
    )
    score.set_defaults(handler=_score)

    info = subparsers.add_parser("model-info", help="print the model card as JSON")
    info.add_argument("--output", help="optional JSON output path")
    info.set_defaults(handler=_model_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
