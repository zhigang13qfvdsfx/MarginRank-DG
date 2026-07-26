from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Cohort:
    name: str
    expression: pd.DataFrame
    clinical: pd.DataFrame

    @property
    def sample_ids(self) -> list[str]:
        return self.expression.index.astype(str).tolist()


def load_cohort(directory: str | Path, stem: str, name: str) -> Cohort:
    directory = Path(directory)
    expression = pd.read_parquet(directory / f"{stem}_expression.parquet")
    clinical = pd.read_csv(directory / f"{stem}_clinical.csv")
    id_column = "PATIENT_ID" if "PATIENT_ID" in clinical else "SAMPLE_ID"
    clinical = clinical.copy()
    clinical[id_column] = clinical[id_column].astype(str)
    clinical = clinical.set_index(id_column, drop=False).loc[expression.index.astype(str)]
    if clinical.index.astype(str).tolist() != expression.index.astype(str).tolist():
        raise ValueError(f"{name}: expression and clinical sample order mismatch")
    return Cohort(name=name, expression=expression, clinical=clinical)

