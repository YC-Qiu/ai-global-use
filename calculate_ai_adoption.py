"""Compute the AI Adoption composite score for the survey dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


# --- User-configurable settings ------------------------------------------------
INPUT_PATH = Path("gpo-ai-data.csv")
OUTPUT_PATH = Path("gpo-ai-data_with_ai_adoption.csv")
WEIGHTS: Dict[str, float] = {
    "GPT_use_future": 0.35,
    "Travel_use": 0.20,
    "Dating_use": 0.20,
    "Grocery_use": 0.15,
    "Clothes_likely": 0.10,
}
NEW_COLUMN = "AI_Adoption"
DEFAULT_FILL_VALUE = 0.0
# -------------------------------------------------------------------------------


def compute_ai_adoption(
    input_path: Path | str = INPUT_PATH,
    output_path: Path | str | None = OUTPUT_PATH,
    weights: Dict[str, float] = WEIGHTS,
    new_column: str = NEW_COLUMN,
    fill_value: float = DEFAULT_FILL_VALUE,
) -> pd.DataFrame:
    """Read `input_path`, calculate the weighted AI adoption score, and return the DataFrame.

    If `output_path` is provided, the updated DataFrame is written to disk.
    Missing or non-numeric values in the weighted features default to `fill_value`.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file '{input_path}' not found. Update INPUT_PATH if needed."
        )

    df = pd.read_csv(input_path)

    missing_columns = [column for column in weights if column not in df.columns]
    if missing_columns:
        raise KeyError(
            "The following required columns are missing from the input dataset: "
            + ", ".join(missing_columns)
        )

    for column in weights:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(fill_value)

    df[new_column] = sum(df[column] * weight for column, weight in weights.items())
    df[new_column] = df[new_column].round(4)

    if output_path is not None:
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    result_df = compute_ai_adoption()
    print(
        f"Calculated '{NEW_COLUMN}' scores for {len(result_df)} rows "
        f"and wrote them to '{OUTPUT_PATH}'."
    )

