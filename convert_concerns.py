"""Utility script for counting concern selections in the survey data.

Edit the `CONCERN_COLUMN` and file paths below if your dataset uses different
names or locations. Run the script from the project root so relative paths
resolve correctly.
"""

from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm


# --- User-configurable settings ------------------------------------------------
INPUT_PATH = Path("gpo-ai-data.csv")
OUTPUT_PATH = Path("gpo-ai-data_concern_counts.csv")
CONCERN_COLUMN = "risk_daily"
# -------------------------------------------------------------------------------


def count_selections(cell_value: object) -> int:
    """Count how many concern options were selected for a single response."""
    if pd.isna(cell_value):
        return 0

    text = str(cell_value).strip()
    if not text:
        return 0

    if text.casefold() == "None of the above":
        return 0

    # Split on one-or-more commas and ignore empty fragments.
    options = [part.strip() for part in re.split(r",+", text) if part.strip()]
    return len(options)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file '{INPUT_PATH}' not found. "
            "Update INPUT_PATH if your data lives elsewhere."
        )

    df = pd.read_csv(INPUT_PATH)

    if CONCERN_COLUMN not in df.columns:
        available = ", ".join(df.columns)
        raise KeyError(
            f"Column '{CONCERN_COLUMN}' not found in '{INPUT_PATH}'. "
            f"Available columns: {available}"
        )

    tqdm.pandas(desc="Counting concern selections")
    output_column = f"{CONCERN_COLUMN}_selection_count"
    df[output_column] = df[CONCERN_COLUMN].progress_apply(count_selections)

    df.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Wrote concern counts to '{OUTPUT_PATH}' in column '{output_column}'. "
        f"Rows processed: {len(df)}."
    )


if __name__ == "__main__":
    main()

