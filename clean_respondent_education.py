"""Clean the `respondent_education` column to match the approved vocabulary.

Update the constants below if your file paths or column name differ. Run the
script from the project root so the relative paths resolve correctly.
"""

from pathlib import Path

import pandas as pd


# --- User-configurable settings ------------------------------------------------
INPUT_PATH = Path("gpo-ai-data.csv")
OUTPUT_PATH = Path("gpo-ai-data_cleaned_education.csv")
EDUCATION_COLUMN = "respondent_education"
ALLOWED_VALUES_PATH = Path("info/respondent_education.txt")
FALLBACK_VALUE = "Post-secondary education"
# -------------------------------------------------------------------------------


def load_allowed_values() -> set[str]:
    """Load the list of accepted education labels from the text file."""
    if not ALLOWED_VALUES_PATH.exists():
        raise FileNotFoundError(
            f"Allowed values file '{ALLOWED_VALUES_PATH}' not found. "
            "Update ALLOWED_VALUES_PATH if it lives elsewhere."
        )

    allowed = {
        line.strip()
        for line in ALLOWED_VALUES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    if not allowed:
        raise ValueError(
            f"No allowed values were found in '{ALLOWED_VALUES_PATH}'. "
            "Ensure the file lists one label per line."
        )

    # Make sure the fallback value is considered valid.
    allowed.add(FALLBACK_VALUE)
    return allowed


def normalize_value(raw_value: object, allowed_values: set[str]) -> str:
    """Return `raw_value` if it is accepted, otherwise the fallback label."""
    if pd.isna(raw_value):
        return FALLBACK_VALUE

    cleaned = str(raw_value).strip()
    if not cleaned:
        return FALLBACK_VALUE

    return cleaned if cleaned in allowed_values else FALLBACK_VALUE


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file '{INPUT_PATH}' not found. Update INPUT_PATH if needed."
        )

    allowed_values = load_allowed_values()

    df = pd.read_csv(INPUT_PATH)
    if EDUCATION_COLUMN not in df.columns:
        available = ", ".join(df.columns)
        raise KeyError(
            f"Column '{EDUCATION_COLUMN}' not found in '{INPUT_PATH}'. "
            f"Available columns: {available}"
        )

    original_values = df[EDUCATION_COLUMN].copy()
    df[EDUCATION_COLUMN] = df[EDUCATION_COLUMN].apply(
        lambda value: normalize_value(value, allowed_values)
    )

    replaced_count = (df[EDUCATION_COLUMN] != original_values).sum()
    df.to_csv(OUTPUT_PATH, index=False)

    print(
        f"Cleaned '{EDUCATION_COLUMN}' and wrote results to '{OUTPUT_PATH}'. "
        f"Rows processed: {len(df)}. Values replaced: {replaced_count}."
    )


if __name__ == "__main__":
    main()

