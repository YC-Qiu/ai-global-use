"""
Utilities for harmonising survey columns using online translation services and
sentence embeddings.

The main entry point is :func:`harmonise_column_with_dictionary`, which
normalises the values of a target column so that they align with the canonical
labels listed in an accompanying `.txt` dictionary file.  The workflow is:

1. Detect the original language of each cell (via `langdetect`) and translate it
   to English using the Google Translate web interface (`deep-translator`).
2. Compare the translated value against the allowed labels by embedding the
   strings with a SentenceTransformer model and selecting the closest match.
3. Replace the original cell with the best-matching canonical label (or the
   `Other` bucket when available).

The implementation is written with readability in mind because it is intended
to be showcased in presentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from pandas import DataFrame


try:  # type: ignore[import-not-found]
    from deep_translator import GoogleTranslator
except ImportError as exc:  # pragma: no cover - informative error message
    raise ImportError(
        "deep-translator is required. Install it with "
        "`pip install deep-translator`."
    ) from exc

try:  # type: ignore[import-not-found]
    from sentence_transformers import SentenceTransformer, util
except ImportError as exc:  # pragma: no cover - informative error message
    raise ImportError(
        "sentence-transformers is required. Install it with "
        "`pip install sentence-transformers`."
    ) from exc

try:  # type: ignore[import-not-found]
    from langdetect import DetectorFactory, detect
except ImportError as exc:  # pragma: no cover - informative error message
    raise ImportError(
        "langdetect is required. Install it with `pip install langdetect`."
    ) from exc

try:  # type: ignore[import-not-found]
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover - informative error message
    raise ImportError(
        "tqdm is required for progress reporting. Install it with `pip install tqdm`."
    ) from exc


# Stabilise language detection across runs.
DetectorFactory.seed = 0


# Lazily initialise heavy dependencies so subsequent runs reuse the same
# translator/model instances.
_translator: Optional[GoogleTranslator] = None
_embedding_model: Optional[SentenceTransformer] = None


@dataclass
class HarmonisationDictionary:
    """Container for canonical labels and (optional) catch-all category."""

    labels: List[str]
    other_label: Optional[str]

    @classmethod
    def from_file(cls, path: Path) -> "HarmonisationDictionary":
        """
        Load a dictionary file that lists allowed labels line by line.

        Lines that start with an asterisk (e.g. ``*Other``) are treated as the
        catch-all fallback category.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Could not locate dictionary file at '{path.as_posix()}'. "
                "Make sure the file exists and the name matches the column."
            )

        raw_lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        labels: List[str] = []
        other_label: Optional[str] = None

        for entry in raw_lines:
            if entry.startswith("*"):
                value = entry.lstrip("*").strip()
                other_label = value
                labels.append(value)
            else:
                labels.append(entry)

        if not labels:
            raise ValueError(
                f"The dictionary file '{path.as_posix()}' is empty. "
                "Please provide at least one canonical label."
            )

        return cls(labels=labels, other_label=other_label)


def _initialise_translator() -> GoogleTranslator:
    global _translator
    if _translator is None:
        _translator = GoogleTranslator(source="auto", target="en")
    return _translator


def _initialise_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        # A compact, general-purpose English model that downloads
        # automatically the first time it is used.
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def _infer_dictionary_path(
    column_name: str,
    search_directory: Path,
) -> Path:
    """
    Infer the dictionary file path associated with ``column_name``.

    The function tries a small set of naming conventions and falls back to a
    glob search.  This makes the utility resilient to different naming habits
    (e.g. ``infospondent_industry.txt`` vs ``info_respondent_industry.txt``).
    """
    direct_candidate = search_directory / f"{column_name}.txt"
    if direct_candidate.exists():
        return direct_candidate

    # Fall back to a recursive search so dictionary files can live inside
    # nested folders such as ``info/``.
    matches = sorted(search_directory.rglob(f"*{column_name}*.txt"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find a dictionary file for column '{column_name}'. "
        f"Searched within '{search_directory.as_posix()}'."
    )


@lru_cache(maxsize=128)
def _embed_labels(labels: Tuple[str, ...]):
    """
    Embed canonical labels once and cache the tensor to avoid redundant work.
    """
    model = _initialise_embedding_model()
    # Returning a tensor keeps cosine similarity operations efficient.
    return model.encode(list(labels), convert_to_tensor=True, normalize_embeddings=True)



def _derive_default_output_path(csv_path: Path, column_name: str) -> Path:
    safe_column_name = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in column_name.strip().replace(" ", "_")
    )
    suffix = f"_{safe_column_name or 'column'}_harmonised"
    return csv_path.with_name(f"{csv_path.stem}{suffix}{csv_path.suffix}")


def _translate_to_english(value: str) -> Tuple[str, Optional[str]]:
    """
    Detect language and translate a single value to English.

    Returns
    -------
    translated_value:
        The value translated to English. Falls back to the original text if
        translation fails.
    detected_language:
        Two-letter language code reported by Google Translate, if available.
    """
    translator = _initialise_translator()

    try:
        detected_lang = detect(value)
    except Exception:  # pragma: no cover - heuristic failure
        detected_lang = None

    try:
        translated_text = translator.translate(value)
    except Exception:  # pragma: no cover - network dependent
        translated_text = value

    return translated_text, detected_lang


def _pick_best_label(
    translated_value: str,
    canonical_labels: List[str],
    canonical_embeddings,
    other_label: Optional[str],
    similarity_threshold: float,
) -> Tuple[str, float, bool]:
    """
    Select the canonical label with the highest cosine similarity.
    """
    model = _initialise_embedding_model()
    value_embedding = model.encode(
        translated_value, convert_to_tensor=True, normalize_embeddings=True
    )
    similarities = util.cos_sim(value_embedding, canonical_embeddings)[0]
    best_index = int(similarities.argmax().item())
    best_score = float(similarities[best_index].item())

    if best_score < similarity_threshold and other_label is not None:
        return other_label, best_score, True

    return canonical_labels[best_index], best_score, False


def harmonise_column_with_dictionary(
    column_name: str,
    *,
    csv_path: Path | str = Path("cleaned-gpo-ai-data-nov-27.csv"),
    info_directory: Path | str = Path("."),
    output_path: Optional[Path | str] = None,
    similarity_threshold: float = 0.45,
) -> Tuple[DataFrame, DataFrame]:
    """
    Harmonise the values of ``column_name`` using a companion dictionary file.

    Parameters
    ----------
    column_name:
        Name of the column to be harmonised (e.g. ``"respondent_industry"``).
    csv_path:
        Path to the CSV file that contains the survey data.
    info_directory:
        Directory that stores the dictionary text files.
    output_path:
        Optional destination for the harmonised CSV. When ``None`` a new file
        is created alongside the original input with the suffix
        ``_<column>_harmonised.csv``.
    similarity_threshold:
        Minimum cosine similarity required to accept a match. If the best match
        falls below this threshold and an ``Other`` label exists in the
        dictionary, the ``Other`` category is used as a fallback.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.DataFrame]
        The harmonised DataFrame and a similarity report with one row per input
        entry.

    Notes
    -----
    * If the dictionary does not define an ``Other`` label, the closest match is
      always selected, even when the similarity is low.
    * Translations and embeddings are cached per unique value to keep the
      method efficient when columns contain repeated entries.
    * Network access is required the first time translations or model downloads
      occur.
    """
    csv_path = Path(csv_path)
    info_directory = Path(info_directory)
    if output_path is not None:
        output_path = Path(output_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file '{csv_path.as_posix()}' not found. Please double-check "
            "the `csv_path` argument."
        )

    dictionary_path = _infer_dictionary_path(column_name, info_directory)
    harmonisation_dict = HarmonisationDictionary.from_file(dictionary_path)

    df = pd.read_csv(csv_path)
    if column_name not in df.columns:
        raise KeyError(
            f"Column '{column_name}' does not exist in '{csv_path.as_posix()}'."
        )

    canonical_labels = harmonisation_dict.labels
    canonical_embeddings = _embed_labels(tuple(canonical_labels))

    translation_cache: Dict[str, Tuple[str, Optional[str]]] = {}
    normalisation_cache: Dict[
        str, Tuple[str, Optional[float], str, Optional[str], bool]
    ] = {}

    report_rows: List[Dict[str, object]] = []
    report_columns = [
        "original_value",
        "translated_value",
        "detected_language",
        "harmonised_value",
        "cosine_similarity",
        "used_fallback",
    ]

    def _record_row(
        *,
        original_value: Optional[str],
        translated_value: Optional[str],
        detected_language: Optional[str],
        harmonised_value: str,
        cosine_similarity: Optional[float],
        used_fallback: bool,
    ) -> None:
        report_rows.append(
            {
                "original_value": original_value,
                "translated_value": translated_value,
                "detected_language": detected_language,
                "harmonised_value": harmonised_value,
                "cosine_similarity": cosine_similarity,
                "used_fallback": used_fallback,
            }
        )

    def _fallback_label() -> str:
        if harmonisation_dict.other_label is not None:
            return harmonisation_dict.other_label
        return canonical_labels[0]

    def _normalise_missing(original_value: Optional[str]) -> str:
        chosen_label = _fallback_label()
        _record_row(
            original_value=original_value,
            translated_value=None,
            detected_language=None,
            harmonised_value=chosen_label,
            cosine_similarity=None,
            used_fallback=harmonisation_dict.other_label is not None,
        )
        return chosen_label

    def _normalise_text(text: str) -> str:
        if text in normalisation_cache:
            (
                cached_label,
                cached_similarity,
                cached_translation,
                cached_language,
                cached_fallback,
            ) = normalisation_cache[text]
            _record_row(
                original_value=text,
                translated_value=cached_translation,
                detected_language=cached_language,
                harmonised_value=cached_label,
                cosine_similarity=cached_similarity,
                used_fallback=cached_fallback,
            )
            return cached_label

        if text not in translation_cache:
            translation_cache[text] = _translate_to_english(text)

        translated_text, detected_language = translation_cache[text]

        chosen_label, similarity, used_fallback = _pick_best_label(
            translated_text,
            canonical_labels,
            canonical_embeddings,
            harmonisation_dict.other_label,
            similarity_threshold,
        )

        normalisation_cache[text] = (
            chosen_label,
            similarity,
            translated_text,
            detected_language,
            used_fallback,
        )

        _record_row(
            original_value=text,
            translated_value=translated_text,
            detected_language=detected_language,
            harmonised_value=chosen_label,
            cosine_similarity=similarity,
            used_fallback=used_fallback,
        )
        return chosen_label

    def _normalise_cell(cell_value) -> str:
        if pd.isna(cell_value):
            return _normalise_missing(None)

        text = str(cell_value).strip()
        if not text:
            return _normalise_missing("")

        return _normalise_text(text)

    normalised_values: List[str] = []
    for value in tqdm(
        df[column_name],
        desc=f"Harmonising '{column_name}'",
        total=len(df[column_name]),
    ):
        normalised_values.append(_normalise_cell(value))

    df[column_name] = normalised_values

    if output_path is None:
        output_path = _derive_default_output_path(csv_path, column_name)

    df.to_csv(output_path, index=False)

    report_df = pd.DataFrame(report_rows, columns=report_columns)

    # Return both the DataFrame and a similarity summary.
    return df, report_df


__all__ = ["harmonise_column_with_dictionary", "HarmonisationDictionary"]


# === Edit the values below before running the module directly =================
COLUMN_NAME = "respondent_industry"
CSV_PATH = "gpo-ai-data.csv"
INFO_DIRECTORY = "info"
OUTPUT_PATH: Optional[str] = None  # e.g. "output/respondent_industry_clean.csv"
SIMILARITY_THRESHOLD = 0.1
REPORT_DIRECTORY = Path("output")
REPORT_PATH: Optional[str] = None
# ==============================================================================


def main() -> None:
    csv_path = Path(CSV_PATH)
    info_directory = Path(INFO_DIRECTORY)
    output_path: Optional[Path]
    if OUTPUT_PATH:
        output_path = Path(OUTPUT_PATH)
    else:
        output_path = _derive_default_output_path(csv_path, COLUMN_NAME)

    df, report_df = harmonise_column_with_dictionary(
        COLUMN_NAME,
        csv_path=csv_path,
        info_directory=info_directory,
        output_path=output_path,
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    print(
        f"Harmonised column '{COLUMN_NAME}'. Result saved to "
        f"{output_path.as_posix()}."
    )
    if not report_df.empty:
        REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

        if REPORT_PATH:
            report_path = Path(REPORT_PATH)
        else:
            safe_column_name = "".join(
                char if char.isalnum() or char == "_" else "_"
                for char in COLUMN_NAME.strip().replace(" ", "_")
            )
            report_filename = f"{safe_column_name or 'column'}_similarity_report.csv"
            report_path = REPORT_DIRECTORY / report_filename

        report_df.to_csv(report_path, index=False)
        print(f"Similarity report saved to {report_path.as_posix()}.")


if __name__ == "__main__":
    main()
