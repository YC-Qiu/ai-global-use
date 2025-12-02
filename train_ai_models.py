"""
Train separate regression models for multiple AI adoption targets with 3-fold CV.

The script reads `gpo-ai-data.csv`, trains one HistGradientBoostingRegressor per
target, performs cross-validation with hyperparameter search, exports
cross-validated prediction diagnostics, and writes summary metrics (including
bin-based accuracy) to the `output/` directory. A progress bar reports overall
status.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, ParameterGrid, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm

DATA_PATH = Path("gpo-ai-data.csv")
OUTPUT_DIR = Path("output")
TARGET_COLUMNS = [
    "Clothes_likely",
    "Travel_use",
    "Dating_use",
    "Grocery_use",
    "GPT_use_future",
    "AI_Adoption",
]
CV_FOLDS = 3
TOP_FEATURES = 10
BIN_EDGES = np.array([0.125, 0.375, 0.625, 0.875])
BIN_LABELS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
PARAM_GRID = list(
    ParameterGrid(
        {
            "regressor__learning_rate": [0.01, 0.03, 0.05, 0.08],
            "regressor__max_iter": [100, 200, 300, 500],
            "regressor__min_samples_leaf": [10, 20, 40],
        }
    )
)


def build_one_hot_encoder() -> OneHotEncoder:
    """Return a dense OneHotEncoder compatible with sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Fallback for older sklearn versions where sparse_output is unsupported.
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the survey dataset, ensuring the file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path.resolve()}")
    return pd.read_csv(path)


def split_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Prepare feature matrix and target vector for the specified target column.

    Returns:
        X: Feature dataframe with rows where target is not missing.
        y: Target series converted to float.
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
    """
    features = [col for col in df.columns if col not in TARGET_COLUMNS]
    if "respondent_country" in features:
        features.remove("respondent_country")
    available = df[target].notna()
    X = df.loc[available, features].copy()
    y = df.loc[available, target].astype(float)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return X, y, numeric_features, categorical_features


def make_pipeline(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    """Create a preprocessing and regression pipeline tailored to the input schema."""
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", build_one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    random_state=42,
                    max_depth=None,
                    learning_rate=0.05,
                    max_iter=400,
                    min_samples_leaf=20,
                ),
            ),
        ]
    )


def summarise_cv_results(cv_results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Convert sklearn cross_validate outputs into readable aggregate metrics."""
    metrics: Dict[str, float] = {}
    rmse_scores = -cv_results["test_rmse"]
    mae_scores = -cv_results["test_mae"]
    r2_scores = cv_results["test_r2"]

    metrics["rmse_mean"] = float(rmse_scores.mean())
    metrics["rmse_std"] = float(rmse_scores.std())
    metrics["mae_mean"] = float(mae_scores.mean())
    metrics["mae_std"] = float(mae_scores.std())
    metrics["r2_mean"] = float(r2_scores.mean())
    metrics["r2_std"] = float(r2_scores.std())
    return metrics


def discretize_scores(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map continuous scores into 0.25-wide bins."""
    indices = np.digitize(values, BIN_EDGES, right=False)
    indices = np.clip(indices, 0, len(BIN_LABELS) - 1)
    return indices, BIN_LABELS[indices]


def summarise_bin_accuracy(
    actual_indices: np.ndarray,
    predicted_indices: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Count how many predictions are exact matches vs off by N bins."""
    if actual_indices.size == 0:
        return {"counts": {}, "percentages": {}}

    deltas = np.abs(actual_indices - predicted_indices)
    counts: Dict[str, int] = {
        f"diff_{i}": int(np.sum(deltas == i)) for i in range(len(BIN_LABELS))
    }
    total = float(actual_indices.size)
    percentages: Dict[str, float] = {key: counts[key] / total for key in counts}
    return {"counts": counts, "percentages": percentages}


def run_parameter_search(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Dict[str, str],
    cv: KFold,
    target: str,
) -> Tuple[Dict[str, Any], Dict[str, float], List[Dict[str, Any]]]:
    """
    Evaluate the parameter grid and return the best configuration and history.
    """
    best_params: Dict[str, Any] | None = None
    best_metrics: Dict[str, float] | None = None
    history: List[Dict[str, Any]] = []

    param_bar = tqdm(PARAM_GRID, desc=f"{target}: tuning", unit="combo", leave=False)
    for params in param_bar:
        candidate = clone(pipeline)
        candidate.set_params(**params)

        cv_results = cross_validate(
            candidate,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_estimator=False,
            n_jobs=-1,
        )
        metrics = summarise_cv_results(cv_results)
        history.append(
            {
                "params": params,
                "metrics": metrics,
            }
        )

        param_bar.set_postfix(rmse=f"{metrics['rmse_mean']:.3f}", r2=f"{metrics['r2_mean']:.3f}")

        if best_metrics is None or metrics["rmse_mean"] < best_metrics["rmse_mean"]:
            best_params = params
            best_metrics = metrics

    param_bar.close()

    assert best_params is not None and best_metrics is not None, "Hyperparameter search failed."
    return best_params, best_metrics, history


def extract_feature_names(pipeline: Pipeline, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Retrieve the transformed feature names after fitting the pipeline."""
    feature_names: List[str] = []
    if numeric_features:
        feature_names.extend(numeric_features)

    if categorical_features:
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        if "categorical" in preprocessor.named_transformers_:
            cat_pipeline: Pipeline = preprocessor.named_transformers_["categorical"]
        else:
            return feature_names
        encoder: OneHotEncoder = cat_pipeline.named_steps["encoder"]
        encoded_names = encoder.get_feature_names_out(categorical_features)
        feature_names.extend(encoded_names.tolist())

    return feature_names


def identify_top_features(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str],
    top_k: int,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Use permutation importance to rank features and correlations to determine direction.
    """
    if not feature_names:
        return [], []

    perm = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=10,
        random_state=42,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    importances = perm.importances_mean

    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(X)
    if isinstance(transformed, np.ndarray):
        transformed_matrix = transformed
    else:
        transformed_matrix = transformed.toarray()
    predictions = pipeline.predict(X)

    feature_count = min(len(feature_names), importances.shape[0], transformed_matrix.shape[1])

    correlations = []
    for idx in range(feature_count):
        feature_column = transformed_matrix[:, idx]
        if np.std(feature_column) == 0:
            correlations.append(0.0)
            continue
        corr = np.corrcoef(feature_column, predictions)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        correlations.append(float(corr))
    correlations = np.array(correlations)

    feature_data = []
    for i in range(feature_count):
        feature_data.append(
            {
                "feature": feature_names[i],
                "importance": float(importances[i]),
                "correlation": float(correlations[i]),
            }
        )

    feature_data.sort(key=lambda item: item["importance"], reverse=True)

    top_positive = [item for item in feature_data if item["correlation"] > 0][:top_k]
    top_negative = [item for item in feature_data if item["correlation"] < 0][:top_k]

    return top_positive, top_negative


def save_results(
    output_dir: Path,
    target: str,
    metrics: Dict[str, float],
    top_positive: List[Dict[str, float]],
    top_negative: List[Dict[str, float]],
    sample_count: int,
    best_params: Dict[str, Any],
    tuning_history: List[Dict[str, Any]],
    bin_accuracy: Dict[str, Dict[str, float]],
    cv_predictions_path: Path,
) -> Path:
    """Persist evaluation summary and feature signals to a JSON file."""
    history_sorted = sorted(tuning_history, key=lambda item: item["metrics"]["rmse_mean"])
    history_compact = [
        {
            "params": entry["params"],
            "metrics": {
                "rmse_mean": entry["metrics"]["rmse_mean"],
                "mae_mean": entry["metrics"]["mae_mean"],
                "r2_mean": entry["metrics"]["r2_mean"],
            },
        }
        for entry in history_sorted[:5]
    ]

    payload = {
        "model": "HistGradientBoostingRegressor",
        "target": target,
        "samples": sample_count,
        "metrics": metrics,
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "best_params": best_params,
        "tuning_history_top5": history_compact,
        "bin_accuracy": bin_accuracy,
        "cv_predictions_csv": str(cv_predictions_path),
    }
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{target}_results.json"
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path


def main() -> None:
    df = load_dataset(DATA_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    overall_bar = tqdm(total=len(TARGET_COLUMNS), desc="Models completed", unit="model")
    for target in TARGET_COLUMNS:
        X, y, numeric_features, categorical_features = split_features(df, target)
        if X.empty:
            tqdm.write(f"Skipping {target}: no available samples.")
            overall_bar.update(1)
            continue

        pipeline = make_pipeline(numeric_features, categorical_features)
        stage_bar = tqdm(total=5, desc=f"{target}: tuning", unit="step", leave=False)

        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        best_params, metrics, tuning_history = run_parameter_search(
            pipeline,
            X,
            y,
            scoring,
            cv,
            target,
        )
        stage_bar.update()
        overall_bar.set_postfix(
            {"target": target, "rmse": f"{metrics['rmse_mean']:.3f}", "lr": best_params.get("regressor__learning_rate")}
        )

        final_model = clone(pipeline)
        final_model.set_params(**best_params)
        stage_bar.set_description(f"{target}: final fit")
        final_model.fit(X, y)
        stage_bar.update()

        feature_names = extract_feature_names(final_model, numeric_features, categorical_features)
        stage_bar.set_description(f"{target}: importance")
        top_positive, top_negative = identify_top_features(
            final_model,
            X,
            y,
            feature_names,
            TOP_FEATURES,
        )
        stage_bar.update()

        stage_bar.set_description(f"{target}: cv predict")
        cv_model = clone(pipeline)
        cv_model.set_params(**best_params)
        cv_predictions = cross_val_predict(
            cv_model,
            X,
            y,
            cv=cv,
            n_jobs=-1,
        )
        stage_bar.update()

        actual_indices, actual_bins = discretize_scores(y.values)
        predicted_indices, predicted_bins = discretize_scores(cv_predictions)
        bin_summary = summarise_bin_accuracy(actual_indices, predicted_indices)
        metrics["bin_match_rate"] = float(bin_summary["percentages"].get("diff_0", 0.0))
        metrics["rmse_cv_predictions"] = float(np.sqrt(np.mean((y.values - cv_predictions) ** 2)))
        metrics["mae_cv_predictions"] = float(np.mean(np.abs(y.values - cv_predictions)))

        residuals = y.values - cv_predictions
        cv_df = pd.DataFrame(
            {
                "actual": y.values,
                "predicted": cv_predictions,
                "residual": residuals,
                "actual_bin": actual_bins,
                "predicted_bin": predicted_bins,
                "bin_difference": np.abs(actual_indices - predicted_indices),
            }
        )

        stage_bar.set_description(f"{target}: outputs")
        cv_predictions_path = OUTPUT_DIR / f"{target}_cv_predictions.csv"
        cv_df.to_csv(cv_predictions_path, index=False)
        save_results(
            OUTPUT_DIR,
            target,
            metrics,
            top_positive,
            top_negative,
            sample_count=len(y),
            best_params=best_params,
            tuning_history=tuning_history,
            bin_accuracy=bin_summary,
            cv_predictions_path=cv_predictions_path,
        )
        stage_bar.update()
        stage_bar.close()

        # Print a clean summary beneath the progress bar.
        pos_summary = ", ".join(
            f"{item['feature']} (imp={item['importance']:.3f}, corr={item['correlation']:.2f})"
            for item in top_positive
        ) or "None"
        neg_summary = ", ".join(
            f"{item['feature']} (imp={item['importance']:.3f}, corr={item['correlation']:.2f})"
            for item in top_negative
        ) or "None"
        match_rate = bin_summary["percentages"].get("diff_0", 0.0)
        off_by_one = bin_summary["percentages"].get("diff_1", 0.0)

        tqdm.write(
            f"\nTarget: {target}\n"
            f"  Samples: {len(y)}\n"
            f"  RMSE (mean ± std): {metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}\n"
            f"  MAE (mean ± std): {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}\n"
            f"  R² (mean ± std): {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}\n"
            f"  CV RMSE (full): {metrics['rmse_cv_predictions']:.4f} | CV MAE (full): {metrics['mae_cv_predictions']:.4f}\n"
            f"  Bin match rate: {match_rate:.3f} | Off-by-1 rate: {off_by_one:.3f}\n"
            f"  Bin diff counts: {bin_summary['counts']}\n"
            f"  Best params: {best_params}\n"
            f"  Top factors increasing {target}: {pos_summary}\n"
            f"  Top factors decreasing {target}: {neg_summary}\n"
        )

        overall_bar.update(1)

    overall_bar.close()


if __name__ == "__main__":
    main()

