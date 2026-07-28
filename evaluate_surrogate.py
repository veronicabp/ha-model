"""Holdout diagnostics for curvature and sampling support in an HA surrogate.

The script compares exact and surrogate MPCs on explicit baseline/treated pairs,
then asks three questions:

1. Along which state variables does the surrogate have too little or too much
   curvature relative to the exact policies?
2. Which structural parameters or parameter interactions systematically predict
   model-level holdout errors?
3. Are large errors concentrated in state-space regions that are sparsely
   represented by the training sample?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ha_policy_surrogate import PolicySurrogateBundle, read_policy_dataset

EPS = 1.0e-12


def _parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas interval/category extension columns to plain strings."""

    out = frame.copy()
    for column in out.columns:
        series = out[column]
        dtype = series.dtype
        interval = isinstance(dtype, pd.IntervalDtype)
        interval_category = (
            isinstance(dtype, pd.CategoricalDtype)
            and isinstance(series.cat.categories, pd.IntervalIndex)
        )
        if interval or interval_category:
            out[column] = series.map(
                lambda value: None if pd.isna(value) else str(value)
            ).astype("string")
    return out


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write a Parquet-safe table, falling back to compressed pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _parquet_safe_frame(frame)
    try:
        safe.to_parquet(path, index=False)
        return path
    except Exception as exc:
        # Arrow extension-type incompatibilities are common across cluster
        # environments. Preserve the complete diagnostic rather than aborting.
        fallback = path.with_suffix(".pkl.gz")
        safe.to_pickle(fallback, compression="gzip")
        print(
            f"Parquet write failed for {path.name} ({type(exc).__name__}); "
            f"wrote {fallback.name} instead.",
            flush=True,
        )
        return fallback


def _load_true_holdout_bundle(path: Path) -> tuple[PolicySurrogateBundle, Path]:
    """Load the development-only bundle used for genuine test evaluation."""

    requested = Path(path).expanduser().resolve()
    bundle = PolicySurrogateBundle.load(requested)
    metadata = dict(bundle.training_metadata or {})

    if metadata.get("fit_scope") != "development_only":
        sibling_name = metadata.get("evaluation_bundle_filename")
        candidates = []
        if sibling_name:
            candidates.append(requested.parent / str(sibling_name))
        candidates.append(requested.parent / "policy_surrogate_holdout.joblib")
        candidate = next((value for value in candidates if value.exists()), None)
        if candidate is None:
            raise ValueError(
                "The supplied bundle was refit on all models, so it cannot be used "
                "for holdout evaluation. Supply policy_surrogate_holdout.joblib."
            )
        requested = candidate.resolve()
        bundle = PolicySurrogateBundle.load(requested)
        metadata = dict(bundle.training_metadata or {})

    fit_ids = set(map(str, metadata.get("fit_models", [])))
    test_ids = set(map(str, metadata.get("test_models", [])))
    overlap = fit_ids & test_ids
    if metadata.get("fit_scope") != "development_only" or not test_ids or overlap:
        raise ValueError(
            "Evaluation requires a development-only bundle with nonempty test "
            f"models and zero fit/test overlap. Overlap count: {len(overlap)}."
        )
    return bundle, requested


def _heldout_ids(bundle: PolicySurrogateBundle) -> set[str]:
    metadata = dict(bundle.training_metadata or {})
    fit_ids = set(map(str, metadata.get("fit_models", [])))
    test_ids = set(map(str, metadata.get("test_models", [])))
    overlap = fit_ids & test_ids
    if metadata.get("fit_scope") != "development_only":
        raise ValueError("The loaded bundle is not a development-only evaluation fit.")
    if not test_ids:
        raise ValueError("The evaluation bundle does not identify test model IDs.")
    if overlap:
        raise ValueError(
            f"Evaluation bundle has {len(overlap)} model IDs in both fit and test sets."
        )
    return test_ids


def _predict_consumption_chunked(
    bundle: PolicySurrogateBundle,
    frame: pd.DataFrame,
    chunk_size: int,
) -> np.ndarray:
    out = np.empty(len(frame), dtype=float)
    for start in range(0, len(frame), int(chunk_size)):
        stop = min(start + int(chunk_size), len(frame))
        pred = bundle.predict(frame.iloc[start:stop], project=True)
        out[start:stop] = pd.to_numeric(
            pred["consumption"], errors="coerce"
        ).to_numpy(dtype=float)
    return out


def build_holdout_mpc_comparison(
    bundle: PolicySurrogateBundle,
    sample: pd.DataFrame,
    *,
    prediction_chunk_size: int = 25_000,
) -> pd.DataFrame:
    required = {
        "model_id",
        "mpc_pair_id",
        "mpc_pair_role",
        "mpc_check_units",
        "consumption",
    }
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(
            "Sample is missing MPC-pair columns: " + ", ".join(sorted(missing))
        )

    heldout = _heldout_ids(bundle)
    if not heldout:
        raise ValueError("The bundle does not identify held-out model IDs.")

    data = sample.copy()
    data["model_id"] = data["model_id"].astype(str)
    pair_id = pd.to_numeric(data["mpc_pair_id"], errors="coerce")
    role = pd.to_numeric(data["mpc_pair_role"], errors="coerce")
    check = pd.to_numeric(data["mpc_check_units"], errors="coerce")
    keep = (
        data["model_id"].isin(heldout)
        & pair_id.ge(0)
        & role.isin([0, 1])
        & check.gt(0)
    )
    pair_rows = data.loc[keep].reset_index(drop=True)
    if pair_rows.empty:
        raise ValueError("No explicit MPC pairs exist for held-out models.")

    pair_rows["__surrogate_consumption"] = _predict_consumption_chunked(
        bundle,
        pair_rows,
        prediction_chunk_size,
    )
    keys = ["model_id", "mpc_pair_id"]
    baseline = (
        pair_rows[pair_rows["mpc_pair_role"] == 0]
        .sort_values(keys)
        .drop_duplicates(keys)
        .copy()
    )
    treated = (
        pair_rows[pair_rows["mpc_pair_role"] == 1][
            keys + ["consumption", "__surrogate_consumption", "mpc_check_units"]
        ]
        .sort_values(keys)
        .drop_duplicates(keys)
        .rename(
            columns={
                "consumption": "true_consumption_treated",
                "__surrogate_consumption": "surrogate_consumption_treated",
                "mpc_check_units": "check_treated",
            }
        )
    )
    baseline = baseline.rename(
        columns={
            "consumption": "true_consumption_baseline",
            "__surrogate_consumption": "surrogate_consumption_baseline",
            "mpc_check_units": "check_baseline",
        }
    )
    comparison = baseline.merge(treated, on=keys, how="inner", validate="one_to_one")
    comparison["mpc_check_units"] = 0.5 * (
        pd.to_numeric(comparison["check_baseline"], errors="coerce")
        + pd.to_numeric(comparison["check_treated"], errors="coerce")
    )
    comparison["true_mpc"] = (
        comparison["true_consumption_treated"]
        - comparison["true_consumption_baseline"]
    ) / comparison["mpc_check_units"]
    comparison["surrogate_mpc"] = (
        comparison["surrogate_consumption_treated"]
        - comparison["surrogate_consumption_baseline"]
    ) / comparison["mpc_check_units"]
    comparison["mpc_error"] = comparison["surrogate_mpc"] - comparison["true_mpc"]
    comparison["absolute_mpc_error"] = comparison["mpc_error"].abs()
    comparison["squared_mpc_error"] = comparison["mpc_error"] ** 2

    prepared = bundle.prepare_inputs(comparison)
    candidate_columns = list(
        dict.fromkeys(
            list(bundle.feature_spec.state_columns)
            + list(bundle.feature_spec.engineered_state_columns)
            + list(bundle.feature_spec.parameter_columns)
            + list(bundle.feature_spec.categorical_state_columns)
            + list(bundle.feature_spec.categorical_parameter_columns)
            + ["population_type", "current_income"]
        )
    )
    for column in candidate_columns:
        if column in prepared:
            comparison[column] = prepared[column].to_numpy()
        elif column in baseline:
            comparison[column] = baseline[column].to_numpy()

    finite = (
        np.isfinite(pd.to_numeric(comparison["true_mpc"], errors="coerce"))
        & np.isfinite(pd.to_numeric(comparison["surrogate_mpc"], errors="coerce"))
        & comparison["mpc_check_units"].gt(0)
    )
    return comparison.loc[finite].reset_index(drop=True)


def _binned_curvature(
    frame: pd.DataFrame,
    variable: str,
    n_bins: int,
) -> pd.DataFrame:
    data = frame[[variable, "true_mpc", "surrogate_mpc", "mpc_error"]].copy()
    for column in data:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if data.empty or data[variable].nunique() < 4:
        return pd.DataFrame()

    if data[variable].nunique() <= n_bins:
        data["bin"] = data[variable]
    else:
        data["bin"] = pd.qcut(data[variable], q=n_bins, duplicates="drop")
    table = (
        data.groupby("bin", observed=True)
        .agg(
            x=(variable, "mean"),
            x_min=(variable, "min"),
            x_max=(variable, "max"),
            true_mpc=("true_mpc", "mean"),
            surrogate_mpc=("surrogate_mpc", "mean"),
            mean_error=("mpc_error", "mean"),
            rmse=("mpc_error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            n=("true_mpc", "size"),
        )
        .sort_values("x")
        .reset_index(drop=True)
    )
    if len(table) < 4:
        return table
    x = table["x"].to_numpy(dtype=float)
    true = table["true_mpc"].to_numpy(dtype=float)
    surrogate = table["surrogate_mpc"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        true_curvature = np.gradient(np.gradient(true, x), x)
        surrogate_curvature = np.gradient(np.gradient(surrogate, x), x)
    table["true_curvature"] = true_curvature
    table["surrogate_curvature"] = surrogate_curvature
    table["curvature_gap"] = true_curvature - surrogate_curvature
    table["absolute_curvature_gap"] = np.abs(table["curvature_gap"])
    return table


def analyze_state_curvature(
    comparison: pd.DataFrame,
    output_dir: Path,
    variables: Iterable[str],
    n_bins: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    figure_dir = output_dir / "state_curvature_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for variable in variables:
        if variable not in comparison:
            continue
        table = _binned_curvature(comparison, variable, n_bins)
        if table.empty:
            continue
        table.insert(0, "variable", variable)
        rows.append(table)

        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(table["x"], table["true_mpc"], marker="o", label="True MPC")
        ax.plot(
            table["x"],
            table["surrogate_mpc"],
            marker="o",
            label="Surrogate MPC",
        )
        ax.set_xlabel(variable.replace("_", " ").title())
        ax.set_ylabel("Binned mean MPC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"mpc_by_{variable}.png", dpi=180)
        plt.close(fig)

        if "curvature_gap" in table:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(table["x"], table["curvature_gap"], marker="o")
            ax.axhline(0.0, linestyle="--")
            ax.set_xlabel(variable.replace("_", " ").title())
            ax.set_ylabel("True minus surrogate curvature")
            fig.tight_layout()
            fig.savefig(figure_dir / f"curvature_gap_{variable}.png", dpi=180)
            plt.close(fig)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result.to_csv(output_dir / "state_curvature_diagnostics.csv", index=False)
    return result


def _cv_r2(X: np.ndarray, y: np.ndarray, folds: int, seed: int) -> float:
    if len(y) < max(12, folds * 2) or np.std(y) <= EPS:
        return np.nan
    splitter = KFold(n_splits=min(folds, len(y)), shuffle=True, random_state=seed)
    scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=splitter, scoring="r2")
    return float(np.nanmean(scores))


def analyze_parameter_curvature(
    comparison: pd.DataFrame,
    output_dir: Path,
    parameter_columns: list[str],
    folds: int,
    seed: int,
    max_interactions: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_table = (
        comparison.groupby("model_id", as_index=False)
        .agg(
            mpc_bias=("mpc_error", "mean"),
            mpc_rmse=("mpc_error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            mean_true_mpc=("true_mpc", "mean"),
            mean_surrogate_mpc=("surrogate_mpc", "mean"),
        )
    )
    parameter_frame = comparison[["model_id", *parameter_columns]].drop_duplicates(
        "model_id"
    )
    model_table = model_table.merge(parameter_frame, on="model_id", how="inner")

    usable_parameters: list[str] = []
    for column in parameter_columns:
        values = pd.to_numeric(model_table[column], errors="coerce")
        if values.notna().mean() >= 0.8 and values.nunique(dropna=True) > 1:
            model_table[column] = values.fillna(values.median())
            usable_parameters.append(column)
    if not usable_parameters:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "parameter_curvature_rankings.csv", index=False)
        empty.to_csv(output_dir / "parameter_interaction_rankings.csv", index=False)
        return empty, empty

    scaler = StandardScaler()
    X = scaler.fit_transform(model_table[usable_parameters].to_numpy(dtype=float))
    targets = ["mpc_bias", "mpc_rmse"]
    curvature_rows: list[dict[str, object]] = []
    interaction_rows: list[dict[str, object]] = []

    for target in targets:
        y = model_table[target].to_numpy(dtype=float)
        baseline_score = _cv_r2(X, y, folds, seed)
        for j, parameter in enumerate(usable_parameters):
            if model_table[parameter].nunique() < 3:
                continue
            candidate = np.column_stack([X, X[:, j] ** 2])
            score = _cv_r2(candidate, y, folds, seed)
            curvature_rows.append(
                {
                    "target": target,
                    "parameter": parameter,
                    "baseline_cv_r2": baseline_score,
                    "augmented_cv_r2": score,
                    "delta_cv_r2": score - baseline_score,
                }
            )

        pairs = [
            (i, j)
            for i in range(len(usable_parameters))
            for j in range(i + 1, len(usable_parameters))
        ]
        if max_interactions is not None:
            pairs = pairs[: int(max_interactions)]
        for i, j in pairs:
            candidate = np.column_stack([X, X[:, i] * X[:, j]])
            score = _cv_r2(candidate, y, folds, seed)
            interaction_rows.append(
                {
                    "target": target,
                    "parameter_1": usable_parameters[i],
                    "parameter_2": usable_parameters[j],
                    "baseline_cv_r2": baseline_score,
                    "augmented_cv_r2": score,
                    "delta_cv_r2": score - baseline_score,
                }
            )

    curvature = pd.DataFrame(curvature_rows)
    if not curvature.empty:
        curvature = curvature.sort_values("delta_cv_r2", ascending=False)
    interactions = pd.DataFrame(interaction_rows)
    if not interactions.empty:
        interactions = interactions.sort_values("delta_cv_r2", ascending=False)
    curvature.to_csv(output_dir / "parameter_curvature_rankings.csv", index=False)
    interactions.to_csv(output_dir / "parameter_interaction_rankings.csv", index=False)
    model_table.to_csv(output_dir / "model_level_mpc_errors.csv", index=False)
    return curvature, interactions


def analyze_sampling_support(
    bundle: PolicySurrogateBundle,
    sample: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
    variables: list[str],
    *,
    max_training_rows: int,
    max_holdout_rows: int,
    neighbors: int,
    seed: int,
    n_bins: int,
) -> pd.DataFrame:
    training_ids = set(map(str, bundle.training_metadata.get("training_models", [])))
    tuning_ids = set(map(str, bundle.training_metadata.get("tuning_models", [])))
    development_ids = training_ids | tuning_ids
    train = sample[sample["model_id"].astype(str).isin(development_ids)].copy()
    train_prepared = bundle.prepare_inputs(train)
    heldout_prepared = bundle.prepare_inputs(comparison)
    valid_variables = [
        column
        for column in variables
        if column in train_prepared and column in heldout_prepared
    ]
    if not valid_variables:
        return pd.DataFrame()

    for column in valid_variables:
        train[column] = train_prepared[column].to_numpy()
        comparison[column] = heldout_prepared[column].to_numpy()

    rng = np.random.default_rng(seed)
    if len(train) > max_training_rows:
        train = train.iloc[
            np.sort(rng.choice(len(train), size=max_training_rows, replace=False))
        ]
    heldout = comparison.copy()
    if len(heldout) > max_holdout_rows:
        heldout = heldout.iloc[
            np.sort(rng.choice(len(heldout), size=max_holdout_rows, replace=False))
        ]

    X_train = train[valid_variables].apply(pd.to_numeric, errors="coerce")
    X_holdout = heldout[valid_variables].apply(pd.to_numeric, errors="coerce")
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_holdout = X_holdout.fillna(medians)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    holdout_scaled = scaler.transform(X_holdout)

    nn = NearestNeighbors(
        n_neighbors=min(max(1, neighbors), len(train_scaled)),
        n_jobs=-1,
    )
    nn.fit(train_scaled)
    distances, _ = nn.kneighbors(holdout_scaled)
    heldout["nearest_training_distance"] = distances[:, 0]
    heldout["mean_neighbor_distance"] = distances.mean(axis=1)
    support_summary = pd.DataFrame(
        {
            "statistic": [
                "holdout observations",
                "training support observations",
                "correlation abs error with nearest distance",
                "correlation abs error with mean neighbor distance",
                "median nearest distance",
                "95th percentile nearest distance",
            ],
            "value": [
                len(heldout),
                len(train),
                heldout["absolute_mpc_error"].corr(
                    heldout["nearest_training_distance"]
                ),
                heldout["absolute_mpc_error"].corr(
                    heldout["mean_neighbor_distance"]
                ),
                heldout["nearest_training_distance"].median(),
                heldout["nearest_training_distance"].quantile(0.95),
            ],
        }
    )
    support_summary.to_csv(output_dir / "sampling_support_summary.csv", index=False)

    heldout["support_bin"] = pd.qcut(
        heldout["nearest_training_distance"], q=n_bins, duplicates="drop"
    )
    by_support = (
        heldout.groupby("support_bin", observed=True)
        .agg(
            nearest_distance=("nearest_training_distance", "mean"),
            mean_abs_mpc_error=("absolute_mpc_error", "mean"),
            mpc_rmse=(
                "mpc_error",
                lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))),
            ),
            mean_bias=("mpc_error", "mean"),
            n=("mpc_error", "size"),
        )
        .reset_index(drop=True)
    )
    by_support.to_csv(output_dir / "errors_by_support_distance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(
        by_support["nearest_distance"],
        by_support["mean_abs_mpc_error"],
        marker="o",
    )
    ax.set_xlabel("Distance to nearest training state")
    ax.set_ylabel("Mean absolute MPC error")
    fig.tight_layout()
    fig.savefig(output_dir / "mpc_error_by_training_distance.png", dpi=180)
    plt.close(fig)

    if "population_type" in heldout:
        by_type = (
            heldout.groupby("population_type", observed=True)
            .agg(
                n=("mpc_error", "size"),
                true_mpc=("true_mpc", "mean"),
                surrogate_mpc=("surrogate_mpc", "mean"),
                bias=("mpc_error", "mean"),
                rmse=(
                    "mpc_error",
                    lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))),
                ),
                nearest_distance=("nearest_training_distance", "mean"),
            )
            .reset_index()
        )
        by_type.to_csv(output_dir / "errors_by_population_type.csv", index=False)

    heldout["support_bin"] = heldout["support_bin"].astype(str)
    _write_table(heldout, output_dir / "holdout_mpc_with_support.parquet")
    return by_support



def _quantile_edges(values: pd.Series, n_bins: int) -> np.ndarray | None:
    finite = pd.to_numeric(values, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if finite.nunique() < 2:
        return None
    edges = np.unique(
        np.quantile(finite.to_numpy(dtype=float), np.linspace(0.0, 1.0, n_bins + 1))
    )
    if len(edges) < 3:
        return None
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def analyze_joint_sampling_cells(
    bundle: PolicySurrogateBundle,
    sample: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
    variable_pairs: list[tuple[str, str]],
    *,
    n_bins: int,
    max_training_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Locate joint state-space cells with low training coverage and high error."""

    development_ids = set(
        map(str, bundle.training_metadata.get("training_models", []))
    ) | set(map(str, bundle.training_metadata.get("tuning_models", [])))
    training = sample[sample["model_id"].astype(str).isin(development_ids)].copy()
    heldout = comparison.copy()
    if training.empty or heldout.empty:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "joint_sampling_cells.csv", index=False)
        return empty

    training_prepared = bundle.prepare_inputs(training)
    heldout_prepared = bundle.prepare_inputs(heldout)
    requested = list(dict.fromkeys(v for pair in variable_pairs for v in pair))
    variables = [
        variable
        for variable in requested
        if variable in training_prepared and variable in heldout_prepared
    ]
    if not variables:
        empty = pd.DataFrame()
        empty.to_csv(output_dir / "joint_sampling_cells.csv", index=False)
        return empty
    for variable in variables:
        training[variable] = training_prepared[variable].to_numpy()
        heldout[variable] = heldout_prepared[variable].to_numpy()

    rng = np.random.default_rng(seed)
    if len(training) > max_training_rows:
        positions = np.sort(
            rng.choice(len(training), size=max_training_rows, replace=False)
        )
        training = training.iloc[positions].copy()

    rows: list[pd.DataFrame] = []
    for left, right in variable_pairs:
        if left not in training or right not in training:
            continue
        left_edges = _quantile_edges(training[left], n_bins)
        right_edges = _quantile_edges(training[right], n_bins)
        if left_edges is None or right_edges is None:
            continue

        train_cells = pd.DataFrame(
            {
                "left": pd.to_numeric(training[left], errors="coerce"),
                "right": pd.to_numeric(training[right], errors="coerce"),
            }
        ).dropna()
        hold_cells = pd.DataFrame(
            {
                "left": pd.to_numeric(heldout[left], errors="coerce"),
                "right": pd.to_numeric(heldout[right], errors="coerce"),
                "mpc_error": pd.to_numeric(heldout["mpc_error"], errors="coerce"),
                "absolute_mpc_error": pd.to_numeric(
                    heldout["absolute_mpc_error"], errors="coerce"
                ),
            }
        ).dropna()
        train_cells["left_bin"] = pd.cut(
            train_cells["left"], left_edges, labels=False, include_lowest=True
        )
        train_cells["right_bin"] = pd.cut(
            train_cells["right"], right_edges, labels=False, include_lowest=True
        )
        hold_cells["left_bin"] = pd.cut(
            hold_cells["left"], left_edges, labels=False, include_lowest=True
        )
        hold_cells["right_bin"] = pd.cut(
            hold_cells["right"], right_edges, labels=False, include_lowest=True
        )
        train_summary = (
            train_cells.dropna(subset=["left_bin", "right_bin"])
            .groupby(["left_bin", "right_bin"], observed=True)
            .agg(
                training_rows=("left", "size"),
                training_left_mean=("left", "mean"),
                training_right_mean=("right", "mean"),
            )
            .reset_index()
        )
        hold_summary = (
            hold_cells.dropna(subset=["left_bin", "right_bin"])
            .groupby(["left_bin", "right_bin"], observed=True)
            .agg(
                holdout_pairs=("mpc_error", "size"),
                holdout_left_mean=("left", "mean"),
                holdout_right_mean=("right", "mean"),
                mean_bias=("mpc_error", "mean"),
                mean_absolute_error=("absolute_mpc_error", "mean"),
                rmse=(
                    "mpc_error",
                    lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))),
                ),
            )
            .reset_index()
        )
        cells = hold_summary.merge(
            train_summary,
            on=["left_bin", "right_bin"],
            how="left",
        )
        cells["training_rows"] = cells["training_rows"].fillna(0).astype(int)
        cells["left_variable"] = left
        cells["right_variable"] = right
        cells["left_bin_min"] = cells["left_bin"].map(
            lambda j: left_edges[int(j)]
        )
        cells["left_bin_max"] = cells["left_bin"].map(
            lambda j: left_edges[int(j) + 1]
        )
        cells["right_bin_min"] = cells["right_bin"].map(
            lambda j: right_edges[int(j)]
        )
        cells["right_bin_max"] = cells["right_bin"].map(
            lambda j: right_edges[int(j) + 1]
        )
        # High values identify cells that combine meaningful holdout error with
        # little direct training support.
        cells["undersampling_score"] = (
            cells["mean_absolute_error"]
            / np.sqrt(cells["training_rows"].astype(float) + 1.0)
        )
        rows.append(cells)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(
            ["undersampling_score", "mean_absolute_error"], ascending=False
        ).reset_index(drop=True)
    result.to_csv(output_dir / "joint_sampling_cells.csv", index=False)
    return result

def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose holdout curvature and state-space sampling support."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prediction-chunk-size", type=int, default=25_000)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--max-interactions", type=int, default=None)
    parser.add_argument("--max-training-support-rows", type=int, default=200_000)
    parser.add_argument("--max-holdout-support-rows", type=int, default=100_000)
    parser.add_argument("--support-neighbors", type=int, default=5)
    parser.add_argument(
        "--state-variables",
        default=(
            "age,current_income,liquid_assets,illiquid_assets,"
            "liquid_slack_to_income,illiquid_assets_to_income,"
            "pre_tax_income_to_income"
        ),
    )
    args = parser.parse_args()

    bundle, evaluated_bundle_path = _load_true_holdout_bundle(args.bundle)
    sample_path = args.sample or evaluated_bundle_path.parent / "policy_grid_sample.pkl.gz"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    sample = read_policy_dataset(sample_path)
    sample["model_id"] = sample["model_id"].astype(str)
    comparison = build_holdout_mpc_comparison(
        bundle,
        sample,
        prediction_chunk_size=args.prediction_chunk_size,
    )
    _write_table(comparison, output / "holdout_mpc_comparison.parquet")

    for subgroup, filename in (
        ("employment_state", "errors_by_employment_state.csv"),
        ("population_type", "errors_by_population_type_unconditional.csv"),
    ):
        if subgroup in comparison:
            table = (
                comparison.groupby(subgroup, observed=True)
                .agg(
                    n=("mpc_error", "size"),
                    true_mpc=("true_mpc", "mean"),
                    surrogate_mpc=("surrogate_mpc", "mean"),
                    bias=("mpc_error", "mean"),
                    rmse=(
                        "mpc_error",
                        lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))),
                    ),
                )
                .reset_index()
            )
            table.to_csv(output / filename, index=False)

    state_variables = _parse_list(args.state_variables)
    curvature = analyze_state_curvature(
        comparison,
        output,
        state_variables,
        args.bins,
    )
    parameter_columns = [
        column
        for column in bundle.feature_spec.parameter_columns
        if not column.endswith("__missing") and column in comparison
    ]
    parameter_curvature, interactions = analyze_parameter_curvature(
        comparison,
        output,
        parameter_columns,
        args.cv_folds,
        args.random_state,
        args.max_interactions,
    )
    support = analyze_sampling_support(
        bundle,
        sample,
        comparison,
        output,
        [
            "age",
            "current_income",
            "liquid_slack_to_income",
            "illiquid_assets_to_income",
        ],
        max_training_rows=args.max_training_support_rows,
        max_holdout_rows=args.max_holdout_support_rows,
        neighbors=args.support_neighbors,
        seed=args.random_state,
        n_bins=args.bins,
    )
    joint_cells = analyze_joint_sampling_cells(
        bundle,
        sample,
        comparison,
        output,
        [
            ("liquid_slack_to_income", "illiquid_assets_to_income"),
            ("liquid_slack_to_income", "age"),
            ("liquid_slack_to_income", "current_income"),
            ("illiquid_assets_to_income", "age"),
        ],
        n_bins=max(4, min(10, args.bins // 2)),
        max_training_rows=args.max_training_support_rows,
        seed=args.random_state,
    )

    true = comparison["true_mpc"].to_numpy(dtype=float)
    surrogate = comparison["surrogate_mpc"].to_numpy(dtype=float)
    error = surrogate - true
    metadata = dict(bundle.training_metadata or {})
    fit_ids = set(map(str, metadata.get("fit_models", [])))
    test_ids = set(map(str, metadata.get("test_models", [])))
    summary = {
        "evaluated_bundle": str(evaluated_bundle_path),
        "fit_scope": metadata.get("fit_scope"),
        "n_fit_models": int(len(fit_ids)),
        "n_declared_test_models": int(len(test_ids)),
        "fit_test_overlap": int(len(fit_ids & test_ids)),
        "n_pairs": int(len(comparison)),
        "n_models": int(comparison["model_id"].nunique()),
        "mpc_r2": float(r2_score(true, surrogate)),
        "mpc_rmse": float(np.sqrt(np.mean(error**2))),
        "mpc_bias": float(np.mean(error)),
        "largest_state_curvature_gaps": (
            curvature.nlargest(20, "absolute_curvature_gap")[
                ["variable", "x", "true_curvature", "surrogate_curvature", "curvature_gap"]
            ].to_dict(orient="records")
            if not curvature.empty and "absolute_curvature_gap" in curvature
            else []
        ),
        "top_parameter_curvature_candidates": (
            parameter_curvature.head(20).to_dict(orient="records")
            if not parameter_curvature.empty
            else []
        ),
        "top_parameter_interaction_candidates": (
            interactions.head(20).to_dict(orient="records")
            if not interactions.empty
            else []
        ),
        "support_bins": support.to_dict(orient="records") if not support.empty else [],
        "top_undersampled_joint_cells": (
            joint_cells.head(30).to_dict(orient="records")
            if not joint_cells.empty
            else []
        ),
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
