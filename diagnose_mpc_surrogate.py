"""Diagnose MPC bias in a smooth HA policy-function surrogate.

This script compares check MPCs from solved HA consumption grids with MPCs from
an accounting-restricted smooth surrogate.  It is designed to diagnose cases in
which the surrogate shifts the MPC distribution to the right or otherwise fits
consumption levels better than consumption slopes.

The main diagnostics are:

1. Exact versus surrogate MPC distributions for every held-out model.
2. Baseline and treated consumption errors, whose difference determines MPC bias.
3. MPC error by liquid wealth, liquid slack, income, illiquid wealth, age, and
   exact MPC.
4. Model-level MPC bias/RMSE/correlation and their relationship to held-out
   consumption/deposit R-squared and parameter-space distance.
5. Sensitivity to check size.
6. Household-level paired output for further analysis.

By default, the script refits the *evaluation* surrogate using only the original
training+tuning models recorded in the fitted bundle.  This makes comparisons
for the recorded test models genuinely out of sample.  Use
``--surrogate-source stored`` to diagnose the production bundle, which may have
been refit on all models.

Example
-------

python diagnose_mpc_surrogate.py \
    --bundle app_data/policy_surrogate.joblib \
    --output-dir diagnostics/mpc \
    --check-amounts 500,1000,5000,10000,25000 \
    --primary-check 10000 \
    --n-households 10000

The script expects ``model_catalog.csv`` and ``policy_grid_sample.pkl.gz`` (or
another supported sample format) beside the bundle.  Exact model directories
may be absolute paths, relative paths in the catalog, or directories under
``held_out_models/<model_id>`` beside the bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ha_mpc_distribution import (
    PopulationAssumptions,
    align_population_to_exact_grid,
    compute_check_mpcs,
    compute_exact_grid_check_mpcs,
    generate_synthetic_population,
    infer_money_unit_info,
    load_exact_policy_grid,
    set_parameter_values,
)
from ha_policy_surrogate import (
    PRIMITIVE_POLICY_TARGETS,
    PolicySurrogateBundle,
    SmoothRFFMap,
    StructuredPolynomialMap,
    TargetTransform,
    prepare_policy_inputs,
    read_policy_dataset,
)

EPS = 1.0e-12


# =============================================================================
# Small utilities
# =============================================================================


def status(message: str) -> None:
    print(message, flush=True)


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        number = float(item)
        if not np.isfinite(number) or number <= 0:
            raise argparse.ArgumentTypeError(
                "Check amounts must be finite and strictly positive."
            )
        values.append(number)
    if not values:
        raise argparse.ArgumentTypeError("At least one check amount is required.")
    return values


def safe_corr(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if len(x_arr) < 2 or np.std(x_arr) <= EPS or np.std(y_arr) <= EPS:
        return np.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def finite_quantile(values: Sequence[float], q: float, default: float = np.nan) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float(default)


def safe_r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    y = y[keep]
    p = p[keep]
    if len(y) < 2 or np.std(y) <= EPS:
        return np.nan
    return float(r2_score(y, p))


def safe_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    if not np.any(keep):
        return np.nan
    return float(math.sqrt(mean_squared_error(y[keep], p[keep])))


def safe_mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y) & np.isfinite(p)
    if not np.any(keep):
        return np.nan
    return float(mean_absolute_error(y[keep], p[keep]))


def resolve_sibling_file(bundle_path: Path, names: Sequence[str]) -> Path | None:
    parent = bundle_path.resolve().parent
    for name in names:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def resolve_model_dir(
    bundle_path: Path,
    raw_model_dir: Any,
    model_id: str,
) -> Path:
    """Resolve local, catalog-relative, and deployment-relative model paths."""

    candidate = Path(str(raw_model_dir)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    parent = bundle_path.resolve().parent
    candidates = [
        parent / candidate,
        parent / "held_out_models" / str(model_id),
        parent / str(model_id),
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def write_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except Exception:
        fallback = path.with_suffix(".pkl.gz")
        frame.to_pickle(fallback, compression="gzip")
        return fallback


def held_out_model_ids(bundle: PolicySurrogateBundle) -> list[str]:
    metadata = dict(bundle.training_metadata or {})
    values = metadata.get("test_models") or metadata.get("validation_models")
    if values:
        return sorted({str(value) for value in values})

    metrics = bundle.validation_by_model.copy()
    if metrics.empty or "model_id" not in metrics:
        return []
    if "split" in metrics:
        mask = metrics["split"].astype(str).str.contains(
            "held|test", case=False, regex=True, na=False
        )
        metrics = metrics.loc[mask]
    return sorted(metrics["model_id"].astype(str).unique().tolist())


def validation_r2_table(bundle: PolicySurrogateBundle) -> pd.DataFrame:
    metrics = bundle.validation_by_model.copy()
    if metrics.empty:
        return pd.DataFrame(columns=["model_id", "consumption_r2", "deposit_r2"])
    metrics["model_id"] = metrics["model_id"].astype(str)
    if "split" in metrics:
        metrics = metrics[
            metrics["split"].astype(str).str.contains(
                "held|test", case=False, regex=True, na=False
            )
        ]
    metrics = metrics[metrics["target"].isin(["consumption", "deposit"])]
    out = (
        metrics.pivot_table(
            index="model_id",
            columns="target",
            values="r2",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                "consumption": "consumption_r2",
                "deposit": "deposit_r2",
            }
        )
    )
    for column in ("consumption_r2", "deposit_r2"):
        if column not in out:
            out[column] = np.nan
    return out


def model_parameter_values(
    bundle: PolicySurrogateBundle,
    catalog_row: pd.Series,
) -> dict[str, Any]:
    values = dict(bundle.default_values)
    for column in bundle.feature_spec.parameter_columns:
        if column.endswith("__missing"):
            values[column] = 0
        elif column in catalog_row.index and pd.notna(catalog_row[column]):
            values[column] = float(catalog_row[column])
    for column in bundle.feature_spec.categorical_parameter_columns:
        if column in catalog_row.index and pd.notna(catalog_row[column]):
            values[column] = str(catalog_row[column])
    return values


# =============================================================================
# Refit the genuine held-out evaluation surrogate
# =============================================================================


def make_feature_map_like_bundle(bundle: PolicySurrogateBundle) -> Any:
    metadata = dict(bundle.training_metadata or {})
    model_type = str(metadata.get("model_type", "polynomial")).lower()
    if model_type == "polynomial":
        return StructuredPolynomialMap(
            bundle.feature_spec,
            degree=int(metadata.get("polynomial_degree", 2)),
            state_state_interactions=True,
            state_parameter_interactions=True,
            parameter_parameter_interactions=True,
            categorical_slopes=True,
        )
    if model_type in {"rff", "rbf"}:
        gamma = getattr(bundle.feature_map, "gamma", None)
        return SmoothRFFMap(
            bundle.feature_spec,
            n_components=int(metadata.get("rff_components", 512)),
            gamma=gamma,
            include_linear=True,
            random_state=int(metadata.get("random_state", 123)),
        )
    raise ValueError(f"Unsupported stored model_type={model_type!r}.")


def target_transform_kind(target: str) -> str:
    return "log1p" if target == "consumption" else "asinh"


def refit_evaluation_surrogate(
    stored_bundle: PolicySurrogateBundle,
    sampled_data: pd.DataFrame,
) -> PolicySurrogateBundle:
    """Refit on recorded training+tuning models, excluding recorded test models."""

    test_ids = set(held_out_model_ids(stored_bundle))
    if not test_ids:
        raise ValueError(
            "The bundle does not record test model IDs, so an evaluation "
            "surrogate cannot be reconstructed."
        )
    if "model_id" not in sampled_data:
        raise KeyError("The sampled policy dataset has no model_id column.")

    raw = sampled_data.copy().reset_index(drop=True)
    raw["model_id"] = raw["model_id"].astype(str)
    development_raw = raw[~raw["model_id"].isin(test_ids)].copy()
    if development_raw.empty:
        raise ValueError("No non-test rows remain for refitting the evaluation model.")

    money_units = str(stored_bundle.training_metadata.get("money_units", "model"))
    development = prepare_policy_inputs(
        development_raw,
        feature_spec=stored_bundle.feature_spec,
        default_values=stored_bundle.default_values,
        accounting_defaults=stored_bundle.accounting_defaults,
        money_units=money_units,
    )

    missing_targets = [
        target for target in PRIMITIVE_POLICY_TARGETS if target not in development
    ]
    if missing_targets:
        raise KeyError(
            f"Sampled policy data are missing primitive targets: {missing_targets}"
        )

    feature_map = make_feature_map_like_bundle(stored_bundle)
    status(
        "Refitting evaluation surrogate on "
        f"{development['model_id'].nunique():,} non-test models "
        f"({len(development):,} sampled rows)..."
    )
    Z = feature_map.fit_transform(development)

    alphas = dict(stored_bundle.training_metadata.get("ridge_alphas", {}) or {})
    regressions: dict[str, Ridge] = {}
    transforms: dict[str, TargetTransform] = {}
    for target in PRIMITIVE_POLICY_TARGETS:
        transform = TargetTransform(target_transform_kind(target)).fit(
            pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        )
        y = transform.transform(
            pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        )
        alpha = float(alphas.get(target, 1.0))
        regression = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        regression.fit(Z, y)
        transforms[target] = transform
        regressions[target] = regression
        status(f"  fitted {target}: alpha={alpha:g}")

    metadata = dict(stored_bundle.training_metadata)
    metadata.update(
        {
            "diagnostic_evaluation_refit": True,
            "diagnostic_excluded_test_models": sorted(test_ids),
            "diagnostic_development_model_count": int(
                development["model_id"].nunique()
            ),
        }
    )
    return PolicySurrogateBundle(
        feature_map=feature_map,
        regressions=regressions,
        feature_spec=stored_bundle.feature_spec,
        internal_targets=tuple(PRIMITIVE_POLICY_TARGETS),
        target_transforms=transforms,
        target_parameterization=stored_bundle.target_parameterization,
        output_bounds=dict(stored_bundle.output_bounds),
        feature_ranges=dict(stored_bundle.feature_ranges),
        default_values=dict(stored_bundle.default_values),
        accounting_defaults=dict(stored_bundle.accounting_defaults),
        validation_metrics=stored_bundle.validation_metrics.copy(),
        validation_by_model=stored_bundle.validation_by_model.copy(),
        training_metadata=metadata,
    )


# =============================================================================
# Population and paired exact/surrogate evaluation
# =============================================================================


def make_population_assumptions(args: argparse.Namespace) -> PopulationAssumptions:
    return PopulationAssumptions(
        n_households=int(args.n_households),
        seed=int(args.population_seed),
        median_income_dollars=float(args.median_income_dollars),
        permanent_income_log_sd=float(args.permanent_income_log_sd),
        transitory_income_log_sd=float(args.transitory_income_log_sd),
        high_education_share=float(args.high_education_share),
        nonemployment_share=float(args.nonemployment_share),
        retirement_age=float(args.retirement_age),
        poor_hand_to_mouth_share=float(args.poor_hand_to_mouth_share),
        wealthy_hand_to_mouth_share=float(args.wealthy_hand_to_mouth_share),
        poor_liquid_ratio_median=float(args.poor_liquid_ratio_median),
        poor_illiquid_ratio_median=float(args.poor_illiquid_ratio_median),
        wealthy_htm_liquid_ratio_median=float(
            args.wealthy_htm_liquid_ratio_median
        ),
        wealthy_htm_illiquid_ratio_median=float(
            args.wealthy_htm_illiquid_ratio_median
        ),
        regular_liquid_ratio_median=float(args.regular_liquid_ratio_median),
        regular_illiquid_ratio_median=float(args.regular_illiquid_ratio_median),
        liquid_ratio_log_sd=float(args.liquid_ratio_log_sd),
        illiquid_ratio_log_sd=float(args.illiquid_ratio_log_sd),
        negative_liquid_share=float(args.negative_liquid_share),
    )


def paired_mpc_frame(
    surrogate: PolicySurrogateBundle,
    exact_grid: Any,
    catalog_row: pd.Series,
    *,
    assumptions: PopulationAssumptions,
    check_amount_dollars: float,
    dollars_per_model_unit: float,
    money_units: str,
) -> pd.DataFrame:
    """Evaluate exact and surrogate MPCs at identical mapped household states."""

    parameter_values = model_parameter_values(surrogate, catalog_row)
    population = generate_synthetic_population(
        surrogate,
        base_values=parameter_values,
        assumptions=assumptions,
        dollars_per_model_unit=dollars_per_model_unit,
        money_units=money_units,
    )
    population = set_parameter_values(population, surrogate, parameter_values)

    aligned = align_population_to_exact_grid(
        exact_grid,
        population,
        money_units=money_units,
    )
    aligned = set_parameter_values(aligned, surrogate, parameter_values)

    exact = compute_exact_grid_check_mpcs(
        exact_grid,
        aligned,
        check_amount_dollars=check_amount_dollars,
        dollars_per_model_unit=dollars_per_model_unit,
        money_units=money_units,
    )
    surrogate_response = compute_check_mpcs(
        surrogate,
        aligned,
        check_amount_dollars=check_amount_dollars,
        dollars_per_model_unit=dollars_per_model_unit,
        money_units=money_units,
    )

    n = len(exact)
    if len(surrogate_response) != n:
        raise RuntimeError("Exact and surrogate response tables have different lengths.")

    out = pd.DataFrame(index=np.arange(n))
    retained_columns = [
        "population_type",
        "age",
        "education",
        "employment_state",
        "is_retired",
        "current_income",
        "after_tax_income",
        "labor_income_state",
        "liquid_assets",
        "illiquid_assets",
        "liquid_grid_min",
        "liquid_grid_max",
        "illiquid_grid_min",
        "illiquid_grid_max",
        "exact_state_g",
        "exact_state_h",
        "exact_state_k",
        "exact_state_e",
        "exact_exogenous_state_mapped",
        "exact_asset_state_clipped",
    ]
    for column in retained_columns:
        if column in exact:
            out[column] = exact[column].to_numpy()

    out["exact_baseline_consumption"] = exact[
        "baseline_consumption"
    ].to_numpy(dtype=float)
    out["exact_treated_consumption"] = exact[
        "post_check_consumption"
    ].to_numpy(dtype=float)
    out["exact_consumption_response"] = exact[
        "consumption_response"
    ].to_numpy(dtype=float)
    out["exact_mpc"] = exact["mpc"].to_numpy(dtype=float)

    out["surrogate_baseline_consumption"] = surrogate_response[
        "baseline_consumption"
    ].to_numpy(dtype=float)
    out["surrogate_treated_consumption"] = surrogate_response[
        "post_check_consumption"
    ].to_numpy(dtype=float)
    out["surrogate_consumption_response"] = surrogate_response[
        "consumption_response"
    ].to_numpy(dtype=float)
    out["surrogate_mpc"] = surrogate_response["mpc"].to_numpy(dtype=float)

    out["baseline_consumption_error"] = (
        out["surrogate_baseline_consumption"] - out["exact_baseline_consumption"]
    )
    out["treated_consumption_error"] = (
        out["surrogate_treated_consumption"] - out["exact_treated_consumption"]
    )
    out["consumption_error_gradient"] = (
        out["treated_consumption_error"] - out["baseline_consumption_error"]
    )
    out["mpc_error"] = out["surrogate_mpc"] - out["exact_mpc"]
    out["check_amount_dollars"] = float(check_amount_dollars)
    out["check_amount_model_units"] = exact[
        "check_amount_model_units"
    ].to_numpy(dtype=float)

    out["exact_check_state_clipped"] = exact.get(
        "check_state_clipped", pd.Series(False, index=exact.index)
    ).to_numpy(dtype=bool)
    out["surrogate_check_state_clipped"] = surrogate_response.get(
        "check_state_clipped", pd.Series(False, index=surrogate_response.index)
    ).to_numpy(dtype=bool)

    out["liquid_slack"] = (
        pd.to_numeric(out["liquid_assets"], errors="coerce")
        - pd.to_numeric(out["liquid_grid_min"], errors="coerce")
    )
    income_scale = np.maximum(
        np.abs(pd.to_numeric(out["labor_income_state"], errors="coerce")), EPS
    )
    out["liquid_assets_to_income"] = (
        pd.to_numeric(out["liquid_assets"], errors="coerce") / income_scale
    )
    out["liquid_slack_to_income"] = out["liquid_slack"] / income_scale
    out["illiquid_assets_to_income"] = (
        pd.to_numeric(out["illiquid_assets"], errors="coerce") / income_scale
    )

    # This should hold exactly up to floating-point error.
    implied_mpc_error = (
        out["consumption_error_gradient"]
        / np.maximum(out["check_amount_model_units"], EPS)
    )
    discrepancy = np.nanmax(np.abs(implied_mpc_error - out["mpc_error"]))
    if np.isfinite(discrepancy) and discrepancy > 1.0e-8:
        warnings.warn(
            "MPC-error decomposition differs from the direct MPC error by "
            f"{discrepancy:.3g}.",
            stacklevel=2,
        )
    return out


# =============================================================================
# Summary tables
# =============================================================================


def model_summary_row(
    paired: pd.DataFrame,
    *,
    model_id: str,
    check_amount: float,
) -> dict[str, Any]:
    exact = pd.to_numeric(paired["exact_mpc"], errors="coerce").to_numpy(dtype=float)
    surrogate = pd.to_numeric(
        paired["surrogate_mpc"], errors="coerce"
    ).to_numpy(dtype=float)
    error = surrogate - exact
    baseline_error = pd.to_numeric(
        paired["baseline_consumption_error"], errors="coerce"
    ).to_numpy(dtype=float)
    treated_error = pd.to_numeric(
        paired["treated_consumption_error"], errors="coerce"
    ).to_numpy(dtype=float)
    gradient = treated_error - baseline_error

    keep = np.isfinite(exact) & np.isfinite(surrogate)
    exact = exact[keep]
    surrogate = surrogate[keep]
    error = error[keep]
    baseline_error = baseline_error[keep]
    treated_error = treated_error[keep]
    gradient = gradient[keep]

    if not len(exact):
        raise ValueError(f"No finite MPC pairs for model {model_id}.")

    return {
        "model_id": str(model_id),
        "check_amount_dollars": float(check_amount),
        "n_households": int(len(exact)),
        "exact_mean_mpc": float(np.mean(exact)),
        "surrogate_mean_mpc": float(np.mean(surrogate)),
        "mean_mpc_bias": float(np.mean(error)),
        "exact_median_mpc": float(np.median(exact)),
        "surrogate_median_mpc": float(np.median(surrogate)),
        "median_mpc_bias": float(np.median(surrogate) - np.median(exact)),
        "mpc_rmse": float(np.sqrt(np.mean(error**2))),
        "mpc_mae": float(np.mean(np.abs(error))),
        "mpc_correlation": safe_corr(exact, surrogate),
        "mpc_r2": safe_r2(exact, surrogate),
        "baseline_consumption_bias": float(np.mean(baseline_error)),
        "treated_consumption_bias": float(np.mean(treated_error)),
        "baseline_consumption_rmse": float(np.sqrt(np.mean(baseline_error**2))),
        "treated_consumption_rmse": float(np.sqrt(np.mean(treated_error**2))),
        "mean_consumption_error_gradient": float(np.mean(gradient)),
        "share_exact_mpc_below_005": float(np.mean(exact < 0.05)),
        "share_surrogate_mpc_below_005": float(np.mean(surrogate < 0.05)),
        "share_exact_mpc_above_025": float(np.mean(exact > 0.25)),
        "share_surrogate_mpc_above_025": float(np.mean(surrogate > 0.25)),
        "share_exact_mpc_above_050": float(np.mean(exact > 0.50)),
        "share_surrogate_mpc_above_050": float(np.mean(surrogate > 0.50)),
        "share_exact_mpc_above_090": float(np.mean(exact > 0.90)),
        "share_surrogate_mpc_above_090": float(np.mean(surrogate > 0.90)),
        "share_exact_check_clipped": float(
            pd.to_numeric(
                paired["exact_check_state_clipped"], errors="coerce"
            ).fillna(0.0).mean()
        ),
        "share_surrogate_check_clipped": float(
            pd.to_numeric(
                paired["surrogate_check_state_clipped"], errors="coerce"
            ).fillna(0.0).mean()
        ),
        "share_asset_state_clipped": float(
            pd.to_numeric(
                paired.get("exact_asset_state_clipped", False), errors="coerce"
            ).fillna(0.0).mean()
        ),
        "share_exogenous_state_mapped": float(
            pd.to_numeric(
                paired.get("exact_exogenous_state_mapped", False), errors="coerce"
            ).fillna(0.0).mean()
        ),
    }


def parameter_distance_table(
    catalog: pd.DataFrame,
    *,
    test_ids: set[str],
    parameter_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> pd.DataFrame:
    """Distance from each test model to its nearest development parameter vector."""

    table = catalog.copy()
    table["model_id"] = table["model_id"].astype(str)
    numeric_columns = [
        column
        for column in parameter_columns
        if column in table and not column.endswith("__missing")
    ]
    categorical = [column for column in categorical_columns if column in table]

    if numeric_columns:
        numeric = table[numeric_columns].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0)
        numeric = numeric.fillna(medians)
        scales = numeric.std(axis=0).replace(0.0, 1.0).fillna(1.0)
        standardized = (numeric - numeric.mean(axis=0)) / scales
    else:
        standardized = pd.DataFrame(index=table.index)

    development_mask = ~table["model_id"].isin(test_ids)
    test_mask = table["model_id"].isin(test_ids)
    development_indices = table.index[development_mask].to_numpy()
    test_indices = table.index[test_mask].to_numpy()
    if not len(development_indices) or not len(test_indices):
        return pd.DataFrame(columns=["model_id", "nearest_parameter_distance"])

    rows: list[dict[str, Any]] = []
    for index in test_indices:
        if numeric_columns:
            diff = standardized.loc[development_indices].to_numpy(dtype=float) - standardized.loc[
                index
            ].to_numpy(dtype=float)
            squared = np.sum(diff**2, axis=1)
        else:
            squared = np.zeros(len(development_indices), dtype=float)
        for column in categorical:
            mismatch = (
                table.loc[development_indices, column].astype(str).to_numpy()
                != str(table.loc[index, column])
            )
            squared += mismatch.astype(float)
        distance = float(np.sqrt(np.min(squared)))
        rows.append(
            {
                "model_id": str(table.loc[index, "model_id"]),
                "nearest_parameter_distance": distance,
            }
        )
    return pd.DataFrame(rows)


def quantile_bin_summary(
    frame: pd.DataFrame,
    variable: str,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    data = frame.copy()
    x = pd.to_numeric(data[variable], errors="coerce")
    finite = np.isfinite(x)
    data = data.loc[finite].copy()
    x = x.loc[finite]
    if data.empty or x.nunique() < 2:
        return pd.DataFrame()
    try:
        data["bin"] = pd.qcut(x, q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    grouped = data.groupby("bin", observed=True)
    summary = grouped.agg(
        n=("exact_mpc", "size"),
        x_mean=(variable, "mean"),
        x_median=(variable, "median"),
        exact_mean_mpc=("exact_mpc", "mean"),
        surrogate_mean_mpc=("surrogate_mpc", "mean"),
        mean_mpc_error=("mpc_error", "mean"),
        mpc_rmse=("mpc_error", lambda values: float(np.sqrt(np.mean(values**2)))),
        baseline_consumption_bias=("baseline_consumption_error", "mean"),
        treated_consumption_bias=("treated_consumption_error", "mean"),
        mean_consumption_error_gradient=("consumption_error_gradient", "mean"),
        exact_check_clipped=("exact_check_state_clipped", "mean"),
        surrogate_check_clipped=("surrogate_check_state_clipped", "mean"),
    ).reset_index()
    summary.insert(0, "variable", variable)
    summary["bin"] = summary["bin"].astype(str)
    return summary


def parameter_correlations(
    model_summary: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    metrics: Sequence[str],
) -> pd.DataFrame:
    parameter_columns = [
        column
        for column in catalog.columns
        if column.startswith("param__")
        and not column.endswith("__missing")
        and pd.to_numeric(catalog[column], errors="coerce").notna().mean() >= 0.80
    ]
    # model_summary may already contain the catalog columns.  Merge only the
    # missing parameter columns to avoid pandas suffixing them to *_x and *_y.
    missing_parameters = [
        column for column in parameter_columns if column not in model_summary.columns
    ]
    if missing_parameters:
        merged = model_summary.merge(
            catalog[["model_id", *missing_parameters]],
            on="model_id",
            how="left",
        )
    else:
        merged = model_summary.copy()
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in merged:
            continue
        metric_values = pd.to_numeric(merged[metric], errors="coerce")
        for parameter in parameter_columns:
            parameter_values = pd.to_numeric(merged[parameter], errors="coerce")
            keep = metric_values.notna() & parameter_values.notna()
            if keep.sum() < 4 or parameter_values[keep].nunique() < 2:
                continue
            pearson = metric_values[keep].corr(parameter_values[keep], method="pearson")
            spearman = metric_values[keep].corr(parameter_values[keep], method="spearman")
            rows.append(
                {
                    "metric": metric,
                    "parameter": parameter,
                    "n_models": int(keep.sum()),
                    "pearson_correlation": float(pearson),
                    "spearman_correlation": float(spearman),
                    "abs_spearman_correlation": float(abs(spearman)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["metric", "abs_spearman_correlation"], ascending=[True, False]
    ) if rows else pd.DataFrame()


# =============================================================================
# Figures: one figure per file, no subplots
# =============================================================================


def save_figure(fig: plt.Figure, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def figure_exact_vs_surrogate_mean(model_summary: pd.DataFrame, path: Path) -> None:
    data = model_summary[
        ["exact_mean_mpc", "surrogate_mean_mpc"]
    ].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.scatter(data["exact_mean_mpc"], data["surrogate_mean_mpc"], alpha=0.75)
    lo = min(data.min().min(), 0.0)
    hi = max(data.max().max(), lo + 0.01)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Exact-model mean MPC")
    ax.set_ylabel("Surrogate mean MPC")
    ax.set_title("Mean MPC by held-out model")
    save_figure(fig, path)


def figure_mpc_bias_vs_r2(model_summary: pd.DataFrame, path: Path) -> None:
    data = model_summary[["consumption_r2", "mean_mpc_bias"]].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(data["consumption_r2"], data["mean_mpc_bias"], alpha=0.75)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Held-out consumption R-squared")
    ax.set_ylabel("Mean surrogate minus exact MPC")
    ax.set_title("MPC bias versus consumption-level fit")
    save_figure(fig, path)


def figure_mpc_bias_vs_distance(model_summary: pd.DataFrame, path: Path) -> None:
    data = model_summary[
        ["nearest_parameter_distance", "mean_mpc_bias"]
    ].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(data["nearest_parameter_distance"], data["mean_mpc_bias"], alpha=0.75)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Distance to nearest development parameterization")
    ax.set_ylabel("Mean surrogate minus exact MPC")
    ax.set_title("MPC bias versus parameter-space distance")
    save_figure(fig, path)


def figure_ranked_model_metric(
    model_summary: pd.DataFrame,
    *,
    metric: str,
    label: str,
    path: Path,
    top_n: int = 40,
) -> None:
    data = model_summary[["model_id", metric]].dropna().copy()
    if data.empty:
        return
    data = data.sort_values(metric, ascending=False).head(top_n).sort_values(metric)
    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.22 * len(data))))
    ax.barh(data["model_id"].astype(str), data[metric])
    ax.set_xlabel(label)
    ax.set_ylabel("Held-out model")
    ax.set_title(f"Largest {label.lower()} values")
    save_figure(fig, path)


def figure_pooled_distribution(
    household: pd.DataFrame,
    *,
    check_amount: float,
    path: Path,
    bins: int = 70,
) -> None:
    exact = pd.to_numeric(household["exact_mpc"], errors="coerce").to_numpy(dtype=float)
    surrogate = pd.to_numeric(
        household["surrogate_mpc"], errors="coerce"
    ).to_numpy(dtype=float)
    exact = exact[np.isfinite(exact)]
    surrogate = surrogate[np.isfinite(surrogate)]
    if not len(exact) or not len(surrogate):
        return
    pooled = np.concatenate([exact, surrogate])
    lo, hi = np.quantile(pooled, [0.0025, 0.9975])
    lo = min(float(lo), -0.05)
    hi = max(float(hi), 1.05)
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(exact, bins=edges, density=True, histtype="step", linewidth=2.0, label="Exact model")
    ax.hist(
        surrogate,
        bins=edges,
        density=True,
        histtype="step",
        linewidth=2.0,
        label="Surrogate",
    )
    ax.axvline(np.mean(exact), linestyle="--", linewidth=1.2, label="Exact mean")
    ax.axvline(
        np.mean(surrogate), linestyle=":", linewidth=1.2, label="Surrogate mean"
    )
    ax.set_xlabel("MPC")
    ax.set_ylabel("Density")
    ax.set_title(f"Pooled MPC distributions: ${check_amount:,.0f} check")
    ax.legend()
    save_figure(fig, path)


def figure_pooled_mpc_scatter(
    household: pd.DataFrame,
    *,
    check_amount: float,
    path: Path,
    max_points: int = 20_000,
    seed: int = 123,
) -> None:
    data = household[["exact_mpc", "surrogate_mpc"]].dropna()
    if data.empty:
        return
    if len(data) > max_points:
        data = data.sample(max_points, random_state=seed)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.scatter(data["exact_mpc"], data["surrogate_mpc"], alpha=0.20, s=8)
    lo = min(data.min().min(), -0.05)
    hi = max(data.max().max(), 1.05)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Exact-model MPC")
    ax.set_ylabel("Surrogate MPC")
    ax.set_title(f"Household MPC fit: ${check_amount:,.0f} check")
    save_figure(fig, path)


def figure_consumption_error_gradient(
    household: pd.DataFrame,
    *,
    check_amount: float,
    path: Path,
    max_points: int = 20_000,
    seed: int = 123,
) -> None:
    data = household[
        ["baseline_consumption_error", "treated_consumption_error"]
    ].dropna()
    if data.empty:
        return
    if len(data) > max_points:
        data = data.sample(max_points, random_state=seed)
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.scatter(
        data["baseline_consumption_error"],
        data["treated_consumption_error"],
        alpha=0.20,
        s=8,
    )
    lo = min(data.min().min(), 0.0)
    hi = max(data.max().max(), lo + 0.01)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.3)
    ax.set_xlabel("Surrogate error at baseline liquid assets")
    ax.set_ylabel("Surrogate error after the check")
    ax.set_title(
        "Consumption-error decomposition\n"
        f"Points above the diagonal imply upward MPC bias (${check_amount:,.0f})"
    )
    save_figure(fig, path)


def figure_binned_line(
    summary: pd.DataFrame,
    *,
    x: str,
    y_columns: Sequence[tuple[str, str]],
    x_label: str,
    y_label: str,
    title: str,
    path: Path,
) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for column, label in y_columns:
        if column in summary:
            ax.plot(summary[x], summary[column], marker="o", label=label)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if len(y_columns) > 1:
        ax.legend()
    save_figure(fig, path)


def figure_check_size_summary(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    grouped = summary.groupby("check_amount_dollars", as_index=False).agg(
        exact_mean_mpc=("exact_mean_mpc", "mean"),
        surrogate_mean_mpc=("surrogate_mean_mpc", "mean"),
        mean_mpc_bias=("mean_mpc_bias", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(
        grouped["check_amount_dollars"],
        grouped["exact_mean_mpc"],
        marker="o",
        label="Exact-model mean MPC",
    )
    ax.plot(
        grouped["check_amount_dollars"],
        grouped["surrogate_mean_mpc"],
        marker="o",
        label="Surrogate mean MPC",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Check amount ($, log scale)")
    ax.set_ylabel("Mean MPC across models")
    ax.set_title("MPC sensitivity to check size")
    ax.legend()
    save_figure(fig, path)


def figure_check_size_bias(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    grouped = summary.groupby("check_amount_dollars", as_index=False).agg(
        mean_mpc_bias=("mean_mpc_bias", "mean"),
        median_model_bias=("mean_mpc_bias", "median"),
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(
        grouped["check_amount_dollars"],
        grouped["mean_mpc_bias"],
        marker="o",
        label="Mean model bias",
    )
    ax.plot(
        grouped["check_amount_dollars"],
        grouped["median_model_bias"],
        marker="o",
        label="Median model bias",
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Check amount ($, log scale)")
    ax.set_ylabel("Surrogate minus exact MPC")
    ax.set_title("MPC bias sensitivity to check size")
    ax.legend()
    save_figure(fig, path)


def figure_worst_model_distribution(
    household: pd.DataFrame,
    *,
    model_id: str,
    check_amount: float,
    path: Path,
    bins: int = 60,
) -> None:
    data = household[household["model_id"].astype(str) == str(model_id)]
    if data.empty:
        return
    figure_pooled_distribution(
        data,
        check_amount=check_amount,
        path=path,
        bins=bins,
    )


# =============================================================================
# Main pipeline
# =============================================================================


def load_catalog(bundle_path: Path) -> pd.DataFrame:
    path = resolve_sibling_file(bundle_path, ["model_catalog.csv"])
    if path is None:
        raise FileNotFoundError(
            f"Could not find model_catalog.csv beside {bundle_path}."
        )
    catalog = pd.read_csv(path)
    catalog["model_id"] = catalog["model_id"].astype(str)
    if "usable" in catalog:
        usable = catalog["usable"].astype(str).str.lower().isin(
            {"true", "1", "yes"}
        )
        catalog = catalog.loc[usable].copy()
    return catalog.reset_index(drop=True)


def load_sampled_data(bundle_path: Path, explicit: Path | None) -> pd.DataFrame:
    if explicit is not None:
        return read_policy_dataset(explicit)
    candidate = resolve_sibling_file(
        bundle_path,
        [
            "policy_grid_sample.pkl.gz",
            "policy_grid_sample.parquet",
            "policy_grid_sample.pkl",
            "policy_grid_sample.csv",
        ],
    )
    if candidate is None:
        raise FileNotFoundError(
            "Could not find a sampled policy dataset beside the bundle. "
            "Pass --sampled-data or use --surrogate-source stored."
        )
    return read_policy_dataset(candidate)


def choose_models(
    catalog: pd.DataFrame,
    test_ids: Sequence[str],
    *,
    max_models: int | None,
    seed: int,
) -> pd.DataFrame:
    selected = catalog[catalog["model_id"].isin({str(x) for x in test_ids})].copy()
    if selected.empty:
        raise ValueError("No recorded held-out model IDs appear in model_catalog.csv.")
    selected = selected.sort_values("model_id").reset_index(drop=True)
    if max_models is not None and max_models < len(selected):
        selected = selected.sample(max_models, random_state=seed).sort_values("model_id")
    return selected.reset_index(drop=True)


def run(args: argparse.Namespace) -> dict[str, Path]:
    start = time.perf_counter()
    output_dir = Path(args.output_dir).expanduser().resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    household_dir = output_dir / "household_data"
    for directory in (tables_dir, figures_dir, household_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bundle_path = Path(args.bundle).expanduser().resolve()
    status(f"Loading stored bundle: {bundle_path}")
    stored_bundle = PolicySurrogateBundle.load(bundle_path)
    catalog = load_catalog(bundle_path)
    test_ids = held_out_model_ids(stored_bundle)
    if not test_ids:
        raise ValueError("No held-out model IDs are recorded in the bundle.")

    if args.surrogate_source == "evaluation":
        sampled = load_sampled_data(bundle_path, args.sampled_data)
        surrogate = refit_evaluation_surrogate(stored_bundle, sampled)
        evaluation_path = output_dir / "evaluation_surrogate.joblib"
        joblib.dump(surrogate, evaluation_path, compress=3)
        status(f"Saved reconstructed evaluation surrogate: {evaluation_path}")
    else:
        surrogate = stored_bundle
        status(
            "Using the stored production surrogate. Note: it may have been refit "
            "on all models, including the recorded test models."
        )

    models = choose_models(
        catalog,
        test_ids,
        max_models=args.max_models,
        seed=args.model_seed,
    )
    status(f"Evaluating {len(models):,} held-out models.")

    inferred = infer_money_unit_info(surrogate, bundle_path=bundle_path)
    money_units = inferred.money_units if args.money_units == "auto" else args.money_units
    dollars_per_unit = (
        float(args.dollars_per_model_unit)
        if args.dollars_per_model_unit is not None
        else float(inferred.dollars_per_model_unit)
    )
    if money_units == "data":
        dollars_per_unit = 1.0
    status(
        f"Money units: {money_units}; dollars per model unit: {dollars_per_unit:,.6g} "
        f"(inference: {inferred.source})."
    )

    assumptions = make_population_assumptions(args)
    primary_check = float(args.primary_check)
    check_amounts = sorted(set(float(x) for x in args.check_amounts))
    if primary_check not in check_amounts:
        check_amounts.append(primary_check)
        check_amounts.sort()

    policy_layout = str(surrogate.training_metadata.get("policy_layout", "GHKEBA"))
    r2 = validation_r2_table(stored_bundle)
    model_rows: list[dict[str, Any]] = []
    primary_household_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for position, (_, row) in enumerate(models.iterrows(), start=1):
        model_id = str(row["model_id"])
        status(f"[{position}/{len(models)}] Model {model_id}")
        model_dir = resolve_model_dir(bundle_path, row.get("model_dir"), model_id)
        try:
            exact_grid = load_exact_policy_grid(
                model_dir,
                policy_layout=policy_layout,
            )
            for check_amount in check_amounts:
                paired = paired_mpc_frame(
                    surrogate,
                    exact_grid,
                    row,
                    assumptions=assumptions,
                    check_amount_dollars=check_amount,
                    dollars_per_model_unit=dollars_per_unit,
                    money_units=money_units,
                )
                paired.insert(0, "model_id", model_id)
                summary = model_summary_row(
                    paired,
                    model_id=model_id,
                    check_amount=check_amount,
                )
                model_rows.append(summary)
                if np.isclose(check_amount, primary_check):
                    primary_household_frames.append(paired)
                    if args.save_model_household_files:
                        write_dataframe(
                            paired,
                            household_dir / f"model_{model_id}_check_{check_amount:g}",
                        )
        except Exception as exc:
            failures.append(
                {
                    "model_id": model_id,
                    "model_dir": str(model_dir),
                    "reason": repr(exc),
                }
            )
            status(f"  FAILED: {exc}")

    model_summary = pd.DataFrame(model_rows)
    if model_summary.empty:
        failure_text = pd.DataFrame(failures).to_string(index=False)
        raise RuntimeError(f"Every held-out model failed.\n{failure_text}")

    model_summary = model_summary.merge(r2, on="model_id", how="left")
    distance = parameter_distance_table(
        catalog,
        test_ids=set(test_ids),
        parameter_columns=surrogate.feature_spec.parameter_columns,
        categorical_columns=surrogate.feature_spec.categorical_parameter_columns,
    )
    model_summary = model_summary.merge(distance, on="model_id", how="left")
    model_summary = model_summary.merge(
        catalog,
        on="model_id",
        how="left",
        suffixes=("", "_catalog"),
    )
    model_summary.to_csv(tables_dir / "model_mpc_diagnostics_all_checks.csv", index=False)

    primary_summary = model_summary[
        np.isclose(model_summary["check_amount_dollars"], primary_check)
    ].copy()
    primary_summary.to_csv(
        tables_dir / "model_mpc_diagnostics_primary_check.csv", index=False
    )

    check_summary = model_summary.groupby("check_amount_dollars", as_index=False).agg(
        n_models=("model_id", "nunique"),
        exact_mean_mpc=("exact_mean_mpc", "mean"),
        surrogate_mean_mpc=("surrogate_mean_mpc", "mean"),
        mean_mpc_bias=("mean_mpc_bias", "mean"),
        median_model_bias=("mean_mpc_bias", "median"),
        mean_mpc_rmse=("mpc_rmse", "mean"),
        median_mpc_correlation=("mpc_correlation", "median"),
        mean_baseline_consumption_bias=("baseline_consumption_bias", "mean"),
        mean_treated_consumption_bias=("treated_consumption_bias", "mean"),
    )
    check_summary.to_csv(tables_dir / "check_size_summary.csv", index=False)

    failure_frame = pd.DataFrame(failures)
    failure_frame.to_csv(tables_dir / "model_failures.csv", index=False)

    if primary_household_frames:
        household = pd.concat(primary_household_frames, ignore_index=True, sort=False)
        if not args.skip_pooled_household_export:
            write_dataframe(household, household_dir / "pooled_primary_check_households")

        bin_variables = [
            "liquid_slack",
            "liquid_slack_to_income",
            "liquid_assets",
            "liquid_assets_to_income",
            "illiquid_assets",
            "illiquid_assets_to_income",
            "current_income",
            "labor_income_state",
            "age",
            "exact_mpc",
        ]
        bin_frames: list[pd.DataFrame] = []
        for variable in bin_variables:
            if variable in household:
                summary = quantile_bin_summary(
                    household,
                    variable,
                    n_bins=args.n_bins,
                )
                if not summary.empty:
                    bin_frames.append(summary)
                    summary.to_csv(
                        tables_dir / f"binned_diagnostics_{variable}.csv",
                        index=False,
                    )
        binned = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
        if not binned.empty:
            binned.to_csv(tables_dir / "binned_diagnostics_all.csv", index=False)

        subgroup_columns = [
            column
            for column in [
                "population_type",
                "employment_state",
                "is_retired",
                "exact_state_k",
            ]
            if column in household
        ]
        subgroup_rows: list[pd.DataFrame] = []
        for column in subgroup_columns:
            table = (
                household.groupby(column, dropna=False)
                .agg(
                    n=("exact_mpc", "size"),
                    exact_mean_mpc=("exact_mpc", "mean"),
                    surrogate_mean_mpc=("surrogate_mpc", "mean"),
                    mean_mpc_error=("mpc_error", "mean"),
                    mpc_rmse=(
                        "mpc_error",
                        lambda values: float(np.sqrt(np.mean(values**2))),
                    ),
                    baseline_consumption_bias=("baseline_consumption_error", "mean"),
                    treated_consumption_bias=("treated_consumption_error", "mean"),
                )
                .reset_index()
                .rename(columns={column: "group_value"})
            )
            table.insert(0, "group_variable", column)
            subgroup_rows.append(table)
        if subgroup_rows:
            pd.concat(subgroup_rows, ignore_index=True).to_csv(
                tables_dir / "subgroup_diagnostics.csv", index=False
            )

        figure_pooled_distribution(
            household,
            check_amount=primary_check,
            path=figures_dir / "pooled_mpc_distribution.png",
        )
        figure_pooled_mpc_scatter(
            household,
            check_amount=primary_check,
            path=figures_dir / "pooled_exact_vs_surrogate_mpc.png",
            max_points=args.max_scatter_points,
            seed=args.model_seed,
        )
        figure_consumption_error_gradient(
            household,
            check_amount=primary_check,
            path=figures_dir / "baseline_vs_treated_consumption_error.png",
            max_points=args.max_scatter_points,
            seed=args.model_seed,
        )

        for variable, x_label in [
            ("liquid_slack", "Liquid assets above borrowing limit"),
            ("liquid_slack_to_income", "Liquid slack / permanent income"),
            ("liquid_assets", "Liquid assets"),
            ("exact_mpc", "Exact-model MPC"),
        ]:
            summary_path = tables_dir / f"binned_diagnostics_{variable}.csv"
            if summary_path.exists():
                summary = pd.read_csv(summary_path)
                figure_binned_line(
                    summary,
                    x="x_mean",
                    y_columns=[
                        ("exact_mean_mpc", "Exact model"),
                        ("surrogate_mean_mpc", "Surrogate"),
                    ],
                    x_label=x_label,
                    y_label="Mean MPC",
                    title=f"MPC fit by {x_label.lower()}",
                    path=figures_dir / f"mpc_by_{variable}.png",
                )
                figure_binned_line(
                    summary,
                    x="x_mean",
                    y_columns=[("mean_mpc_error", "MPC error")],
                    x_label=x_label,
                    y_label="Surrogate minus exact MPC",
                    title=f"MPC bias by {x_label.lower()}",
                    path=figures_dir / f"mpc_error_by_{variable}.png",
                )
                figure_binned_line(
                    summary,
                    x="x_mean",
                    y_columns=[
                        ("baseline_consumption_bias", "Baseline state"),
                        ("treated_consumption_bias", "After-check state"),
                    ],
                    x_label=x_label,
                    y_label="Mean surrogate consumption error",
                    title=f"Consumption-error decomposition by {x_label.lower()}",
                    path=figures_dir / f"consumption_errors_by_{variable}.png",
                )

        worst = primary_summary.reindex(
            primary_summary["mean_mpc_bias"].abs().sort_values(ascending=False).index
        ).head(args.n_worst_model_plots)
        for _, row in worst.iterrows():
            model_id = str(row["model_id"])
            figure_worst_model_distribution(
                household,
                model_id=model_id,
                check_amount=primary_check,
                path=figures_dir / "worst_models" / f"mpc_distribution_{model_id}.png",
            )

    figure_exact_vs_surrogate_mean(
        primary_summary,
        figures_dir / "model_exact_vs_surrogate_mean_mpc.png",
    )
    figure_mpc_bias_vs_r2(
        primary_summary,
        figures_dir / "model_mpc_bias_vs_consumption_r2.png",
    )
    figure_mpc_bias_vs_distance(
        primary_summary,
        figures_dir / "model_mpc_bias_vs_parameter_distance.png",
    )
    figure_ranked_model_metric(
        primary_summary,
        metric="mpc_rmse",
        label="MPC RMSE",
        path=figures_dir / "largest_model_mpc_rmse.png",
        top_n=args.ranked_models,
    )
    ranked_bias = primary_summary.copy()
    ranked_bias["absolute_mean_mpc_bias"] = ranked_bias["mean_mpc_bias"].abs()
    figure_ranked_model_metric(
        ranked_bias,
        metric="absolute_mean_mpc_bias",
        label="Absolute mean MPC bias",
        path=figures_dir / "largest_model_mean_mpc_bias.png",
        top_n=args.ranked_models,
    )
    figure_check_size_summary(
        model_summary,
        figures_dir / "mean_mpc_by_check_size.png",
    )
    figure_check_size_bias(
        model_summary,
        figures_dir / "mpc_bias_by_check_size.png",
    )

    correlations = parameter_correlations(
        primary_summary,
        catalog,
        metrics=[
            "mean_mpc_bias",
            "mpc_rmse",
            "mpc_correlation",
            "baseline_consumption_bias",
            "treated_consumption_bias",
            "mean_consumption_error_gradient",
        ],
    )
    if not correlations.empty:
        correlations.to_csv(
            tables_dir / "parameter_correlations_with_diagnostics.csv",
            index=False,
        )

    run_metadata = {
        "bundle": str(bundle_path),
        "output_dir": str(output_dir),
        "surrogate_source": args.surrogate_source,
        "check_amounts": check_amounts,
        "primary_check": primary_check,
        "money_units": money_units,
        "dollars_per_model_unit": dollars_per_unit,
        "n_models_requested": len(models),
        "n_models_succeeded": int(model_summary["model_id"].nunique()),
        "n_models_failed": len(failures),
        "population_assumptions": asdict(assumptions),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    status("\nPrimary-check summary:")
    summary_lines = {
        "models": int(primary_summary["model_id"].nunique()),
        "mean exact MPC": float(primary_summary["exact_mean_mpc"].mean()),
        "mean surrogate MPC": float(primary_summary["surrogate_mean_mpc"].mean()),
        "mean MPC bias": float(primary_summary["mean_mpc_bias"].mean()),
        "median model MPC RMSE": float(primary_summary["mpc_rmse"].median()),
        "median model MPC correlation": float(
            primary_summary["mpc_correlation"].median()
        ),
    }
    for key, value in summary_lines.items():
        status(f"  {key}: {value:,.6g}" if isinstance(value, float) else f"  {key}: {value}")
    status(f"\nDiagnostics written to: {output_dir}")

    return {
        "output_dir": output_dir,
        "model_summary": tables_dir / "model_mpc_diagnostics_primary_check.csv",
        "check_summary": tables_dir / "check_size_summary.csv",
        "figures": figures_dir,
    }


# =============================================================================
# Command line
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose exact-grid versus smooth-surrogate MPC differences."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sampled-data",
        type=Path,
        default=None,
        help="Optional path to policy_grid_sample; inferred beside the bundle.",
    )
    parser.add_argument(
        "--surrogate-source",
        choices=["evaluation", "stored"],
        default="evaluation",
        help=(
            "'evaluation' refits on training+tuning models and excludes recorded "
            "test models; 'stored' uses the saved production bundle."
        ),
    )
    parser.add_argument(
        "--check-amounts",
        type=parse_float_list,
        default=parse_float_list("500,1000,5000,10000,25000"),
    )
    parser.add_argument("--primary-check", type=float, default=10_000.0)
    parser.add_argument("--n-households", type=int, default=10_000)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--model-seed", type=int, default=123)
    parser.add_argument("--population-seed", type=int, default=20260717)
    parser.add_argument(
        "--money-units", choices=["auto", "model", "data"], default="auto"
    )
    parser.add_argument("--dollars-per-model-unit", type=float, default=None)

    # Population controls.
    parser.add_argument("--median-income-dollars", type=float, default=60_000.0)
    parser.add_argument("--permanent-income-log-sd", type=float, default=0.50)
    parser.add_argument("--transitory-income-log-sd", type=float, default=0.20)
    parser.add_argument("--high-education-share", type=float, default=0.55)
    parser.add_argument("--nonemployment-share", type=float, default=0.08)
    parser.add_argument("--retirement-age", type=float, default=65.0)
    parser.add_argument("--poor-hand-to-mouth-share", type=float, default=0.25)
    parser.add_argument("--wealthy-hand-to-mouth-share", type=float, default=0.15)
    parser.add_argument("--poor-liquid-ratio-median", type=float, default=0.015)
    parser.add_argument("--poor-illiquid-ratio-median", type=float, default=0.10)
    parser.add_argument(
        "--wealthy-htm-liquid-ratio-median", type=float, default=0.010
    )
    parser.add_argument(
        "--wealthy-htm-illiquid-ratio-median", type=float, default=2.50
    )
    parser.add_argument("--regular-liquid-ratio-median", type=float, default=0.25)
    parser.add_argument("--regular-illiquid-ratio-median", type=float, default=1.25)
    parser.add_argument("--liquid-ratio-log-sd", type=float, default=0.75)
    parser.add_argument("--illiquid-ratio-log-sd", type=float, default=0.85)
    parser.add_argument("--negative-liquid-share", type=float, default=0.08)

    # Output and plot controls.
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--max-scatter-points", type=int, default=20_000)
    parser.add_argument("--n-worst-model-plots", type=int, default=8)
    parser.add_argument("--ranked-models", type=int, default=40)
    parser.add_argument(
        "--save-model-household-files",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--skip-pooled-household-export",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.primary_check <= 0 or not np.isfinite(args.primary_check):
        raise ValueError("--primary-check must be finite and positive.")
    if args.n_households <= 0:
        raise ValueError("--n-households must be positive.")
    if args.max_models is not None and args.max_models <= 0:
        raise ValueError("--max-models must be positive when supplied.")
    if args.n_bins < 2:
        raise ValueError("--n-bins must be at least two.")
    if (
        args.poor_hand_to_mouth_share + args.wealthy_hand_to_mouth_share
        > 1.0 + EPS
    ):
        raise ValueError(
            "Poor and wealthy hand-to-mouth shares cannot sum to more than one."
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    run(args)


if __name__ == "__main__":
    main()
