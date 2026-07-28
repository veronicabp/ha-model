"""Streamlit explorer for smooth heterogeneous-agent policy functions.

Run with either backend:

    streamlit run policy_explorer.py -- \
        --surrogate-type polynomial \
        --bundle OUTPUT/policy_surrogate.joblib

    streamlit run policy_explorer.py -- \
        --surrogate-type neural \
        --bundle OUTPUT/neural_policy_surrogate.pt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

APP_ROOT = Path(__file__).resolve().parent
APP_DATA_DIR = APP_ROOT / "app_data"

DEFAULT_POLYNOMIAL_BUNDLE_PATH = APP_DATA_DIR / "policy_surrogate.joblib"
DEFAULT_POLYNOMIAL_EVALUATION_BUNDLE_PATH = (
    APP_DATA_DIR / "policy_surrogate_holdout.joblib"
)
DEFAULT_NEURAL_BUNDLE_PATH = APP_DATA_DIR / "neural_policy_surrogate.pt"
DEFAULT_NEURAL_EVALUATION_BUNDLE_PATH = (
    APP_DATA_DIR / "neural_policy_surrogate_holdout.pt"
)

SURROGATE_TYPE_LABELS = {
    "polynomial": "Polynomial",
    "neural": "Neural network",
}

try:
    import streamlit as st
except ImportError as exc:  # clearer error outside Streamlit
    raise ImportError(
        "Streamlit is required for the interactive explorer. Install it with "
        "`pip install streamlit`."
    ) from exc

from ha_policy_surrogate import read_policy_dataset
from ha_surrogate_backend import (
    SurrogateBundle,
    WebsiteSurrogateBundle,
    infer_surrogate_type,
    load_surrogate_bundle,
    normalize_surrogate_type,
)
from ha_policy_visualization import (
    DERIVED_ASSET_AXES,
    complete_derived_state,
    load_exact_grid_slice,
    make_policy_surface,
    set_plot_axis,
)
from ha_mpc_distribution import (
    ExactPolicyGrid,
    PopulationAssumptions,
    compute_check_mpcs,
    compute_exact_grid_check_mpcs,
    align_population_to_exact_grid,
    predict_exact_grid_consumption,
    generate_synthetic_population,
    infer_money_unit_info,
    load_exact_policy_grid,
    mpc_summary,
    set_parameter_values,
)

POLICY_OUTPUTS = [
    "consumption",
    "deposit",
    "delta_liquid_assets",
    "delta_illiquid_assets",
    "next_liquid_assets",
    "next_illiquid_assets",
]
GRID_COMPARISON_OUTPUTS = POLICY_OUTPUTS + ["mpc"]

SPECIAL_LABELS = {
    "consumption": "Consumption",
    "deposit": "Net Transfer into Illiquid Account (d)",
    "delta_liquid_assets": "Change in Liquid Assets (Liquid Saving)",
    "delta_illiquid_assets": "Change in Illiquid Assets (Illiquid Saving)",
    "next_liquid_assets": "Next Liquid Assets",
    "next_illiquid_assets": "Next Illiquid Assets",
    "current_income": "Pre-Tax Income",
    "after_tax_income": "After-Tax Income",
    "employment_state": "Employment",
    "param__policy__tax__kind": "Tax Schedule",
    "illiquid_assets_to_income": "Illiquid Assets / Permanent Income",
    "liquid_slack_to_income": "Liquid Slack / Permanent Income",
    "mpc": "Marginal Propensity to Consume",
    "income_deviation": "Income Deviation",
}


INCOME_DEVIATION_COLUMN = "income_deviation"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--surrogate-type",
        choices=("auto", "polynomial", "neural"),
        default="neural",
        help="Surrogate implementation used by the website.",
    )
    # Backward-compatible generic overrides. When omitted, the type-specific
    # defaults below are used.
    parser.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Optional production-bundle override.",
    )
    parser.add_argument(
        "--evaluation-bundle",
        type=str,
        default=None,
        help="Optional true-holdout bundle override.",
    )
    parser.add_argument(
        "--polynomial-bundle",
        type=str,
        default=str(DEFAULT_POLYNOMIAL_BUNDLE_PATH),
    )
    parser.add_argument(
        "--polynomial-evaluation-bundle",
        type=str,
        default=str(DEFAULT_POLYNOMIAL_EVALUATION_BUNDLE_PATH),
    )
    parser.add_argument(
        "--neural-bundle",
        type=str,
        default=str(DEFAULT_NEURAL_BUNDLE_PATH),
    )
    parser.add_argument(
        "--neural-evaluation-bundle",
        type=str,
        default=str(DEFAULT_NEURAL_EVALUATION_BUNDLE_PATH),
    )
    parser.add_argument(
        "--neural-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Inference device for a neural surrogate.",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


def _initial_surrogate_type(args: argparse.Namespace) -> str:
    requested = normalize_surrogate_type(args.surrogate_type)
    if requested != "auto":
        return requested
    if args.bundle:
        try:
            return infer_surrogate_type(args.bundle)
        except ValueError:
            pass
    # Prefer the only deployed backend when exactly one default bundle exists;
    # otherwise preserve the historical polynomial default.
    polynomial_exists = Path(args.polynomial_bundle).expanduser().exists()
    neural_exists = Path(args.neural_bundle).expanduser().exists()
    if neural_exists and not polynomial_exists:
        return "neural"
    return "polynomial"


def _default_bundle_path(args: argparse.Namespace, surrogate_type: str) -> str:
    if args.bundle:
        try:
            if infer_surrogate_type(args.bundle) == surrogate_type:
                return str(args.bundle)
        except ValueError:
            if normalize_surrogate_type(args.surrogate_type) == surrogate_type:
                return str(args.bundle)
    return str(
        args.neural_bundle if surrogate_type == "neural" else args.polynomial_bundle
    )


def _default_evaluation_bundle_path(
    args: argparse.Namespace,
    surrogate_type: str,
) -> str:
    if args.evaluation_bundle:
        try:
            if infer_surrogate_type(args.evaluation_bundle) == surrogate_type:
                return str(args.evaluation_bundle)
        except ValueError:
            if normalize_surrogate_type(args.surrogate_type) == surrogate_type:
                return str(args.evaluation_bundle)
    return str(
        args.neural_evaluation_bundle
        if surrogate_type == "neural"
        else args.polynomial_evaluation_bundle
    )


@st.cache_resource(show_spinner=False)
def _load_bundle(
    path: str,
    surrogate_type: str,
    neural_device: str,
) -> WebsiteSurrogateBundle:
    return load_surrogate_bundle(
        path,
        surrogate_type=surrogate_type,
        neural_device=neural_device,
    )


def _validate_evaluation_bundle(bundle: SurrogateBundle) -> None:
    metadata = dict(bundle.training_metadata or {})
    fit_ids = set(map(str, metadata.get("fit_models", [])))
    test_ids = set(map(str, metadata.get("test_models", [])))
    if metadata.get("fit_scope") != "development_only":
        raise ValueError(
            "The evaluation bundle was not fit on development models only."
        )
    if not test_ids or fit_ids & test_ids:
        raise ValueError("The evaluation bundle does not contain a disjoint test set.")


@st.cache_resource(show_spinner=False)
def _load_exact_policy_grid_cached(
    model_dir: str,
    policy_layout: str,
) -> ExactPolicyGrid:
    return load_exact_policy_grid(
        model_dir,
        policy_layout=policy_layout,
    )


def _training_sample_path(bundle_path: str | Path) -> Path | None:
    """Return the sampled-policy file stored beside the fitted bundle."""

    resolved = Path(bundle_path).expanduser().resolve()
    parent = resolved.parent
    deployment = [
        parent / "heldout_mpc_pairs.parquet",
        parent / "heldout_mpc_pairs.pkl.gz",
    ]
    polynomial = [
        parent / "policy_grid_sample.parquet",
        parent / "policy_grid_sample.pkl.gz",
        parent / "policy_grid_sample.pkl",
        parent / "policy_grid_sample.csv",
    ]
    neural = [
        parent / "neural_policy_grid_sample.parquet",
        parent / "neural_policy_grid_sample.pkl.gz",
        parent / "neural_policy_grid_sample.pkl",
        parent / "neural_policy_grid_sample.csv",
    ]
    model_specific = (
        neural + polynomial
        if resolved.suffix.lower()
        in {
            ".pt",
            ".pth",
            ".ckpt",
        }
        else polynomial + neural
    )
    candidates = deployment + model_specific
    return next((path for path in candidates if path.exists()), None)


@st.cache_data(show_spinner=False, max_entries=1)
def _load_training_sample(bundle_path: str) -> pd.DataFrame:
    """Load the sampled grid data saved beside the surrogate bundle."""

    path = _training_sample_path(bundle_path)
    return read_policy_dataset(path) if path is not None else pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=1)
def _load_output_axis_ranges(
    bundle_path: str,
    file_signature: int | None,
) -> dict[str, tuple[float, float]]:
    """Load compact quantile-based output ranges saved beside the bundle."""

    del file_signature  # Used only to invalidate the cache when the file changes.
    path = Path(bundle_path).expanduser().resolve().parent / "output_axis_ranges.csv"
    if not path.exists():
        return {}

    table = pd.read_csv(path)
    required = {"output", "lower", "upper"}
    if not required.issubset(table.columns):
        return {}

    ranges: dict[str, tuple[float, float]] = {}
    for row in table.itertuples(index=False):
        lo = float(row.lower)
        hi = float(row.upper)
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            ranges[str(row.output)] = (lo, hi)
    return ranges


def _stored_output_axis_ranges(
    bundle_path: str | Path,
) -> dict[str, tuple[float, float]]:
    path = Path(bundle_path).expanduser().resolve().parent / "output_axis_ranges.csv"
    signature = int(path.stat().st_mtime_ns) if path.exists() else None
    return _load_output_axis_ranges(str(bundle_path), signature)


@st.cache_data(show_spinner=False)
def _load_model_catalog(bundle_path: str) -> pd.DataFrame:
    parent = Path(bundle_path).expanduser().resolve().parent
    candidates = [
        parent / "model_catalog.csv",
        parent / "neural_model_catalog.csv",
    ]
    catalog_path = next((path for path in candidates if path.exists()), None)

    if catalog_path is None:
        names = ", ".join(path.name for path in candidates)
        raise FileNotFoundError(
            f"Could not find a model catalog beside the bundle. Expected one "
            f"of: {names}."
        )

    catalog = pd.read_csv(catalog_path)

    if "usable" in catalog:
        usable = catalog["usable"].astype(str).str.lower().isin({"true", "1", "yes"})
        catalog = catalog.loc[usable].copy()

    if catalog.empty:
        raise ValueError("model_catalog.csv contains no usable models.")

    return catalog.reset_index(drop=True)


def _out_of_sample_model_summary(
    bundle: SurrogateBundle,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """One row per held-out model with consumption and deposit R²."""

    metrics = bundle.validation_by_model.copy()

    required = {
        "split",
        "model_id",
        "target",
        "r2",
    }
    missing = required.difference(metrics.columns)

    if missing:
        raise ValueError(
            "validation_by_model is missing columns: " + ", ".join(sorted(missing))
        )

    metrics["model_id"] = metrics["model_id"].astype(str)
    metrics["target"] = metrics["target"].astype(str)

    held_out = metrics[
        metrics["split"]
        .astype(str)
        .str.contains(
            "held_out",
            case=False,
            na=False,
        )
    ].copy()

    held_out = held_out[held_out["target"].isin(["consumption", "deposit"])]

    if held_out.empty:
        raise ValueError("No held-out consumption or deposit metrics were found.")

    summary = (
        held_out.pivot_table(
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

    if "consumption_r2" not in summary:
        summary["consumption_r2"] = np.nan

    if "deposit_r2" not in summary:
        summary["deposit_r2"] = np.nan

    summary["mean_r2"] = summary[["consumption_r2", "deposit_r2"]].mean(axis=1)

    catalog_small = catalog.copy()
    catalog_small["model_id"] = catalog_small["model_id"].astype(str)

    catalog_small = catalog_small[["model_id", "model_dir"]].drop_duplicates(
        subset=["model_id"]
    )

    summary = summary.merge(
        catalog_small,
        on="model_id",
        how="inner",
        validate="one_to_one",
    )
    summary = summary[
        summary["model_dir"].notna()
        & summary["model_dir"].astype(str).str.strip().ne("")
    ].copy()

    if summary.empty:
        raise ValueError(
            "No held-out validation models are present in model_catalog.csv."
        )

    summary = summary.sort_values(
        ["mean_r2", "model_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return summary


def _default_output_range(
    bundle: SurrogateBundle,
    sample: pd.DataFrame,
    output: str,
    *,
    stored_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Return a stable initial display range for one policy outcome."""

    stored_ranges = stored_ranges or {}

    if output == "mpc" and output not in stored_ranges:
        lo, hi = -0.05, 1.05

    elif output in stored_ranges:
        lo, hi = map(float, stored_ranges[output])

    elif not sample.empty and output in sample:
        values = pd.to_numeric(sample[output], errors="coerce")
        values = values[np.isfinite(values)]

        if len(values):
            lo = float(values.quantile(0.005))
            hi = float(values.quantile(0.995))
        else:
            lo, hi = -1.0, 1.0

    elif output in bundle.output_bounds:
        lo, hi = map(float, bundle.output_bounds[output])

    elif output == "delta_liquid_assets":
        next_lo, next_hi = bundle.output_bounds["next_liquid_assets"]
        current = bundle.feature_ranges["liquid_assets"]
        lo = float(next_lo) - float(current["max"])
        hi = float(next_hi) - float(current["min"])

    elif output == "delta_illiquid_assets":
        next_lo, next_hi = bundle.output_bounds["next_illiquid_assets"]
        current = bundle.feature_ranges["illiquid_assets"]
        lo = float(next_lo) - float(current["max"])
        hi = float(next_hi) - float(current["min"])

    else:
        lo, hi = -1.0, 1.0

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = -1.0, 1.0

    if hi <= lo + 1.0e-12:
        center = 0.5 * (lo + hi)
        lo, hi = center - 0.5, center + 0.5

    padding = 0.05 * (hi - lo)
    return lo - padding, hi + padding


def _reset_output_axis_state(
    range_key: str,
    min_widget_key: str,
    max_widget_key: str,
    default_range: tuple[float, float],
) -> None:
    """Reset axis widgets before Streamlit instantiates them on the rerun."""

    lo, hi = map(float, default_range)
    st.session_state[range_key] = (lo, hi)
    st.session_state[min_widget_key] = lo
    st.session_state[max_widget_key] = hi


def _output_axis_control(
    bundle: SurrogateBundle,
    sample: pd.DataFrame,
    output: str,
    *,
    context: str,
    bundle_path: str | Path,
) -> tuple[float, float]:
    """Persistent axis limits that do not change when other controls move."""

    stored_ranges = _stored_output_axis_ranges(bundle_path)
    default_range = _default_output_range(
        bundle,
        sample,
        output,
        stored_ranges=stored_ranges,
    )

    range_key = f"{context}__fixed_output_range__{output}"
    min_widget_key = f"{context}__ymin__{output}"
    max_widget_key = f"{context}__ymax__{output}"
    default_key = f"{context}__axis_default__{output}"

    normalized_default = tuple(round(float(x), 12) for x in default_range)
    previous_default = st.session_state.get(default_key)

    # Refresh stale full-range state when output_axis_ranges.csv first appears
    # or is replaced. User edits remain persistent while the default is unchanged.
    if previous_default != normalized_default:
        st.session_state[default_key] = normalized_default
        st.session_state[range_key] = tuple(map(float, default_range))
        st.session_state[min_widget_key] = float(default_range[0])
        st.session_state[max_widget_key] = float(default_range[1])
    elif range_key not in st.session_state:
        st.session_state[range_key] = tuple(map(float, default_range))

    current_lo, current_hi = st.session_state[range_key]

    if min_widget_key not in st.session_state:
        st.session_state[min_widget_key] = float(current_lo)
    if max_widget_key not in st.session_state:
        st.session_state[max_widget_key] = float(current_hi)

    with st.popover("Change y-axis"):
        st.caption(
            "These limits remain fixed when household states or model "
            "parameters change."
        )

        lo = st.number_input(
            "Minimum",
            key=min_widget_key,
            format="%.6g",
        )
        hi = st.number_input(
            "Maximum",
            key=max_widget_key,
            format="%.6g",
        )

        if float(hi) <= float(lo):
            st.error("The maximum must exceed the minimum.")
        else:
            st.session_state[range_key] = (float(lo), float(hi))

        st.button(
            "Reset to data range",
            key=f"{context}__reset_y__{output}__v2",
            width="stretch",
            on_click=_reset_output_axis_state,
            args=(
                range_key,
                min_widget_key,
                max_widget_key,
                default_range,
            ),
        )

    return tuple(st.session_state[range_key])


def _continuous_plot_columns(
    bundle: SurrogateBundle,
) -> list[str]:
    plot_engineered_columns = [
        "liquid_slack_to_income",
        "illiquid_assets_to_income",
    ]

    columns = (
        list(bundle.feature_spec.state_columns)
        + [
            column
            for column in plot_engineered_columns
            if (
                column
                in getattr(
                    bundle.feature_spec,
                    "engineered_state_columns",
                    (),
                )
                and column in bundle.feature_ranges
            )
        ]
        + [
            column
            for column in bundle.feature_spec.parameter_columns
            if not column.endswith("__missing")
        ]
    )
    excluded = {
        "after_tax_income",
        "labor_income_state",
        "total_assets",
        "cash_on_hand",
        "years_to_retirement",
    }

    columns = [
        column
        for column in columns
        if column in bundle.feature_ranges and column not in excluded
    ]

    preferred = [
        "current_income",
        "age",
        "liquid_assets",
        "liquid_slack_to_income",
        "illiquid_assets",
        "illiquid_assets_to_income",
    ]

    return [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]


def _retirement_age(
    bundle: SurrogateBundle,
    base: dict[str, Any] | pd.Series | None = None,
) -> float:
    """Return the retirement threshold used by the fitted HA accounting."""

    values = {} if base is None else dict(base)
    candidates = (
        values.get("param__config__retirement_age"),
        values.get("acct__retirement_age"),
        bundle.default_values.get("param__config__retirement_age"),
        bundle.default_values.get("acct__retirement_age"),
        bundle.accounting_defaults.get("acct__retirement_age"),
    )
    for value in candidates:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return 65.0


def _working_age_max(
    bundle: SurrogateBundle,
    base: dict[str, Any] | pd.Series | None = None,
) -> float:
    """Largest integer age that the solver treats as working age."""

    return float(np.ceil(_retirement_age(bundle, base)) - 1.0)


def _working_age_range_info(
    bundle: SurrogateBundle,
    info: dict[str, Any],
    base: dict[str, Any] | pd.Series | None = None,
) -> dict[str, Any]:
    """Cap age-widget metadata at the working-age boundary."""

    out = dict(info)
    cap = _working_age_max(bundle, base)
    for key in ("max", "q99"):
        if key in out:
            out[key] = min(float(out[key]), cap)
    if "median" in out:
        out["median"] = min(float(out["median"]), cap)
    if "min" in out and float(out["min"]) > cap:
        raise ValueError("The fitted bundle has no working-age support.")
    return out


def _employment_option_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"0", "0.0", "employed", "working", "true"}:
        return "Employed"
    return "Unemployed"


def _borrowing_limit_values(
    frame: pd.DataFrame,
    *,
    fallback: float = 0.0,
) -> np.ndarray:
    """Return the row-specific lower bound for liquid assets."""

    for column in (
        "param__config__borrowing_limit",
        "liquid_grid_min",
    ):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            return np.where(np.isfinite(values), values, float(fallback))
    return np.full(len(frame), float(fallback), dtype=float)


def _borrowing_limit_from_base(
    bundle: SurrogateBundle,
    base: dict[str, Any] | pd.Series,
) -> float:
    """Resolve the borrowing limit used by a one-dimensional plot."""

    values = dict(base)
    for column in (
        "param__config__borrowing_limit",
        "liquid_grid_min",
    ):
        value = values.get(column, bundle.default_values.get(column))
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return 0.0


def _shade_liquid_infeasible_region(
    fig: go.Figure,
    *,
    axis_variable: str,
    axis_range: tuple[float, float],
    bundle: SurrogateBundle,
    base: dict[str, Any] | pd.Series,
) -> None:
    """Shade the part of a one-dimensional axis violating b >= borrowing limit."""

    lo, hi = map(float, axis_range)
    limit = _borrowing_limit_from_base(bundle, base)
    if axis_variable == "liquid_assets" and lo < limit:
        fig.add_vrect(
            x0=lo,
            x1=min(limit, hi),
            fillcolor="#bdbdbd",
            opacity=0.55,
            line_width=0,
            layer="below",
            annotation_text="Infeasible",
            annotation_position="top left",
        )
    elif axis_variable == "param__config__borrowing_limit":
        liquid = float(
            dict(base).get(
                "liquid_assets",
                bundle.default_values.get("liquid_assets", 0.0),
            )
        )
        if hi > liquid:
            fig.add_vrect(
                x0=max(liquid, lo),
                x1=hi,
                fillcolor="#bdbdbd",
                opacity=0.55,
                line_width=0,
                layer="below",
                annotation_text="Infeasible",
                annotation_position="top right",
            )


def _pretty(name: str) -> str:
    if name in SPECIAL_LABELS:
        return SPECIAL_LABELS[name]
    return (
        name.replace("param__config__", "")
        .replace("param__policy__", "policy: ")
        .replace("param__derived__", "derived: ")
        .replace("__", " / ")
        .replace("_", " ")
        .title()
    )


def _button_label(value: Any) -> str:
    """Format mixed parameter/state values for display tables."""

    if value is None or value is pd.NA:
        return ""

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float) and not np.isfinite(value):
        return ""

    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if np.isfinite(number):
            return f"{number:.6g}"

    return str(value)


def _continuous_control(
    label: str,
    info: dict[str, Any],
    default: float,
    *,
    key: str,
    use_full_range: bool,
    disabled: bool = False,
) -> float:
    lo_key, hi_key = ("min", "max") if use_full_range else ("q01", "q99")
    lo = float(info.get(lo_key, info["min"]))
    hi = float(info.get(hi_key, info["max"]))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return float(default)

    if hi <= lo + 1.0e-14:
        st.sidebar.number_input(
            label,
            value=float(default),
            disabled=True,
            key=key,
        )
        return float(default)

    value = float(np.clip(default, lo, hi))
    step = max((hi - lo) / 250.0, 1.0e-12)

    return float(
        st.sidebar.slider(
            label,
            min_value=lo,
            max_value=hi,
            value=value,
            step=step,
            key=key,
            disabled=disabled,
        )
    )


def _income_deviation_range(
    bundle: SurrogateBundle,
) -> tuple[float, float]:
    """Return a stable range for current-income deviations from permanent income.

    The public control is

        current pre-tax income / permanent income - 1.

    Thus zero means that current and permanent income are equal.  The range is
    inferred from the fitted ``pre_tax_income_to_income`` feature but is always
    constrained to keep permanent income positive.
    """

    info = bundle.feature_ranges.get("pre_tax_income_to_income", {})
    ratio_lo = float(info.get("q01", info.get("min", 0.25)))
    ratio_hi = float(info.get("q99", info.get("max", 1.75)))
    if not np.isfinite(ratio_lo):
        ratio_lo = 0.25
    if not np.isfinite(ratio_hi):
        ratio_hi = 1.75
    lo = max(ratio_lo - 1.0, -0.95)
    hi = max(ratio_hi - 1.0, lo + 0.05)
    # The no-deviation state must always be available, even when it is just
    # outside an empirical quantile range.
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    return float(lo), float(hi)


def _sidebar_axis_bindings(
    axis_columns: set[str] | list[str] | tuple[str, ...],
) -> dict[str, str]:
    """Map plot axes to the primitive sidebar controls they determine.

    Normalized asset axes determine the corresponding asset level conditional
    on permanent income.  They therefore lock only the asset control; pre-tax
    income remains adjustable and changes permanent income through the income-
    deviation relationship.
    """

    primitive_for_axis = {
        "liquid_slack_to_income": "liquid_assets",
        "illiquid_assets_to_income": "illiquid_assets",
    }
    bindings: dict[str, str] = {}
    for axis in axis_columns:
        if not axis:
            continue
        bindings[primitive_for_axis.get(axis, axis)] = axis
    return bindings


def _exact_comparison_x_options(
    bundle: SurrogateBundle,
) -> list[str]:
    return [
        column
        for column in (
            "age",
            "current_income",
            "labor_income_state",
            "liquid_assets",
            "liquid_slack_to_income",
            "illiquid_assets",
            "illiquid_assets_to_income",
        )
        if column in bundle.feature_ranges
    ]


def _page_axis_columns(
    bundle: SurrogateBundle,
    selected_page: str,
) -> set[str]:
    """Resolve the axes that should lock sidebar controls on this rerun."""

    continuous = _continuous_plot_columns(bundle)
    if not continuous:
        return set()

    default_x = "current_income" if "current_income" in continuous else continuous[0]

    if selected_page == "One-dimensional slices":
        x = str(st.session_state.get("one_d_x", default_x))
        return {x if x in continuous else default_x}

    if selected_page == "Two-dimensional surfaces":
        x = str(st.session_state.get("surface_x", default_x))
        if x not in continuous:
            x = default_x
        family = {
            "liquid_assets": "liquid",
            "liquid_slack_to_income": "liquid",
            "illiquid_assets": "illiquid",
            "illiquid_assets_to_income": "illiquid",
        }
        x_family = family.get(x)
        y_options = [
            column
            for column in continuous
            if column != x and (x_family is None or family.get(column) != x_family)
        ]
        if not y_options:
            return {x}
        default_y = "liquid_assets" if "liquid_assets" in y_options else y_options[0]
        y = str(st.session_state.get("surface_y", default_y))
        if y not in y_options:
            y = default_y
        return {x, y}

    if selected_page == "Grid versus surrogate":
        options = _exact_comparison_x_options(bundle)
        if not options:
            return set()
        default = "current_income" if "current_income" in options else options[0]
        x = str(st.session_state.get("exact_comparison_x", default))
        return {x if x in options else default}

    return set()


def _reset_parameter_controls(bundle: SurrogateBundle) -> None:
    """Reset every structural and policy parameter widget."""

    for column in bundle.feature_spec.parameter_columns:
        if column.endswith("__missing"):
            continue
        st.session_state.pop(f"param_{column}", None)
    for column in bundle.feature_spec.categorical_parameter_columns:
        st.session_state.pop(f"cat_param_v2_{column}", None)


def _base_controls(
    bundle: SurrogateBundle,
    *,
    excluded_columns: set[str] | None = None,
    disabled_state_columns: set[str] | None = None,
    axis_controlled_by: dict[str, str] | None = None,
) -> dict[str, Any]:
    excluded_columns = set(excluded_columns or ())
    disabled_state_columns = set(disabled_state_columns or ())
    axis_controlled_by = dict(axis_controlled_by or {})
    for column in excluded_columns:
        axis_controlled_by.setdefault(column, column)

    base = dict(bundle.default_values)
    original_parameters = [
        c for c in bundle.feature_spec.parameter_columns if not c.endswith("__missing")
    ]

    st.sidebar.header("Structural and policy parameters")
    st.sidebar.button(
        "Default Parameters",
        on_click=_reset_parameter_controls,
        args=(bundle,),
        width="stretch",
        help="Reset all structural and policy parameters to bundle defaults.",
    )
    st.sidebar.caption(
        "Controls are restricted to values represented in the solved-model sample."
    )
    for col in original_parameters:
        info = bundle.feature_ranges[col]

        if col in axis_controlled_by:
            base[col] = bundle.default_values[col]
            _continuous_control(
                _pretty(col),
                info,
                float(base[col]),
                key=f"axis_locked_param_{col}",
                use_full_range=True,
                disabled=True,
            )
            st.sidebar.caption(
                f"{_pretty(col)} is controlled by the "
                f"{_pretty(axis_controlled_by[col])} plot axis."
            )
            continue

        base[col] = _continuous_control(
            _pretty(col),
            info,
            float(base[col]),
            key=f"param_{col}",
            use_full_range=True,
        )
        missing_col = f"{col}__missing"
        if missing_col in bundle.feature_spec.parameter_columns:
            base[missing_col] = 0

    for col in bundle.feature_spec.categorical_parameter_columns:
        info = bundle.feature_ranges[col]
        values = list(info.get("values", []))
        if not values:
            continue
        default_str = str(base.get(col, info.get("mode", values[0])))
        index = values.index(default_str) if default_str in values else 0
        widget_key = f"cat_param_v2_{col}"
        if (
            widget_key in st.session_state
            and st.session_state[widget_key] not in values
        ):
            st.session_state.pop(widget_key, None)
        base[col] = st.sidebar.selectbox(
            _pretty(col),
            values,
            index=index,
            key=widget_key,
        )

    st.sidebar.header("State held fixed")

    derived_or_recomputed = {
        "after_tax_income",
        "labor_income_state",
        "total_assets",
        "cash_on_hand",
        "years_to_retirement",
    }

    for col in bundle.feature_spec.state_columns:
        if col in derived_or_recomputed:
            continue

        if col in axis_controlled_by:
            base[col] = bundle.default_values[col]
            info = bundle.feature_ranges[col]
            if col == "age":
                info = _working_age_range_info(bundle, info, base)
            _continuous_control(
                _pretty(col),
                info,
                float(base[col]),
                key=f"axis_locked_state_{col}",
                use_full_range=False,
                disabled=True,
            )
            st.sidebar.caption(
                f"{_pretty(col)} is controlled by the "
                f"{_pretty(axis_controlled_by[col])} plot axis."
            )
            continue

        info = bundle.feature_ranges[col]
        if col == "age":
            info = _working_age_range_info(bundle, info, base)
        disabled = col in disabled_state_columns

        base[col] = _continuous_control(
            _pretty(col),
            info,
            float(base[col]),
            key=f"state_{col}",
            use_full_range=False,
            disabled=disabled,
        )

        if disabled:
            st.sidebar.caption(
                f"{_pretty(col)} moves proportionally with current income."
            )

    deviation_lo, deviation_hi = _income_deviation_range(bundle)
    deviation_step = max((deviation_hi - deviation_lo) / 250.0, 1.0e-4)
    deviation_key = "state_income_deviation_v1"
    if deviation_key in st.session_state:
        try:
            stored_deviation = float(st.session_state[deviation_key])
        except (TypeError, ValueError):
            st.session_state.pop(deviation_key, None)
        else:
            if not deviation_lo <= stored_deviation <= deviation_hi:
                st.session_state.pop(deviation_key, None)
    base[INCOME_DEVIATION_COLUMN] = float(
        st.sidebar.slider(
            "Income Deviation",
            min_value=float(deviation_lo),
            max_value=float(deviation_hi),
            value=0.0,
            step=float(deviation_step),
            key=deviation_key,
            help=(
                "Current pre-tax income divided by permanent income, minus one. "
                "Zero means current pre-tax income equals permanent income."
            ),
        )
    )
    st.sidebar.caption(
        "Permanent income is linked to Pre-Tax Income using this deviation."
    )

    st.sidebar.header("Discrete household state")
    for col in bundle.feature_spec.categorical_state_columns:
        if col == "is_retired":
            # The website is intentionally restricted to working-age states.
            values = list(bundle.feature_ranges.get(col, {}).get("values", []))
            not_retired = next(
                (
                    value
                    for value in values
                    if str(value).strip().lower()
                    in {"0", "0.0", "false", "working_age", "not_retired"}
                ),
                "0",
            )
            base[col] = not_retired
            continue
        info = bundle.feature_ranges[col]
        values = list(info.get("values", []))
        if not values:
            continue
        default_str = str(base.get(col, info.get("mode", values[0])))
        if col == "employment_state":
            employed_value = next(
                (
                    value
                    for value in values
                    if str(value).strip().lower() in {"0", "0.0", "employed", "true"}
                ),
                None,
            )
            if employed_value is not None:
                default_str = str(employed_value)
        index = values.index(default_str) if default_str in values else 0
        widget_key = f"cat_state_v4_{col}"
        if (
            widget_key in st.session_state
            and st.session_state[widget_key] not in values
        ):
            st.session_state.pop(widget_key, None)
        if col == "employment_state":
            base[col] = st.sidebar.radio(
                "Employment",
                values,
                index=index,
                key=widget_key,
                format_func=_employment_option_label,
                horizontal=True,
            )
        else:
            base[col] = st.sidebar.selectbox(
                _pretty(col),
                values,
                index=index,
                key=widget_key,
            )

    for col in (
        "liquid_grid_min",
        "liquid_grid_max",
        "illiquid_grid_min",
        "illiquid_grid_max",
        "money_scale",
    ):
        if col in bundle.default_values:
            base[col] = bundle.default_values[col]
    return base


def _range_selector(bundle: SurrogateBundle, col: str, key: str) -> tuple[float, float]:
    info = bundle.feature_ranges[col]
    if col == "age":
        info = _working_age_range_info(bundle, info)
    lo = float(info.get("q01", info["min"]))
    hi = float(info.get("q99", info["max"]))
    if hi <= lo:
        return lo, hi
    minimum = float(info["min"])
    maximum = float(info["max"])
    widget_key = f"{key}__{col}"
    existing = st.session_state.get(widget_key)
    if existing is not None:
        try:
            existing_lo, existing_hi = map(float, existing)
        except (TypeError, ValueError):
            st.session_state.pop(widget_key, None)
        else:
            if not (
                minimum <= existing_lo <= existing_hi <= maximum
                and existing_hi > existing_lo
            ):
                st.session_state.pop(widget_key, None)
    return st.slider(
        f"{_pretty(col)} range",
        min_value=minimum,
        max_value=maximum,
        value=(lo, hi),
        step=max((maximum - minimum) / 250.0, 1.0e-12),
        key=widget_key,
    )


def _exact_grid_range_selector(
    exact_support: pd.DataFrame,
    x: str,
    *,
    model_id: str,
    key: str,
) -> tuple[float, float]:
    """Select a range whose endpoints are solved points for one exact model."""

    values = (
        pd.to_numeric(exact_support["__x"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .unique()
    )
    values = np.sort(values.astype(float))
    if len(values) == 0:
        raise ValueError(
            f"The selected exact model has no finite values for {_pretty(x)}."
        )
    if len(values) == 1:
        value = float(values[0])
        st.caption(
            f"{_pretty(x)} has only one solved value in this slice: {value:.6g}."
        )
        return value, value

    signature = hashlib.sha1(np.round(values, 12).tobytes()).hexdigest()[:12]
    selected = st.select_slider(
        f"{_pretty(x)} range",
        options=values.tolist(),
        value=(float(values[0]), float(values[-1])),
        key=f"{key}__{model_id}__{x}__{signature}",
        format_func=lambda value: f"{float(value):.6g}",
        help=(
            "The endpoints are restricted to values solved for the selected "
            "exact model and fixed household states."
        ),
    )
    return float(selected[0]), float(selected[1])


def _link_income_states_to_pre_tax_income(
    bundle: SurrogateBundle,
    frame: pd.DataFrame,
    current_income_values: np.ndarray,
    *,
    base_values: dict[str, Any] | None = None,
    proportional_after_tax: bool = True,
) -> pd.DataFrame:
    """Link permanent income to current pre-tax income and its deviation.

    ``income_deviation`` is defined as current pre-tax income divided by
    permanent income, minus one.  Holding it fixed while current income moves
    gives

        permanent income = current income / (1 + income_deviation).

    For plotted income slices, disposable income can additionally move with the
    selected baseline net-to-gross ratio.
    """

    out = frame.copy()
    base = dict(base_values or {})
    current = np.asarray(current_income_values, dtype=float)

    current_info = bundle.feature_ranges.get("current_income", {})
    anchor_current = float(
        base.get(
            "current_income",
            bundle.default_values.get(
                "current_income", current_info.get("median", 1.0)
            ),
        )
    )
    if not np.isfinite(anchor_current) or abs(anchor_current) <= 1.0e-12:
        anchor_current = float(current_info.get("median", 1.0))
    if not np.isfinite(anchor_current) or abs(anchor_current) <= 1.0e-12:
        anchor_current = 1.0

    if "labor_income_state" in bundle.feature_spec.state_columns:
        labor_info = bundle.feature_ranges.get("labor_income_state", {})
        anchor_labor = float(
            base.get(
                "labor_income_state",
                bundle.default_values.get(
                    "labor_income_state", labor_info.get("median", anchor_current)
                ),
            )
        )
        if not np.isfinite(anchor_labor):
            anchor_labor = anchor_current

        if INCOME_DEVIATION_COLUMN in out:
            deviation = pd.to_numeric(
                out[INCOME_DEVIATION_COLUMN], errors="coerce"
            ).to_numpy(dtype=float)
        elif INCOME_DEVIATION_COLUMN in base:
            deviation = np.full(
                len(out),
                float(base[INCOME_DEVIATION_COLUMN]),
                dtype=float,
            )
        else:
            anchor_ratio = anchor_current / max(abs(anchor_labor), 1.0e-12)
            deviation = np.full(len(out), anchor_ratio - 1.0, dtype=float)
        deviation = np.where(np.isfinite(deviation), deviation, 0.0)
        income_ratio = np.maximum(1.0 + deviation, 1.0e-6)
        out["labor_income_state"] = current / income_ratio

    if proportional_after_tax:
        scale = current / anchor_current
        anchor_after_tax = float(
            base.get(
                "after_tax_income",
                bundle.default_values.get("after_tax_income", anchor_current),
            )
        )
        if np.isfinite(anchor_after_tax):
            out["__after_tax_income_override"] = np.maximum(
                anchor_after_tax * scale, 0.0
            )

    return out


def _line_slice(
    bundle: SurrogateBundle,
    base: dict[str, Any],
    *,
    x: str,
    x_range: tuple[float, float],
    overlay: str | None,
    overlay_values: list[Any],
    outputs: list[str],
    y_range: tuple[float, float],
    n_points: int = 220,
) -> go.Figure:
    x_values = np.linspace(float(x_range[0]), float(x_range[1]), n_points)
    fig = go.Figure()
    curve_values = overlay_values if overlay is not None else [None]

    for curve_value in curve_values:

        frame = pd.DataFrame([base] * n_points)
        label = "baseline"

        # Set the primitive x-axis first. Derived ratios are translated below.
        if x not in DERIVED_ASSET_AXES:
            frame = set_plot_axis(
                frame,
                x,
                x_values,
            )

        if overlay is not None:
            frame[overlay] = curve_value
            label = f"{_pretty(overlay)} = {curve_value}"

        if x == "current_income" or overlay == "current_income":
            frame = _link_income_states_to_pre_tax_income(
                bundle,
                frame,
                pd.to_numeric(frame["current_income"], errors="coerce").to_numpy(
                    dtype=float
                ),
                base_values=base,
            )

        # Translate the requested ratio into the corresponding asset level.
        if x in DERIVED_ASSET_AXES:
            frame = set_plot_axis(
                frame,
                x,
                x_values,
            )

        frame = complete_derived_state(frame)

        pred = bundle.predict(frame)
        liquid = pd.to_numeric(frame["liquid_assets"], errors="coerce").to_numpy(
            dtype=float
        )
        borrowing_limit = _borrowing_limit_values(
            frame,
            fallback=_borrowing_limit_from_base(bundle, base),
        )
        feasible = (
            np.isfinite(liquid)
            & np.isfinite(borrowing_limit)
            & (liquid >= borrowing_limit - 1.0e-12)
        )
        for output in outputs:
            plotted_values = pd.to_numeric(pred[output], errors="coerce").to_numpy(
                dtype=float
            )
            plotted_values = np.where(feasible, plotted_values, np.nan)
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=plotted_values,
                    mode="lines",
                    name=(
                        _pretty(output)
                        if overlay is None
                        else f"{_pretty(output)}; {label}"
                    ),
                    legendgroup=label,
                )
            )

    _shade_liquid_infeasible_region(
        fig,
        axis_variable=x,
        axis_range=x_range,
        bundle=bundle,
        base=base,
    )

    fig.update_layout(
        template="plotly_white",
        height=620,
        xaxis_title=_pretty(x),
        yaxis_title=_pretty(outputs[0]),
        yaxis=dict(
            range=[float(y_range[0]), float(y_range[1])],
            autorange=False,
        ),
        legend_title="Output / variation",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _one_dimensional_page(
    bundle: SurrogateBundle,
    base: dict[str, Any],
    *,
    bundle_path: str | Path,
) -> None:
    continuous = _continuous_plot_columns(bundle)

    default_x = "current_income" if "current_income" in continuous else continuous[0]

    current_x = st.session_state.get(
        "one_d_x",
        default_x,
    )

    if current_x not in continuous:
        current_x = default_x

    c1, c2, c3 = st.columns([1.3, 1.2, 1.5])

    with c1:
        x = st.selectbox(
            "Horizontal axis",
            continuous,
            index=continuous.index(current_x),
            format_func=_pretty,
            key="one_d_x",
        )
    with c2:
        output = st.selectbox(
            "Policy choice",
            POLICY_OUTPUTS,
            index=0,
            format_func=_pretty,
            key="one_d_output",
        )
    with c3:
        overlay_options: list[str | None] = [None] + [
            col
            for col in (continuous + list(bundle.feature_spec.categorical_columns))
            if col != x and col != "is_retired"
        ]
        overlay = st.selectbox(
            "Overlay several values of",
            overlay_options,
            format_func=lambda x: "No overlay" if x is None else _pretty(x),
        )

    x_range = _range_selector(bundle, x, "one_d_x_range")
    overlay_values: list[Any] = []
    if overlay is not None:
        info = bundle.feature_ranges[overlay]
        if overlay == "age":
            info = _working_age_range_info(bundle, info, base)
        if info["kind"] == "categorical":
            overlay_values = st.multiselect(
                f"{_pretty(overlay)} values",
                info["values"],
                default=info["values"],
                format_func=(
                    _employment_option_label if overlay == "employment_state" else str
                ),
            )
        else:
            lo, hi = float(info.get("q01", info["min"])), float(
                info.get("q99", info["max"])
            )
            suggested = np.linspace(lo, hi, 3).tolist()
            text = st.text_input(
                f"{_pretty(overlay)} values (comma separated)",
                value=", ".join(f"{x:.6g}" for x in suggested),
            )
            try:
                overlay_values = [
                    float(x.strip()) for x in text.split(",") if x.strip()
                ]
            except ValueError:
                st.error("Overlay values must be numeric.")
                overlay_values = suggested

    if overlay is not None and not overlay_values:
        st.info("Select at least one overlay value.")
        return

    # Do not load the household-level training sample for ordinary plots.
    # Bundle-level output bounds provide stable axes without a large DataFrame.
    sample = pd.DataFrame()
    y_range = _output_axis_control(
        bundle,
        sample,
        output,
        context="one_dimensional",
        bundle_path=bundle_path,
    )

    fig = _line_slice(
        bundle,
        base,
        x=x,
        x_range=x_range,
        overlay=overlay,
        overlay_values=overlay_values,
        outputs=[output],
        y_range=y_range,
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Fixed values used in this slice"):
        display = pd.DataFrame(
            {
                "Variable": [_pretty(name) for name in base],
                "Value": [_button_label(base[name]) for name in base],
            },
            dtype="string",
        )
        st.dataframe(display, hide_index=True, width="stretch")


def _surface_page(
    bundle: SurrogateBundle, base: dict[str, Any], *, bundle_path: str | Path
) -> None:
    continuous = _continuous_plot_columns(bundle)
    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.selectbox(
            "Horizontal axis",
            continuous,
            index=(
                continuous.index("current_income")
                if "current_income" in continuous
                else 0
            ),
            format_func=_pretty,
            key="surface_x",
        )

    axis_family = {
        "liquid_assets": "liquid",
        "liquid_slack_to_income": "liquid",
        "illiquid_assets": "illiquid",
        "illiquid_assets_to_income": "illiquid",
    }

    x_family = axis_family.get(x)

    y_options = [
        column
        for column in continuous
        if (column != x and (x_family is None or axis_family.get(column) != x_family))
    ]

    with c2:
        y_default = (
            y_options.index("liquid_assets") if "liquid_assets" in y_options else 0
        )
        if (
            "surface_y" in st.session_state
            and st.session_state["surface_y"] not in y_options
        ):
            st.session_state.pop("surface_y", None)
        y = st.selectbox(
            "Vertical axis",
            y_options,
            index=y_default,
            format_func=_pretty,
            key="surface_y",
        )
    with c3:
        output = st.selectbox(
            "Policy choice",
            POLICY_OUTPUTS,
            format_func=_pretty,
            key="surface_output",
        )

    xr = _range_selector(bundle, x, "surface_x_range")
    yr = _range_selector(bundle, y, "surface_y_range")
    x_values = np.linspace(xr[0], xr[1], 65)
    y_values = np.linspace(yr[0], yr[1], 65)
    surface = make_policy_surface(
        bundle,
        x=x,
        y=y,
        x_values=x_values,
        y_values=y_values,
        base=base,
        linked_income=True,
    )

    liquid = pd.to_numeric(surface["liquid_assets"], errors="coerce").to_numpy(
        dtype=float
    )
    borrowing_limit = _borrowing_limit_values(
        surface,
        fallback=_borrowing_limit_from_base(bundle, base),
    )
    feasible = (
        np.isfinite(liquid)
        & np.isfinite(borrowing_limit)
        & (liquid >= borrowing_limit - 1.0e-12)
    )
    value_column = f"pred_{output}"
    surface[value_column] = np.where(
        feasible,
        pd.to_numeric(surface[value_column], errors="coerce").to_numpy(dtype=float),
        np.nan,
    )
    table = surface.pivot(index="__y", columns="__x", values=value_column)

    # Avoid loading the full sampled grid merely to initialize the color scale.
    sample = pd.DataFrame()
    z_range = _output_axis_control(
        bundle,
        sample,
        output,
        context="two_dimensional",
        bundle_path=bundle_path,
    )
    fig = go.Figure(
        data=go.Heatmap(
            x=table.columns,
            y=table.index,
            z=table.to_numpy(),
            zmin=float(z_range[0]),
            zmax=float(z_range[1]),
            zauto=False,
            colorbar=dict(title=_pretty(output)),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=650,
        xaxis_title=_pretty(x),
        yaxis_title=_pretty(y),
        plot_bgcolor=("#bdbdbd" if (~feasible).any() else "white"),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, width="stretch")


def _mpc_histogram_figure(
    response: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    bins: int = 55,
    check_amount_dollars: float = 10_000.0,
) -> go.Figure:
    current = pd.to_numeric(response["mpc"], errors="coerce").to_numpy(dtype=float)
    current = current[np.isfinite(current)]
    arrays = [current]
    ref = None
    if reference is not None:
        ref = pd.to_numeric(reference["mpc"], errors="coerce").to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        if len(ref):
            arrays.append(ref)
    nonempty = [x for x in arrays if len(x)]
    if not nonempty:
        raise ValueError("No finite MPC predictions are available for the histogram.")
    pooled = np.concatenate(nonempty)
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    lo = min(float(lo), -0.05)
    hi = max(float(hi), 1.05)
    if hi <= lo + 1.0e-12:
        lo, hi = float(lo - 0.05), float(hi + 0.05)
    edges = np.linspace(lo, hi, int(bins) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    density, _ = np.histogram(current, bins=edges, density=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=centers,
            y=density,
            width=widths,
            name="Selected parameters",
            opacity=0.72,
            hovertemplate="MPC: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>",
        )
    )
    if ref is not None and len(ref):
        ref_density, _ = np.histogram(ref, bins=edges, density=True)
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=ref_density,
                mode="lines",
                name="Default parameters",
                line=dict(width=2.5),
                hovertemplate="MPC: %{x:.3f}<br>Reference density: %{y:.3f}<extra></extra>",
            )
        )
    current_mean = float(np.mean(current))
    current_median = float(np.median(current))
    fig.add_vline(
        x=current_mean,
        line_dash="dash",
        annotation_text=f"Mean {current_mean:.3f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=current_median,
        line_dash="dot",
        annotation_text=f"Median {current_median:.3f}",
        annotation_position="top left",
    )
    fig.update_layout(
        template="plotly_white",
        height=610,
        title=f"MPC distribution for an unexpected ${check_amount_dollars:,.0f} check",
        xaxis_title="Marginal propensity to consume",
        yaxis_title="Density",
        barmode="overlay",
        legend_title="Parameter setting",
        margin=dict(l=45, r=20, t=70, b=45),
    )
    fig.update_xaxes(range=[lo, hi])
    return fig


def _mpc_distribution_page(
    bundle: SurrogateBundle,
    base: dict[str, Any],
    *,
    bundle_path: str | Path,
) -> None:
    st.subheader("Distribution of MPCs from an unexpected check")
    st.caption(
        "The same synthetic households are used after every slider change. "
        "Thus movements in the histogram reflect the policy function, not a "
        "new population draw. The state-held-fixed sliders in the sidebar apply "
        "to the slice tabs; this tab replaces them with a population distribution."
    )

    inferred = infer_money_unit_info(bundle, bundle_path=bundle_path)
    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.25, 1.1])
    with c1:
        check_amount = st.number_input(
            "Check amount ($)",
            min_value=100.0,
            max_value=100_000.0,
            value=10_000.0,
            step=500.0,
            key="mpc_check_amount",
        )
    with c2:
        n_households = st.select_slider(
            "Synthetic households",
            options=[2_000, 5_000, 10_000, 20_000, 50_000],
            value=20_000,
            key="mpc_population_size",
        )
    with c3:
        normalized_units = st.checkbox(
            "Bundle uses normalized model units",
            value=(inferred.money_units == "model"),
            key="mpc_normalized_units",
            help=(
                "When selected, dollar values are divided by the saved-model "
                "normalization scale before the surrogate is evaluated."
            ),
        )
        money_units = "model" if normalized_units else "data"
    with c4:
        dollars_per_unit = st.number_input(
            "Dollars per model unit",
            min_value=1.0,
            max_value=10_000_000.0,
            value=float(inferred.dollars_per_model_unit),
            step=1_000.0,
            disabled=not normalized_units,
            key="mpc_dollars_per_unit",
            help=f"Initial value source: {inferred.source}.",
        )
        if not normalized_units:
            dollars_per_unit = 1.0

    compare_default = st.checkbox(
        "Overlay the distribution at the bundle's default parameters",
        value=True,
        key="mpc_compare_default",
    )

    with st.expander("Population distribution assumptions", expanded=False):
        st.markdown(
            "Income has permanent and transitory lognormal components. Wealth is "
            "a mixture of poor hand-to-mouth, wealthy hand-to-mouth, and regular "
            "saver households. Dollar states are clipped to the ranges represented "
            "in the solved policy grids."
        )
        a1, a2, a3 = st.columns(3)
        with a1:
            median_income = st.number_input(
                "Median annual income ($)",
                min_value=10_000.0,
                max_value=250_000.0,
                value=60_000.0,
                step=5_000.0,
                key="mpc_median_income",
            )
            permanent_sd = st.slider(
                "Permanent log-income SD",
                min_value=0.10,
                max_value=1.20,
                value=0.50,
                step=0.02,
                key="mpc_perm_sd",
            )
            transitory_sd = st.slider(
                "Transitory log-income SD",
                min_value=0.00,
                max_value=0.70,
                value=0.20,
                step=0.02,
                key="mpc_transitory_sd",
            )
        with a2:
            poor_share = st.slider(
                "Poor hand-to-mouth share",
                min_value=0.0,
                max_value=0.60,
                value=0.25,
                step=0.01,
                key="mpc_poor_htm_share",
            )
            wealthy_share = st.slider(
                "Wealthy hand-to-mouth share",
                min_value=0.0,
                max_value=0.50,
                value=0.15,
                step=0.01,
                key="mpc_wealthy_htm_share",
            )
            high_education_share = st.slider(
                "High-education share",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                key="mpc_high_education_share",
            )
        with a3:
            regular_liquid_ratio = st.slider(
                "Regular saver: median liquid wealth / income",
                min_value=0.01,
                max_value=1.50,
                value=0.25,
                step=0.01,
                key="mpc_regular_liquid_ratio",
            )
            regular_illiquid_ratio = st.slider(
                "Regular saver: median illiquid wealth / income",
                min_value=0.10,
                max_value=6.00,
                value=1.25,
                step=0.05,
                key="mpc_regular_illiquid_ratio",
            )
            seed = st.number_input(
                "Population seed",
                min_value=0,
                max_value=2_147_483_647,
                value=20260717,
                step=1,
                key="mpc_population_seed",
            )

    if poor_share + wealthy_share > 1.0:
        st.error(
            "Poor and wealthy hand-to-mouth shares sum to more than one. "
            "Reduce at least one share."
        )
        return

    assumptions = PopulationAssumptions(
        n_households=int(n_households),
        seed=int(seed),
        median_income_dollars=float(median_income),
        permanent_income_log_sd=float(permanent_sd),
        transitory_income_log_sd=float(transitory_sd),
        high_education_share=float(high_education_share),
        poor_hand_to_mouth_share=float(poor_share),
        wealthy_hand_to_mouth_share=float(wealthy_share),
        regular_liquid_ratio_median=float(regular_liquid_ratio),
        regular_illiquid_ratio_median=float(regular_illiquid_ratio),
    )

    population = generate_synthetic_population(
        bundle,
        base_values=base,
        assumptions=assumptions,
        dollars_per_model_unit=float(dollars_per_unit),
        money_units=money_units,
    )
    population = set_parameter_values(population, bundle, base)
    response = compute_check_mpcs(
        bundle,
        population,
        check_amount_dollars=float(check_amount),
        dollars_per_model_unit=float(dollars_per_unit),
        money_units=money_units,
    )

    reference = None
    if compare_default:
        reference_population = set_parameter_values(
            population, bundle, bundle.default_values
        )
        reference = compute_check_mpcs(
            bundle,
            reference_population,
            check_amount_dollars=float(check_amount),
            dollars_per_model_unit=float(dollars_per_unit),
            money_units=money_units,
        )

    summary = mpc_summary(response)
    if summary["n"] == 0:
        st.error(
            "The surrogate returned no finite MPC predictions for this population "
            "and parameter setting. Inspect the held-out diagnostics and feature support."
        )
        return
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean MPC", f"{summary['mean']:.3f}")
    m2.metric("Median MPC", f"{summary['median']:.3f}")
    m3.metric("10th percentile", f"{summary['p10']:.3f}")
    m4.metric("90th percentile", f"{summary['p90']:.3f}")
    m5.metric("Share MPC > 0.8", f"{summary['share_above_08']:.1%}")

    fig = _mpc_histogram_figure(
        response,
        reference=reference,
        check_amount_dollars=float(check_amount),
    )
    st.plotly_chart(fig, width="stretch")
    st.download_button(
        "Download household MPCs as CSV",
        data=response.to_csv(index=False).encode("utf-8"),
        file_name="ha_surrogate_mpc_distribution.csv",
        mime="text/csv",
        key="download_mpc_distribution",
    )

    clipped_population = float(
        pd.to_numeric(
            population["state_clipped_to_training_support"], errors="coerce"
        ).mean()
    )
    clipped_check = summary["share_check_state_clipped"]
    if clipped_population > 0.01 or clipped_check > 0.01:
        st.warning(
            f"{clipped_population:.1%} of population states were clipped to the "
            f"surrogate's training ranges, and {clipped_check:.1%} of post-check "
            "liquid states hit the grid/training maximum. Treat tail behavior "
            "cautiously or enlarge the original asset grids."
        )
    if summary["share_negative"] > 0.01 or summary["share_above_one"] > 0.01:
        st.warning(
            f"The smooth surrogate predicts MPC < 0 for "
            f"{summary['share_negative']:.1%} and MPC > 1 for "
            f"{summary['share_above_one']:.1%} of households. These are shown "
            "rather than silently truncated; inspect held-out accuracy and the "
            "corresponding state regions."
        )

    subgroup = (
        response.groupby("population_type", as_index=False)["mpc"]
        .agg(
            n="size",
            mean_mpc="mean",
            median_mpc="median",
            p10=lambda x: x.quantile(0.10),
            p90=lambda x: x.quantile(0.90),
        )
        .sort_values("mean_mpc", ascending=False)
    )
    with st.expander("MPCs and state diagnostics by population type"):
        st.dataframe(subgroup, hide_index=True, width="stretch")
        diagnostics = pd.DataFrame(
            {
                "statistic": [
                    "Population states clipped to training support",
                    "Post-check liquid state clipped",
                    "MPC below zero",
                    "MPC above one",
                ],
                "share": [
                    clipped_population,
                    clipped_check,
                    summary["share_negative"],
                    summary["share_above_one"],
                ],
            }
        )
        st.dataframe(diagnostics, hide_index=True, width="stretch")


def _resolve_model_dir(
    bundle_path: str | Path,
    raw_model_dir: Any,
) -> Path:
    """Resolve either an absolute catalog path or an app-data-relative path."""

    model_dir = Path(str(raw_model_dir)).expanduser()
    if not model_dir.is_absolute():
        model_dir = Path(bundle_path).expanduser().resolve().parent / model_dir
    return model_dir.resolve()


def _model_base_from_catalog_row(
    bundle: SurrogateBundle,
    base: dict[str, Any],
    selected_row: pd.Series,
) -> dict[str, Any]:
    """Replace sidebar parameters with one solved model's parameter vector."""

    model_base = dict(base)
    for column in bundle.feature_spec.parameter_columns:
        if column.endswith("__missing"):
            model_base[column] = 0
        elif column in selected_row.index and pd.notna(selected_row[column]):
            model_base[column] = float(selected_row[column])
    for column in bundle.feature_spec.categorical_parameter_columns:
        if column in selected_row.index and pd.notna(selected_row[column]):
            model_base[column] = str(selected_row[column])
    return model_base


def _model_parameter_table(
    catalog: pd.DataFrame,
    selected_row: pd.Series,
) -> pd.DataFrame:
    parameter_columns = [
        column
        for column in catalog.columns
        if column.startswith("param__") and not column.endswith("__missing")
    ]
    rows: list[dict[str, str]] = []
    for column in parameter_columns:
        value = selected_row.get(column)
        if pd.isna(value):
            continue
        rows.append(
            {
                "Parameter": _pretty(column),
                "Value": _button_label(value),
            }
        )
    return pd.DataFrame(rows, columns=["Parameter", "Value"], dtype="string")


def _held_out_model_selector(
    bundle: SurrogateBundle,
    *,
    bundle_path: str | Path,
    key_prefix: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, Path] | None:
    """Display held-out performance and return the selected solved model."""

    try:
        catalog = _load_model_catalog(str(bundle_path))
        model_summary = _out_of_sample_model_summary(bundle, catalog)
    except Exception as exc:
        st.error(str(exc))
        return None

    display_summary = model_summary[
        ["model_id", "consumption_r2", "deposit_r2", "mean_r2"]
    ].rename(
        columns={
            "model_id": "Model",
            "consumption_r2": "Consumption R²",
            "deposit_r2": "Deposit R²",
            "mean_r2": "Mean R²",
        }
    )
    st.dataframe(
        display_summary,
        hide_index=True,
        width="stretch",
        column_config={
            "Consumption R²": st.column_config.NumberColumn(format="%.4f"),
            "Deposit R²": st.column_config.NumberColumn(format="%.4f"),
            "Mean R²": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    summary_lookup = model_summary.set_index("model_id")
    model_ids = model_summary["model_id"].astype(str).tolist()

    def format_model_option(model_id: str) -> str:
        row = summary_lookup.loc[str(model_id)]
        consumption = row["consumption_r2"]
        deposit = row["deposit_r2"]
        consumption_text = f"{consumption:.4f}" if pd.notna(consumption) else "NA"
        deposit_text = f"{deposit:.4f}" if pd.notna(deposit) else "NA"
        return (
            f"{model_id}  |  Consumption R²: {consumption_text}  |  "
            f"Deposit R²: {deposit_text}"
        )

    selected_model_id = st.selectbox(
        "Select an out-of-sample model",
        model_ids,
        format_func=format_model_option,
        key=f"{key_prefix}_selected_model_id",
    )
    selected_summary = summary_lookup.loc[str(selected_model_id)]
    catalog_copy = catalog.copy()
    catalog_copy["model_id"] = catalog_copy["model_id"].astype(str)
    rows = catalog_copy[catalog_copy["model_id"] == str(selected_model_id)]
    if rows.empty:
        st.error(f"Model {selected_model_id} is missing from model_catalog.csv.")
        return None
    selected_row = rows.iloc[0]
    raw_model_dir = selected_row.get("model_dir", selected_summary.get("model_dir"))
    if pd.isna(raw_model_dir):
        st.error("The selected model has no model_dir entry.")
        return None
    model_dir = _resolve_model_dir(bundle_path, raw_model_dir)
    if not model_dir.exists():
        st.error(f"The selected model directory does not exist: {model_dir}")
        return None

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        "Consumption R²",
        (
            f"{selected_summary['consumption_r2']:.4f}"
            if pd.notna(selected_summary["consumption_r2"])
            else "NA"
        ),
    )
    metric_columns[1].metric(
        "Deposit R²",
        (
            f"{selected_summary['deposit_r2']:.4f}"
            if pd.notna(selected_summary["deposit_r2"])
            else "NA"
        ),
    )
    metric_columns[2].metric(
        "Mean R²",
        (
            f"{selected_summary['mean_r2']:.4f}"
            if pd.notna(selected_summary["mean_r2"])
            else "NA"
        ),
    )

    with st.expander("Selected model parameters", expanded=True):
        st.dataframe(
            _model_parameter_table(catalog, selected_row),
            hide_index=True,
            width="stretch",
        )
    return catalog, selected_row, selected_summary, model_dir


def _exact_grid_comparison_page(
    bundle: SurrogateBundle,
    base: dict[str, Any],
    *,
    bundle_path: str | Path,
) -> None:
    """Compare exact-grid slices with the development-only surrogate."""

    st.subheader(f"Held-out grid versus {bundle.surrogate_label.lower()}")
    st.caption(
        f"The {bundle.short_surrogate_label.lower()} model on this page is the "
        "development-only evaluation fit. Every selectable exact model is "
        "therefore genuinely absent from its estimation sample."
    )

    selection = _held_out_model_selector(
        bundle,
        bundle_path=bundle_path,
        key_prefix="exact_policy",
    )
    if selection is None:
        return
    _, selected_row, _, model_dir = selection

    c1, c2 = st.columns(2)
    exact_x_options = _exact_comparison_x_options(bundle)
    with c1:
        x = st.selectbox(
            "Horizontal axis",
            exact_x_options,
            index=(
                exact_x_options.index("current_income")
                if "current_income" in exact_x_options
                else 0
            ),
            format_func=_pretty,
            key="exact_comparison_x",
        )
    with c2:
        output = st.selectbox(
            "Policy choice",
            GRID_COMPARISON_OUTPUTS,
            format_func=_pretty,
            key="exact_comparison_output",
        )

    mpc_check_units = 0.188679
    if output == "mpc":
        units_label = (
            "model units"
            if str(bundle.training_metadata.get("money_units", "model")) == "model"
            else "data units"
        )
        mpc_check_units = float(
            st.number_input(
                f"Transfer used to calculate the MPC ({units_label})",
                min_value=1.0e-6,
                value=0.188679,
                step=0.01,
                format="%.6f",
                key="exact_comparison_mpc_check_units",
            )
        )

    model_base = _model_base_from_catalog_row(bundle, base, selected_row)
    policy_layout = str(bundle.training_metadata.get("policy_layout", "GHKEBA"))
    money_units = str(bundle.training_metadata.get("money_units", "model"))

    # Load the complete selected-model slice first. The range selector must be
    # based on this model's solved support rather than the surrogate's global
    # marginal support.
    try:
        exact_all = load_exact_grid_slice(
            model_dir,
            base=model_base,
            x=x,
            x_range=None,
            policy_layout=policy_layout,
            money_units=money_units,
        )
    except Exception as exc:
        st.error(f"Could not load the exact policy grid: {exc}")
        return
    if exact_all.empty:
        st.warning("The selected exact model has no solved points for this slice.")
        return

    exact_support = exact_all.copy()
    if output == "mpc":
        if not {"liquid_assets", "liquid_grid_max"}.issubset(exact_support.columns):
            st.error(
                "The exact slice does not contain liquid-grid bounds needed "
                "to determine MPC feasibility."
            )
            return
        liquid_assets = pd.to_numeric(exact_support["liquid_assets"], errors="coerce")
        liquid_max = pd.to_numeric(exact_support["liquid_grid_max"], errors="coerce")
        feasible = (
            liquid_assets.notna()
            & liquid_max.notna()
            & (liquid_assets + float(mpc_check_units) <= liquid_max + 1.0e-12)
        )
        exact_support = exact_support.loc[feasible].copy()
        if exact_support.empty:
            st.warning(
                "The transfer is too large for every point in this exact slice. "
                "Reduce the MPC transfer or select a lower-liquid state."
            )
            return

    selected_model_id = str(selected_row.get("model_id", model_dir.name))
    try:
        x_range = _exact_grid_range_selector(
            exact_support,
            x,
            model_id=selected_model_id,
            key="exact_comparison_x_range_v2",
        )
    except Exception as exc:
        st.error(f"Could not construct the exact-support range selector: {exc}")
        return

    exact = exact_support.loc[
        pd.to_numeric(exact_support["__x"], errors="coerce").between(
            float(x_range[0]),
            float(x_range[1]),
        )
    ].copy()
    exact = exact.sort_values("__x").reset_index(drop=True)
    if exact.empty:
        st.warning("No exact grid points fall inside the selected x-axis range.")
        return

    y_range = _output_axis_control(
        bundle,
        pd.DataFrame(),
        output,
        context="exact_comparison",
        bundle_path=bundle_path,
    )

    surrogate_input = pd.DataFrame([model_base] * len(exact))
    state_columns = set(
        bundle.feature_spec.state_columns
        + bundle.feature_spec.categorical_state_columns
    )
    for column in state_columns:
        if column in exact:
            surrogate_input[column] = exact[column].to_numpy()
    for column in [
        "liquid_grid_min",
        "liquid_grid_max",
        "illiquid_grid_min",
        "illiquid_grid_max",
        "money_scale",
    ]:
        if column in exact:
            surrogate_input[column] = exact[column].to_numpy()

    if output == "mpc":
        try:
            exact_grid = _load_exact_policy_grid_cached(
                str(model_dir),
                policy_layout,
            )
            aligned = align_population_to_exact_grid(
                exact_grid,
                surrogate_input,
                money_units=money_units,
            )
            exact_baseline, _ = predict_exact_grid_consumption(
                exact_grid,
                aligned,
                liquid_asset_increment=0.0,
                money_units=money_units,
            )
            exact_treated, treated_clipped = predict_exact_grid_consumption(
                exact_grid,
                aligned,
                liquid_asset_increment=mpc_check_units,
                money_units=money_units,
            )
            surrogate_baseline = bundle.predict(aligned)["consumption"].to_numpy(
                dtype=float
            )
            treated_input = aligned.copy()
            treated_input["liquid_assets"] = (
                pd.to_numeric(treated_input["liquid_assets"], errors="coerce")
                + mpc_check_units
            )
            surrogate_treated = bundle.predict(treated_input)["consumption"].to_numpy(
                dtype=float
            )

            exact_values = (exact_treated - exact_baseline) / mpc_check_units
            surrogate_values = (
                surrogate_treated - surrogate_baseline
            ) / mpc_check_units
            valid = (
                ~treated_clipped
                & np.isfinite(exact_values)
                & np.isfinite(surrogate_values)
            )
        except Exception as exc:
            st.error(f"Could not calculate exact and surrogate MPCs: {exc}")
            return
    else:
        surrogate = bundle.predict(surrogate_input)
        exact_values = pd.to_numeric(exact[output], errors="coerce").to_numpy(
            dtype=float
        )
        surrogate_values = pd.to_numeric(surrogate[output], errors="coerce").to_numpy(
            dtype=float
        )
        valid = np.isfinite(exact_values) & np.isfinite(surrogate_values)

    comparison = pd.DataFrame(
        {
            x: exact["__x"].to_numpy(dtype=float),
            "exact_grid": exact_values,
            "surrogate": surrogate_values,
            "valid": valid,
        }
    )
    comparison = comparison.loc[comparison["valid"]].drop(columns="valid")
    if comparison.empty:
        st.warning(
            "No feasible comparison points remain. For MPCs, reduce the transfer "
            "so treated liquid wealth stays inside the exact grid."
        )
        return
    comparison["error"] = comparison["surrogate"] - comparison["exact_grid"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comparison[x],
            y=comparison["exact_grid"],
            mode="lines+markers",
            name="Solved grid",
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comparison[x],
            y=comparison["surrogate"],
            mode="lines",
            name=bundle.surrogate_label,
            line=dict(width=3),
        )
    )
    _shade_liquid_infeasible_region(
        fig,
        axis_variable=x,
        axis_range=x_range,
        bundle=bundle,
        base=model_base,
    )
    fig.update_layout(
        template="plotly_white",
        height=620,
        xaxis_title=_pretty(x),
        xaxis=dict(
            range=[float(x_range[0]), float(x_range[1])],
            autorange=False,
        ),
        yaxis_title=_pretty(output),
        yaxis=dict(range=[float(y_range[0]), float(y_range[1])], autorange=False),
        hovermode="x unified",
        legend_title="Policy source",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    st.plotly_chart(fig, width="stretch")

    truth = comparison["exact_grid"].to_numpy(dtype=float)
    predicted = comparison["surrogate"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean((predicted - truth) ** 2)))
    max_error = float(np.max(np.abs(predicted - truth)))
    r2 = (
        float(
            1.0 - np.sum((predicted - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)
        )
        if len(truth) > 1 and np.var(truth) > 1.0e-14
        else np.nan
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Points compared", f"{len(comparison):,}")
    m2.metric("Slice R²", f"{r2:.4f}" if np.isfinite(r2) else "NA")
    m3.metric("Slice RMSE", f"{rmse:.6g}")
    m4.metric("Maximum absolute error", f"{max_error:.6g}")
    with st.expander("Comparison data"):
        st.dataframe(comparison, hide_index=True, width="stretch")


def _model_surrogate_mpc_histogram(
    exact_response: pd.DataFrame,
    surrogate_response: pd.DataFrame,
    *,
    check_amount_dollars: float,
    surrogate_label: str,
    bins: int = 60,
) -> go.Figure:
    """Overlapping, unsmoothed histograms using identical bins."""

    exact = pd.to_numeric(exact_response["mpc"], errors="coerce").to_numpy(dtype=float)
    surrogate = pd.to_numeric(surrogate_response["mpc"], errors="coerce").to_numpy(
        dtype=float
    )

    exact = exact[np.isfinite(exact)]
    surrogate = surrogate[np.isfinite(surrogate)]

    if not len(exact) or not len(surrogate):
        raise ValueError("No finite exact-model or surrogate MPCs are available.")

    pooled = np.concatenate([exact, surrogate])
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    lo = min(float(lo), -0.05)
    hi = max(float(hi), 1.05)

    if hi <= lo + 1.0e-12:
        lo, hi = lo - 0.05, hi + 0.05

    bin_size = (hi - lo) / int(bins)
    shared_bins = dict(
        start=lo,
        end=hi,
        size=bin_size,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=exact,
            xbins=shared_bins,
            histnorm="probability density",
            name="Solved model",
            opacity=0.5,
            hovertemplate=("MPC bin: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>"),
        )
    )

    fig.add_trace(
        go.Histogram(
            x=surrogate,
            xbins=shared_bins,
            histnorm="probability density",
            name=surrogate_label,
            opacity=0.5,
            hovertemplate=("MPC bin: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>"),
        )
    )

    fig.add_vline(
        x=float(np.mean(exact)),
        line_dash="dash",
        annotation_text=f"Model mean {np.mean(exact):.3f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=float(np.mean(surrogate)),
        line_dash="dot",
        annotation_text=f"Surrogate mean {np.mean(surrogate):.3f}",
        annotation_position="top right",
    )

    fig.update_layout(
        template="plotly_white",
        height=620,
        title=(
            "Solved-model and surrogate MPC distributions for an unexpected "
            f"${check_amount_dollars:,.0f} check"
        ),
        xaxis_title="Marginal propensity to consume",
        yaxis_title="Density",
        barmode="overlay",
        bargap=0.02,
        legend_title="Policy source",
        margin=dict(l=45, r=20, t=75, b=45),
    )
    fig.update_xaxes(range=[lo, hi])
    return fig


def _mpc_grid_comparison_page(
    bundle: SurrogateBundle,
    base: dict[str, Any],
    *,
    bundle_path: str | Path,
) -> None:
    st.subheader(
        f"Held-out model MPCs versus {bundle.short_surrogate_label.lower()} MPCs"
    )
    st.caption(
        f"The exact model and {bundle.short_surrogate_label.lower()} are evaluated "
        "on the same synthetic households. Ages, persistent-income states, "
        "employment states, and groups are first mapped to the selected model's "
        "solved grid; liquid and illiquid assets remain continuous and are "
        "interpolated."
    )

    selection = _held_out_model_selector(
        bundle,
        bundle_path=bundle_path,
        key_prefix="mpc_exact",
    )
    if selection is None:
        return
    _, selected_row, _, model_dir = selection
    model_base = _model_base_from_catalog_row(bundle, base, selected_row)
    policy_layout = str(bundle.training_metadata.get("policy_layout", "GHKEBA"))

    inferred = infer_money_unit_info(bundle, bundle_path=bundle_path)
    c1, c2, c3, c4 = st.columns([1.0, 1.0, 1.25, 1.1])
    with c1:
        check_amount = st.number_input(
            "Check amount ($)",
            min_value=100.0,
            max_value=100_000.0,
            value=10_000.0,
            step=500.0,
            key="mpc_exact_check_amount",
        )
    with c2:
        n_households = st.select_slider(
            "Synthetic households",
            options=[2_000, 5_000, 10_000, 20_000, 50_000],
            value=20_000,
            key="mpc_exact_population_size",
        )
    with c3:
        normalized_units = st.checkbox(
            "Bundle uses normalized model units",
            value=(inferred.money_units == "model"),
            key="mpc_exact_normalized_units",
        )
        money_units = "model" if normalized_units else "data"
    with c4:
        dollars_per_unit = st.number_input(
            "Dollars per model unit",
            min_value=1.0,
            max_value=10_000_000.0,
            value=float(inferred.dollars_per_model_unit),
            step=1_000.0,
            disabled=not normalized_units,
            key="mpc_exact_dollars_per_unit",
            help=f"Initial value source: {inferred.source}.",
        )
        if not normalized_units:
            dollars_per_unit = 1.0

    with st.expander("Population distribution assumptions", expanded=False):
        a1, a2, a3 = st.columns(3)
        with a1:
            median_income = st.number_input(
                "Median annual income ($)",
                min_value=10_000.0,
                max_value=250_000.0,
                value=60_000.0,
                step=5_000.0,
                key="mpc_exact_median_income",
            )
            permanent_sd = st.slider(
                "Permanent log-income SD",
                min_value=0.10,
                max_value=1.20,
                value=0.50,
                step=0.02,
                key="mpc_exact_perm_sd",
            )
            transitory_sd = st.slider(
                "Transitory log-income SD",
                min_value=0.00,
                max_value=0.70,
                value=0.20,
                step=0.02,
                key="mpc_exact_transitory_sd",
            )
        with a2:
            poor_share = st.slider(
                "Poor hand-to-mouth share",
                min_value=0.0,
                max_value=0.60,
                value=0.25,
                step=0.01,
                key="mpc_exact_poor_share",
            )
            wealthy_share = st.slider(
                "Wealthy hand-to-mouth share",
                min_value=0.0,
                max_value=0.50,
                value=0.15,
                step=0.01,
                key="mpc_exact_wealthy_share",
            )
            high_education_share = st.slider(
                "High-education share",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                key="mpc_exact_education_share",
            )
        with a3:
            regular_liquid_ratio = st.slider(
                "Regular saver: median liquid wealth / income",
                min_value=0.01,
                max_value=1.50,
                value=0.25,
                step=0.01,
                key="mpc_exact_liquid_ratio",
            )
            regular_illiquid_ratio = st.slider(
                "Regular saver: median illiquid wealth / income",
                min_value=0.10,
                max_value=6.00,
                value=1.25,
                step=0.05,
                key="mpc_exact_illiquid_ratio",
            )
            seed = st.number_input(
                "Population seed",
                min_value=0,
                max_value=2_147_483_647,
                value=20260717,
                step=1,
                key="mpc_exact_seed",
            )

    if poor_share + wealthy_share > 1.0:
        st.error("Poor and wealthy hand-to-mouth shares cannot sum to more than one.")
        return
    assumptions = PopulationAssumptions(
        n_households=int(n_households),
        seed=int(seed),
        median_income_dollars=float(median_income),
        permanent_income_log_sd=float(permanent_sd),
        transitory_income_log_sd=float(transitory_sd),
        high_education_share=float(high_education_share),
        poor_hand_to_mouth_share=float(poor_share),
        wealthy_hand_to_mouth_share=float(wealthy_share),
        regular_liquid_ratio_median=float(regular_liquid_ratio),
        regular_illiquid_ratio_median=float(regular_illiquid_ratio),
    )

    population = generate_synthetic_population(
        bundle,
        base_values=model_base,
        assumptions=assumptions,
        dollars_per_model_unit=float(dollars_per_unit),
        money_units=money_units,
    )
    try:
        exact_grid = _load_exact_policy_grid_cached(str(model_dir), policy_layout)
        exact_response = compute_exact_grid_check_mpcs(
            exact_grid,
            population,
            check_amount_dollars=float(check_amount),
            dollars_per_model_unit=float(dollars_per_unit),
            money_units=money_units,
        )
    except Exception as exc:
        st.error(f"Could not evaluate exact-grid MPCs: {exc}")
        return

    # The exact response contains the population aligned to the solved model's
    # exogenous states. Evaluate the surrogate on precisely those same states.
    surrogate_population = set_parameter_values(
        exact_response.copy(),
        bundle,
        model_base,
    )
    surrogate_response = compute_check_mpcs(
        bundle,
        surrogate_population,
        check_amount_dollars=float(check_amount),
        dollars_per_model_unit=float(dollars_per_unit),
        money_units=money_units,
    )

    exact_summary = mpc_summary(exact_response)
    surrogate_summary = mpc_summary(surrogate_response)
    exact_mpc = pd.to_numeric(exact_response["mpc"], errors="coerce").to_numpy(
        dtype=float
    )
    surrogate_mpc = pd.to_numeric(surrogate_response["mpc"], errors="coerce").to_numpy(
        dtype=float
    )
    finite = np.isfinite(exact_mpc) & np.isfinite(surrogate_mpc)
    if not finite.any():
        st.error("No finite paired MPC predictions are available.")
        return
    mpc_error = surrogate_mpc[finite] - exact_mpc[finite]
    mpc_rmse = float(np.sqrt(np.mean(mpc_error**2)))
    mpc_bias = float(np.mean(mpc_error))
    correlation = (
        float(np.corrcoef(exact_mpc[finite], surrogate_mpc[finite])[0, 1])
        if finite.sum() > 1
        and np.std(exact_mpc[finite]) > 0
        and np.std(surrogate_mpc[finite]) > 0
        else np.nan
    )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Model mean MPC", f"{exact_summary['mean']:.3f}")
    m2.metric("Surrogate mean MPC", f"{surrogate_summary['mean']:.3f}")
    m3.metric("Model median", f"{exact_summary['median']:.3f}")
    m4.metric("Surrogate median", f"{surrogate_summary['median']:.3f}")
    m5.metric("Paired MPC RMSE", f"{mpc_rmse:.4f}")
    m6.metric(
        "MPC correlation", f"{correlation:.3f}" if np.isfinite(correlation) else "NA"
    )

    fig = _model_surrogate_mpc_histogram(
        exact_response,
        surrogate_response,
        check_amount_dollars=float(check_amount),
        surrogate_label=bundle.surrogate_label,
    )
    st.plotly_chart(fig, width="stretch")

    comparison = pd.DataFrame(
        {
            "model_mpc": exact_mpc,
            "surrogate_mpc": surrogate_mpc,
            "error": surrogate_mpc - exact_mpc,
            "population_type": exact_response.get("population_type", "unknown"),
        }
    )
    paired = comparison.loc[finite].copy()
    paired_summary = pd.DataFrame(
        {
            "Statistic": [
                "Paired households",
                "Mean surrogate minus model MPC",
                "Paired MPC RMSE",
                "Paired MPC correlation",
                "Exact post-check state clipped",
                "Exact asset state initially clipped",
                "Exact exogenous state mapped to nearest grid point",
            ],
            "Value": [
                f"{finite.sum():,}",
                f"{mpc_bias:.6g}",
                f"{mpc_rmse:.6g}",
                f"{correlation:.6g}" if np.isfinite(correlation) else "NA",
                f"{exact_response['check_state_clipped'].mean():.2%}",
                f"{exact_response['exact_asset_state_clipped'].mean():.2%}",
                f"{exact_response['exact_exogenous_state_mapped'].mean():.2%}",
            ],
        },
        dtype="string",
    )
    subgroup = (
        paired.groupby("population_type", as_index=False)
        .agg(
            n=("model_mpc", "size"),
            model_mean_mpc=("model_mpc", "mean"),
            surrogate_mean_mpc=("surrogate_mpc", "mean"),
            mean_error=("error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
        )
        .sort_values("model_mean_mpc", ascending=False)
    )
    with st.expander("Paired MPC diagnostics", expanded=False):
        st.dataframe(paired_summary, hide_index=True, width="stretch")
        st.dataframe(subgroup, hide_index=True, width="stretch")
        st.download_button(
            "Download paired household MPCs as CSV",
            data=paired.to_csv(index=False).encode("utf-8"),
            file_name="exact_model_vs_surrogate_mpcs.csv",
            mime="text/csv",
            key="download_exact_surrogate_mpcs",
        )


def _held_out_model_ids(bundle: SurrogateBundle) -> set[str]:
    """Return the model IDs reserved for untouched validation."""

    for key in ("test_models", "validation_models"):
        values = bundle.training_metadata.get(key)
        if values:
            return {str(value) for value in values}

    metrics = bundle.validation_by_model
    if metrics.empty or "model_id" not in metrics:
        return set()

    held_out = metrics.copy()
    if "split" in held_out:
        held_out = held_out[
            held_out["split"].astype(str).str.contains("held_out", case=False, na=False)
        ]
    return set(held_out["model_id"].astype(str))


def _mpc_validation_columns(
    bundle: SurrogateBundle,
    frame: pd.DataFrame,
) -> list[str]:
    """Continuous states and parameters suitable for the MPC binscatter."""

    preferred = [
        "age",
        "years_to_retirement_positive",
        "current_income",
        "liquid_assets",
        "illiquid_assets",
        "liquid_slack",
        "liquid_slack_to_income",
        "illiquid_assets_to_income",
        "cash_on_hand_to_income",
        "mpc_check_units",
    ]
    preferred += [
        column
        for column in bundle.feature_spec.parameter_columns
        if not column.endswith("__missing")
    ]
    preferred += list(getattr(bundle.feature_spec, "engineered_state_columns", ()))

    blocked = {
        "mpc_pair_id",
        "mpc_pair_role",
        "grid_g",
        "grid_h",
        "grid_k",
        "grid_e",
        "grid_b",
        "grid_a",
        "after_tax_income",
        "labor_income_state",
        "after_tax_income_to_income",
        "pre_tax_income_to_income",
        "is_retired",
        "years_since_retirement",
        "years_to_terminal",
    }
    out: list[str] = []
    for column in dict.fromkeys(preferred):
        if column in blocked or column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() >= 2 and values.nunique(dropna=True) > 1:
            out.append(column)
    return out


@st.cache_data(show_spinner=False)
def _load_holdout_mpc_validation(
    bundle_path: str,
    bundle_signature: int,
    sample_path: str,
    sample_signature: int,
    prediction_chunk_size: int = 20_000,
) -> pd.DataFrame:
    """Construct exact and surrogate MPCs from held-out sampled grid pairs."""

    # Signatures are included only to invalidate the Streamlit cache whenever
    # either file is replaced at the same path.
    del bundle_signature, sample_signature

    bundle = load_surrogate_bundle(bundle_path, surrogate_type="auto")
    sample = read_policy_dataset(sample_path)

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
            "policy_grid_sample is missing MPC-pair columns: "
            + ", ".join(sorted(missing))
        )

    held_out_ids = _held_out_model_ids(bundle)
    if not held_out_ids:
        raise ValueError("The bundle does not identify any held-out model IDs.")

    sample = sample.copy()
    sample["model_id"] = sample["model_id"].astype(str)
    pair_id = pd.to_numeric(sample["mpc_pair_id"], errors="coerce")
    role = pd.to_numeric(sample["mpc_pair_role"], errors="coerce")
    check = pd.to_numeric(sample["mpc_check_units"], errors="coerce")

    keep = (
        sample["model_id"].isin(held_out_ids)
        & pair_id.ge(0)
        & role.isin([0, 1])
        & check.gt(0)
    )
    pair_rows = sample.loc[keep].copy().reset_index(drop=True)
    del sample

    if pair_rows.empty:
        raise ValueError(
            "No explicit MPC pairs were found for the held-out models. "
            "Retrain with a positive --mpc-pair-share."
        )

    pair_rows["mpc_pair_id"] = pd.to_numeric(
        pair_rows["mpc_pair_id"], errors="raise"
    ).astype(np.int64)
    pair_rows["mpc_pair_role"] = pd.to_numeric(
        pair_rows["mpc_pair_role"], errors="raise"
    ).astype(np.int8)

    predicted = np.empty(len(pair_rows), dtype=float)
    for start in range(0, len(pair_rows), int(prediction_chunk_size)):
        stop = min(start + int(prediction_chunk_size), len(pair_rows))
        chunk_prediction = bundle.predict(
            pair_rows.iloc[start:stop],
            project=True,
        )
        predicted[start:stop] = pd.to_numeric(
            chunk_prediction["consumption"], errors="coerce"
        ).to_numpy(dtype=float)
    pair_rows["__surrogate_consumption"] = predicted

    keys = ["model_id", "mpc_pair_id"]
    baseline = (
        pair_rows[pair_rows["mpc_pair_role"] == 0]
        .sort_values(keys)
        .drop_duplicates(keys, keep="first")
        .rename(
            columns={
                "consumption": "true_consumption_baseline",
                "__surrogate_consumption": "surrogate_consumption_baseline",
                "mpc_check_units": "check_baseline",
            }
        )
    )
    treated = (
        pair_rows[pair_rows["mpc_pair_role"] == 1][
            keys + ["consumption", "__surrogate_consumption", "mpc_check_units"]
        ]
        .sort_values(keys)
        .drop_duplicates(keys, keep="first")
        .rename(
            columns={
                "consumption": "true_consumption_treated",
                "__surrogate_consumption": "surrogate_consumption_treated",
                "mpc_check_units": "check_treated",
            }
        )
    )
    del pair_rows

    comparison = baseline.merge(
        treated,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    comparison["mpc_check_units"] = 0.5 * (
        pd.to_numeric(comparison["check_baseline"], errors="coerce")
        + pd.to_numeric(comparison["check_treated"], errors="coerce")
    )
    comparison["true_mpc"] = (
        comparison["true_consumption_treated"] - comparison["true_consumption_baseline"]
    ) / comparison["mpc_check_units"]
    comparison["surrogate_mpc"] = (
        comparison["surrogate_consumption_treated"]
        - comparison["surrogate_consumption_baseline"]
    ) / comparison["mpc_check_units"]
    comparison["mpc_error"] = comparison["surrogate_mpc"] - comparison["true_mpc"]
    comparison["absolute_mpc_error"] = comparison["mpc_error"].abs()
    comparison["squared_mpc_error"] = comparison["mpc_error"] ** 2

    finite = (
        np.isfinite(pd.to_numeric(comparison["true_mpc"], errors="coerce"))
        & np.isfinite(pd.to_numeric(comparison["surrogate_mpc"], errors="coerce"))
        & np.isfinite(pd.to_numeric(comparison["mpc_check_units"], errors="coerce"))
        & comparison["mpc_check_units"].gt(0)
    )
    comparison = comparison.loc[finite].reset_index(drop=True)
    if comparison.empty:
        raise ValueError("No finite held-out MPC comparisons could be constructed.")

    # Recreate the exact engineered states used by the fitted surrogate at the
    # baseline observation. This makes variables such as liquid slack / income
    # available even though they are not necessarily stored in the sampled file.
    prepared = bundle.prepare_inputs(comparison)
    candidate_columns = list(
        dict.fromkeys(
            list(bundle.feature_spec.state_columns)
            + list(getattr(bundle.feature_spec, "engineered_state_columns", ()))
            + list(bundle.feature_spec.parameter_columns)
            + list(bundle.feature_spec.categorical_state_columns)
            + list(bundle.feature_spec.categorical_parameter_columns)
            + [
                "years_to_terminal",
                "years_since_retirement",
                "years_to_retirement_positive",
                "after_tax_income",
                "liquid_slack",
                "liquid_slack_to_income",
                "illiquid_assets_to_income",
                "cash_on_hand_to_income",
            ]
        )
    )
    for column in candidate_columns:
        if column in prepared:
            comparison[column] = prepared[column].to_numpy()

    keep_columns = list(
        dict.fromkeys(
            keys
            + [
                "mpc_check_units",
                "true_mpc",
                "surrogate_mpc",
                "mpc_error",
                "absolute_mpc_error",
                "squared_mpc_error",
            ]
            + [column for column in candidate_columns if column in comparison]
        )
    )
    return comparison[keep_columns].reset_index(drop=True)


def _mpc_binscatter_table(
    data: pd.DataFrame,
    *,
    x: str,
    n_bins: int,
    trim_share: float,
    equal_weight_models: bool,
) -> pd.DataFrame:
    """Quantile bins with true MPC, surrogate MPC, bias, and RMSE."""

    columns = [
        "model_id",
        x,
        "true_mpc",
        "surrogate_mpc",
        "mpc_error",
        "absolute_mpc_error",
        "squared_mpc_error",
    ]
    frame = data[columns].copy()
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.DataFrame()

    unique_values = frame[x].nunique(dropna=True)
    if trim_share > 0 and unique_values > int(n_bins):
        lo, hi = frame[x].quantile([trim_share, 1.0 - trim_share])
        frame = frame[frame[x].between(float(lo), float(hi))].copy()
    if frame.empty:
        return pd.DataFrame()

    unique_values = frame[x].nunique(dropna=True)
    if unique_values <= int(n_bins):
        frame["__bin"] = frame[x]
    else:
        q = min(int(n_bins), int(unique_values))
        frame["__bin"] = pd.qcut(frame[x], q=q, duplicates="drop")

    if equal_weight_models:
        model_bin = (
            frame.groupby(["__bin", "model_id"], observed=True)
            .agg(
                x_mean=(x, "mean"),
                x_min=(x, "min"),
                x_max=(x, "max"),
                true_mpc=("true_mpc", "mean"),
                surrogate_mpc=("surrogate_mpc", "mean"),
                mpc_error=("mpc_error", "mean"),
                mean_absolute_error=("absolute_mpc_error", "mean"),
                mean_squared_error=("squared_mpc_error", "mean"),
                n_pairs=("true_mpc", "size"),
            )
            .reset_index()
        )
        grouped = model_bin.groupby("__bin", observed=True)
        summary = grouped.agg(
            x=("x_mean", "mean"),
            x_min=("x_min", "min"),
            x_max=("x_max", "max"),
            true_mpc=("true_mpc", "mean"),
            surrogate_mpc=("surrogate_mpc", "mean"),
            mean_error=("mpc_error", "mean"),
            mean_absolute_error=("mean_absolute_error", "mean"),
            mean_squared_error=("mean_squared_error", "mean"),
            true_sd=("true_mpc", "std"),
            surrogate_sd=("surrogate_mpc", "std"),
            error_sd=("mpc_error", "std"),
            models=("model_id", "nunique"),
            pairs=("n_pairs", "sum"),
        ).reset_index(drop=True)
        denominator = np.sqrt(summary["models"].clip(lower=1))
    else:
        grouped = frame.groupby("__bin", observed=True)
        summary = grouped.agg(
            x=(x, "mean"),
            x_min=(x, "min"),
            x_max=(x, "max"),
            true_mpc=("true_mpc", "mean"),
            surrogate_mpc=("surrogate_mpc", "mean"),
            mean_error=("mpc_error", "mean"),
            mean_absolute_error=("absolute_mpc_error", "mean"),
            mean_squared_error=("squared_mpc_error", "mean"),
            true_sd=("true_mpc", "std"),
            surrogate_sd=("surrogate_mpc", "std"),
            error_sd=("mpc_error", "std"),
            models=("model_id", "nunique"),
            pairs=("true_mpc", "size"),
        ).reset_index(drop=True)
        denominator = np.sqrt(summary["pairs"].clip(lower=1))

    summary["rmse"] = np.sqrt(summary["mean_squared_error"])
    summary["true_se"] = summary["true_sd"].fillna(0.0) / denominator
    summary["surrogate_se"] = summary["surrogate_sd"].fillna(0.0) / denominator
    summary["error_se"] = summary["error_sd"].fillna(0.0) / denominator
    return summary.sort_values("x").reset_index(drop=True)


def _validation_mpc_binscatter(
    bundle: SurrogateBundle,
    *,
    bundle_path: str | Path,
) -> None:
    st.subheader("MPC divergence by state and model parameter")
    st.caption(
        "This uses explicit baseline/treated pairs from entirely held-out model "
        "parameterizations. True MPCs come from the saved policy grids; surrogate "
        "MPCs are finite differences of the fitted consumption function on those "
        "same rows."
    )

    enabled = st.checkbox(
        "Load household-level held-out MPC diagnostics",
        value=False,
        key="validation_enable_mpc_binscatter",
        help=(
            "The first load reads policy_grid_sample and evaluates the surrogate "
            "on all paired rows in the holdout set. The result is cached."
        ),
    )
    if not enabled:
        return

    sample_path = _training_sample_path(bundle_path)
    if sample_path is None:
        st.error(
            "No policy_grid_sample file was found beside the surrogate bundle. "
            "This household-level validation plot requires that sampled file."
        )
        return

    resolved_bundle = Path(bundle_path).expanduser().resolve()
    try:
        with st.spinner("Constructing held-out true and surrogate MPC pairs..."):
            comparison = _load_holdout_mpc_validation(
                str(resolved_bundle),
                int(resolved_bundle.stat().st_mtime_ns),
                str(sample_path.resolve()),
                int(sample_path.stat().st_mtime_ns),
            )
    except Exception as exc:
        st.error(f"Could not construct held-out MPC diagnostics: {exc}")
        return

    x_columns = _mpc_validation_columns(bundle, comparison)
    if not x_columns:
        st.error("No varying numeric states or parameters are available for plotting.")
        return

    default_x = "age" if "age" in x_columns else x_columns[0]
    c1, c2, c3, c4 = st.columns([1.6, 1.0, 1.0, 1.1])
    with c1:
        x = st.selectbox(
            "Horizontal axis",
            x_columns,
            index=x_columns.index(default_x),
            format_func=_pretty,
            key="validation_mpc_x",
        )
    with c2:
        n_bins = st.slider(
            "Number of bins",
            min_value=5,
            max_value=40,
            value=20,
            step=1,
            key="validation_mpc_bins",
        )
    with c3:
        trim_label = st.selectbox(
            "Trim x-axis tails",
            ["None", "0.1%", "0.5%", "1%"],
            index=2,
            key="validation_mpc_trim",
        )
        trim_share = {
            "None": 0.0,
            "0.1%": 0.001,
            "0.5%": 0.005,
            "1%": 0.01,
        }[trim_label]
    with c4:
        equal_weight_models = st.checkbox(
            "Equal weight per model",
            value=True,
            key="validation_mpc_equal_model_weight",
            help=(
                "Within each bin, first average within each held-out model and "
                "then average across models."
            ),
        )

    filtered = comparison.copy()
    if "is_retired" in filtered:
        retired = pd.to_numeric(filtered["is_retired"], errors="coerce").fillna(1)
        filtered = filtered.loc[retired.eq(0)].copy()
    st.caption("Validation diagnostics are restricted to working-age households.")
    filter_columns = st.columns(2)

    check_values = (
        pd.to_numeric(filtered["mpc_check_units"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(check_values) and check_values.max() > check_values.min():
        with filter_columns[1]:
            check_min = float(check_values.min())
            check_max = float(check_values.max())
            check_range = st.slider(
                "Actual check size in bundle units",
                min_value=check_min,
                max_value=check_max,
                value=(check_min, check_max),
                step=max((check_max - check_min) / 250.0, 1.0e-8),
                key="validation_mpc_check_range",
            )
        filtered = filtered[
            pd.to_numeric(filtered["mpc_check_units"], errors="coerce").between(
                float(check_range[0]), float(check_range[1])
            )
        ].copy()

    if filtered.empty:
        st.warning("No MPC pairs remain after applying the filters.")
        return

    bins = _mpc_binscatter_table(
        filtered,
        x=x,
        n_bins=int(n_bins),
        trim_share=float(trim_share),
        equal_weight_models=bool(equal_weight_models),
    )
    if bins.empty:
        st.warning("The selected variable does not have enough finite observations.")
        return

    true = pd.to_numeric(filtered["true_mpc"], errors="coerce").to_numpy(dtype=float)
    surrogate = pd.to_numeric(filtered["surrogate_mpc"], errors="coerce").to_numpy(
        dtype=float
    )
    finite = np.isfinite(true) & np.isfinite(surrogate)
    true, surrogate = true[finite], surrogate[finite]
    error = surrogate - true
    sst = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / sst) if sst > 0 else np.nan
    rmse = float(np.sqrt(np.mean(error**2)))
    bias = float(np.mean(error))
    correlation = (
        float(np.corrcoef(true, surrogate)[0, 1])
        if len(true) > 1 and np.std(true) > 0 and np.std(surrogate) > 0
        else np.nan
    )
    worst_index = bins["mean_error"].abs().idxmax()
    worst = bins.loc[worst_index]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("MPC pairs", f"{len(true):,}")
    m2.metric("Held-out models", f"{filtered['model_id'].nunique():,}")
    m3.metric("MPC R²", f"{r2:.3f}" if np.isfinite(r2) else "NA")
    m4.metric(
        "MPC correlation", f"{correlation:.3f}" if np.isfinite(correlation) else "NA"
    )
    m5.metric("MPC RMSE", f"{rmse:.4f}")
    m6.metric("MPC bias", f"{bias:+.4f}")

    custom = np.column_stack(
        [
            bins["x_min"],
            bins["x_max"],
            bins["pairs"],
            bins["models"],
            bins["mean_error"],
            bins["rmse"],
        ]
    )
    show_ci = st.checkbox(
        "Show 95% uncertainty intervals",
        value=False,
        key="validation_mpc_show_ci",
        help=(
            "With equal model weights, intervals use variation in model-level "
            "bin means. Otherwise they use variation across MPC pairs."
        ),
    )
    true_error_y = (
        dict(type="data", array=1.96 * bins["true_se"], visible=True)
        if show_ci
        else None
    )
    surrogate_error_y = (
        dict(type="data", array=1.96 * bins["surrogate_se"], visible=True)
        if show_ci
        else None
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["true_mpc"],
            mode="lines+markers",
            name="True MPC",
            error_y=true_error_y,
            customdata=custom,
            hovertemplate=(
                f"{_pretty(x)} mean: %{{x:.5g}}<br>"
                "Bin range: %{customdata[0]:.5g} to %{customdata[1]:.5g}<br>"
                "True MPC: %{y:.4f}<br>"
                "Pairs: %{customdata[2]:,.0f}<br>"
                "Models: %{customdata[3]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["surrogate_mpc"],
            mode="lines+markers",
            name="Surrogate MPC",
            error_y=surrogate_error_y,
            customdata=custom,
            hovertemplate=(
                f"{_pretty(x)} mean: %{{x:.5g}}<br>"
                "Bin range: %{customdata[0]:.5g} to %{customdata[1]:.5g}<br>"
                "Surrogate MPC: %{y:.4f}<br>"
                "Bias: %{customdata[4]:+.4f}<br>"
                "Bin RMSE: %{customdata[5]:.4f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=560,
        title="Binned mean true and surrogate MPC",
        xaxis_title=_pretty(x),
        yaxis_title="Marginal propensity to consume",
        hovermode="x unified",
        legend_title="Policy source",
        margin=dict(l=45, r=20, t=65, b=45),
    )
    st.plotly_chart(fig, width="stretch")

    fig_error = go.Figure()
    fig_error.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["mean_error"],
            mode="lines+markers",
            name="Mean error",
            error_y=(
                dict(type="data", array=1.96 * bins["error_se"], visible=True)
                if show_ci
                else None
            ),
            customdata=custom,
            hovertemplate=(
                f"{_pretty(x)} mean: %{{x:.5g}}<br>"
                "Mean surrogate − true: %{y:+.4f}<br>"
                "Bin RMSE: %{customdata[5]:.4f}<br>"
                "Pairs: %{customdata[2]:,.0f}<br>"
                "Models: %{customdata[3]:,.0f}<extra></extra>"
            ),
        )
    )
    fig_error.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["rmse"],
            mode="lines+markers",
            name="RMSE",
            customdata=custom,
            hovertemplate=(
                f"{_pretty(x)} mean: %{{x:.5g}}<br>" "RMSE: %{y:.4f}<extra></extra>"
            ),
        )
    )
    fig_error.add_hline(y=0.0, line_dash="dash")
    fig_error.update_layout(
        template="plotly_white",
        height=460,
        title="Binned MPC error",
        xaxis_title=_pretty(x),
        yaxis_title="MPC error",
        hovermode="x unified",
        legend_title="Diagnostic",
        margin=dict(l=45, r=20, t=65, b=45),
    )
    st.plotly_chart(fig_error, width="stretch")

    st.caption(
        f"Largest absolute binned bias: {worst['mean_error']:+.4f} around "
        f"{_pretty(x)} = {worst['x']:.5g} "
        f"(bin RMSE {worst['rmse']:.4f}, {int(worst['models']):,} models)."
    )

    display = bins.rename(
        columns={
            "x": _pretty(x),
            "x_min": "Bin minimum",
            "x_max": "Bin maximum",
            "true_mpc": "True MPC",
            "surrogate_mpc": "Surrogate MPC",
            "mean_error": "Mean error",
            "mean_absolute_error": "Mean absolute error",
            "rmse": "RMSE",
            "pairs": "Pairs",
            "models": "Models",
        }
    )[
        [
            _pretty(x),
            "Bin minimum",
            "Bin maximum",
            "True MPC",
            "Surrogate MPC",
            "Mean error",
            "Mean absolute error",
            "RMSE",
            "Pairs",
            "Models",
        ]
    ]
    with st.expander("Binned MPC comparison data", expanded=False):
        st.dataframe(display, hide_index=True, width="stretch")
        st.download_button(
            "Download binned MPC comparison as CSV",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name=f"holdout_mpc_binscatter_{x}.csv",
            mime="text/csv",
            key="download_holdout_mpc_binscatter",
        )


def _load_dual_holdout_mpc_validation(
    polynomial_bundle_path: str | Path,
    neural_bundle_path: str | Path,
) -> pd.DataFrame:
    """Evaluate both holdout bundles on the same explicit MPC pairs."""

    polynomial_path = Path(polynomial_bundle_path).expanduser().resolve()
    neural_path = Path(neural_bundle_path).expanduser().resolve()

    polynomial_sample = _training_sample_path(polynomial_path)
    neural_sample = _training_sample_path(neural_path)

    if polynomial_sample is None:
        raise FileNotFoundError(
            "No heldout_mpc_pairs or policy_grid_sample file was found beside "
            "the polynomial holdout bundle."
        )

    if neural_sample is None:
        raise FileNotFoundError(
            "No heldout_mpc_pairs or policy_grid_sample file was found beside "
            "the neural holdout bundle."
        )

    polynomial = _load_holdout_mpc_validation(
        str(polynomial_path),
        int(polynomial_path.stat().st_mtime_ns),
        str(polynomial_sample.resolve()),
        int(polynomial_sample.stat().st_mtime_ns),
    ).rename(
        columns={
            "surrogate_mpc": "polynomial_mpc",
        }
    )

    neural = _load_holdout_mpc_validation(
        str(neural_path),
        int(neural_path.stat().st_mtime_ns),
        str(neural_sample.resolve()),
        int(neural_sample.stat().st_mtime_ns),
    )

    keys = [
        "model_id",
        "mpc_pair_id",
    ]

    neural = neural[
        keys
        + [
            "true_mpc",
            "mpc_check_units",
            "surrogate_mpc",
        ]
    ].rename(
        columns={
            "true_mpc": "neural_true_mpc",
            "mpc_check_units": "neural_check_units",
            "surrogate_mpc": "neural_mpc",
        }
    )

    comparison = polynomial.merge(
        neural,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if comparison.empty:
        raise ValueError(
            "The polynomial and neural bundles have no common held-out MPC pairs. "
            "They must use the same model split and compatible pair files."
        )

    true_gap = (
        pd.to_numeric(
            comparison["true_mpc"],
            errors="coerce",
        )
        - pd.to_numeric(
            comparison["neural_true_mpc"],
            errors="coerce",
        )
    ).abs()

    check_gap = (
        pd.to_numeric(
            comparison["mpc_check_units"],
            errors="coerce",
        )
        - pd.to_numeric(
            comparison["neural_check_units"],
            errors="coerce",
        )
    ).abs()

    if true_gap.notna().any() and float(true_gap.max()) > 1.0e-8:
        raise ValueError(
            "The polynomial and neural pair files disagree on the true MPC "
            "for at least one common model_id/mpc_pair_id."
        )

    if check_gap.notna().any() and float(check_gap.max()) > 1.0e-8:
        raise ValueError(
            "The polynomial and neural pair files disagree on the check size "
            "for at least one common model_id/mpc_pair_id."
        )

    comparison = comparison.drop(
        columns=[
            "neural_true_mpc",
            "neural_check_units",
        ],
        errors="ignore",
    )

    comparison["polynomial_error"] = (
        comparison["polynomial_mpc"] - comparison["true_mpc"]
    )

    comparison["neural_error"] = comparison["neural_mpc"] - comparison["true_mpc"]

    comparison["polynomial_squared_error"] = comparison["polynomial_error"] ** 2

    comparison["neural_squared_error"] = comparison["neural_error"] ** 2

    finite = np.ones(
        len(comparison),
        dtype=bool,
    )

    for column in (
        "true_mpc",
        "polynomial_mpc",
        "neural_mpc",
        "mpc_check_units",
    ):
        finite &= np.isfinite(
            pd.to_numeric(
                comparison[column],
                errors="coerce",
            ).to_numpy(dtype=float)
        )

    finite &= (
        pd.to_numeric(
            comparison["mpc_check_units"],
            errors="coerce",
        ).to_numpy(dtype=float)
        > 0.0
    )

    comparison = comparison.loc[finite].reset_index(drop=True)

    if comparison.empty:
        raise ValueError("No finite common MPC pairs remain.")

    return comparison


def _dual_mpc_binscatter_table(
    data: pd.DataFrame,
    *,
    x: str,
    n_bins: int,
    trim_share: float,
    equal_weight_models: bool,
) -> pd.DataFrame:
    """Bin truth and both surrogate MPCs using one common bin assignment."""

    frame = data[
        [
            "model_id",
            x,
            "true_mpc",
            "polynomial_mpc",
            "neural_mpc",
            "polynomial_error",
            "neural_error",
            "polynomial_squared_error",
            "neural_squared_error",
        ]
    ].copy()

    for column in frame.columns:
        if column != "model_id":
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    frame = frame.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if frame.empty:
        return pd.DataFrame()

    if trim_share > 0.0 and frame[x].nunique(dropna=True) > int(n_bins):
        lower, upper = frame[x].quantile(
            [
                trim_share,
                1.0 - trim_share,
            ]
        )

        frame = frame[
            frame[x].between(
                float(lower),
                float(upper),
            )
        ].copy()

    if frame.empty:
        return pd.DataFrame()

    unique_values = int(frame[x].nunique(dropna=True))

    if unique_values <= int(n_bins):
        frame["__bin"] = frame[x]
    else:
        frame["__bin"] = pd.qcut(
            frame[x],
            q=min(
                int(n_bins),
                unique_values,
            ),
            duplicates="drop",
        )

    if equal_weight_models:
        model_bin = (
            frame.groupby(
                [
                    "__bin",
                    "model_id",
                ],
                observed=True,
            )
            .agg(
                x=(x, "mean"),
                true_mpc=("true_mpc", "mean"),
                polynomial_mpc=("polynomial_mpc", "mean"),
                neural_mpc=("neural_mpc", "mean"),
                polynomial_error=("polynomial_error", "mean"),
                neural_error=("neural_error", "mean"),
                polynomial_mse=(
                    "polynomial_squared_error",
                    "mean",
                ),
                neural_mse=(
                    "neural_squared_error",
                    "mean",
                ),
                pairs=("true_mpc", "size"),
            )
            .reset_index()
        )

        bins = (
            model_bin.groupby(
                "__bin",
                observed=True,
            )
            .agg(
                x=("x", "mean"),
                true_mpc=("true_mpc", "mean"),
                polynomial_mpc=(
                    "polynomial_mpc",
                    "mean",
                ),
                neural_mpc=(
                    "neural_mpc",
                    "mean",
                ),
                polynomial_bias=(
                    "polynomial_error",
                    "mean",
                ),
                neural_bias=(
                    "neural_error",
                    "mean",
                ),
                polynomial_mse=(
                    "polynomial_mse",
                    "mean",
                ),
                neural_mse=(
                    "neural_mse",
                    "mean",
                ),
                models=(
                    "model_id",
                    "nunique",
                ),
                pairs=(
                    "pairs",
                    "sum",
                ),
            )
            .reset_index(drop=True)
        )

    else:
        bins = (
            frame.groupby(
                "__bin",
                observed=True,
            )
            .agg(
                x=(x, "mean"),
                true_mpc=("true_mpc", "mean"),
                polynomial_mpc=(
                    "polynomial_mpc",
                    "mean",
                ),
                neural_mpc=(
                    "neural_mpc",
                    "mean",
                ),
                polynomial_bias=(
                    "polynomial_error",
                    "mean",
                ),
                neural_bias=(
                    "neural_error",
                    "mean",
                ),
                polynomial_mse=(
                    "polynomial_squared_error",
                    "mean",
                ),
                neural_mse=(
                    "neural_squared_error",
                    "mean",
                ),
                models=(
                    "model_id",
                    "nunique",
                ),
                pairs=(
                    "true_mpc",
                    "size",
                ),
            )
            .reset_index(drop=True)
        )

    bins["polynomial_rmse"] = np.sqrt(bins["polynomial_mse"])

    bins["neural_rmse"] = np.sqrt(bins["neural_mse"])

    return bins.sort_values(
        "x",
        kind="mergesort",
    ).reset_index(drop=True)


def _validation_mpc_binscatter(
    polynomial_bundle: SurrogateBundle,
    neural_bundle: SurrogateBundle,
    *,
    polynomial_bundle_path: str | Path,
    neural_bundle_path: str | Path,
) -> None:
    """Show common-pair MPC diagnostics for truth and both surrogates."""

    polynomial_color = "#1f77b4"
    neural_color = "#ff7f0e"
    truth_color = "#222222"

    st.markdown("#### MPC binscatters on common held-out pairs")

    st.caption(
        "Truth, polynomial predictions, and neural-network predictions use "
        "the same held-out model/pair observations and the same bins."
    )

    enabled = st.checkbox(
        "Load household-level held-out MPC diagnostics",
        value=False,
        key="validation_enable_mpc_binscatter",
        help=(
            "This evaluates both holdout bundles on the explicit "
            "baseline/treated MPC pairs."
        ),
    )

    if not enabled:
        return

    try:
        with st.spinner("Evaluating both surrogates on common MPC pairs..."):
            comparison = _load_dual_holdout_mpc_validation(
                polynomial_bundle_path,
                neural_bundle_path,
            )

    except Exception as exc:
        st.error("Could not construct dual-surrogate MPC diagnostics: " f"{exc}")
        return

    polynomial_columns = _mpc_validation_columns(
        polynomial_bundle,
        comparison,
    )

    neural_columns = set(
        _mpc_validation_columns(
            neural_bundle,
            comparison,
        )
    )

    x_columns = [column for column in polynomial_columns if column in neural_columns]

    if not x_columns:
        st.error("No common varying numeric state or parameter is available.")
        return

    default_x = "age" if "age" in x_columns else x_columns[0]

    left, middle_1, middle_2, right = st.columns(
        [
            1.7,
            1.0,
            1.0,
            1.2,
        ]
    )

    with left:
        x = st.selectbox(
            "Horizontal axis",
            x_columns,
            index=x_columns.index(default_x),
            format_func=_pretty,
            key="validation_mpc_x",
        )

    with middle_1:
        n_bins = st.slider(
            "Number of bins",
            min_value=5,
            max_value=40,
            value=20,
            step=1,
            key="validation_mpc_bins",
        )

    with middle_2:
        trim_label = st.selectbox(
            "Trim x-axis tails",
            [
                "None",
                "0.1%",
                "0.5%",
                "1%",
            ],
            index=2,
            key="validation_mpc_trim",
        )

        trim_share = {
            "None": 0.0,
            "0.1%": 0.001,
            "0.5%": 0.005,
            "1%": 0.01,
        }[trim_label]

    with right:
        equal_weight_models = st.checkbox(
            "Equal weight per model",
            value=True,
            key="validation_mpc_equal_model_weight",
            help=(
                "Average first within each model and bin, then average "
                "equally across models."
            ),
        )

    filtered = comparison.copy()

    if "is_retired" in filtered:
        retired = pd.to_numeric(
            filtered["is_retired"],
            errors="coerce",
        ).fillna(1)

        filtered = filtered.loc[retired.eq(0)].copy()

    check_values = (
        pd.to_numeric(
            filtered["mpc_check_units"],
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if len(check_values) and check_values.max() > check_values.min():
        check_min = float(check_values.min())

        check_max = float(check_values.max())

        check_range = st.slider(
            "Actual check size in bundle units",
            min_value=check_min,
            max_value=check_max,
            value=(
                check_min,
                check_max,
            ),
            step=max(
                (check_max - check_min) / 250.0,
                1.0e-8,
            ),
            key="validation_mpc_check_range",
        )

        filtered = filtered[
            pd.to_numeric(
                filtered["mpc_check_units"],
                errors="coerce",
            ).between(
                float(check_range[0]),
                float(check_range[1]),
            )
        ].copy()

    if filtered.empty:
        st.warning("No common MPC pairs remain after filtering.")
        return

    bins = _dual_mpc_binscatter_table(
        filtered,
        x=x,
        n_bins=int(n_bins),
        trim_share=float(trim_share),
        equal_weight_models=bool(equal_weight_models),
    )

    if bins.empty:
        st.warning("The selected variable does not have enough finite observations.")
        return

    def metric_row(
        label: str,
        prediction_column: str,
    ) -> dict[str, Any]:
        true = pd.to_numeric(
            filtered["true_mpc"],
            errors="coerce",
        ).to_numpy(dtype=float)

        predicted = pd.to_numeric(
            filtered[prediction_column],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite = np.isfinite(true) & np.isfinite(predicted)

        true = true[finite]
        predicted = predicted[finite]

        error = predicted - true

        denominator = float(np.sum((true - np.mean(true)) ** 2))

        r2 = (
            float(1.0 - np.sum(error**2) / denominator) if denominator > 0.0 else np.nan
        )

        correlation = (
            float(
                np.corrcoef(
                    true,
                    predicted,
                )[0, 1]
            )
            if (len(true) > 1 and np.std(true) > 0.0 and np.std(predicted) > 0.0)
            else np.nan
        )

        return {
            "Surrogate": label,
            "Pairs": int(len(true)),
            "MPC R²": r2,
            "MPC correlation": correlation,
            "MPC RMSE": (float(np.sqrt(np.mean(error**2))) if len(error) else np.nan),
            "MPC bias": (float(np.mean(error)) if len(error) else np.nan),
        }

    metrics = pd.DataFrame(
        [
            metric_row(
                "Polynomial",
                "polynomial_mpc",
            ),
            metric_row(
                "Neural network",
                "neural_mpc",
            ),
        ]
    )

    st.dataframe(
        metrics.style.format(
            {
                "MPC R²": "{:.4f}",
                "MPC correlation": "{:.4f}",
                "MPC RMSE": "{:.5f}",
                "MPC bias": "{:+.5f}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
    )

    st.caption(
        f"Common pairs: {len(filtered):,}; "
        f"common held-out models: "
        f"{filtered['model_id'].nunique():,}."
    )

    mpc_figure = go.Figure()

    mpc_figure.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["true_mpc"],
            mode="lines+markers",
            name="True MPC",
            line=dict(
                color=truth_color,
                width=3,
            ),
            marker=dict(
                color=truth_color,
                size=7,
            ),
            hovertemplate=(
                f"{_pretty(x)}: %{{x:.5g}}<br>" "True MPC: %{y:.4f}" "<extra></extra>"
            ),
        )
    )

    mpc_figure.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["polynomial_mpc"],
            mode="lines+markers",
            name="Polynomial",
            line=dict(
                color=polynomial_color,
                width=2.5,
            ),
            marker=dict(
                color=polynomial_color,
                size=7,
            ),
            hovertemplate=(
                f"{_pretty(x)}: %{{x:.5g}}<br>"
                "Polynomial MPC: %{y:.4f}"
                "<extra></extra>"
            ),
        )
    )

    mpc_figure.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["neural_mpc"],
            mode="lines+markers",
            name="Neural network",
            line=dict(
                color=neural_color,
                width=2.5,
            ),
            marker=dict(
                color=neural_color,
                size=7,
            ),
            hovertemplate=(
                f"{_pretty(x)}: %{{x:.5g}}<br>" "Neural MPC: %{y:.4f}" "<extra></extra>"
            ),
        )
    )

    mpc_figure.update_layout(
        template="plotly_white",
        height=580,
        title=("Binned mean MPC: " "truth versus both surrogates"),
        xaxis_title=_pretty(x),
        yaxis_title=("Marginal propensity to consume"),
        hovermode="x unified",
        legend_title="Policy source",
        margin=dict(
            l=45,
            r=20,
            t=65,
            b=45,
        ),
    )

    st.plotly_chart(
        mpc_figure,
        width="stretch",
    )

    error_figure = go.Figure()

    error_figure.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["polynomial_bias"],
            mode="lines+markers",
            name="Polynomial bias",
            line=dict(
                color=polynomial_color,
                width=2.5,
            ),
            marker=dict(
                color=polynomial_color,
            ),
        )
    )

    error_figure.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["neural_bias"],
            mode="lines+markers",
            name="Neural bias",
            line=dict(
                color=neural_color,
                width=2.5,
            ),
            marker=dict(
                color=neural_color,
            ),
        )
    )

    error_figure.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["polynomial_rmse"],
            mode="lines+markers",
            name="Polynomial RMSE",
            line=dict(
                color=polynomial_color,
                width=2,
                dash="dash",
            ),
            marker=dict(
                color=polynomial_color,
            ),
        )
    )

    error_figure.add_trace(
        go.Scatter(
            x=bins["x"],
            y=bins["neural_rmse"],
            mode="lines+markers",
            name="Neural RMSE",
            line=dict(
                color=neural_color,
                width=2,
                dash="dash",
            ),
            marker=dict(
                color=neural_color,
            ),
        )
    )

    error_figure.add_hline(
        y=0.0,
        line_dash="dot",
        line_color="#777777",
    )

    error_figure.update_layout(
        template="plotly_white",
        height=470,
        title="Binned MPC error",
        xaxis_title=_pretty(x),
        yaxis_title="MPC error",
        hovermode="x unified",
        legend_title="Diagnostic",
        margin=dict(
            l=45,
            r=20,
            t=65,
            b=45,
        ),
    )

    st.plotly_chart(
        error_figure,
        width="stretch",
    )

    display = bins.rename(
        columns={
            "x": _pretty(x),
            "true_mpc": "True MPC",
            "polynomial_mpc": "Polynomial MPC",
            "neural_mpc": "Neural MPC",
            "polynomial_bias": "Polynomial bias",
            "neural_bias": "Neural bias",
            "polynomial_rmse": "Polynomial RMSE",
            "neural_rmse": "Neural RMSE",
            "pairs": "Pairs",
            "models": "Models",
        }
    )

    with st.expander(
        "Binned MPC comparison data",
        expanded=False,
    ):
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
        )

        st.download_button(
            "Download binned MPC comparison as CSV",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name=("holdout_mpc_binscatter_" f"both_surrogates_{x}.csv"),
            mime="text/csv",
            key=("download_holdout_" "mpc_binscatter_both"),
        )


def _validation_page(
    polynomial_bundle: SurrogateBundle,
    neural_bundle: SurrogateBundle,
    *,
    polynomial_bundle_path: str | Path,
    neural_bundle_path: str | Path,
) -> None:
    """Compare polynomial and neural holdout performance in one validation tab."""

    polynomial_color = "#1f77b4"  # blue
    neural_color = "#ff7f0e"  # orange

    st.subheader("Validation on entirely held-out parameterizations")
    st.caption(
        "Polynomial and neural results are shown on the same held-out metrics. "
        "R² is reported only for fitted surrogates; reference rules, when "
        "available, are displayed only as RMSE benchmarks."
    )

    def _aggregate(
        bundle: SurrogateBundle,
        label: str,
    ) -> pd.DataFrame:
        frame = bundle.validation_metrics.copy()

        required = {"target", "r2", "rmse"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{label} validation metrics are missing columns: " f"{sorted(missing)}"
            )

        frame["target"] = frame["target"].astype(str)

        for column in frame.columns:
            if column != "target":
                frame[column] = pd.to_numeric(
                    frame[column],
                    errors="coerce",
                )

        return frame.drop_duplicates("target", keep="first").reset_index(drop=True)

    try:
        polynomial = _aggregate(
            polynomial_bundle,
            "Polynomial",
        )
        neural = _aggregate(
            neural_bundle,
            "Neural",
        )
    except Exception as exc:
        st.error(str(exc))
        return

    # Preserve the target order in the validation files.
    target_order = list(
        dict.fromkeys(polynomial["target"].tolist() + neural["target"].tolist())
    )
    order_lookup = {target: position for position, target in enumerate(target_order)}

    combined = polynomial.merge(
        neural,
        on="target",
        how="outer",
        suffixes=("_polynomial", "_neural"),
        validate="one_to_one",
    )
    combined["__order"] = combined["target"].map(order_lookup)
    combined = combined.sort_values("__order", kind="mergesort").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Aggregate table
    # ------------------------------------------------------------------

    table = pd.DataFrame(
        {
            "Policy choice": combined["target"].map(_pretty),
            "Polynomial R²": pd.to_numeric(
                combined["r2_polynomial"],
                errors="coerce",
            ),
            "Neural R²": pd.to_numeric(
                combined["r2_neural"],
                errors="coerce",
            ),
            "Polynomial RMSE": pd.to_numeric(
                combined["rmse_polynomial"],
                errors="coerce",
            ),
            "Neural RMSE": pd.to_numeric(
                combined["rmse_neural"],
                errors="coerce",
            ),
        }
    )

    table["Neural − polynomial R²"] = table["Neural R²"] - table["Polynomial R²"]
    table["Neural − polynomial RMSE"] = table["Neural RMSE"] - table["Polynomial RMSE"]

    st.markdown("#### Aggregate holdout metrics")

    styled_table = (
        table.style.format(
            {
                "Polynomial R²": "{:.4f}",
                "Neural R²": "{:.4f}",
                "Polynomial RMSE": "{:.6g}",
                "Neural RMSE": "{:.6g}",
                "Neural − polynomial R²": "{:+.4f}",
                "Neural − polynomial RMSE": "{:+.6g}",
            },
            na_rep="—",
        )
        .set_properties(
            subset=[
                "Polynomial R²",
                "Polynomial RMSE",
            ],
            **{"background-color": "rgba(31,119,180,0.12)"},
        )
        .set_properties(
            subset=[
                "Neural R²",
                "Neural RMSE",
            ],
            **{"background-color": "rgba(255,127,14,0.12)"},
        )
    )

    st.dataframe(
        styled_table,
        hide_index=True,
        width="stretch",
    )

    # ------------------------------------------------------------------
    # Aggregate R² plot
    # ------------------------------------------------------------------

    r2_data = table.dropna(
        subset=[
            "Polynomial R²",
            "Neural R²",
        ],
        how="all",
    )

    if not r2_data.empty:
        fig_r2 = go.Figure()

        fig_r2.add_trace(
            go.Bar(
                x=r2_data["Policy choice"],
                y=r2_data["Polynomial R²"],
                name="Polynomial",
                marker_color=polynomial_color,
                hovertemplate=("%{x}<br>" "Polynomial R²: %{y:.4f}" "<extra></extra>"),
            )
        )

        fig_r2.add_trace(
            go.Bar(
                x=r2_data["Policy choice"],
                y=r2_data["Neural R²"],
                name="Neural network",
                marker_color=neural_color,
                hovertemplate=("%{x}<br>" "Neural R²: %{y:.4f}" "<extra></extra>"),
            )
        )

        fig_r2.add_hline(
            y=0.0,
            line_dash="dash",
            line_color="#777777",
        )

        fig_r2.update_layout(
            template="plotly_white",
            height=480,
            title="Held-out R²",
            xaxis_title="Policy choice",
            yaxis_title="R²",
            barmode="group",
            legend_title="Surrogate",
            margin=dict(
                l=45,
                r=20,
                t=65,
                b=45,
            ),
        )

        st.plotly_chart(
            fig_r2,
            width="stretch",
        )

    # ------------------------------------------------------------------
    # Aggregate RMSE plot
    # ------------------------------------------------------------------

    rmse_data = table.dropna(
        subset=[
            "Polynomial RMSE",
            "Neural RMSE",
        ],
        how="all",
    )

    if not rmse_data.empty:
        fig_rmse = go.Figure()

        fig_rmse.add_trace(
            go.Bar(
                x=rmse_data["Policy choice"],
                y=rmse_data["Polynomial RMSE"],
                name="Polynomial",
                marker_color=polynomial_color,
                hovertemplate=(
                    "%{x}<br>" "Polynomial RMSE: %{y:.6g}" "<extra></extra>"
                ),
            )
        )

        fig_rmse.add_trace(
            go.Bar(
                x=rmse_data["Policy choice"],
                y=rmse_data["Neural RMSE"],
                name="Neural network",
                marker_color=neural_color,
                hovertemplate=("%{x}<br>" "Neural RMSE: %{y:.6g}" "<extra></extra>"),
            )
        )

        # Reference rules are common benchmarks, not fitted surrogates.
        reference_specs = [
            (
                "reference_70_20_10_rmse",
                "70/20/10 reference",
                "#9e9e9e",
            ),
            (
                "reference_consume_all_rmse",
                "Consume-all reference",
                "#c7c7c7",
            ),
        ]

        for root, label, color in reference_specs:
            polynomial_column = f"{root}_polynomial"
            neural_column = f"{root}_neural"

            if polynomial_column in combined.columns:
                source_column = polynomial_column
            elif neural_column in combined.columns:
                source_column = neural_column
            else:
                continue

            reference = pd.DataFrame(
                {
                    "Policy choice": combined["target"].map(_pretty),
                    "RMSE": pd.to_numeric(
                        combined[source_column],
                        errors="coerce",
                    ),
                }
            ).set_index("Policy choice")

            values = rmse_data["Policy choice"].map(reference["RMSE"])

            if values.notna().any():
                fig_rmse.add_trace(
                    go.Bar(
                        x=rmse_data["Policy choice"],
                        y=values,
                        name=label,
                        marker_color=color,
                        opacity=0.75,
                    )
                )

        fig_rmse.update_layout(
            template="plotly_white",
            height=480,
            title="Held-out RMSE",
            xaxis_title="Policy choice",
            yaxis_title="RMSE",
            barmode="group",
            legend_title="Model or benchmark",
            margin=dict(
                l=45,
                r=20,
                t=65,
                b=45,
            ),
        )

        st.plotly_chart(
            fig_rmse,
            width="stretch",
        )

    # ------------------------------------------------------------------
    # Model-level comparison
    # ------------------------------------------------------------------

    polynomial_models = polynomial_bundle.validation_by_model.copy()
    neural_models = neural_bundle.validation_by_model.copy()

    if polynomial_models.empty or neural_models.empty:
        st.info(
            "Model-level comparison requires both validation-by-model "
            "tables. For the neural bundle, place "
            "neural_validation_by_model.csv beside the holdout .pt file."
        )
        return

    for frame in (
        polynomial_models,
        neural_models,
    ):
        frame["model_id"] = frame["model_id"].astype(str)
        frame["target"] = frame["target"].astype(str)
        frame["r2"] = pd.to_numeric(
            frame["r2"],
            errors="coerce",
        )
        frame["rmse"] = pd.to_numeric(
            frame["rmse"],
            errors="coerce",
        )

    polynomial_test_models = _held_out_model_ids(polynomial_bundle)
    neural_test_models = _held_out_model_ids(neural_bundle)

    if polynomial_test_models != neural_test_models:
        st.warning(
            "The polynomial and neural bundles do not contain exactly "
            "the same held-out model IDs. The model-level table uses "
            "the intersection where possible."
        )

    common_targets = sorted(
        set(polynomial_models["target"]) & set(neural_models["target"])
    )

    if not common_targets:
        st.info(
            "The polynomial and neural model-level validation files "
            "have no common policy targets."
        )
        return

    st.markdown("#### Error distribution across held-out models")

    chosen_target = st.selectbox(
        "Policy choice",
        common_targets,
        format_func=_pretty,
        key="validation_comparison_target",
    )

    polynomial_target = polynomial_models[
        polynomial_models["target"] == chosen_target
    ].copy()

    neural_target = neural_models[neural_models["target"] == chosen_target].copy()

    # ------------------------------------------------------------------
    # Model-level RMSE box plot
    # ------------------------------------------------------------------

    fig_models = go.Figure()

    fig_models.add_trace(
        go.Box(
            y=polynomial_target["rmse"],
            name="Polynomial",
            boxpoints="all",
            marker_color=polynomial_color,
            line_color=polynomial_color,
            jitter=0.25,
            hovertemplate=("Polynomial RMSE: %{y:.6g}" "<extra></extra>"),
        )
    )

    fig_models.add_trace(
        go.Box(
            y=neural_target["rmse"],
            name="Neural network",
            boxpoints="all",
            marker_color=neural_color,
            line_color=neural_color,
            jitter=0.25,
            hovertemplate=("Neural RMSE: %{y:.6g}" "<extra></extra>"),
        )
    )

    fig_models.update_layout(
        template="plotly_white",
        height=480,
        title=("Held-out model RMSE: " f"{_pretty(chosen_target)}"),
        xaxis_title="Surrogate",
        yaxis_title="RMSE",
        showlegend=False,
        margin=dict(
            l=45,
            r=20,
            t=65,
            b=45,
        ),
    )

    st.plotly_chart(
        fig_models,
        width="stretch",
    )

    # ------------------------------------------------------------------
    # Model-level table
    # ------------------------------------------------------------------

    model_table = (
        polynomial_target[
            [
                "model_id",
                "r2",
                "rmse",
            ]
        ]
        .rename(
            columns={
                "r2": "Polynomial R²",
                "rmse": "Polynomial RMSE",
            }
        )
        .merge(
            neural_target[
                [
                    "model_id",
                    "r2",
                    "rmse",
                ]
            ].rename(
                columns={
                    "r2": "Neural R²",
                    "rmse": "Neural RMSE",
                }
            ),
            on="model_id",
            how="outer",
            validate="one_to_one",
        )
    )

    model_table["Neural − polynomial R²"] = (
        model_table["Neural R²"] - model_table["Polynomial R²"]
    )
    model_table["Neural − polynomial RMSE"] = (
        model_table["Neural RMSE"] - model_table["Polynomial RMSE"]
    )

    model_table = model_table.sort_values(
        "Neural − polynomial RMSE",
        kind="mergesort",
    )

    styled_models = (
        model_table.style.format(
            {
                "Polynomial R²": "{:.4f}",
                "Neural R²": "{:.4f}",
                "Polynomial RMSE": "{:.6g}",
                "Neural RMSE": "{:.6g}",
                "Neural − polynomial R²": "{:+.4f}",
                "Neural − polynomial RMSE": "{:+.6g}",
            },
            na_rep="—",
        )
        .set_properties(
            subset=[
                "Polynomial R²",
                "Polynomial RMSE",
            ],
            **{"background-color": "rgba(31,119,180,0.12)"},
        )
        .set_properties(
            subset=[
                "Neural R²",
                "Neural RMSE",
            ],
            **{"background-color": "rgba(255,127,14,0.12)"},
        )
    )

    st.dataframe(
        styled_models,
        hide_index=True,
        width="stretch",
    )

    _validation_mpc_binscatter(
        polynomial_bundle,
        neural_bundle,
        polynomial_bundle_path=polynomial_bundle_path,
        neural_bundle_path=neural_bundle_path,
    )


def main() -> None:
    st.set_page_config(page_title="HA Policy Surrogate", layout="wide")
    args = _parse_args()
    st.title("Smooth HA policy-function explorer")
    st.caption(
        "The page evaluates a fitted surrogate; it does not re-solve the HA model. "
        "Sliders stay within marginal training ranges, but unusual combinations of "
        "parameters can still lie outside the joint support of solved models."
    )

    initial_type = _initial_surrogate_type(args)
    surrogate_types = ["polynomial", "neural"]
    surrogate_type = st.sidebar.selectbox(
        "Surrogate type",
        surrogate_types,
        index=surrogate_types.index(initial_type),
        format_func=lambda value: SURROGATE_TYPE_LABELS[value],
        key="surrogate_type",
    )

    neural_device = str(args.neural_device)
    if surrogate_type == "neural":
        neural_device = st.sidebar.selectbox(
            "Neural inference device",
            ["auto", "cpu", "cuda"],
            index=["auto", "cpu", "cuda"].index(str(args.neural_device)),
            key="neural_inference_device",
            help=(
                "Auto uses CUDA when it is available and otherwise uses CPU. "
                "CPU is generally sufficient for interactive slices."
            ),
        )

    bundle_path = st.sidebar.text_input(
        f"{SURROGATE_TYPE_LABELS[surrogate_type]} production bundle",
        value=_default_bundle_path(args, surrogate_type),
        key=f"production_bundle_path__{surrogate_type}",
    )
    try:
        bundle = _load_bundle(
            str(Path(bundle_path).expanduser()),
            surrogate_type,
            neural_device,
        )
    except Exception as exc:
        st.error(
            f"Could not load the {SURROGATE_TYPE_LABELS[surrogate_type].lower()} "
            f"production bundle: {exc}"
        )
        st.stop()

    if bundle.surrogate_type != surrogate_type:
        st.error(
            f"The selected type is {surrogate_type!r}, but the loaded bundle is "
            f"{bundle.surrogate_type!r}."
        )
        st.stop()

    default_evaluation_path = Path(
        _default_evaluation_bundle_path(args, surrogate_type)
    )
    metadata_name = bundle.training_metadata.get("evaluation_bundle_filename")
    if metadata_name:
        default_evaluation_path = Path(bundle_path).expanduser().resolve().parent / str(
            metadata_name
        )
    evaluation_bundle_path = st.sidebar.text_input(
        f"{SURROGATE_TYPE_LABELS[surrogate_type]} true-holdout bundle",
        value=str(default_evaluation_path),
        key=f"evaluation_bundle_path__{surrogate_type}",
    )
    evaluation_bundle: SurrogateBundle | None = None
    evaluation_error: str | None = None
    try:
        evaluation_bundle = _load_bundle(
            str(Path(evaluation_bundle_path).expanduser()),
            surrogate_type,
            neural_device,
        )
        if evaluation_bundle.surrogate_type != surrogate_type:
            raise ValueError(
                "Production and evaluation bundles use different surrogate types."
            )
        _validate_evaluation_bundle(evaluation_bundle)
    except Exception as exc:
        evaluation_error = str(exc)

    st.sidebar.caption(f"Loaded: {bundle.surrogate_label}")

    continuous_columns = _continuous_plot_columns(bundle)

    if not continuous_columns:
        st.error("The surrogate has no continuous plotting variables.")
        return

    page_names = [
        "One-dimensional slices",
        "Two-dimensional surfaces",
        "MPC distribution",
        "Grid versus surrogate",
        "MPCs: grid versus surrogate",
        "Validation",
    ]

    # Use an ordinary widget rather than dynamic tabs. Only the selected page is
    # executed, preserving the low-memory behavior across reruns.
    selected_page = st.radio(
        "Page",
        page_names,
        horizontal=True,
        label_visibility="collapsed",
        key="main_page",
    )

    # Exact-grid pages use the evaluation bundle's ranges, while ordinary pages
    # use the production bundle.
    axis_bundle = (
        evaluation_bundle
        if selected_page == "Grid versus surrogate" and evaluation_bundle is not None
        else bundle
    )
    active_axes = _page_axis_columns(axis_bundle, selected_page)
    axis_bindings = _sidebar_axis_bindings(active_axes)

    base = _base_controls(
        bundle,
        disabled_state_columns=set(),
        axis_controlled_by=axis_bindings,
    )
    base_frame = pd.DataFrame([base])
    if "current_income" in base_frame:
        base_frame = _link_income_states_to_pre_tax_income(
            bundle,
            base_frame,
            pd.to_numeric(base_frame["current_income"], errors="coerce").to_numpy(
                dtype=float
            ),
            base_values=base,
            proportional_after_tax=False,
        )
    base_frame = bundle.prepare_inputs(base_frame)
    # Preserve the public deviation control after the bundle has recomputed all
    # model inputs; it is intentionally not a regression feature itself.
    base_frame[INCOME_DEVIATION_COLUMN] = float(base.get(INCOME_DEVIATION_COLUMN, 0.0))
    base.update(base_frame.iloc[0].to_dict())

    if selected_page == "One-dimensional slices":
        _one_dimensional_page(
            bundle,
            base,
            bundle_path=bundle_path,
        )
    elif selected_page == "Two-dimensional surfaces":
        _surface_page(
            bundle,
            base,
            bundle_path=bundle_path,
        )
    elif selected_page == "MPC distribution":
        _mpc_distribution_page(
            bundle,
            base,
            bundle_path=bundle_path,
        )
    elif selected_page in {
        "Grid versus surrogate",
        "MPCs: grid versus surrogate",
        "Validation",
    }:
        if evaluation_bundle is None:
            expected = (
                "neural_policy_surrogate_holdout.pt"
                if surrogate_type == "neural"
                else "policy_surrogate_holdout.joblib"
            )
            st.error(
                f"True-holdout pages require {expected}. "
                f"Could not load it: {evaluation_error}"
            )
            return
        if selected_page == "Grid versus surrogate":
            _exact_grid_comparison_page(
                evaluation_bundle,
                base,
                bundle_path=evaluation_bundle_path,
            )
        elif selected_page == "MPCs: grid versus surrogate":
            _mpc_grid_comparison_page(
                evaluation_bundle,
                base,
                bundle_path=evaluation_bundle_path,
            )
        else:
            polynomial_validation_path = st.sidebar.text_input(
                "Polynomial true-holdout bundle",
                value=str(args.polynomial_evaluation_bundle),
                key="validation_polynomial_bundle_path",
            )

            neural_validation_path = st.sidebar.text_input(
                "Neural true-holdout bundle",
                value=str(args.neural_evaluation_bundle),
                key="validation_neural_bundle_path",
            )

            try:
                polynomial_validation_bundle = _load_bundle(
                    str(Path(polynomial_validation_path).expanduser()),
                    "polynomial",
                    "cpu",
                )
                _validate_evaluation_bundle(polynomial_validation_bundle)
            except Exception as exc:
                st.error("Could not load the polynomial holdout bundle: " f"{exc}")
                return

            try:
                neural_validation_bundle = _load_bundle(
                    str(Path(neural_validation_path).expanduser()),
                    "neural",
                    neural_device,
                )
                _validate_evaluation_bundle(neural_validation_bundle)
            except Exception as exc:
                st.error("Could not load the neural holdout bundle: " f"{exc}")
                return

            _validation_page(
                polynomial_validation_bundle,
                neural_validation_bundle,
                polynomial_bundle_path=polynomial_validation_path,
                neural_bundle_path=neural_validation_path,
            )


if __name__ == "__main__":
    main()
