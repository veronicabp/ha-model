"""Streamlit explorer for smooth heterogeneous-agent policy functions.

Run with:

    streamlit run policy_explorer.py -- --bundle OUTPUT/policy_surrogate.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

APP_ROOT = Path(__file__).resolve().parent
APP_DATA_DIR = APP_ROOT / "app_data"
DEFAULT_BUNDLE_PATH = APP_DATA_DIR / "policy_surrogate.joblib"

try:
    import streamlit as st
except ImportError as exc:  # clearer error outside Streamlit
    raise ImportError(
        "Streamlit is required for the interactive explorer. Install it with "
        "`pip install streamlit`."
    ) from exc

from ha_policy_surrogate import (
    PolicySurrogateBundle,
    read_policy_dataset,
)
from ha_policy_visualization import (
    complete_derived_state,
    link_labor_income_state,
    load_exact_grid_slice,
    make_policy_surface,
)
from ha_mpc_distribution import (
    PopulationAssumptions,
    compute_check_mpcs,
    generate_synthetic_population,
    infer_money_unit_info,
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

SPECIAL_LABELS = {
    "consumption": "Consumption",
    "deposit": "Net Transfer into Illiquid Account (d)",
    "delta_liquid_assets": "Change in Liquid Assets (Liquid Saving)",
    "delta_illiquid_assets": "Change in Illiquid Assets (Illiquid Saving)",
    "next_liquid_assets": "Next Liquid Assets",
    "next_illiquid_assets": "Next Illiquid Assets",
    "after_tax_income": "After-Tax Income",
    "param__policy__tax__kind": "Tax Schedule",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--bundle",
        type=str,
        default=str(DEFAULT_BUNDLE_PATH),
        help="Path to the fitted policy-surrogate bundle.",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


@st.cache_resource(show_spinner=False)
def _load_bundle(path: str) -> PolicySurrogateBundle:
    return PolicySurrogateBundle.load(path)


@st.cache_data(show_spinner=False)
def _load_training_sample(bundle_path: str) -> pd.DataFrame:
    """Load the sampled grid data saved beside the surrogate bundle."""

    parent = Path(bundle_path).expanduser().resolve().parent
    candidates = [
        parent / "policy_grid_sample.pkl.gz",
        parent / "policy_grid_sample.parquet",
        parent / "policy_grid_sample.pkl",
        parent / "policy_grid_sample.csv",
    ]

    for path in candidates:
        if path.exists():
            return read_policy_dataset(path)

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_model_catalog(bundle_path: str) -> pd.DataFrame:
    catalog_path = Path(bundle_path).expanduser().resolve().parent / "model_catalog.csv"

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Could not find {catalog_path}. The exact-grid tab "
            "requires model_catalog.csv beside the surrogate bundle."
        )

    catalog = pd.read_csv(catalog_path)

    if "usable" in catalog:
        usable = catalog["usable"].astype(str).str.lower().isin({"true", "1", "yes"})
        catalog = catalog.loc[usable].copy()

    if catalog.empty:
        raise ValueError("model_catalog.csv contains no usable models.")

    return catalog.reset_index(drop=True)


def _out_of_sample_model_summary(
    bundle: PolicySurrogateBundle,
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

    summary = summary.merge(
        catalog_small[["model_id", "model_dir"]],
        on="model_id",
        how="left",
        validate="one_to_one",
    )

    summary = summary.sort_values(
        ["mean_r2", "model_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return summary


def _catalog_values_equal(
    series: pd.Series,
    value: Any,
) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")

    try:
        numeric_value = float(value)
        value_is_numeric = np.isfinite(numeric_value)
    except (TypeError, ValueError):
        value_is_numeric = False

    if value_is_numeric and numeric.notna().any():
        return np.isclose(
            numeric.to_numpy(dtype=float),
            numeric_value,
            rtol=1.0e-10,
            atol=1.0e-12,
            equal_nan=False,
        )

    return series.astype(str).to_numpy() == str(value)


def _filter_catalog(
    catalog: pd.DataFrame,
    selections: dict[str, Any],
) -> pd.DataFrame:
    mask = np.ones(len(catalog), dtype=bool)

    for column, value in selections.items():
        if column not in catalog:
            continue
        mask &= _catalog_values_equal(
            catalog[column],
            value,
        )

    return catalog.loc[mask].copy()


def _sorted_catalog_values(
    series: pd.Series,
) -> list[Any]:
    values = series.dropna().unique().tolist()

    if not values:
        return []

    numeric = pd.to_numeric(
        pd.Series(values),
        errors="coerce",
    )

    if numeric.notna().all():
        return [float(value) for value in sorted(numeric.tolist())]

    return sorted([str(value) for value in values])


def _contains_catalog_value(
    values: list[Any],
    candidate: Any,
) -> bool:
    if not values:
        return False

    return bool(
        _catalog_values_equal(
            pd.Series(values),
            candidate,
        ).any()
    )


def _initial_catalog_value(
    bundle: PolicySurrogateBundle,
    column: str,
    available: list[Any],
) -> Any:
    if not available:
        raise ValueError(f"No available values for {column}.")

    default = bundle.default_values.get(column)

    if default is not None and _contains_catalog_value(
        available,
        default,
    ):
        for value in available:
            if _contains_catalog_value([value], default):
                return value

    try:
        default_numeric = float(default)
        numeric = np.asarray(available, dtype=float)
        return available[int(np.argmin(np.abs(numeric - default_numeric)))]
    except (TypeError, ValueError):
        return available[0]


def _parameter_button_order(
    bundle: PolicySurrogateBundle,
    catalog: pd.DataFrame,
) -> list[str]:
    candidates = [
        column
        for column in (
            list(bundle.feature_spec.parameter_columns)
            + list(bundle.feature_spec.categorical_parameter_columns)
        )
        if (
            column in catalog
            and not column.endswith("__missing")
            and catalog[column].nunique(dropna=True) > 1
        )
    ]

    preferred = [
        "param__config__discount_factor",
        "param__config__risk_aversion",
        "param__config__allow_borrowing",
        "param__config__borrowing_limit",
        "param__config__liquid_interest_rate",
        "param__config__illiquid_interest_rate",
        "param__config__borrowing_interest_rate",
        "param__config__ct_linear_adjustment_cost",
        "param__config__ct_convex_adjustment_cost",
        "param__config__ct_automatic_illiquid_income_share",
        "param__policy__income__persistence",
        "param__policy__income__innovation_std",
        "param__policy__employment__job_loss_probability",
        "param__policy__employment__job_finding_probability",
        "param__policy__employment__unemployment_replacement_rate",
        "param__policy__tax__kind",
        "param__policy__tax__flat_rate",
        "param__policy__tax__progressive_rate",
        "param__policy__pension__replacement_rate",
    ]

    ordered = [column for column in preferred if column in candidates]

    ordered.extend(sorted(column for column in candidates if column not in ordered))

    return ordered


def _button_label(value: Any) -> str:
    try:
        number = float(value)
        if np.isfinite(number):
            return f"{number:.6g}"
    except (TypeError, ValueError):
        pass

    return str(value)


def _parameter_button_selector(
    bundle: PolicySurrogateBundle,
    catalog: pd.DataFrame,
    *,
    bundle_path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sequential parameter selector with unavailable choices disabled."""

    columns = _parameter_button_order(
        bundle,
        catalog,
    )

    selector_identity = str(Path(bundle_path).expanduser().resolve())
    identity_key = "exact_selector_bundle_identity"
    selection_key = "exact_parameter_selection"

    if st.session_state.get(identity_key) != selector_identity:
        st.session_state[identity_key] = selector_identity
        st.session_state[selection_key] = {}

    selections = dict(st.session_state.get(selection_key, {}))
    prior_selections: dict[str, Any] = {}

    for index, column in enumerate(columns):
        eligible = _filter_catalog(
            catalog,
            prior_selections,
        )

        all_values = _sorted_catalog_values(catalog[column])
        available_values = _sorted_catalog_values(eligible[column])

        current = selections.get(column)
        if current is None or not _contains_catalog_value(
            available_values,
            current,
        ):
            current = _initial_catalog_value(
                bundle,
                column,
                available_values,
            )
            selections[column] = current

        st.markdown(f"**{_pretty(column)}**")

        for start in range(0, len(all_values), 6):
            chunk = all_values[start : start + 6]
            button_columns = st.columns(len(chunk))

            for position, value in enumerate(chunk):
                available = _contains_catalog_value(
                    available_values,
                    value,
                )
                selected = _contains_catalog_value(
                    [current],
                    value,
                )

                with button_columns[position]:
                    clicked = st.button(
                        _button_label(value),
                        key=(f"exact_param_button__" f"{index}__{start + position}"),
                        disabled=not available,
                        type="primary" if selected else "secondary",
                        use_container_width=True,
                    )

                if clicked:
                    selections[column] = value

                    # Later parameter choices are conditional on this one.
                    for later_column in columns[index + 1 :]:
                        selections.pop(later_column, None)

                    st.session_state[selection_key] = selections
                    st.rerun()

        prior_selections[column] = current

    st.session_state[selection_key] = selections

    matches = _filter_catalog(
        catalog,
        prior_selections,
    )

    return matches, prior_selections


def _default_output_range(
    bundle: PolicySurrogateBundle,
    sample: pd.DataFrame,
    output: str,
) -> tuple[float, float]:
    """Return a stable initial display range for one policy outcome."""

    if not sample.empty and output in sample:
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


def _output_axis_control(
    bundle: PolicySurrogateBundle,
    sample: pd.DataFrame,
    output: str,
    *,
    context: str,
) -> tuple[float, float]:
    """Persistent axis limits that do not change when other controls move."""

    default_range = _default_output_range(bundle, sample, output)

    range_key = f"{context}__fixed_output_range__{output}"
    min_widget_key = f"{context}__ymin__{output}"
    max_widget_key = f"{context}__ymax__{output}"

    if range_key not in st.session_state:
        st.session_state[range_key] = default_range

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

        if st.button(
            "Reset to data range",
            key=f"{context}__reset_y__{output}",
            use_container_width=True,
        ):
            st.session_state[range_key] = default_range
            st.session_state[min_widget_key] = float(default_range[0])
            st.session_state[max_widget_key] = float(default_range[1])
            st.rerun()

    return tuple(st.session_state[range_key])


def _continuous_plot_columns(
    bundle: PolicySurrogateBundle,
) -> list[str]:
    columns = list(bundle.feature_spec.state_columns) + [
        column
        for column in bundle.feature_spec.parameter_columns
        if not column.endswith("__missing")
    ]

    excluded = {
        "after_tax_income",
        "total_assets",
        "cash_on_hand",
        "years_to_retirement",
    }

    return [
        column
        for column in columns
        if column in bundle.feature_ranges and column not in excluded
    ]


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


def _base_controls(
    bundle: PolicySurrogateBundle,
    *,
    excluded_columns: set[str] | None = None,
    disabled_state_columns: set[str] | None = None,
) -> dict[str, Any]:
    excluded_columns = set(excluded_columns or ())
    disabled_state_columns = set(disabled_state_columns or ())

    base = dict(bundle.default_values)
    original_parameters = [
        c for c in bundle.feature_spec.parameter_columns if not c.endswith("__missing")
    ]

    st.sidebar.header("Structural and policy parameters")
    st.sidebar.caption(
        "Controls are restricted to values represented in the solved-model sample."
    )
    for col in original_parameters:
        info = bundle.feature_ranges[col]

        if col in excluded_columns:
            base[col] = bundle.default_values[col]
            st.sidebar.caption(f"{_pretty(col)} is controlled by the plot axis.")
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
        base[col] = st.sidebar.selectbox(
            _pretty(col),
            values,
            index=index,
            key=f"cat_param_{col}",
        )

    st.sidebar.header("State held fixed")

    derived_or_recomputed = {
        "after_tax_income",
        "total_assets",
        "cash_on_hand",
        "years_to_retirement",
    }

    for col in bundle.feature_spec.state_columns:
        if col in derived_or_recomputed:
            continue

        if col in excluded_columns:
            base[col] = bundle.default_values[col]
            st.sidebar.caption(f"{_pretty(col)} is controlled by the plot axis.")
            continue

        info = bundle.feature_ranges[col]
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

    st.sidebar.header("Discrete household state")
    for col in bundle.feature_spec.categorical_state_columns:
        if col == "is_retired":
            continue
        info = bundle.feature_ranges[col]
        values = list(info.get("values", []))
        if not values:
            continue
        default_str = str(base.get(col, info.get("mode", values[0])))
        index = values.index(default_str) if default_str in values else 0
        base[col] = st.sidebar.selectbox(
            _pretty(col), values, index=index, key=f"cat_state_{col}"
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


def _range_selector(
    bundle: PolicySurrogateBundle, col: str, key: str
) -> tuple[float, float]:
    info = bundle.feature_ranges[col]
    lo = float(info.get("q01", info["min"]))
    hi = float(info.get("q99", info["max"]))
    if hi <= lo:
        return lo, hi
    return st.slider(
        f"{_pretty(col)} range",
        min_value=float(info["min"]),
        max_value=float(info["max"]),
        value=(lo, hi),
        step=max((float(info["max"]) - float(info["min"])) / 250.0, 1.0e-12),
        key=key,
    )


def _link_labor_income_state_to_current_income(
    bundle: PolicySurrogateBundle,
    frame: pd.DataFrame,
    current_income_values: np.ndarray,
) -> pd.DataFrame:
    """Move labor-income state proportionally with current gross income.

    The proportionality ratio is anchored to the bundle's baseline state,
    rather than to the disabled sidebar slider.
    """

    out = frame.copy()

    if "labor_income_state" not in bundle.feature_spec.state_columns:
        return out

    current_info = bundle.feature_ranges["current_income"]
    labor_info = bundle.feature_ranges["labor_income_state"]

    anchor_current_income = float(
        bundle.default_values.get(
            "current_income",
            current_info.get("median", 0.0),
        )
    )

    anchor_labor_income = float(
        bundle.default_values.get(
            "labor_income_state",
            labor_info.get("median", 0.0),
        )
    )

    # Guard against a zero median caused by unemployed or retired observations.
    if not np.isfinite(anchor_current_income) or abs(anchor_current_income) <= 1.0e-12:
        anchor_current_income = float(
            current_info.get(
                "q99",
                current_info.get("max", 1.0),
            )
        )

    if not np.isfinite(anchor_labor_income) or abs(anchor_labor_income) <= 1.0e-12:
        anchor_labor_income = float(
            labor_info.get(
                "median",
                labor_info.get("q99", 1.0),
            )
        )

    if abs(anchor_current_income) <= 1.0e-12:
        ratio = 1.0
    else:
        ratio = anchor_labor_income / anchor_current_income

    out["labor_income_state"] = np.asarray(current_income_values, dtype=float) * ratio

    return out


def _line_slice(
    bundle: PolicySurrogateBundle,
    base: dict[str, Any],
    *,
    x: str,
    x_range: tuple[float, float],
    overlay: str | None,
    overlay_values: list[Any],
    outputs: list[str],
    linked_income: bool,
    y_range: tuple[float, float],
    n_points: int = 220,
) -> go.Figure:
    x_values = np.linspace(float(x_range[0]), float(x_range[1]), n_points)
    fig = go.Figure()
    curve_values = overlay_values if overlay is not None else [None]

    for curve_value in curve_values:
        frame = pd.DataFrame([base] * n_points)
        frame[x] = x_values
        label = "baseline"
        if overlay is not None:
            frame[overlay] = curve_value
            label = f"{_pretty(overlay)} = {curve_value}"
        if linked_income and x == "current_income":
            frame = _link_labor_income_state_to_current_income(
                bundle,
                frame,
                x_values,
            )
        frame = complete_derived_state(frame)
        pred = bundle.predict(frame)
        for output in outputs:
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=pred[output],
                    mode="lines",
                    name=(
                        _pretty(output)
                        if overlay is None
                        else f"{_pretty(output)}; {label}"
                    ),
                    legendgroup=label,
                )
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
    bundle: PolicySurrogateBundle,
    base: dict[str, Any],
    *,
    bundle_path: str | Path,
) -> None:
    continuous = _continuous_plot_columns(bundle)

    linked_columns = [
        col
        for col in ("labor_income_state",)
        if col in bundle.feature_spec.state_columns
    ]

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
        linked_state_locked = x == "current_income" and bool(
            st.session_state.get(
                "one_d_linked_income",
                bool(linked_columns),
            )
        )

        overlay_options: list[str | None] = [None] + [
            col
            for col in (continuous + list(bundle.feature_spec.categorical_columns))
            if (
                col != x
                and col != "is_retired"
                and not (linked_state_locked and col == "labor_income_state")
            )
        ]
        overlay = st.selectbox(
            "Overlay several values of",
            overlay_options,
            format_func=lambda x: "No overlay" if x is None else _pretty(x),
        )

    x_range = _range_selector(bundle, x, "one_d_x_range")
    linked_columns = [
        col
        for col in ("labor_income_state",)
        if col in bundle.feature_spec.state_columns
    ]

    linked_label = (
        "Move labor income state proportionally with current income"
        if linked_columns
        else "No labor-income state is included in this model"
    )

    linked_income_requested = st.checkbox(
        linked_label,
        disabled=(x != "current_income" or not linked_columns),
        key="one_d_linked_income",
    )

    linked_income = bool(
        linked_income_requested and x == "current_income" and linked_columns
    )
    overlay_values: list[Any] = []
    if overlay is not None:
        info = bundle.feature_ranges[overlay]
        if info["kind"] == "categorical":
            overlay_values = st.multiselect(
                f"{_pretty(overlay)} values",
                info["values"],
                default=info["values"],
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

    sample = _load_training_sample(str(bundle_path))
    y_range = _output_axis_control(
        bundle,
        sample,
        output,
        context="one_dimensional",
    )

    fig = _line_slice(
        bundle,
        base,
        x=x,
        x_range=x_range,
        overlay=overlay,
        overlay_values=overlay_values,
        outputs=[output],
        linked_income=linked_income,
        y_range=y_range,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Fixed values used in this slice"):
        display = pd.DataFrame(
            {"variable": list(base), "value": [base[k] for k in base]}
        )
        st.dataframe(display, hide_index=True, use_container_width=True)


def _surface_page(
    bundle: PolicySurrogateBundle, base: dict[str, Any], *, bundle_path: str | Path
) -> None:
    continuous = list(bundle.feature_spec.state_columns) + [
        c for c in bundle.feature_spec.parameter_columns if not c.endswith("__missing")
    ]
    continuous = [
        c
        for c in continuous
        if c in bundle.feature_ranges
        and c
        not in {
            "after_tax_income",
            "total_assets",
            "cash_on_hand",
            "years_to_retirement",
        }
    ]
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
    y_options = [c for c in continuous if c != x]
    with c2:
        y_default = (
            y_options.index("liquid_assets") if "liquid_assets" in y_options else 0
        )
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
    linked_columns = [
        col
        for col in ("labor_income_state",)
        if col in bundle.feature_spec.state_columns and col not in {x, y}
    ]
    linked_income = st.checkbox(
        "Move other included income-state variables proportionally when current income varies",
        value=bool(linked_columns),
        disabled=("current_income" not in {x, y} or not linked_columns),
        key="surface_linked_income",
    )
    x_values = np.linspace(xr[0], xr[1], 65)
    y_values = np.linspace(yr[0], yr[1], 65)
    surface = make_policy_surface(
        bundle,
        x=x,
        y=y,
        x_values=x_values,
        y_values=y_values,
        base=base,
        linked_income=linked_income,
    )
    table = surface.pivot(index="__y", columns="__x", values=f"pred_{output}")

    sample = _load_training_sample(str(bundle_path))
    z_range = _output_axis_control(
        bundle,
        sample,
        output,
        context="two_dimensional",
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
        margin=dict(l=40, r=20, t=30, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


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
    bundle: PolicySurrogateBundle,
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
    st.plotly_chart(fig, use_container_width=True)
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
        st.dataframe(subgroup, hide_index=True, use_container_width=True)
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
        st.dataframe(diagnostics, hide_index=True, use_container_width=True)


def _exact_grid_comparison_page(
    bundle: PolicySurrogateBundle,
    base: dict[str, Any],
    *,
    bundle_path: str | Path,
) -> None:
    st.subheader("Held-out grid versus polynomial surrogate")

    st.caption(
        "Each model below was excluded when the evaluation surrogate "
        "was fitted. The reported R² values are therefore out of sample."
    )

    try:
        catalog = _load_model_catalog(str(bundle_path))
        model_summary = _out_of_sample_model_summary(
            bundle,
            catalog,
        )
    except Exception as exc:
        st.error(str(exc))
        return

    display_summary = model_summary[
        [
            "model_id",
            "consumption_r2",
            "deposit_r2",
            "mean_r2",
        ]
    ].copy()

    display_summary = display_summary.rename(
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
        use_container_width=True,
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

        consumption_r2 = row["consumption_r2"]
        deposit_r2 = row["deposit_r2"]

        consumption_text = f"{consumption_r2:.4f}" if pd.notna(consumption_r2) else "NA"
        deposit_text = f"{deposit_r2:.4f}" if pd.notna(deposit_r2) else "NA"

        return (
            f"{model_id}  |  "
            f"Consumption R²: {consumption_text}  |  "
            f"Deposit R²: {deposit_text}"
        )

    selected_model_id = st.selectbox(
        "Select an out-of-sample model",
        model_ids,
        format_func=format_model_option,
        key="exact_selected_model_id",
    )

    selected_summary = summary_lookup.loc[str(selected_model_id)]

    catalog_for_selection = catalog.copy()
    catalog_for_selection["model_id"] = catalog_for_selection["model_id"].astype(str)

    selected_catalog_rows = catalog_for_selection[
        catalog_for_selection["model_id"] == str(selected_model_id)
    ]

    if selected_catalog_rows.empty:
        st.error(f"Model {selected_model_id} is missing from model_catalog.csv.")
        return

    selected_row = selected_catalog_rows.iloc[0]

    model_dir = selected_row.get(
        "model_dir",
        selected_summary.get("model_dir"),
    )

    if pd.isna(model_dir):
        st.error("The selected model has no model_dir entry.")
        return

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

    # ------------------------------------------------------------------
    # Display the selected model's parameters.
    # ------------------------------------------------------------------

    parameter_columns = [
        column
        for column in catalog.columns
        if column.startswith("param__") and not column.endswith("__missing")
    ]

    parameter_rows: list[dict[str, str]] = []

    for column in parameter_columns:
        value = selected_row.get(column)

        if pd.isna(value):
            continue

        parameter_rows.append(
            {
                "Parameter": _pretty(column),
                # Convert all values to strings to avoid Arrow mixed-type
                # serialization warnings.
                "Value": _button_label(value),
            }
        )

    parameter_table = pd.DataFrame(
        parameter_rows,
        columns=["Parameter", "Value"],
        dtype="string",
    )

    with st.expander(
        "Selected model parameters",
        expanded=True,
    ):
        st.dataframe(
            parameter_table,
            hide_index=True,
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # Choose the state slice and plotted policy.
    # ------------------------------------------------------------------

    c1, c2 = st.columns(2)

    exact_x_options = [
        column
        for column in [
            "age",
            "current_income",
            "labor_income_state",
            "liquid_assets",
            "illiquid_assets",
        ]
        if column in bundle.feature_ranges
    ]

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
            POLICY_OUTPUTS,
            format_func=_pretty,
            key="exact_comparison_output",
        )

    x_range = _range_selector(
        bundle,
        x,
        "exact_comparison_x_range",
    )

    sample = _load_training_sample(str(bundle_path))

    y_range = _output_axis_control(
        bundle,
        sample,
        output,
        context="exact_comparison",
    )

    # Start from the sidebar household state and replace every model
    # parameter with the values from the selected held-out model.
    model_base = dict(base)

    for column in bundle.feature_spec.parameter_columns:
        if column in selected_row.index and pd.notna(selected_row[column]):
            model_base[column] = float(selected_row[column])

    for column in bundle.feature_spec.categorical_parameter_columns:
        if column in selected_row.index and pd.notna(selected_row[column]):
            model_base[column] = str(selected_row[column])

    policy_layout = str(
        bundle.training_metadata.get(
            "policy_layout",
            "GHKEBA",
        )
    )

    money_units = str(
        bundle.training_metadata.get(
            "money_units",
            "model",
        )
    )

    try:
        exact = load_exact_grid_slice(
            model_dir,
            base=model_base,
            x=x,
            x_range=x_range,
            policy_layout=policy_layout,
            money_units=money_units,
        )
    except Exception as exc:
        st.error(f"Could not load the exact policy grid: {exc}")
        return

    if exact.empty:
        st.warning("No exact grid points fall inside the selected x-axis range.")
        return

    # Evaluate the polynomial at precisely the same state points as the
    # grid-based policy.
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

    surrogate = bundle.predict(surrogate_input)

    comparison = pd.DataFrame(
        {
            x: exact["__x"].to_numpy(),
            "exact_grid": exact[output].to_numpy(),
            "polynomial": surrogate[output].to_numpy(),
        }
    )

    comparison["error"] = comparison["polynomial"] - comparison["exact_grid"]

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
            y=comparison["polynomial"],
            mode="lines",
            name="Polynomial surrogate",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=620,
        xaxis_title=_pretty(x),
        yaxis_title=_pretty(output),
        yaxis=dict(
            range=[
                float(y_range[0]),
                float(y_range[1]),
            ],
            autorange=False,
        ),
        hovermode="x unified",
        legend_title="Policy source",
        margin=dict(
            l=40,
            r=20,
            t=40,
            b=40,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    rmse = float(np.sqrt(np.mean(comparison["error"] ** 2)))

    max_error = float(comparison["error"].abs().max())

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Points compared",
        f"{len(comparison):,}",
    )
    m2.metric(
        "Slice RMSE",
        f"{rmse:.6g}",
    )
    m3.metric(
        "Maximum absolute error",
        f"{max_error:.6g}",
    )

    with st.expander("Comparison data"):
        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )


def _validation_page(bundle: PolicySurrogateBundle) -> None:
    st.subheader("Validation on entirely held-out parameterizations")
    st.caption(
        "Consumption and deposit are fitted directly. Both asset levels and both "
        "asset changes are reconstructed from the accounting equations. Asset "
        "rows include persistence or zero-change benchmarks."
    )
    st.dataframe(bundle.validation_metrics, hide_index=True, use_container_width=True)
    metrics = bundle.validation_metrics
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=metrics["target"].map(_pretty),
            y=metrics["r2"],
            name="Surrogate R-squared",
        )
    )
    if "benchmark_r2" in metrics:
        benchmark = metrics[metrics["benchmark_r2"].notna()]
        if not benchmark.empty:
            fig.add_trace(
                go.Bar(
                    x=benchmark["target"].map(_pretty),
                    y=benchmark["benchmark_r2"],
                    name="Persistence / zero-change benchmark",
                )
            )
    fig.add_hline(y=0.0)
    fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis_title="R-squared",
        xaxis_title="Policy choice",
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True)

    if "skill_vs_benchmark" in metrics:
        skill = metrics[metrics["skill_vs_benchmark"].notna()]
        if not skill.empty:
            fig_skill = go.Figure(
                go.Bar(
                    x=skill["target"].map(_pretty),
                    y=skill["skill_vs_benchmark"],
                    name="Skill",
                )
            )
            fig_skill.add_hline(y=0.0)
            fig_skill.update_layout(
                template="plotly_white",
                height=390,
                yaxis_title="1 - MSE / benchmark MSE",
                xaxis_title="Policy choice",
                title="Improvement over persistence or zero change",
            )
            st.plotly_chart(fig_skill, use_container_width=True)

    if not bundle.validation_by_model.empty:
        st.subheader("Error distribution across held-out models")
        chosen = st.selectbox(
            "Policy choice",
            sorted(bundle.validation_by_model["target"].unique()),
            format_func=_pretty,
            key="validation_target",
        )
        d = bundle.validation_by_model[
            bundle.validation_by_model["target"] == chosen
        ].sort_values("rmse")
        fig2 = go.Figure(go.Box(y=d["rmse"], boxpoints="all", name=_pretty(chosen)))
        fig2.update_layout(template="plotly_white", height=430, yaxis_title="RMSE")
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(d, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="HA Policy Surrogate", layout="wide")
    args = _parse_args()
    st.title("Smooth HA policy-function explorer")
    st.caption(
        "The page evaluates a fitted surrogate; it does not re-solve the HA model. "
        "Sliders stay within marginal training ranges, but unusual combinations of "
        "parameters can still lie outside the joint support of solved models."
    )

    bundle_path = st.sidebar.text_input("Surrogate bundle", value=str(args.bundle))
    try:
        bundle = _load_bundle(str(Path(bundle_path).expanduser()))
    except Exception as exc:
        st.error(f"Could not load surrogate bundle: {exc}")
        st.stop()
    continuous_columns = _continuous_plot_columns(bundle)

    if not continuous_columns:
        st.error("The surrogate has no continuous plotting variables.")
        return

    default_x = (
        "current_income"
        if "current_income" in continuous_columns
        else continuous_columns[0]
    )

    current_x = st.session_state.get(
        "one_d_x",
        default_x,
    )

    if current_x not in continuous_columns:
        st.session_state.pop("one_d_x", None)
        current_x = default_x

    # This must be defined before it is used below.
    has_labor_income_state = "labor_income_state" in bundle.feature_spec.state_columns

    linked_income_requested = bool(
        st.session_state.get(
            "one_d_linked_income",
            has_labor_income_state,
        )
    )

    linked_income_active = (
        has_labor_income_state
        and current_x == "current_income"
        and linked_income_requested
    )

    disabled_state_columns = {"labor_income_state"} if linked_income_active else set()

    base = _base_controls(
        bundle,
        excluded_columns={current_x},
        disabled_state_columns=disabled_state_columns,
    )
    base_frame = bundle.prepare_inputs(pd.DataFrame([base]))
    base.update(base_frame.iloc[0].to_dict())

    tabs = st.tabs(
        [
            "One-dimensional slices",
            "Two-dimensional surfaces",
            "MPC distribution",
            "Grid versus polynomial",
            "Validation",
        ]
    )

    with tabs[0]:
        _one_dimensional_page(
            bundle,
            base,
            bundle_path=bundle_path,
        )

    with tabs[1]:
        _surface_page(
            bundle,
            base,
            bundle_path=bundle_path,
        )

    with tabs[2]:
        _mpc_distribution_page(
            bundle,
            base,
            bundle_path=bundle_path,
        )

    with tabs[3]:
        _exact_grid_comparison_page(
            bundle,
            base,
            bundle_path=bundle_path,
        )

    with tabs[4]:
        _validation_page(bundle)


if __name__ == "__main__":
    main()
