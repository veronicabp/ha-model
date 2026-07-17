"""Static and HTML visualization helpers for HA policy surrogates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import json

from ha_policy_surrogate import PolicySurrogateBundle


def link_labor_income_state(
    bundle: PolicySurrogateBundle,
    frame: pd.DataFrame,
    varied_income: np.ndarray,
    base: Mapping[str, Any],
) -> pd.DataFrame:
    """Move the latent labor-income state proportionally with current income.

    The proportionality anchor deliberately does not use the sidebar value of
    current_income. That value is irrelevant when current_income itself is the
    plotting axis.
    """

    out = frame.copy()

    if "labor_income_state" not in bundle.feature_spec.state_columns:
        return out

    anchor_income = float(
        bundle.default_values.get(
            "current_income",
            bundle.feature_ranges["current_income"].get("median", 1.0),
        )
    )

    if not np.isfinite(anchor_income) or abs(anchor_income) <= 1.0e-12:
        anchor_income = float(
            bundle.feature_ranges["current_income"].get("median", 1.0)
        )

    if not np.isfinite(anchor_income) or abs(anchor_income) <= 1.0e-12:
        return out

    anchor_labor_state = float(
        base.get(
            "labor_income_state",
            bundle.default_values.get("labor_income_state", anchor_income),
        )
    )

    ratio = anchor_labor_state / anchor_income
    out["labor_income_state"] = np.asarray(varied_income, dtype=float) * ratio

    return out


def _canonical_policy_array(
    array: np.ndarray,
    layout: str,
) -> np.ndarray:
    """Return a policy tensor in canonical GHKEBA order."""

    layout = layout.upper()
    arr = np.asarray(array)

    if len(layout) == 5:
        arr = arr[None, ...]
        layout = "G" + layout

    if len(layout) != 6 or set(layout) != set("GHKEBA"):
        raise ValueError(f"Unsupported policy layout {layout!r}.")

    permutation = [layout.index(axis) for axis in "GHKEBA"]
    return np.transpose(arr, permutation)


def _nearest_grid_index(
    grid: np.ndarray,
    value: float,
) -> int:
    grid = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(grid - float(value))))


def _exact_group_index(
    group_values: np.ndarray,
    value: Any,
) -> int:
    text = np.asarray(group_values).astype(str)
    matches = np.flatnonzero(text == str(value))

    if len(matches):
        return int(matches[0])

    try:
        return _nearest_grid_index(
            np.asarray(group_values, dtype=float),
            float(value),
        )
    except (TypeError, ValueError):
        return 0


def load_exact_grid_slice(
    model_dir: str | Path,
    *,
    base: Mapping[str, Any],
    x: str,
    x_range: tuple[float, float] | None = None,
    policy_layout: str = "GHKEBA",
    money_units: str = "model",
) -> pd.DataFrame:
    """Extract an exact state slice from one solved HA policy grid."""

    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text())

    with np.load(
        model_dir / "arrays.npz",
        allow_pickle=False,
    ) as npz:
        arrays = {name: npz[name] for name in npz.files}

    consumption = _canonical_policy_array(
        arrays["consumption"],
        policy_layout,
    )
    deposit = _canonical_policy_array(
        arrays["deposit"],
        policy_layout,
    )

    if deposit.shape != consumption.shape:
        raise ValueError("Consumption and deposit arrays have different shapes.")

    G, H, K, E, B, A = consumption.shape

    group_values = arrays.get(
        "group_values",
        np.arange(G),
    )
    ages = np.asarray(
        arrays.get("ages", np.arange(H)),
        dtype=float,
    )

    liquid_grid_model = np.asarray(
        arrays["liquid_grid"],
        dtype=float,
    )
    illiquid_grid_model = np.asarray(
        arrays["illiquid_grid"],
        dtype=float,
    )

    log_income_grid = np.asarray(
        arrays["log_income_grid"],
        dtype=float,
    )
    if log_income_grid.ndim == 1:
        log_income_grid = np.repeat(
            log_income_grid[None, :],
            G,
            axis=0,
        )

    gross_income_model = np.asarray(
        arrays["gross_income"],
        dtype=float,
    )
    after_tax_income_model = np.asarray(
        arrays["after_tax_income"],
        dtype=float,
    )

    money_scale = float(metadata.get("money_scale", 1.0) or 1.0)
    factor = 1.0 if money_units == "model" else money_scale

    liquid_grid = liquid_grid_model * factor
    illiquid_grid = illiquid_grid_model * factor
    labor_income_grid = np.exp(log_income_grid) * factor

    g0 = _exact_group_index(
        group_values,
        base.get("education", group_values[0]),
    )
    h0 = _nearest_grid_index(
        ages,
        float(base.get("age", np.median(ages))),
    )
    e0 = int(
        np.clip(
            int(float(base.get("employment_state", 0))),
            0,
            E - 1,
        )
    )
    b0 = _nearest_grid_index(
        liquid_grid,
        float(base.get("liquid_assets", 0.0)),
    )
    a0 = _nearest_grid_index(
        illiquid_grid,
        float(base.get("illiquid_assets", 0.0)),
    )

    if "labor_income_state" in base:
        k0 = _nearest_grid_index(
            labor_income_grid[g0],
            float(base["labor_income_state"]),
        )
    else:
        gross_grid = gross_income_model[g0, h0, :, e0] * factor
        k0 = _nearest_grid_index(
            gross_grid,
            float(base.get("current_income", 0.0)),
        )

    if x == "age":
        h = np.arange(H)
        k = np.full(H, k0)
        b = np.full(H, b0)
        a = np.full(H, a0)

    elif x in {"current_income", "labor_income_state"}:
        k = np.arange(K)
        h = np.full(K, h0)
        b = np.full(K, b0)
        a = np.full(K, a0)

    elif x == "liquid_assets":
        b = np.arange(B)
        h = np.full(B, h0)
        k = np.full(B, k0)
        a = np.full(B, a0)

    elif x == "illiquid_assets":
        a = np.arange(A)
        h = np.full(A, h0)
        k = np.full(A, k0)
        b = np.full(A, b0)

    else:
        raise ValueError(
            "Exact-grid comparisons support only age, "
            "current_income, labor_income_state, "
            "liquid_assets, and illiquid_assets."
        )

    n = len(h)
    g = np.full(n, g0)
    e = np.full(n, e0)

    exact_consumption = consumption[g, h, k, e, b, a].astype(float) * factor
    exact_deposit = deposit[g, h, k, e, b, a].astype(float) * factor

    current_b = liquid_grid[b]
    current_a = illiquid_grid[a]

    gross_income = gross_income_model[g, h, k, e].astype(float) * factor
    after_tax_income = after_tax_income_model[g, h, k, e].astype(float) * factor
    labor_income_state = labor_income_grid[g, k]

    config = metadata.get("config", {}) or {}
    derived = metadata.get("metadata", {}) or {}

    dt = float(config.get("ct_time_step", 1.0))

    rb_pos = float(
        derived.get(
            "rb_pos_ct",
            np.log1p(float(config.get("liquid_interest_rate", 0.0))),
        )
    )
    rb_neg = float(
        derived.get(
            "rb_neg_ct",
            np.log1p(float(config.get("borrowing_interest_rate", 0.0))),
        )
    )
    ra = float(
        derived.get(
            "ra_ct",
            np.log1p(float(config.get("illiquid_interest_rate", 0.0))),
        )
    )

    chi0 = float(
        derived.get(
            "chi0",
            config.get("ct_linear_adjustment_cost", 0.0),
        )
    )
    chi1 = float(
        derived.get(
            "chi1",
            config.get("ct_convex_adjustment_cost", 0.0),
        )
    )
    xi = float(
        derived.get(
            "xi",
            config.get(
                "ct_automatic_illiquid_income_share",
                0.0,
            ),
        )
    )

    a_floor = float(config.get("ct_illiquid_cost_floor", 1.0e-6)) * factor

    adjustment_cost = chi0 * np.abs(
        exact_deposit
    ) + 0.5 * chi1 * exact_deposit**2 / np.maximum(current_a, a_floor)

    liquid_return = np.where(
        current_b >= 0.0,
        rb_pos * current_b,
        rb_neg * current_b,
    )

    liquid_drift = (
        (1.0 - xi) * after_tax_income
        + liquid_return
        - exact_deposit
        - adjustment_cost
        - exact_consumption
    )
    illiquid_drift = ra * current_a + xi * after_tax_income + exact_deposit

    next_b = np.clip(
        current_b + dt * liquid_drift,
        liquid_grid[0],
        liquid_grid[-1],
    )
    next_a = np.clip(
        current_a + dt * illiquid_drift,
        illiquid_grid[0],
        illiquid_grid[-1],
    )

    retirement_age = float(config.get("retirement_age", np.inf))

    frame = pd.DataFrame(
        {
            "education": np.asarray(group_values)[g],
            "age": ages[h],
            "current_income": gross_income,
            "after_tax_income": after_tax_income,
            "labor_income_state": labor_income_state,
            "employment_state": e,
            "is_retired": (ages[h] >= retirement_age).astype(int),
            "liquid_assets": current_b,
            "illiquid_assets": current_a,
            "liquid_grid_min": liquid_grid[0],
            "liquid_grid_max": liquid_grid[-1],
            "illiquid_grid_min": illiquid_grid[0],
            "illiquid_grid_max": illiquid_grid[-1],
            "money_scale": money_scale,
            "consumption": exact_consumption,
            "deposit": exact_deposit,
            "delta_liquid_assets": next_b - current_b,
            "delta_illiquid_assets": next_a - current_a,
            "next_liquid_assets": next_b,
            "next_illiquid_assets": next_a,
        }
    )

    frame["__x"] = frame[x]

    if x_range is not None:
        lo, hi = map(float, x_range)
        frame = frame[frame["__x"].between(lo, hi)]

    return frame.sort_values("__x").reset_index(drop=True)


def complete_derived_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute deterministic state features after changing a primitive state."""

    out = frame.copy()
    if {"liquid_assets", "illiquid_assets"}.issubset(out.columns):
        out["total_assets"] = pd.to_numeric(
            out["liquid_assets"], errors="coerce"
        ) + pd.to_numeric(out["illiquid_assets"], errors="coerce")
    if {"liquid_assets", "after_tax_income"}.issubset(out.columns):
        out["cash_on_hand"] = pd.to_numeric(
            out["liquid_assets"], errors="coerce"
        ) + pd.to_numeric(out["after_tax_income"], errors="coerce")
    return out


def make_policy_slice(
    bundle: PolicySurrogateBundle,
    *,
    x: str,
    values: Sequence[float] | None = None,
    base: Mapping[str, Any] | pd.Series | None = None,
    n_points: int = 200,
    quantile_range: tuple[str, str] = ("q01", "q99"),
    linked_income: bool = False,
) -> pd.DataFrame:
    """Evaluate a one-dimensional policy slice.

    When ``linked_income`` is true and ``x='current_income'``, the latent labor-
    income state moves proportionally with current gross income.  After-tax
    income is always recomputed by the bundle from the selected tax policy.
    """

    if x not in bundle.feature_ranges:
        raise KeyError(f"Unknown feature {x!r}.")
    info = bundle.feature_ranges[x]
    if info.get("kind") != "continuous":
        raise ValueError(f"Slice variable {x!r} is categorical.")
    base_row = dict(bundle.default_values if base is None else dict(base))

    if values is None:
        low = float(info.get(quantile_range[0], info["min"]))
        high = float(info.get(quantile_range[1], info["max"]))
        values_array = np.linspace(low, high, int(n_points))
    else:
        values_array = np.asarray(values, dtype=float)

    frame = pd.DataFrame([base_row] * len(values_array))
    frame[x] = values_array

    if linked_income and x == "current_income":
        frame = link_labor_income_state(
            bundle,
            frame,
            values_array,
            base_row,
        )
    frame = complete_derived_state(frame)
    pred = bundle.predict(frame)
    return pd.concat(
        [frame.reset_index(drop=True), pred.add_prefix("pred_").reset_index(drop=True)],
        axis=1,
    )


def make_policy_surface(
    bundle: PolicySurrogateBundle,
    *,
    x: str,
    y: str,
    x_values: Sequence[float] | None = None,
    y_values: Sequence[float] | None = None,
    base: Mapping[str, Any] | pd.Series | None = None,
    n_x: int = 60,
    n_y: int = 60,
    linked_income: bool = False,
) -> pd.DataFrame:
    """Evaluate a rectangular two-dimensional policy surface.

    When ``linked_income`` is true, varying ``current_income`` also moves the
    latent labor-income state in the same proportion as in the base row.  The
    bundle separately recomputes taxes and after-tax income.
    """

    if x == y:
        raise ValueError("x and y must differ.")
    base_row = dict(bundle.default_values if base is None else dict(base))
    for col in (x, y):
        if col not in bundle.feature_ranges:
            raise KeyError(f"Unknown feature {col!r}.")
        if bundle.feature_ranges[col].get("kind") != "continuous":
            raise ValueError(f"Surface variable {col!r} is categorical.")

    def grid(col: str, supplied: Sequence[float] | None, n: int) -> np.ndarray:
        if supplied is not None:
            return np.asarray(supplied, dtype=float)
        info = bundle.feature_ranges[col]
        return np.linspace(
            float(info.get("q01", info["min"])),
            float(info.get("q99", info["max"])),
            int(n),
        )

    xv = grid(x, x_values, n_x)
    yv = grid(y, y_values, n_y)
    xx, yy = np.meshgrid(xv, yv)
    frame = pd.DataFrame([base_row] * xx.size)
    frame[x] = xx.ravel()
    frame[y] = yy.ravel()
    if linked_income and "current_income" in {x, y}:
        frame = link_labor_income_state(
            bundle,
            frame,
            frame["current_income"].to_numpy(dtype=float),
            base_row,
        )
    frame = complete_derived_state(frame)
    pred = bundle.predict(frame)
    out = pd.concat(
        [frame.reset_index(drop=True), pred.add_prefix("pred_").reset_index(drop=True)],
        axis=1,
    )
    out["__x"] = xx.ravel()
    out["__y"] = yy.ravel()
    return out


def plot_policy_slice(
    slice_data: pd.DataFrame,
    *,
    x: str,
    outputs: Sequence[str] = (
        "consumption",
        "deposit",
        "delta_liquid_assets",
        "delta_illiquid_assets",
    ),
    title: str | None = None,
    output_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for output in outputs:
        col = output if output.startswith("pred_") else f"pred_{output}"
        if col not in slice_data:
            raise KeyError(f"Slice data do not contain {col!r}.")
        ax.plot(slice_data[x], slice_data[col], label=output.removeprefix("pred_"))
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel("Policy choice")
    ax.set_title(title or f"Policy functions as a function of {x.replace('_', ' ')}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200)
    return fig, ax


def plot_policy_surface(
    surface_data: pd.DataFrame,
    *,
    x: str,
    y: str,
    output: str = "consumption",
    title: str | None = None,
    output_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    value_col = output if output.startswith("pred_") else f"pred_{output}"
    table = surface_data.pivot(index="__y", columns="__x", values=value_col)
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    contour = ax.contourf(
        table.columns.to_numpy(dtype=float),
        table.index.to_numpy(dtype=float),
        table.to_numpy(dtype=float),
        levels=30,
    )
    fig.colorbar(contour, ax=ax, label=output.removeprefix("pred_").replace("_", " "))
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(y.replace("_", " ").title())
    ax.set_title(title or f"{output.replace('_', ' ').title()} policy surface")
    fig.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200)
    return fig, ax


def plot_validation_metrics(
    bundle: PolicySurrogateBundle,
    *,
    output_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    metrics = bundle.validation_metrics.copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(metrics["target"], metrics["r2"])
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("R-squared on held-out parameterizations")
    ax.set_xlabel("")
    ax.set_title("Surrogate validation by policy choice")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=200)
    return fig, ax


def export_slice_html(
    slice_data: pd.DataFrame,
    *,
    x: str,
    outputs: Sequence[str] = (
        "consumption",
        "deposit",
        "delta_liquid_assets",
        "delta_illiquid_assets",
    ),
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Export a self-contained interactive Plotly snapshot of one slice."""

    import plotly.graph_objects as go

    fig = go.Figure()
    for output in outputs:
        col = output if output.startswith("pred_") else f"pred_{output}"
        fig.add_trace(
            go.Scatter(
                x=slice_data[x],
                y=slice_data[col],
                mode="lines",
                name=output.removeprefix("pred_").replace("_", " "),
            )
        )
    fig.update_layout(
        title=title or f"Policy functions versus {x.replace('_', ' ')}",
        xaxis_title=x.replace("_", " "),
        yaxis_title="Policy choice",
        template="plotly_white",
    )
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(p, include_plotlyjs=True, full_html=True)
    return p
