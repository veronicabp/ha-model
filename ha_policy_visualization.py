"""Static and HTML visualization helpers for HA policy surrogates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import json

from ha_policy_surrogate import PolicySurrogateBundle

DERIVED_ASSET_AXES = {
    "liquid_slack_to_income",
    "illiquid_assets_to_income",
}


def set_plot_axis(
    frame: pd.DataFrame,
    variable: str,
    values: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Set a plot axis, translating derived asset ratios into primitive states.

    liquid_slack_to_income = (liquid_assets - borrowing_limit) / permanent_income
    illiquid_assets_to_income = illiquid_assets / permanent_income
    """

    out = frame.copy()
    values_array = np.asarray(values, dtype=float)

    if values_array.ndim == 0:
        values_array = np.full(len(out), float(values_array))

    if len(values_array) != len(out):
        raise ValueError(
            f"Axis {variable!r} has {len(values_array)} values "
            f"for a frame with {len(out)} rows."
        )

    if variable not in DERIVED_ASSET_AXES:
        out[variable] = values_array
        return out

    if "labor_income_state" not in out:
        raise ValueError(
            f"{variable!r} requires labor_income_state to recover asset levels."
        )

    permanent_income = pd.to_numeric(
        out["labor_income_state"],
        errors="coerce",
    ).to_numpy(dtype=float)

    finite_income = np.abs(permanent_income[np.isfinite(permanent_income)])
    typical_income = float(np.median(finite_income)) if len(finite_income) else 1.0
    income_floor = max(1.0e-6, 1.0e-3 * typical_income)

    income_scale = np.maximum(
        np.abs(permanent_income),
        income_floor,
    )

    if variable == "liquid_slack_to_income":
        if "param__config__borrowing_limit" in out:
            borrowing_limit = pd.to_numeric(
                out["param__config__borrowing_limit"],
                errors="coerce",
            ).to_numpy(dtype=float)
        elif "liquid_grid_min" in out:
            borrowing_limit = pd.to_numeric(
                out["liquid_grid_min"],
                errors="coerce",
            ).to_numpy(dtype=float)
        else:
            borrowing_limit = np.zeros(len(out), dtype=float)

        borrowing_limit = np.where(
            np.isfinite(borrowing_limit),
            borrowing_limit,
            0.0,
        )

        out["liquid_assets"] = borrowing_limit + values_array * income_scale
        out["liquid_slack_to_income"] = values_array

    elif variable == "illiquid_assets_to_income":
        out["illiquid_assets"] = values_array * income_scale
        out["illiquid_assets_to_income"] = values_array

    elif variable == "cash_on_hand":
        out["cash_on_hand"] = values_array
        out["liquid_assets"] = values_array - pd.to_numeric(
            out.get("after_tax_income", 0.0),
            errors="coerce",
        ).to_numpy(dtype=float)

    return out


def _numeric_income_setting(
    bundle: PolicySurrogateBundle,
    frame: pd.DataFrame,
    base: Mapping[str, Any],
    candidates: Sequence[str],
    *,
    default: float,
) -> np.ndarray:
    """Resolve a row-specific income-process parameter with stable fallbacks."""

    n = len(frame)
    for column in candidates:
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            fallback = float(base.get(column, bundle.default_values.get(column, default)))
            return np.where(np.isfinite(values), values, fallback)

        for source in (
            base,
            bundle.default_values,
            bundle.accounting_defaults,
        ):
            if column not in source:
                continue
            try:
                value = float(source[column])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return np.full(n, value, dtype=float)

        info = bundle.feature_ranges.get(column, {})
        try:
            value = float(info.get("median", np.nan))
        except (TypeError, ValueError):
            value = np.nan
        if np.isfinite(value):
            return np.full(n, value, dtype=float)

    return np.full(n, float(default), dtype=float)


def _employment_is_employed(
    frame: pd.DataFrame,
    base: Mapping[str, Any],
    bundle: PolicySurrogateBundle,
) -> np.ndarray:
    """Return the model's employment indicator: state 0 is employed."""

    if "employment_state" in frame:
        raw = frame["employment_state"]
    else:
        raw = pd.Series(
            base.get(
                "employment_state",
                bundle.default_values.get("employment_state", 0),
            ),
            index=frame.index,
        )

    numeric = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=float)
    text = raw.astype(str).str.strip().str.lower().to_numpy(dtype=object)
    employed_text = np.isin(
        text,
        ["0", "0.0", "employed", "working", "true"],
    )
    return np.where(np.isfinite(numeric), numeric == 0.0, employed_text)


def link_labor_income_state(
    bundle: PolicySurrogateBundle,
    frame: pd.DataFrame,
    varied_income: np.ndarray,
    base: Mapping[str, Any],
) -> pd.DataFrame:
    """Infer persistent income from pre-tax income, age, and employment.

    In the solved model, working-age gross income is

        current_income
        = labor_income_state * age_profile(age)
          * employment_multiplier,

    where the employment multiplier equals one for employed households and the
    unemployment replacement rate for unemployed households.  Consequently,
    the latent persistent-income state is pinned down exactly once pre-tax
    income, age, employment, and the selected income-process parameters are
    known.  The website therefore does not need a separate income-deviation
    control.

    The age profile matches ``ha_model_grid.age_profile``:

        exp(slope * ((age - 45) / 20)
            + quadratic * ((age - 45) / 20)**2).

    The default quadratic and replacement rates below are the structural-model
    defaults and are used only when a constant parameter was omitted from the
    fitted surrogate's varying-parameter metadata.
    """

    out = frame.copy()
    current = np.asarray(varied_income, dtype=float)
    if current.ndim == 0:
        current = np.full(len(out), float(current), dtype=float)
    if len(current) != len(out):
        raise ValueError(
            "Pre-tax income has the wrong number of rows for income-state linking."
        )
    if np.any(~np.isfinite(current)) or np.any(current < 0.0):
        raise ValueError("Pre-tax income must be finite and nonnegative.")

    out["current_income"] = current
    out["gross_income"] = current
    # Remove the old proportional disposable-income override.  The bundle's
    # accounting code should calculate after-tax income from the selected tax
    # schedule and the structurally consistent gross-income state.
    out = out.drop(columns=["__after_tax_income_override"], errors="ignore")

    age = pd.to_numeric(
        out.get(
            "age",
            pd.Series(
                base.get("age", bundle.default_values.get("age", 45.0)),
                index=out.index,
            ),
        ),
        errors="coerce",
    ).to_numpy(dtype=float)
    if np.any(~np.isfinite(age)):
        raise ValueError("Age must be finite to infer the persistent-income state.")

    slope = _numeric_income_setting(
        bundle,
        out,
        base,
        (
            "param__policy__income__age_profile_slope",
            "param__config__age_profile_slope",
            "age_profile_slope",
        ),
        default=0.25,
    )
    quadratic = _numeric_income_setting(
        bundle,
        out,
        base,
        (
            "param__policy__income__age_profile_quadratic",
            "param__config__age_profile_quadratic",
            "age_profile_quadratic",
        ),
        default=-0.10,
    )
    unemployment_replacement = _numeric_income_setting(
        bundle,
        out,
        base,
        (
            "param__policy__employment__unemployment_replacement_rate",
            "param__config__unemployment_replacement_rate",
            "unemployment_replacement_rate",
        ),
        default=0.50,
    )
    pension_replacement = _numeric_income_setting(
        bundle,
        out,
        base,
        (
            "param__policy__pension__replacement_rate",
            "param__config__pension_replacement_rate",
            "pension_replacement_rate",
        ),
        default=0.45,
    )
    retirement_age = _numeric_income_setting(
        bundle,
        out,
        base,
        (
            "param__config__retirement_age",
            "acct__retirement_age",
            "retirement_age",
        ),
        default=65.0,
    )

    age_position = (age - 45.0) / 20.0
    age_multiplier = np.exp(
        slope * age_position + quadratic * age_position**2
    )

    employed = _employment_is_employed(out, base, bundle)
    working_multiplier = age_multiplier * np.where(
        employed,
        1.0,
        unemployment_replacement,
    )

    # The website is normally restricted to working-age households.  This
    # branch keeps the helper correct if a retired row is supplied elsewhere.
    retirement_position = (retirement_age - 45.0) / 20.0
    retirement_profile = np.exp(
        slope * retirement_position + quadratic * retirement_position**2
    )
    retired_multiplier = pension_replacement * retirement_profile
    multiplier = np.where(age >= retirement_age, retired_multiplier, working_multiplier)

    if np.any(~np.isfinite(multiplier)) or np.any(multiplier <= 1.0e-12):
        raise ValueError(
            "The selected age/employment parameters imply a zero or invalid "
            "gross-income multiplier, so persistent income cannot be inferred."
        )

    if "labor_income_state" in bundle.feature_spec.state_columns:
        out["labor_income_state"] = current / multiplier

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

    elif x in {"liquid_assets", "liquid_slack_to_income"}:
        b = np.arange(B)
        h = np.full(B, h0)
        k = np.full(B, k0)
        a = np.full(B, a0)

    elif x in {"illiquid_assets", "illiquid_assets_to_income"}:
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

    income_scale = np.maximum(
        np.abs(frame["labor_income_state"].to_numpy(dtype=float)),
        1.0e-8,
    )

    frame["liquid_slack_to_income"] = (
        frame["liquid_assets"].to_numpy(dtype=float) - float(liquid_grid[0])
    ) / income_scale

    frame["illiquid_assets_to_income"] = (
        frame["illiquid_assets"].to_numpy(dtype=float) / income_scale
    )

    frame["__x"] = frame[x]

    if x_range is not None:
        lo, hi = map(float, x_range)
        frame = frame[frame["__x"].between(lo, hi)]

    return frame.sort_values("__x").reset_index(drop=True)


def complete_derived_state(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute deterministic state features after changing primitive states."""

    out = frame.copy()

    liquid = None
    illiquid = None

    if "liquid_assets" in out:
        liquid = pd.to_numeric(
            out["liquid_assets"],
            errors="coerce",
        )

    if "illiquid_assets" in out:
        illiquid = pd.to_numeric(
            out["illiquid_assets"],
            errors="coerce",
        )

    if liquid is not None and illiquid is not None:
        out["total_assets"] = liquid + illiquid

    if liquid is not None and "after_tax_income" in out:
        after_tax = pd.to_numeric(
            out["after_tax_income"],
            errors="coerce",
        )
        out["cash_on_hand"] = liquid + after_tax

    if "labor_income_state" in out:
        permanent_income = pd.to_numeric(
            out["labor_income_state"],
            errors="coerce",
        )

        finite_income = permanent_income.abs()
        finite_income = finite_income[np.isfinite(finite_income)]

        typical_income = float(finite_income.median()) if len(finite_income) else 1.0
        income_floor = max(
            1.0e-6,
            1.0e-3 * typical_income,
        )
        income_scale = permanent_income.abs().clip(
            lower=income_floor,
        )

        if liquid is not None:
            if "param__config__borrowing_limit" in out:
                borrowing_limit = pd.to_numeric(
                    out["param__config__borrowing_limit"],
                    errors="coerce",
                )
            elif "liquid_grid_min" in out:
                borrowing_limit = pd.to_numeric(
                    out["liquid_grid_min"],
                    errors="coerce",
                )
            else:
                borrowing_limit = pd.Series(
                    0.0,
                    index=out.index,
                )

            borrowing_limit = borrowing_limit.fillna(0.0)
            liquid_slack = liquid - borrowing_limit

            out["liquid_slack"] = liquid_slack
            out["liquid_assets_to_income"] = liquid / income_scale
            out["liquid_slack_to_income"] = liquid_slack / income_scale

        if illiquid is not None:
            out["illiquid_assets_to_income"] = illiquid / income_scale

    return out


def make_policy_slice(
    bundle: PolicySurrogateBundle,
    *,
    x: str,
    values: Sequence[float] | None = None,
    base: Mapping[str, Any] | pd.Series | None = None,
    n_points: int = 200,
    quantile_range: tuple[str, str] = ("q01", "q99"),
    linked_income: bool = True,
) -> pd.DataFrame:
    """Evaluate a one-dimensional policy slice.

    Pre-tax income is the public income control.  Unless persistent income is
    itself the requested axis, the latent labor-income state is inferred from
    each row's pre-tax income, age, employment state, and selected structural
    parameters before normalized asset axes are translated into levels.
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

    # Primitive variables must be assigned before persistent income is inferred.
    if x not in DERIVED_ASSET_AXES:
        frame = set_plot_axis(frame, x, values_array)

    if linked_income and x != "labor_income_state":
        current = pd.to_numeric(
            frame["current_income"],
            errors="coerce",
        ).to_numpy(dtype=float)
        frame = link_labor_income_state(
            bundle,
            frame,
            current,
            base_row,
        )

    # Ratios are translated only after permanent income has been determined.
    if x in DERIVED_ASSET_AXES:
        frame = set_plot_axis(frame, x, values_array)

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
    linked_income: bool = True,
) -> pd.DataFrame:
    """Evaluate a rectangular two-dimensional policy surface.

    For every row of the surface, persistent income is inferred from pre-tax
    income, age, and employment before normalized asset axes are converted into
    primitive asset levels.  This keeps age and employment surfaces on the
    structural income process rather than on an arbitrary income-deviation path.
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
    axis_values = {x: xx.ravel(), y: yy.ravel()}

    # First assign primitive states and parameters, including age and income.
    for variable, axis_value in axis_values.items():
        if variable not in DERIVED_ASSET_AXES:
            frame = set_plot_axis(frame, variable, axis_value)

    if linked_income and "labor_income_state" not in {x, y}:
        current = pd.to_numeric(
            frame["current_income"],
            errors="coerce",
        ).to_numpy(dtype=float)
        frame = link_labor_income_state(
            bundle,
            frame,
            current,
            base_row,
        )

    # Translate normalized asset axes using the structurally implied permanent
    # income for each row of the surface.
    for variable, axis_value in axis_values.items():
        if variable in DERIVED_ASSET_AXES:
            frame = set_plot_axis(frame, variable, axis_value)

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
