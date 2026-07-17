"""Population-level MPC analysis for a smooth HA policy surrogate.

The functions in this module deliberately hold the population state draws fixed
when structural or policy parameters change.  This isolates how the fitted
policy map changes, rather than mixing policy changes with a changing simulated
population.

A transfer is represented in the same way as in the HA model's interpolation
code: it is an unexpected increase in current liquid resources.  Baseline and
post-check consumption are evaluated at the same age, income, education,
employment, and illiquid-asset states.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ha_policy_surrogate import PolicySurrogateBundle
from ha_policy_visualization import complete_derived_state

EPS = 1.0e-12


@dataclass(frozen=True)
class PopulationAssumptions:
    """Assumptions for the synthetic cross-section used in the MPC graph.

    Dollar-valued quantities are converted to the units used by the surrogate
    with ``dollars_per_model_unit``.  The default wealth distribution is a
    three-component mixture intended to include poor hand-to-mouth, wealthy
    hand-to-mouth, and conventional saver households.
    """

    n_households: int = 20_000
    seed: int = 20260717
    median_income_dollars: float = 60_000.0
    permanent_income_log_sd: float = 0.50
    transitory_income_log_sd: float = 0.20
    high_education_share: float = 0.55
    nonemployment_share: float = 0.08
    retirement_age: float = 65.0

    poor_hand_to_mouth_share: float = 0.25
    wealthy_hand_to_mouth_share: float = 0.15

    poor_liquid_ratio_median: float = 0.015
    poor_illiquid_ratio_median: float = 0.10
    wealthy_htm_liquid_ratio_median: float = 0.010
    wealthy_htm_illiquid_ratio_median: float = 2.50
    regular_liquid_ratio_median: float = 0.25
    regular_illiquid_ratio_median: float = 1.25
    liquid_ratio_log_sd: float = 0.75
    illiquid_ratio_log_sd: float = 0.85
    negative_liquid_share: float = 0.08

    def validate(self) -> None:
        if self.n_households <= 0:
            raise ValueError("n_households must be positive.")
        if self.median_income_dollars <= 0:
            raise ValueError("median_income_dollars must be positive.")
        for name in (
            "high_education_share",
            "nonemployment_share",
            "poor_hand_to_mouth_share",
            "wealthy_hand_to_mouth_share",
            "negative_liquid_share",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        if (
            self.poor_hand_to_mouth_share + self.wealthy_hand_to_mouth_share
            > 1.0 + EPS
        ):
            raise ValueError(
                "poor_hand_to_mouth_share + wealthy_hand_to_mouth_share "
                "cannot exceed one."
            )
        for name in (
            "permanent_income_log_sd",
            "transitory_income_log_sd",
            "liquid_ratio_log_sd",
            "illiquid_ratio_log_sd",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be nonnegative.")
        for name in (
            "poor_liquid_ratio_median",
            "poor_illiquid_ratio_median",
            "wealthy_htm_liquid_ratio_median",
            "wealthy_htm_illiquid_ratio_median",
            "regular_liquid_ratio_median",
            "regular_illiquid_ratio_median",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True)
class MoneyUnitInfo:
    """How to translate a dollar check into the surrogate's money units."""

    money_units: str
    dollars_per_model_unit: float
    source: str


def _finite_positive_median(values: pd.Series | np.ndarray) -> float | None:
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.median(x)) if len(x) else None


def infer_money_unit_info(
    bundle: PolicySurrogateBundle,
    *,
    bundle_path: str | Path | None = None,
    normalized_unit_fallback: float = 53_000.0,
) -> MoneyUnitInfo:
    """Infer the conversion from dollars to surrogate money units.

    A saved ``money_scale`` is a reliable dollar conversion when it is the
    empirical mean-income normalization used by the HA model.  Parametric grids
    can instead report ``money_scale == 1`` while all incomes are normalized
    around one.  In that case, one model unit is *not* one dollar, so the page
    uses a transparent editable fallback rather than interpreting a $10,000
    check as 10,000 model units.
    """

    meta = dict(bundle.training_metadata or {})
    units = str(meta.get("money_units", meta.get("schema_money_units", ""))).lower()

    income_info = bundle.feature_ranges.get("current_income", {})
    income_median = income_info.get(
        "median", bundle.default_values.get("current_income")
    )
    try:
        income_median_f = float(income_median)
    except (TypeError, ValueError):
        income_median_f = np.nan

    def interpret_scale(value: object, source: str) -> MoneyUnitInfo | None:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value_f) or value_f <= 0:
            return None
        if units == "data":
            return MoneyUnitInfo("data", 1.0, source)

        # Parametric policy bundles often normalize mean income to one and save
        # money_scale=1.  Detect that combination from the state scale.
        normalized_unitless = (
            units == "model"
            and value_f <= 10.0
            and np.isfinite(income_median_f)
            and abs(income_median_f) <= 100.0
        )
        if normalized_unitless:
            return MoneyUnitInfo(
                "model",
                float(normalized_unit_fallback),
                f"normalized-unit fallback; {source}={value_f:g} is unitless",
            )
        return MoneyUnitInfo("model", value_f, source)

    for key in ("money_scale_median", "dollars_per_model_unit"):
        result = interpret_scale(meta.get(key), f"bundle metadata ({key})")
        if result is not None:
            return result

    result = interpret_scale(
        bundle.default_values.get("money_scale"), "bundle default money_scale"
    )
    if result is not None:
        return result

    if bundle_path is not None:
        catalog_path = (
            Path(bundle_path).expanduser().resolve().parent / "model_catalog.csv"
        )
        if catalog_path.exists():
            try:
                catalog = pd.read_csv(catalog_path)
                if "money_scale" in catalog:
                    value_f = _finite_positive_median(catalog["money_scale"])
                    result = interpret_scale(value_f, "sibling model_catalog.csv")
                    if result is not None:
                        return result
            except Exception:
                pass

    if units == "data" or (
        np.isfinite(income_median_f) and income_median_f > 1_000.0
    ):
        return MoneyUnitInfo("data", 1.0, "income-level heuristic")
    return MoneyUnitInfo(
        "model",
        float(normalized_unit_fallback),
        "normalized-unit fallback; edit in the page if needed",
    )


def _range(bundle: PolicySurrogateBundle, column: str) -> tuple[float, float]:
    info = bundle.feature_ranges.get(column, {})
    lo = float(info.get("q01", info.get("min", -np.inf)))
    hi = float(info.get("q99", info.get("max", np.inf)))
    if not np.isfinite(lo):
        lo = float(info.get("min", -np.inf))
    if not np.isfinite(hi):
        hi = float(info.get("max", np.inf))
    return lo, hi


def _clip_to_training_range(
    values: np.ndarray,
    bundle: PolicySurrogateBundle,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _range(bundle, column)
    raw = np.asarray(values, dtype=float)
    clipped = raw.copy()
    changed = np.zeros(len(raw), dtype=bool)
    if np.isfinite(lo):
        changed |= clipped < lo
        clipped = np.maximum(clipped, lo)
    if np.isfinite(hi):
        changed |= clipped > hi
        clipped = np.minimum(clipped, hi)
    return clipped, changed


def _category_values(bundle: PolicySurrogateBundle, column: str) -> list[str]:
    info = bundle.feature_ranges.get(column, {})
    return [str(x) for x in info.get("values", [])]


def _find_named_category(values: list[str], needles: tuple[str, ...]) -> str | None:
    normalized_values = [
        value.lower().replace("-", "_").replace(" ", "_") for value in values
    ]
    # Prefer exact matches so, for example, ``employed`` does not accidentally
    # match ``nonemployed``.
    for needle in needles:
        for value, normalized in zip(values, normalized_values):
            if normalized == needle:
                return value
    for needle in needles:
        for value, normalized in zip(values, normalized_values):
            tokens = tuple(token for token in normalized.split("_") if token)
            if needle in tokens or normalized.startswith(f"{needle}_"):
                return value
    return None


def _binary_categories(
    values: list[str],
    *,
    zero_needles: tuple[str, ...],
    one_needles: tuple[str, ...],
) -> tuple[str, str]:
    if not values:
        return "0", "1"
    zero = _find_named_category(values, zero_needles)
    one = _find_named_category(values, one_needles)
    if zero is None and "0" in values:
        zero = "0"
    if one is None and "1" in values:
        one = "1"
    zero = zero or values[0]
    one = one or (values[1] if len(values) > 1 else values[0])
    return zero, one


def _education_categories(values: list[str]) -> tuple[str, str]:
    if not values:
        return "low", "high"
    high = _find_named_category(
        values, ("high", "college", "university", "degree", "tertiary")
    )
    low = _find_named_category(values, ("low", "no_college", "secondary", "basic"))
    if high is None and "1" in values:
        high = "1"
    if low is None and "0" in values:
        low = "0"
    low = low or values[0]
    high = high or (values[-1] if len(values) > 1 else values[0])
    return low, high


def generate_synthetic_population(
    bundle: PolicySurrogateBundle,
    *,
    base_values: Mapping[str, Any] | None = None,
    assumptions: PopulationAssumptions = PopulationAssumptions(),
    dollars_per_model_unit: float = 53_000.0,
    money_units: str = "model",
) -> pd.DataFrame:
    """Generate a fixed synthetic population compatible with a surrogate bundle.

    The cross-section is drawn in dollars and then converted to the surrogate's
    units.  The same seed produces exactly the same households after any change
    in structural parameters.
    """

    assumptions.validate()
    if dollars_per_model_unit <= 0 or not np.isfinite(dollars_per_model_unit):
        raise ValueError("dollars_per_model_unit must be finite and positive.")
    if money_units not in {"model", "data"}:
        raise ValueError("money_units must be 'model' or 'data'.")

    base = dict(bundle.default_values if base_values is None else dict(base_values))
    n = int(assumptions.n_households)
    rng = np.random.default_rng(int(assumptions.seed))
    frame = pd.DataFrame([base] * n)

    # Education is drawn first because it shifts permanent income.
    education_high = rng.random(n) < assumptions.high_education_share
    if "education" in bundle.feature_spec.categorical_columns:
        values = _category_values(bundle, "education")
        low_value, high_value = _education_categories(values)
        frame["education"] = np.where(education_high, high_value, low_value)

    # A flexible lifecycle cross-section supported by the age range in the grids.
    if "age" in bundle.feature_spec.state_columns:
        age_lo, age_hi = _range(bundle, "age")
        if not np.isfinite(age_lo):
            age_lo = 25.0
        if not np.isfinite(age_hi) or age_hi <= age_lo:
            age_hi = max(age_lo + 1.0, 82.0)
        age = age_lo + (age_hi - age_lo) * rng.beta(2.2, 2.0, n)
        age = np.rint(age)
    else:
        age = np.repeat(float(base.get("age", 40.0)), n)

    retired = age >= float(assumptions.retirement_age)
    nonemployed = retired | (rng.random(n) < assumptions.nonemployment_share)

    # Permanent and transitory log-income components.  The education loading is
    # intentionally modest; the aggregate median remains close to the control.
    log_permanent = (
        np.log(float(assumptions.median_income_dollars))
        + assumptions.permanent_income_log_sd * rng.normal(size=n)
        + np.where(education_high, 0.14, -0.10)
    )
    permanent_income_dollars = np.exp(log_permanent)
    current_income_dollars = permanent_income_dollars * np.exp(
        assumptions.transitory_income_log_sd * rng.normal(size=n)
    )
    # UI and pension replace part of potential labor income rather than setting
    # current resources to zero.
    current_income_dollars = np.where(
        nonemployed & ~retired,
        0.50 * permanent_income_dollars * np.exp(0.08 * rng.normal(size=n)),
        current_income_dollars,
    )
    current_income_dollars = np.where(
        retired,
        0.45 * permanent_income_dollars * np.exp(0.08 * rng.normal(size=n)),
        current_income_dollars,
    )
    # Do not impose an ad-hoc tax rule here.  The bundle recomputes taxes and
    # after-tax income exactly from the currently selected tax kind and tax
    # parameters after the primitive population states have been constructed.

    # Three wealth types.  This produces a mass near zero liquid wealth and a
    # separate group with low liquidity but substantial illiquid wealth.
    draw = rng.random(n)
    poor = draw < assumptions.poor_hand_to_mouth_share
    wealthy_htm = (draw >= assumptions.poor_hand_to_mouth_share) & (
        draw
        < assumptions.poor_hand_to_mouth_share
        + assumptions.wealthy_hand_to_mouth_share
    )
    regular = ~(poor | wealthy_htm)

    liquid_ratio = np.empty(n)
    illiquid_ratio = np.empty(n)
    for mask, liquid_median, illiquid_median in (
        (
            poor,
            assumptions.poor_liquid_ratio_median,
            assumptions.poor_illiquid_ratio_median,
        ),
        (
            wealthy_htm,
            assumptions.wealthy_htm_liquid_ratio_median,
            assumptions.wealthy_htm_illiquid_ratio_median,
        ),
        (
            regular,
            assumptions.regular_liquid_ratio_median,
            assumptions.regular_illiquid_ratio_median,
        ),
    ):
        count = int(mask.sum())
        if count == 0:
            continue
        liquid_ratio[mask] = rng.lognormal(
            mean=np.log(float(liquid_median)),
            sigma=float(assumptions.liquid_ratio_log_sd),
            size=count,
        )
        illiquid_ratio[mask] = rng.lognormal(
            mean=np.log(float(illiquid_median)),
            sigma=float(assumptions.illiquid_ratio_log_sd),
            size=count,
        )

    liquid_assets_dollars = liquid_ratio * permanent_income_dollars
    if _range(bundle, "liquid_assets")[0] < 0:
        debt = rng.random(n) < assumptions.negative_liquid_share
        debt_ratio = rng.lognormal(mean=np.log(0.04), sigma=0.65, size=int(debt.sum()))
        liquid_assets_dollars[debt] = -debt_ratio * permanent_income_dollars[debt]
    illiquid_assets_dollars = illiquid_ratio * permanent_income_dollars

    divisor = float(dollars_per_model_unit) if money_units == "model" else 1.0
    state_values: dict[str, np.ndarray] = {
        "age": age,
        "current_income": current_income_dollars / divisor,
        "labor_income_state": permanent_income_dollars / divisor,
        "liquid_assets": liquid_assets_dollars / divisor,
        "illiquid_assets": illiquid_assets_dollars / divisor,
        "years_to_retirement": float(assumptions.retirement_age) - age,
        "education_state": education_high.astype(float),
    }

    clipping_flags: list[np.ndarray] = []
    for column in bundle.feature_spec.state_columns:
        if column in state_values:
            clipped, changed = _clip_to_training_range(
                state_values[column], bundle, column
            )
            frame[column] = clipped
            clipping_flags.append(changed)

    # Standard binary categorical states used by the solver.
    if "employment_state" in bundle.feature_spec.categorical_columns:
        values = _category_values(bundle, "employment_state")
        employed_value, nonemployed_value = _binary_categories(
            values,
            zero_needles=("employed", "working", "work"),
            one_needles=("unemployed", "nonemployed", "not_working", "retired"),
        )
        frame["employment_state"] = np.where(
            nonemployed, nonemployed_value, employed_value
        )
    if "is_retired" in bundle.feature_spec.categorical_columns:
        values = _category_values(bundle, "is_retired")
        not_retired_value, retired_value = _binary_categories(
            values,
            zero_needles=("not_retired", "working_age", "false"),
            one_needles=("retired", "true"),
        )
        frame["is_retired"] = np.where(
            retired, retired_value, not_retired_value
        )

    # Other categorical features are held at the selected/base category.
    for column in bundle.feature_spec.categorical_columns:
        if column in frame:
            continue
        frame[column] = str(base.get(column, bundle.feature_ranges[column].get("mode", "0")))

    # Apply the selected continuous and categorical parameters, then rebuild
    # labor income, pension income, taxes, and after-tax income using the same
    # accounting code used by the fitted surrogate.
    frame = set_parameter_values(
        frame,
        bundle,
        base,
        recompute_after_tax=False,
    )
    frame = bundle.prepare_inputs(frame)
    frame = complete_derived_state(frame)

    # Derived variables must remain internally consistent, so we only *flag*
    # support violations for them rather than clipping them independently.
    derived_columns = {
        "after_tax_income",
        "years_to_retirement",
        "total_assets",
        "cash_on_hand",
    }
    for column in bundle.feature_spec.state_columns:
        if column not in frame or column not in bundle.feature_ranges:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        clipped, changed = _clip_to_training_range(values, bundle, column)
        if column not in derived_columns:
            frame[column] = clipped
        clipping_flags.append(changed)

    # If clipping changed a primitive state, refresh all accounting-derived
    # variables one final time.
    frame = bundle.prepare_inputs(frame)
    frame = complete_derived_state(frame)

    frame["population_type"] = np.select(
        [poor, wealthy_htm],
        ["poor hand-to-mouth", "wealthy hand-to-mouth"],
        default="regular saver",
    )
    frame["income_dollars_unclipped"] = current_income_dollars
    frame["permanent_income_dollars_unclipped"] = permanent_income_dollars
    frame["liquid_assets_dollars_unclipped"] = liquid_assets_dollars
    frame["illiquid_assets_dollars_unclipped"] = illiquid_assets_dollars
    if clipping_flags:
        frame["state_clipped_to_training_support"] = np.logical_or.reduce(clipping_flags)
    else:
        frame["state_clipped_to_training_support"] = False
    frame["household_weight"] = 1.0 / n
    return frame


def set_parameter_values(
    frame: pd.DataFrame,
    bundle: PolicySurrogateBundle,
    values: Mapping[str, Any],
    *,
    recompute_after_tax: bool = True,
) -> pd.DataFrame:
    """Set all continuous and categorical policy parameters on ``frame``.

    By default, the function immediately calls ``bundle.prepare_inputs`` so a
    change in ``tax.kind`` or any tax-rate parameter also updates taxes and
    after-tax income.  Set ``recompute_after_tax=False`` only while assembling
    a partially completed population frame.
    """

    out = frame.copy()
    for column in bundle.feature_spec.parameter_columns:
        if column in values:
            out[column] = values[column]
        elif column.endswith("__missing"):
            out[column] = 0
        elif column in bundle.default_values:
            out[column] = bundle.default_values[column]
        else:
            raise KeyError(f"No value available for parameter {column!r}.")
    for column in bundle.feature_spec.categorical_parameter_columns:
        if column in values:
            out[column] = str(values[column])
        elif column in bundle.default_values:
            out[column] = str(bundle.default_values[column])
        else:
            raise KeyError(f"No value available for parameter {column!r}.")
    return bundle.prepare_inputs(out) if recompute_after_tax else out


def compute_check_mpcs(
    bundle: PolicySurrogateBundle,
    population: pd.DataFrame,
    *,
    check_amount_dollars: float = 10_000.0,
    dollars_per_model_unit: float = 53_000.0,
    money_units: str = "model",
    clip_treated_liquid_state: bool = True,
) -> pd.DataFrame:
    """Evaluate household MPCs from an unexpected liquid check.

    MPC is ``(c_after - c_before) / check``.  Because consumption and the check
    are expressed in the same units, this ratio is invariant to normalization.
    When requested, the post-check liquid state is clipped at the saved grid or
    training maximum, matching the solver's interpolation behavior.
    """

    if check_amount_dollars <= 0 or not np.isfinite(check_amount_dollars):
        raise ValueError("check_amount_dollars must be finite and positive.")
    if dollars_per_model_unit <= 0 or not np.isfinite(dollars_per_model_unit):
        raise ValueError("dollars_per_model_unit must be finite and positive.")
    if money_units not in {"model", "data"}:
        raise ValueError("money_units must be 'model' or 'data'.")
    if "liquid_assets" not in population:
        raise KeyError("Population must contain liquid_assets.")

    check_units = (
        float(check_amount_dollars) / float(dollars_per_model_unit)
        if money_units == "model"
        else float(check_amount_dollars)
    )
    # Recompute taxes and after-tax income from the currently selected policy
    # parameters before evaluating either state.
    baseline = bundle.prepare_inputs(population.copy())
    treated = baseline.copy()
    liquid_before = pd.to_numeric(
        baseline["liquid_assets"], errors="coerce"
    ).to_numpy(dtype=float)
    liquid_after_raw = liquid_before + check_units
    liquid_after = liquid_after_raw.copy()

    if clip_treated_liquid_state:
        if "liquid_grid_max" in baseline:
            upper = pd.to_numeric(
                baseline["liquid_grid_max"], errors="coerce"
            ).to_numpy(dtype=float)
            fallback_hi = _range(bundle, "liquid_assets")[1]
            upper = np.where(np.isfinite(upper), upper, fallback_hi)
        else:
            upper = np.repeat(_range(bundle, "liquid_assets")[1], len(baseline))
        lower = np.repeat(_range(bundle, "liquid_assets")[0], len(baseline))
        liquid_after = np.minimum(liquid_after, upper)
        if np.isfinite(lower).all():
            liquid_after = np.maximum(liquid_after, lower)
    treated["liquid_assets"] = liquid_after
    treated = bundle.prepare_inputs(treated)

    pred0 = bundle.predict(baseline)
    pred1 = bundle.predict(treated)
    delta_c = pred1["consumption"].to_numpy(dtype=float) - pred0[
        "consumption"
    ].to_numpy(dtype=float)
    mpc = delta_c / check_units

    out = population.copy()
    out["baseline_consumption"] = pred0["consumption"].to_numpy(dtype=float)
    out["post_check_consumption"] = pred1["consumption"].to_numpy(dtype=float)
    out["consumption_response"] = delta_c
    out["mpc"] = mpc
    out["check_amount_dollars"] = float(check_amount_dollars)
    out["check_amount_model_units"] = check_units
    out["liquid_assets_after_check_raw"] = liquid_after_raw
    out["liquid_assets_after_check"] = liquid_after
    out["check_state_clipped"] = liquid_after < liquid_after_raw - 1.0e-12
    return out


def mpc_summary(response: pd.DataFrame) -> dict[str, float]:
    """Compact diagnostics for an MPC response table."""

    x = pd.to_numeric(response["mpc"], errors="coerce").to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "share_negative": np.nan,
            "share_above_one": np.nan,
            "share_above_08": np.nan,
            "share_check_state_clipped": np.nan,
        }
    clipped_col = response.get("check_state_clipped")
    if clipped_col is None:
        clipped_share = 0.0
    else:
        clipped_share = float(
            pd.to_numeric(clipped_col, errors="coerce").fillna(0.0).mean()
        )
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "share_negative": float(np.mean(x < 0.0)),
        "share_above_one": float(np.mean(x > 1.0)),
        "share_above_08": float(np.mean(x > 0.8)),
        "share_check_state_clipped": clipped_share,
    }


def plot_mpc_distribution(
    response: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    bins: int = 50,
    title: str = "Distribution of MPCs from a $10,000 check",
    output_path: str | Path | None = None,
):
    """Matplotlib histogram for batch/static output."""

    import matplotlib.pyplot as plt

    current = pd.to_numeric(response["mpc"], errors="coerce").to_numpy(dtype=float)
    current = current[np.isfinite(current)]
    arrays = [current]
    labels = ["Selected parameters"]
    if reference is not None:
        ref = pd.to_numeric(reference["mpc"], errors="coerce").to_numpy(dtype=float)
        ref = ref[np.isfinite(ref)]
        if len(ref):
            arrays.append(ref)
            labels.append("Default parameters")

    pooled = np.concatenate([x for x in arrays if len(x)])
    lo, hi = np.quantile(pooled, [0.005, 0.995])
    if hi <= lo:
        lo, hi = float(lo - 0.05), float(hi + 0.05)
    edges = np.linspace(lo, hi, int(bins) + 1)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for values, label in zip(arrays, labels):
        ax.hist(values, bins=edges, density=True, histtype="step", linewidth=2, label=label)
    ax.axvline(float(np.mean(current)), linestyle="--", linewidth=1.5, label="Current mean")
    ax.axvline(float(np.median(current)), linestyle=":", linewidth=1.5, label="Current median")
    ax.set_xlabel("Marginal propensity to consume")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
    return fig, ax
