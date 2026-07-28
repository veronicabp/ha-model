"""Command-line entry point for fitting an MPC-aware HA policy surrogate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from threadpoolctl import threadpool_limits

from ha_policy_surrogate import (
    DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS,
    DEFAULT_CATEGORICAL_STATE_COLUMNS,
    DEFAULT_CATEGORICAL_STATE_POWER_TERMS,
    DEFAULT_ENGINEERED_STATE_COLUMNS,
    DEFAULT_REDUNDANT_STATE_COLUMNS,
    DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS,
    DEFAULT_SPLINE_CATEGORICAL_TENSOR_PAIRS,
    DEFAULT_SPLINE_COLUMN_N_KNOTS,
    DEFAULT_SPLINE_KNOT_POWERS,
    DEFAULT_SPLINE_KNOT_STRATEGIES,
    DEFAULT_PARAMETER_POWER_TERMS,
    DEFAULT_PARAMETER_SELECTION_POWERS,
    DEFAULT_PARAMETER_SELECTION_SPLINE_COLUMNS,
    DEFAULT_SPLINE_COLUMNS,
    DEFAULT_SPLINE_PARAMETER_PAIR_TERMS,
    DEFAULT_SPLINE_PARAMETER_POWER_TERMS,
    DEFAULT_SPLINE_TENSOR_PAIRS,
    DEFAULT_SPLINE_TENSOR_TRIPLES,
    DEFAULT_SPLINE_TENSOR_PARAMETER_TERMS,
    DEFAULT_STATE_COLUMNS,
    GridSchema,
    PopulationSamplingSpec,
    train_from_saved_grids,
)


def _parse_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _parse_float_list(value: str | None) -> list[float] | None:
    values = _parse_list(value)
    if values is None:
        return None
    try:
        out = [float(x) for x in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of numbers."
        ) from exc
    if any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("All MPC check sizes must be positive.")
    return out


def _parse_int_list(
    value: str | None, *, minimum: int | None = None
) -> list[int] | None:
    values = _parse_list(value)
    if values is None:
        return None
    try:
        out = [int(x) for x in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of integers."
        ) from exc
    if minimum is not None and any(x < int(minimum) for x in out):
        raise argparse.ArgumentTypeError(f"All values must be at least {int(minimum)}.")
    return out


def _parse_nonnegative_float_list(value: str | None) -> list[float] | None:
    values = _parse_list(value)
    if values is None:
        return None
    try:
        out = [float(x) for x in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a comma-separated list of numbers."
        ) from exc
    if any(x < 0 for x in out):
        raise argparse.ArgumentTypeError("All values must be nonnegative.")
    return out


def _parse_column_int_terms(value: str | None) -> list[tuple[str, int]]:
    items = _parse_list(value) or []
    terms: list[tuple[str, int]] = []
    for item in items:
        parts = [part.strip() for part in item.rsplit(":", 1)]
        if len(parts) != 2 or not parts[0]:
            raise argparse.ArgumentTypeError(
                "Column knot counts must use column:count syntax."
            )
        try:
            count = int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Column knot counts must be integers."
            ) from exc
        if count < 4:
            raise argparse.ArgumentTypeError(
                "Every spline column must have at least four knots."
            )
        terms.append((parts[0], count))
    return terms


def _parse_column_float_terms(value: str | None) -> list[tuple[str, float]]:
    items = _parse_list(value) or []
    terms: list[tuple[str, float]] = []
    for item in items:
        parts = [part.strip() for part in item.rsplit(":", 1)]
        if len(parts) != 2 or not parts[0]:
            raise argparse.ArgumentTypeError(
                "Column powers must use column:power syntax."
            )
        try:
            power = float(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError("Column powers must be numbers.") from exc
        if power <= 0:
            raise argparse.ArgumentTypeError("Column powers must be strictly positive.")
        terms.append((parts[0], power))
    return terms


def _parse_column_string_terms(value: str | None) -> list[tuple[str, str]]:
    items = _parse_list(value) or []
    terms: list[tuple[str, str]] = []
    allowed = {"quantile", "uniform", "low_tail"}
    for item in items:
        parts = [part.strip() for part in item.rsplit(":", 1)]
        if len(parts) != 2 or not parts[0] or parts[1] not in allowed:
            raise argparse.ArgumentTypeError(
                "Knot strategies must use column:quantile, column:uniform, "
                "or column:low_tail syntax."
            )
        terms.append((parts[0], parts[1]))
    return terms


def _parse_tensor_pairs(value: str | None) -> list[tuple[str, str]]:
    items = _parse_list(value) or []
    pairs: list[tuple[str, str]] = []
    for item in items:
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                "Spline tensor pairs must use left:right syntax."
            )
        left, right = (part.strip() for part in item.split(":", 1))
        if not left or not right:
            raise argparse.ArgumentTypeError(
                "Spline tensor pairs must use nonempty left:right names."
            )
        pairs.append((left, right))
    return pairs


def _parse_tensor_triples(value: str | None) -> list[tuple[str, str, str]]:
    items = _parse_list(value) or []
    triples: list[tuple[str, str, str]] = []
    for item in items:
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3 or any(not part for part in parts):
            raise argparse.ArgumentTypeError(
                "Spline tensor triples must use first:second:third syntax."
            )
        triples.append((parts[0], parts[1], parts[2]))
    return triples


def _parse_parameter_power_terms(value: str | None) -> list[tuple[str, int]]:
    items = _parse_list(value) or []
    terms: list[tuple[str, int]] = []
    for item in items:
        parts = [part.strip() for part in item.rsplit(":", 1)]
        if len(parts) != 2 or not parts[0]:
            raise argparse.ArgumentTypeError(
                "Parameter powers must use parameter:power syntax."
            )
        try:
            power = int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Parameter powers must be integers."
            ) from exc
        if power < 2:
            raise argparse.ArgumentTypeError("Parameter powers must be at least two.")
        terms.append((parts[0], power))
    return terms


def _parse_spline_parameter_power_terms(
    value: str | None,
) -> list[tuple[str, str, int]]:
    items = _parse_list(value) or []
    terms: list[tuple[str, str, int]] = []
    for item in items:
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3 or any(not part for part in parts[:2]):
            raise argparse.ArgumentTypeError(
                "Spline-parameter powers must use spline:parameter:power syntax."
            )
        try:
            power = int(parts[2])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Spline-parameter powers must be integers."
            ) from exc
        if power < 2:
            raise argparse.ArgumentTypeError(
                "Spline-parameter powers must be at least two."
            )
        terms.append((parts[0], parts[1], power))
    return terms


def _parse_spline_parameter_pair_terms(
    value: str | None,
) -> list[tuple[str, str, str]]:
    items = _parse_list(value) or []
    terms: list[tuple[str, str, str]] = []
    for item in items:
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3 or any(not part for part in parts):
            raise argparse.ArgumentTypeError(
                "Spline-parameter pairs must use spline:parameter1:parameter2 syntax."
            )
        terms.append((parts[0], parts[1], parts[2]))
    return terms


def _parse_categorical_state_power_terms(
    value: str | None,
) -> list[tuple[str, str, int]]:
    items = _parse_list(value) or []
    terms: list[tuple[str, str, int]] = []
    for item in items:
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3 or any(not part for part in parts[:2]):
            raise argparse.ArgumentTypeError(
                "Categorical-state powers must use category:state:power syntax."
            )
        try:
            power = int(parts[2])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "Categorical-state powers must be integers."
            ) from exc
        if power < 1:
            raise argparse.ArgumentTypeError(
                "Categorical-state powers must be positive."
            )
        terms.append((parts[0], parts[1], power))
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample saved HA policy grids and fit a smooth state-and-parameter "
            "surrogate. Validation holds out entire solved parameterizations. "
            "The loader treats consumption and deposit as the two primitive "
            "saved policies and reconstructs both asset policies from accounting. "
            "The default specification uses standardized liquid-slack splines, "
            "a joint consumption-level/MPC objective, and an auxiliary "
            "liquid-outflow target."
        )
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--policy-layout", default="GHKEBA")
    parser.add_argument(
        "--money-units",
        choices=("model", "data"),
        default="model",
        help="Fit in normalized model units (recommended) or multiply by money_scale.",
    )
    parser.add_argument(
        "--key-overrides-json",
        type=Path,
        default=None,
        help='Optional JSON mapping, e.g. {"consumption":"policy_c"}.',
    )
    parser.add_argument(
        "--require-compact-policies",
        action="store_true",
        help=(
            "Reject legacy model files that save full drift or next-asset "
            "policy tensors. By default, compact and legacy files are both "
            "accepted, but consumption and deposit are always treated as the "
            "authoritative primitive policies when available."
        ),
    )

    parser.add_argument("--rows-per-model", type=int, default=1_000)
    parser.add_argument("--max-total-rows", type=int, default=3_000_000)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument(
        "--sampling-mode",
        choices=("population", "grid"),
        default="population",
        help=(
            "Population mode draws continuous asset states from a synthetic "
            "cross-section and deliberately oversamples poor and wealthy "
            "hand-to-mouth households. Grid mode retains the older marginal "
            "grid sampler."
        ),
    )
    parser.add_argument(
        "--include-retired",
        action="store_true",
        help=(
            "Include retired states in sampling, fitting, tuning, and validation. "
            "By default, the surrogate is restricted to age < retirement_age, "
            "which is the solver's working-age convention."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        help=(
            "CPU workers used to read and sample solved models. The spline/ridge "
            "fit is CPU-based; GPUs do not materially accelerate this linear "
            "basis estimator."
        ),
    )
    parser.add_argument(
        "--fit-threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        help=(
            "BLAS threads used for feature cross-products and the closed-form "
            "ridge solves after model-level sampling is complete."
        ),
    )
    parser.add_argument("--poor-htm-share", type=float, default=0.35)
    parser.add_argument("--wealthy-htm-share", type=float, default=0.35)
    parser.add_argument("--broad-support-share", type=float, default=0.10)
    parser.add_argument("--nonemployment-share", type=float, default=0.20)
    parser.add_argument("--negative-liquid-share", type=float, default=0.10)
    parser.add_argument(
        "--constraint-anchor-share",
        type=float,
        default=0.25,
        help=(
            "Share of hand-to-mouth draws placed very near the borrowing " "constraint."
        ),
    )
    parser.add_argument("--poor-liquid-ratio-median", type=float, default=0.012)
    parser.add_argument("--poor-illiquid-ratio-median", type=float, default=0.08)
    parser.add_argument("--wealthy-liquid-ratio-median", type=float, default=0.010)
    parser.add_argument("--wealthy-illiquid-ratio-median", type=float, default=2.50)
    parser.add_argument("--regular-liquid-ratio-median", type=float, default=0.30)
    parser.add_argument("--regular-illiquid-ratio-median", type=float, default=1.40)
    parser.add_argument("--liquid-ratio-log-sd", type=float, default=0.75)
    parser.add_argument("--illiquid-ratio-log-sd", type=float, default=0.85)
    parser.add_argument(
        "--mpc-pair-share",
        type=float,
        default=0.60,
        help=(
            "Share of sampled rows devoted to baseline/treated liquid-asset "
            "pairs. The remaining rows provide broad policy-level coverage."
        ),
    )
    parser.add_argument(
        "--mpc-cross-zero-share",
        type=float,
        default=0.30,
        help=(
            "Share of explicit MPC pairs forced to start with negative liquid "
            "wealth and finish above zero after the transfer."
        ),
    )
    parser.add_argument(
        "--mpc-check-sizes",
        default="0.02,0.05,0.10,0.188679,0.25",
        help=(
            "Comma-separated liquid-resource changes in model units used to "
            "construct MPC training pairs."
        ),
    )

    parser.add_argument(
        "--parameter-include",
        default=None,
        help="Comma-separated exact flattened parameter names.",
    )
    parser.add_argument(
        "--parameter-exclude",
        default=None,
        help="Comma-separated exact flattened continuous parameter names.",
    )
    parser.add_argument(
        "--categorical-parameter-include",
        default=None,
        help=(
            "Comma-separated exact categorical parameter names. By default, "
            "varying string/Boolean parameters such as param__policy__tax__kind "
            "are detected automatically."
        ),
    )
    parser.add_argument(
        "--categorical-parameter-exclude",
        default=None,
        help="Comma-separated categorical parameter names to exclude.",
    )
    parser.add_argument(
        "--max-solver-distance",
        type=float,
        default=None,
        help=(
            "Optional exclusion cutoff for diagnostics.max_distance. Omit this "
            "argument to keep every readable model whose solver status is success."
        ),
    )
    parser.add_argument(
        "--max-relative-distance",
        type=float,
        default=None,
        help="Optional cutoff for diagnostics.max_relative_distance.",
    )
    parser.add_argument(
        "--max-hjb-residual",
        type=float,
        default=None,
        help="Optional cutoff for diagnostics.max_hjb_residual.",
    )

    parser.add_argument(
        "--state-columns",
        default=",".join(DEFAULT_STATE_COLUMNS),
        help=(
            "Comma-separated primitive continuous household states. Derived "
            "MPC-oriented states are added separately and recomputed before "
            "every prediction."
        ),
    )
    parser.add_argument(
        "--engineered-state-columns",
        default=",".join(DEFAULT_ENGINEERED_STATE_COLUMNS),
        help=(
            "Comma-separated internally generated states, such as liquid slack "
            "relative to permanent income."
        ),
    )
    parser.add_argument(
        "--redundant-state-columns",
        default=",".join(DEFAULT_REDUNDANT_STATE_COLUMNS),
        help=(
            "Primitive states retained for the interface but excluded from the "
            "regression because they are deterministic/redundant conditional on "
            "other inputs. The default drops gross current income while retaining "
            "after-tax income and the persistent labor-income state."
        ),
    )
    parser.add_argument(
        "--categorical-columns",
        default=",".join(DEFAULT_CATEGORICAL_STATE_COLUMNS),
        help="Comma-separated categorical household-state variables.",
    )

    parser.add_argument(
        "--model-type", choices=("polynomial", "rff"), default="polynomial"
    )
    parser.add_argument(
        "--polynomial-degree", type=int, choices=(1, 2, 3, 4), default=2
    )
    parser.add_argument(
        "--spline-columns",
        default=",".join(DEFAULT_SPLINE_COLUMNS),
        help=(
            "Comma-separated engineered states receiving a B-spline basis in "
            "polynomial mode. Liquid-resource columns use the custom knot "
            "strategies below; other columns use quantile knots."
        ),
    )
    parser.add_argument(
        "--spline-knots",
        type=int,
        default=8,
        help="Default number of knots for spline columns without an override.",
    )
    parser.add_argument(
        "--spline-column-knots",
        default=",".join(
            f"{column}:{count}" for column, count in DEFAULT_SPLINE_COLUMN_N_KNOTS
        ),
        help=(
            "Comma-separated column:count overrides. The defaults allocate "
            "additional knots to low-liquidity and liquid-resource splines."
        ),
    )
    parser.add_argument(
        "--spline-knot-strategies",
        default=",".join(
            f"{column}:{strategy}"
            for column, strategy in DEFAULT_SPLINE_KNOT_STRATEGIES
        ),
        help=(
            "Comma-separated column:strategy values. Strategies are quantile, "
            "uniform, or low_tail. The liquid defaults use low_tail rather "
            "than sample-quantile knots."
        ),
    )
    parser.add_argument(
        "--spline-knot-powers",
        default=",".join(
            f"{column}:{power:g}" for column, power in DEFAULT_SPLINE_KNOT_POWERS
        ),
        help=(
            "Comma-separated column:power values controlling low_tail spacing. "
            "Larger powers place more knots near the lower support endpoint."
        ),
    )
    parser.add_argument(
        "--spline-tensor-pairs",
        default=",".join(
            f"{left}:{right}" for left, right in DEFAULT_SPLINE_TENSOR_PAIRS
        ),
        help=(
            "Comma-separated left:right spline pairs. The defaults let low "
            "liquid slack and disposable liquid resources interact with "
            "illiquid wealth and working-age lifecycle position."
        ),
    )
    parser.add_argument(
        "--spline-tensor-triples",
        default=",".join(
            f"{first}:{second}:{third}"
            for first, second, third in DEFAULT_SPLINE_TENSOR_TRIPLES
        ),
        help=(
            "Optional comma-separated first:second:third spline triples. "
            "The default allows the illiquid-wealth MPC hump to shift jointly "
            "with liquid resources and age. Use additional triples sparingly."
        ),
    )
    parser.add_argument(
        "--spline-categorical-tensor-pairs",
        default=",".join(
            f"{category}:{left}:{right}"
            for category, left, right in DEFAULT_SPLINE_CATEGORICAL_TENSOR_PAIRS
        ),
        help=(
            "Comma-separated category:left:right terms. These allow selected "
            "state tensor products to differ by employment or another categorical "
            "household state."
        ),
    )
    parser.add_argument(
        "--spline-tensor-parameter-terms",
        default=",".join(
            f"{left}:{right}:{parameter}"
            for left, right, parameter in DEFAULT_SPLINE_TENSOR_PARAMETER_TERMS
        ),
        help=(
            "Comma-separated left:right:parameter terms. Each term lets an "
            "entire two-dimensional spline surface vary with a structural "
            "parameter, especially adjustment costs and risk aversion."
        ),
    )

    parser.add_argument(
        "--parameter-power-terms",
        default=",".join(
            f"{parameter}:{power}" for parameter, power in DEFAULT_PARAMETER_POWER_TERMS
        ),
        help=(
            "Comma-separated parameter:power terms forced into every fit beyond "
            "the global polynomial degree. The default is empty because the "
            "automatic tuning-set search screens these terms."
        ),
    )
    parser.add_argument(
        "--spline-parameter-power-terms",
        default=",".join(
            f"{spline}:{parameter}:{power}"
            for spline, parameter, power in DEFAULT_SPLINE_PARAMETER_POWER_TERMS
        ),
        help=(
            "Comma-separated spline:parameter:power terms forced into every fit. "
            "The automatic search separately screens all eligible versions."
        ),
    )
    parser.add_argument(
        "--spline-parameter-pair-terms",
        default=",".join(
            f"{spline}:{parameter_1}:{parameter_2}"
            for spline, parameter_1, parameter_2 in DEFAULT_SPLINE_PARAMETER_PAIR_TERMS
        ),
        help=(
            "Comma-separated spline:parameter1:parameter2 third-order terms "
            "forced into every fit. The defaults retain the strongest "
            "economically motivated risk-aversion and discounting terms."
        ),
    )
    parser.add_argument(
        "--auto-select-parameter-terms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Screen every eligible higher-order parameter power and every "
            "unordered parameter pair on the tuning models, then add only "
            "terms with a practically meaningful and model-bootstrap-significant "
            "R2 improvement to the final feature map. Use "
            "--no-auto-select-parameter-terms to use only manually listed terms."
        ),
    )
    parser.add_argument(
        "--parameter-selection-spline-columns",
        default=",".join(DEFAULT_PARAMETER_SELECTION_SPLINE_COLUMNS),
        help=(
            "Comma-separated spline bases used to modulate candidate parameter "
            "powers and parameter pairs. The defaults are the two liquid-state "
            "bases that can change the MPC."
        ),
    )
    parser.add_argument(
        "--parameter-selection-powers",
        default=",".join(str(x) for x in DEFAULT_PARAMETER_SELECTION_POWERS),
        help=(
            "Comma-separated powers screened for spline x parameter^power terms. "
            "Standalone powers are considered only when they exceed the global "
            "polynomial degree."
        ),
    )
    parser.add_argument(
        "--parameter-selection-patterns",
        default="",
        help=(
            "Optional comma-separated name patterns defining eligible varying "
            "continuous parameters. The default empty value screens every "
            "varying continuous parameter."
        ),
    )
    parser.add_argument(
        "--parameter-selection-exclude-patterns",
        default=None,
        help="Optional comma-separated parameter-name patterns excluded from search.",
    )
    parser.add_argument(
        "--parameter-selection-include-parameter-powers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also screen standalone powers above --polynomial-degree for policy-"
            "level fit. These terms do not directly change finite-difference MPCs."
        ),
    )
    parser.add_argument(
        "--parameter-selection-max-train-rows",
        type=int,
        default=300_000,
        help=(
            "Approximate row cap for the model-preserving parameter-search "
            "training subsample. The selected terms are still refitted on all rows."
        ),
    )
    parser.add_argument(
        "--parameter-selection-max-tune-rows",
        type=int,
        default=150_000,
        help="Approximate row cap for the model-preserving tuning subsample.",
    )
    parser.add_argument(
        "--parameter-selection-min-mpc-r2-gain",
        type=float,
        default=0.005,
        help=(
            "Minimum tuning-set MPC R2 increase required for a spline-modulated "
            "higher-order parameter term."
        ),
    )
    parser.add_argument(
        "--parameter-selection-min-level-r2-gain",
        type=float,
        default=0.001,
        help=(
            "Minimum tuning-set consumption R2 increase required for a standalone "
            "higher-order parameter power."
        ),
    )
    parser.add_argument(
        "--parameter-selection-max-mpc-r2-loss-for-level-term",
        type=float,
        default=0.001,
        help=(
            "Largest allowed MPC R2 deterioration when accepting a standalone "
            "parameter power for consumption-level fit."
        ),
    )
    parser.add_argument(
        "--parameter-selection-min-model-win-share",
        type=float,
        default=0.55,
        help=("Minimum share of tuning models whose mean squared error improves."),
    )
    parser.add_argument(
        "--parameter-selection-fdr",
        type=float,
        default=0.10,
        help=(
            "Benjamini-Hochberg false-discovery-rate cutoff applied to one-sided "
            "model-level improvement tests. The model-bootstrap lower bound must "
            "also be positive."
        ),
    )
    parser.add_argument(
        "--parameter-selection-bootstrap-reps",
        type=int,
        default=200,
        help="Number of model-level bootstrap draws used for significance screening.",
    )
    parser.add_argument(
        "--parameter-selection-bootstrap-confidence",
        type=float,
        default=0.95,
        help="One-sided bootstrap confidence level for error improvement.",
    )
    parser.add_argument(
        "--parameter-selection-max-terms",
        type=int,
        default=0,
        help=(
            "Maximum number of greedily selected higher-order parameter terms added "
            "to the final joint fit. Zero keeps selecting until no remaining term "
            "passes the tuning-set criteria."
        ),
    )

    parser.add_argument(
        "--categorical-state-power-terms",
        default=",".join(
            f"{category}:{state}:{power}"
            for category, state, power in DEFAULT_CATEGORICAL_STATE_POWER_TERMS
        ),
        help=(
            "Comma-separated category:state:power terms. General employment-state "
            "slopes already interact employment with liquid assets and the zero-"
            "asset hinge; the defaults add employment-specific quadratic curvature."
        ),
    )
    parser.add_argument(
        "--categorical-slope-exclude-patterns",
        default=",".join(DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS),
        help=(
            "Category-name patterns excluded from category-specific state slopes. "
            "The default keeps is_retired as a level dummy but removes its "
            "unrestricted slope jump."
        ),
    )
    parser.add_argument(
        "--spline-categorical-exclude-patterns",
        default=",".join(DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS),
        help="Category-name patterns excluded from spline interactions.",
    )
    parser.add_argument(
        "--ridge-alphas",
        default="0.001,0.01,0.1,1",
        help="Comma-separated ridge penalties. Fewer entries make tuning faster.",
    )
    parser.add_argument("--rff-components", type=int, default=512)

    parser.add_argument(
        "--target-parameterization",
        choices=("accounting", "changes", "levels"),
        default="accounting",
        help=(
            "Use the accounting-restricted consumption/deposit target system. "
            "Legacy changes/levels values are accepted as aliases."
        ),
    )
    parser.add_argument(
        "--consumption-transform",
        choices=("identity", "asinh", "log1p"),
        default="identity",
        help=(
            "Identity is recommended because MPC tuning is evaluated in "
            "consumption levels."
        ),
    )
    parser.add_argument(
        "--deposit-transform",
        choices=("identity", "asinh", "log1p"),
        default="asinh",
    )
    parser.add_argument(
        "--liquid-outflow-transform",
        choices=("identity", "asinh", "log1p"),
        default="identity",
        help=(
            "Transformation for c + d + adjustment_cost, the target that "
            "determines liquid saving directly."
        ),
    )
    parser.add_argument(
        "--mpc-loss-weight",
        type=float,
        default=2.0,
        help=(
            "Weight on tuning-set MPC NRMSE when selecting the consumption "
            "ridge penalty and direct MPC-objective weight. The consumption "
            "level NRMSE retains weight one."
        ),
    )
    parser.add_argument(
        "--mpc-difference-weight",
        type=float,
        default=None,
        help=(
            "Fixed nonnegative weight on exact finite-difference MPC errors in "
            "the consumption coefficient objective. Omit to select the weight "
            "from --mpc-difference-weight-grid on tuning models."
        ),
    )
    parser.add_argument(
        "--mpc-difference-weight-grid",
        default="0,0.25,0.50,1,2,4,8",
        help=(
            "Comma-separated candidate weights for directly targeting MPCs. "
            "Used only when --mpc-difference-weight is omitted."
        ),
    )
    parser.add_argument(
        "--mpc-objective-max-pairs",
        type=int,
        default=500_000,
        help=(
            "Maximum number of lifecycle- and wealth-stratified training MPC "
            "pairs used in each joint consumption/MPC solve."
        ),
    )
    parser.add_argument(
        "--mpc-objective-max-iter",
        type=int,
        default=200,
        help="Retained for command compatibility; closed-form moment fitting ignores it.",
    )
    parser.add_argument(
        "--mpc-objective-tol",
        type=float,
        default=1.0e-6,
        help="Retained for command compatibility; closed-form moment fitting ignores it.",
    )
    parser.add_argument(
        "--mpc-objective-trim-quantile",
        type=float,
        default=0.0,
        help=(
            "Optional symmetric tail fraction trimmed from exact MPC targets "
            "inside the training objective. Validation always uses all pairs."
        ),
    )
    parser.add_argument(
        "--mpc-stratify-age-bins",
        default="35,45,55,60",
        help=(
            "Comma-separated age boundaries used when retaining MPC pairs. "
            "The default separates early, middle, and late working ages."
        ),
    )
    parser.add_argument(
        "--mpc-stratify-liquid-bins",
        type=int,
        default=4,
        help="Number of liquid-slack quantile bins used in MPC-pair stratification.",
    )
    parser.add_argument(
        "--mpc-stratify-illiquid-bins",
        type=int,
        default=10,
        help=(
            "Number of illiquid-wealth quantile bins used when balancing MPC "
            "pairs, which protects wealthy hand-to-mouth states."
        ),
    )

    parser.add_argument(
        "--mpc-stratify-target-bins",
        type=int,
        default=5,
        help=(
            "Number of exact-MPC quantile bins used when balancing training "
            "pairs. Set to 4 or 5 to protect low- and high-MPC tails; zero disables."
        ),
    )

    parser.add_argument(
        "--liquid-change-loss-weight",
        type=float,
        default=1.0,
        help="Weight on liquid-saving accuracy when reconciling deposit and outflow.",
    )
    parser.add_argument(
        "--deposit-reconciliation-loss-weight",
        type=float,
        default=0.50,
        help="Weight on deposit accuracy in the reconciliation step.",
    )
    parser.add_argument(
        "--liquid-outflow-weight",
        type=float,
        default=None,
        help=(
            "Fixed weight in [0,1] on the direct liquid-outflow prediction. "
            "By default the weight is selected on tuning models."
        ),
    )
    parser.add_argument(
        "--test-share",
        type=float,
        default=0.20,
        help="Share of solved parameterizations reserved for final test metrics.",
    )
    parser.add_argument(
        "--tuning-share",
        type=float,
        default=0.20,
        help="Share of remaining parameterizations used to choose ridge strength.",
    )
    parser.add_argument(
        "--model-split-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file freezing train/tuning/test model IDs. By default "
            "OUTPUT_DIR/model_split.json is created on the first run and reused."
        ),
    )
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stage messages and metric summaries.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable tqdm progress bars (stage messages remain unless --quiet).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.mpc_pair_share < 1.0:
        parser.error("--mpc-pair-share must be in [0, 1).")
    if not 0.0 <= args.mpc_cross_zero_share <= 1.0:
        parser.error("--mpc-cross-zero-share must be in [0, 1].")
    if args.spline_knots < 4:
        parser.error("--spline-knots must be at least 4.")
    if args.mpc_loss_weight < 0:
        parser.error("--mpc-loss-weight must be nonnegative.")
    if args.mpc_difference_weight is not None and args.mpc_difference_weight < 0:
        parser.error("--mpc-difference-weight must be nonnegative.")
    if args.mpc_objective_max_pairs <= 0:
        parser.error("--mpc-objective-max-pairs must be positive.")
    if args.mpc_objective_max_iter <= 0 or args.mpc_objective_tol <= 0:
        parser.error("MPC objective solver controls must be positive.")
    if not 0.0 <= args.mpc_objective_trim_quantile < 0.5:
        parser.error("--mpc-objective-trim-quantile must be in [0, 0.5).")
    if args.mpc_stratify_liquid_bins < 1:
        parser.error("--mpc-stratify-liquid-bins must be at least one.")
    if args.mpc_stratify_illiquid_bins < 1:
        parser.error("--mpc-stratify-illiquid-bins must be at least one.")
    if args.mpc_stratify_target_bins < 0:
        parser.error("--mpc-stratify-target-bins must be nonnegative.")
    if args.parameter_selection_min_mpc_r2_gain < 0:
        parser.error("--parameter-selection-min-mpc-r2-gain must be nonnegative.")
    if args.parameter_selection_min_level_r2_gain < 0:
        parser.error("--parameter-selection-min-level-r2-gain must be nonnegative.")
    if args.parameter_selection_max_mpc_r2_loss_for_level_term < 0:
        parser.error(
            "--parameter-selection-max-mpc-r2-loss-for-level-term must be nonnegative."
        )
    if not 0.0 <= args.parameter_selection_min_model_win_share <= 1.0:
        parser.error("--parameter-selection-min-model-win-share must be in [0,1].")
    if not 0.0 <= args.parameter_selection_fdr <= 1.0:
        parser.error("--parameter-selection-fdr must be in [0,1].")
    if not 0.0 < args.parameter_selection_bootstrap_confidence < 1.0:
        parser.error("--parameter-selection-bootstrap-confidence must lie in (0,1).")
    if args.parameter_selection_bootstrap_reps < 0:
        parser.error("--parameter-selection-bootstrap-reps must be nonnegative.")
    if args.parameter_selection_max_terms < 0:
        parser.error("--parameter-selection-max-terms must be nonnegative.")
    if args.n_jobs < 1:
        parser.error("--n-jobs must be at least one.")
    if args.fit_threads < 1:
        parser.error("--fit-threads must be at least one.")
    shares = args.poor_htm_share + args.wealthy_htm_share + args.broad_support_share
    if shares > 1.0 + 1.0e-12:
        parser.error(
            "Poor HTM, wealthy HTM, and broad-support shares cannot sum to more than one."
        )
    if (
        args.liquid_change_loss_weight < 0
        or args.deposit_reconciliation_loss_weight < 0
    ):
        parser.error("Reconciliation loss weights must be nonnegative.")
    if args.liquid_outflow_weight is not None and not (
        0.0 <= args.liquid_outflow_weight <= 1.0
    ):
        parser.error("--liquid-outflow-weight must be in [0,1].")

    overrides = {}
    if args.key_overrides_json is not None:
        with args.key_overrides_json.open("r", encoding="utf-8") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict):
            raise ValueError("--key-overrides-json must contain a JSON object.")

    mpc_check_sizes = _parse_float_list(args.mpc_check_sizes)
    if not mpc_check_sizes:
        parser.error("--mpc-check-sizes must contain at least one positive value.")
    mpc_difference_weight_grid = _parse_nonnegative_float_list(
        args.mpc_difference_weight_grid
    )
    if not mpc_difference_weight_grid:
        parser.error("--mpc-difference-weight-grid must contain at least one value.")
    ridge_alphas = _parse_float_list(args.ridge_alphas)
    if not ridge_alphas:
        parser.error("--ridge-alphas must contain at least one positive value.")
    mpc_stratify_age_bins = _parse_float_list(args.mpc_stratify_age_bins)
    if not mpc_stratify_age_bins:
        parser.error("--mpc-stratify-age-bins must contain at least one value.")
    spline_column_n_knots = _parse_column_int_terms(args.spline_column_knots)
    spline_knot_strategies = _parse_column_string_terms(args.spline_knot_strategies)
    spline_knot_powers = _parse_column_float_terms(args.spline_knot_powers)
    spline_tensor_pairs = _parse_tensor_pairs(args.spline_tensor_pairs)
    spline_tensor_triples = _parse_tensor_triples(args.spline_tensor_triples)
    spline_categorical_tensor_pairs = _parse_tensor_triples(
        args.spline_categorical_tensor_pairs
    )
    spline_tensor_parameter_terms = _parse_tensor_triples(
        args.spline_tensor_parameter_terms
    )
    parameter_power_terms = _parse_parameter_power_terms(args.parameter_power_terms)
    spline_parameter_power_terms = _parse_spline_parameter_power_terms(
        args.spline_parameter_power_terms
    )
    spline_parameter_pair_terms = _parse_spline_parameter_pair_terms(
        args.spline_parameter_pair_terms
    )
    parameter_selection_powers = _parse_int_list(
        args.parameter_selection_powers,
        minimum=2,
    )
    if not parameter_selection_powers:
        parser.error(
            "--parameter-selection-powers must contain at least one integer >= 2."
        )
    categorical_state_power_terms = _parse_categorical_state_power_terms(
        args.categorical_state_power_terms
    )

    population_spec = PopulationSamplingSpec(
        poor_hand_to_mouth_share=args.poor_htm_share,
        wealthy_hand_to_mouth_share=args.wealthy_htm_share,
        broad_support_share=args.broad_support_share,
        nonemployment_share=args.nonemployment_share,
        negative_liquid_share=args.negative_liquid_share,
        borrowing_constraint_anchor_share=args.constraint_anchor_share,
        working_age_only=not args.include_retired,
        mpc_cross_zero_share=args.mpc_cross_zero_share,
        poor_liquid_ratio_median=args.poor_liquid_ratio_median,
        poor_illiquid_ratio_median=args.poor_illiquid_ratio_median,
        wealthy_liquid_ratio_median=args.wealthy_liquid_ratio_median,
        wealthy_illiquid_ratio_median=args.wealthy_illiquid_ratio_median,
        regular_liquid_ratio_median=args.regular_liquid_ratio_median,
        regular_illiquid_ratio_median=args.regular_illiquid_ratio_median,
        liquid_ratio_log_sd=args.liquid_ratio_log_sd,
        illiquid_ratio_log_sd=args.illiquid_ratio_log_sd,
    )
    population_spec.validate()

    schema = GridSchema(
        policy_layout=args.policy_layout,
        key_overrides=overrides,
        money_units=args.money_units,
        require_compact_policies=args.require_compact_policies,
    )
    with threadpool_limits(limits=int(args.fit_threads)):
        paths = train_from_saved_grids(
            args.model_root,
            args.output_dir,
            manifest_path=args.manifest,
            schema=schema,
            rows_per_model=args.rows_per_model,
            max_total_rows=args.max_total_rows,
            max_models=args.max_models,
            sampling_mode=args.sampling_mode,
            population_spec=population_spec,
            n_jobs=args.n_jobs,
            mpc_pair_share=args.mpc_pair_share,
            mpc_check_sizes=mpc_check_sizes,
            parameter_include=_parse_list(args.parameter_include),
            parameter_exclude=_parse_list(args.parameter_exclude) or (),
            categorical_parameter_include=_parse_list(
                args.categorical_parameter_include
            ),
            categorical_parameter_exclude=(
                _parse_list(args.categorical_parameter_exclude) or ()
            ),
            max_solver_distance=args.max_solver_distance,
            max_relative_distance=args.max_relative_distance,
            max_hjb_residual=args.max_hjb_residual,
            model_type=args.model_type,
            polynomial_degree=args.polynomial_degree,
            spline_columns=_parse_list(args.spline_columns)
            or list(DEFAULT_SPLINE_COLUMNS),
            spline_n_knots=args.spline_knots,
            spline_column_n_knots=spline_column_n_knots,
            spline_knot_strategies=spline_knot_strategies,
            spline_knot_powers=spline_knot_powers,
            spline_tensor_pairs=spline_tensor_pairs,
            spline_tensor_triples=spline_tensor_triples,
            spline_categorical_tensor_pairs=spline_categorical_tensor_pairs,
            spline_tensor_parameter_terms=spline_tensor_parameter_terms,
            parameter_power_terms=parameter_power_terms,
            spline_parameter_power_terms=spline_parameter_power_terms,
            spline_parameter_pair_terms=spline_parameter_pair_terms,
            auto_select_parameter_terms=args.auto_select_parameter_terms,
            parameter_selection_spline_columns=(
                _parse_list(args.parameter_selection_spline_columns)
                or list(DEFAULT_PARAMETER_SELECTION_SPLINE_COLUMNS)
            ),
            parameter_selection_powers=parameter_selection_powers,
            parameter_selection_parameter_patterns=(
                _parse_list(args.parameter_selection_patterns) or ()
            ),
            parameter_selection_exclude_patterns=(
                _parse_list(args.parameter_selection_exclude_patterns) or ()
            ),
            parameter_selection_include_parameter_powers=(
                args.parameter_selection_include_parameter_powers
            ),
            parameter_selection_max_train_rows=(
                args.parameter_selection_max_train_rows
            ),
            parameter_selection_max_tune_rows=(args.parameter_selection_max_tune_rows),
            parameter_selection_min_mpc_r2_gain=(
                args.parameter_selection_min_mpc_r2_gain
            ),
            parameter_selection_min_level_r2_gain=(
                args.parameter_selection_min_level_r2_gain
            ),
            parameter_selection_max_mpc_r2_loss_for_level_term=(
                args.parameter_selection_max_mpc_r2_loss_for_level_term
            ),
            parameter_selection_min_model_win_share=(
                args.parameter_selection_min_model_win_share
            ),
            parameter_selection_fdr=args.parameter_selection_fdr,
            parameter_selection_bootstrap_reps=(
                args.parameter_selection_bootstrap_reps
            ),
            parameter_selection_bootstrap_confidence=(
                args.parameter_selection_bootstrap_confidence
            ),
            parameter_selection_max_terms=args.parameter_selection_max_terms,
            categorical_state_power_terms=categorical_state_power_terms,
            categorical_slope_exclude_patterns=(
                _parse_list(args.categorical_slope_exclude_patterns) or ()
            ),
            spline_categorical_exclude_patterns=(
                _parse_list(args.spline_categorical_exclude_patterns) or ()
            ),
            rff_components=args.rff_components,
            ridge_alphas=ridge_alphas,
            target_parameterization=args.target_parameterization,
            consumption_transform=args.consumption_transform,
            deposit_transform=args.deposit_transform,
            liquid_outflow_transform=args.liquid_outflow_transform,
            mpc_loss_weight=args.mpc_loss_weight,
            mpc_difference_weight=args.mpc_difference_weight,
            mpc_difference_weight_grid=mpc_difference_weight_grid,
            mpc_objective_max_pairs=args.mpc_objective_max_pairs,
            mpc_objective_max_iter=args.mpc_objective_max_iter,
            mpc_objective_tol=args.mpc_objective_tol,
            mpc_objective_trim_quantile=args.mpc_objective_trim_quantile,
            mpc_stratify_age_bins=mpc_stratify_age_bins,
            mpc_stratify_liquid_bins=args.mpc_stratify_liquid_bins,
            mpc_stratify_illiquid_bins=args.mpc_stratify_illiquid_bins,
            mpc_stratify_target_bins=args.mpc_stratify_target_bins,
            liquid_change_loss_weight=args.liquid_change_loss_weight,
            deposit_reconciliation_loss_weight=(
                args.deposit_reconciliation_loss_weight
            ),
            liquid_outflow_weight=args.liquid_outflow_weight,
            state_columns=_parse_list(args.state_columns)
            or list(DEFAULT_STATE_COLUMNS),
            engineered_state_columns=(
                _parse_list(args.engineered_state_columns)
                or list(DEFAULT_ENGINEERED_STATE_COLUMNS)
            ),
            redundant_state_columns=(
                _parse_list(args.redundant_state_columns)
                or list(DEFAULT_REDUNDANT_STATE_COLUMNS)
            ),
            categorical_columns=(
                _parse_list(args.categorical_columns)
                or list(DEFAULT_CATEGORICAL_STATE_COLUMNS)
            ),
            working_age_only=not args.include_retired,
            validation_share=args.test_share,
            tuning_share=args.tuning_share,
            model_split_path=args.model_split_file,
            random_state=args.random_state,
            verbose=not args.quiet,
            show_progress=not args.no_progress_bar,
        )

    print("\nCreated:")
    for key, path in paths.items():
        print(f"  {key:24s} {path}")


if __name__ == "__main__":
    main()
