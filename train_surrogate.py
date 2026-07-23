"""Command-line entry point for fitting an MPC-aware HA policy surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ha_policy_surrogate import (
    DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS,
    DEFAULT_CATEGORICAL_STATE_COLUMNS,
    DEFAULT_ENGINEERED_STATE_COLUMNS,
    DEFAULT_REDUNDANT_STATE_COLUMNS,
    DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS,
    DEFAULT_SPLINE_COLUMNS,
    DEFAULT_SPLINE_TENSOR_PAIRS,
    DEFAULT_STATE_COLUMNS,
    GridSchema,
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

    parser.add_argument("--rows-per-model", type=int, default=2_000)
    parser.add_argument("--max-total-rows", type=int, default=250_000)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument(
        "--mpc-pair-share",
        type=float,
        default=0.40,
        help=(
            "Share of sampled rows devoted to baseline/treated liquid-asset "
            "pairs. The remaining rows provide broad policy-level coverage."
        ),
    )
    parser.add_argument(
        "--mpc-check-sizes",
        default="0.02,0.05,0.10,0.188679,0.20,0.50",
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
            "Comma-separated engineered states receiving a quantile-knot "
            "B-spline basis in polynomial mode."
        ),
    )
    parser.add_argument(
        "--spline-knots",
        type=int,
        default=6,
        help="Number of quantile knots in each MPC-oriented spline.",
    )
    parser.add_argument(
        "--spline-tensor-pairs",
        default=",".join(f"{left}:{right}" for left, right in DEFAULT_SPLINE_TENSOR_PAIRS),
        help=(
            "Comma-separated left:right spline pairs. The default lets the "
            "liquid-wealth slope vary smoothly with years to the terminal age."
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
        default="0.0001,0.001,0.01,0.1,1,10,100",
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
        default=1.0,
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
        default="0,0.05,0.10,0.25,0.50,1,2",
        help=(
            "Comma-separated candidate weights for directly targeting MPCs. "
            "Used only when --mpc-difference-weight is omitted."
        ),
    )
    parser.add_argument(
        "--mpc-objective-max-pairs",
        type=int,
        default=250_000,
        help=(
            "Maximum number of lifecycle- and wealth-stratified training MPC "
            "pairs used in each joint consumption/MPC solve."
        ),
    )
    parser.add_argument(
        "--mpc-objective-max-iter",
        type=int,
        default=200,
        help="Maximum LSMR iterations for the joint consumption/MPC regression.",
    )
    parser.add_argument(
        "--mpc-objective-tol",
        type=float,
        default=1.0e-6,
        help="Absolute and relative LSMR tolerance for the joint MPC objective.",
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
        default="45,65,70,75",
        help=(
            "Comma-separated age boundaries used when retaining MPC pairs. "
            "The default deliberately separates early retirement and terminal ages."
        ),
    )
    parser.add_argument(
        "--mpc-stratify-liquid-bins",
        type=int,
        default=4,
        help="Number of liquid-slack quantile bins used in MPC-pair stratification.",
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
        parser.error(
            "--mpc-difference-weight-grid must contain at least one value."
        )
    ridge_alphas = _parse_float_list(args.ridge_alphas)
    if not ridge_alphas:
        parser.error("--ridge-alphas must contain at least one positive value.")
    mpc_stratify_age_bins = _parse_float_list(args.mpc_stratify_age_bins)
    if not mpc_stratify_age_bins:
        parser.error("--mpc-stratify-age-bins must contain at least one value.")
    spline_tensor_pairs = _parse_tensor_pairs(args.spline_tensor_pairs)

    schema = GridSchema(
        policy_layout=args.policy_layout,
        key_overrides=overrides,
        money_units=args.money_units,
        require_compact_policies=args.require_compact_policies,
    )
    paths = train_from_saved_grids(
        args.model_root,
        args.output_dir,
        manifest_path=args.manifest,
        schema=schema,
        rows_per_model=args.rows_per_model,
        max_total_rows=args.max_total_rows,
        max_models=args.max_models,
        mpc_pair_share=args.mpc_pair_share,
        mpc_check_sizes=mpc_check_sizes,
        parameter_include=_parse_list(args.parameter_include),
        parameter_exclude=_parse_list(args.parameter_exclude) or (),
        categorical_parameter_include=_parse_list(args.categorical_parameter_include),
        categorical_parameter_exclude=(
            _parse_list(args.categorical_parameter_exclude) or ()
        ),
        max_solver_distance=args.max_solver_distance,
        model_type=args.model_type,
        polynomial_degree=args.polynomial_degree,
        spline_columns=_parse_list(args.spline_columns) or list(DEFAULT_SPLINE_COLUMNS),
        spline_n_knots=args.spline_knots,
        spline_tensor_pairs=spline_tensor_pairs,
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
        liquid_change_loss_weight=args.liquid_change_loss_weight,
        deposit_reconciliation_loss_weight=(
            args.deposit_reconciliation_loss_weight
        ),
        liquid_outflow_weight=args.liquid_outflow_weight,
        state_columns=_parse_list(args.state_columns) or list(DEFAULT_STATE_COLUMNS),
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
        validation_share=args.test_share,
        tuning_share=args.tuning_share,
        random_state=args.random_state,
        verbose=not args.quiet,
        show_progress=not args.no_progress_bar,
    )

    print("\nCreated:")
    for key, path in paths.items():
        print(f"  {key:24s} {path}")


if __name__ == "__main__":
    main()
