

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import argparse

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.optimize import brentq
from scipy.sparse import linalg as spla

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover

    def tqdm(x=None, **kwargs):
        return x if x is not None else range(0)


EPS = 1.0e-12
VERY_SMALL_DERIVATIVE = 1.0e-10


# =============================================================================
# Configuration objects
# =============================================================================


@dataclass(frozen=True)
class HAConfig:
    """Core configuration for the lifecycle two-asset HA model.

    Monetary quantities are in model units. With synthetic/no-data policy
    functions, model units usually mean ``mean annual income = 1``. With an
    empirical panel, ``money_scale`` is estimated and saved; the loader divides
    input dollars by this scale and converts predictions back to dollars.
    """

    # Lifecycle.
    min_age: int = 25
    retirement_age: int = 63
    max_age: int = 82

    # Preferences.
    discount_factor: float = 0.96
    risk_aversion: float = 4.0
    consumption_floor: float = 1.0e-8
    liquid_service_flow: float = 0.0
    illiquid_service_flow: float = 0.0

    # Annual real returns. These are converted internally to continuous-time
    # rates with log(1+r).
    liquid_interest_rate: float = 0.01
    borrowing_interest_rate: float = 0.08
    illiquid_interest_rate: float = 0.035

    # Borrowing and grids.
    allow_borrowing: bool = False
    borrowing_limit: float = 0.0
    income_grid_size: int = 8
    liquid_grid_size: int = 80
    illiquid_grid_size: int = 40
    liquid_max: float = 4.0
    illiquid_max: float = 12.0
    dense_liquid_upper: float = 0.75
    illiquid_min: float = 1.0e-6

    # Continuous-time transfer cost between liquid and illiquid assets:
    # chi0 |d| + 0.5 chi1 d^2 / max(a, illiquid_cost_floor).
    ct_time_step: float = 1.0
    ct_discount_rate: Optional[float] = None
    ct_linear_adjustment_cost: float = 0.03
    ct_convex_adjustment_cost: float = 2.0
    ct_automatic_illiquid_income_share: float = 0.0
    ct_illiquid_cost_floor: float = 1.0e-6

    # HJB iteration controls. The Delta schedule follows the tentative-guess
    # algorithm in Sabet-Schneider's solveHousehold.m.
    ct_max_iterations: int = 200
    ct_convergence_tol: float = 1.0e-7
    ct_relative_convergence_tol: float = 1.0e-8
    ct_hjb_residual_tol: float = 1.0e-7
    ct_delta_min: float = 1.0e-6
    ct_delta_base: float = 0.5
    ct_delta_max: float = 1.0e8
    ct_delta_backoff: float = 0.2
    ct_slow_iterations: int = 20
    ct_max_restarts: int = 50
    ct_regime_stability_iterations: int = 2
    ct_regime_change_tol: float = 0.0

    # Numerical validation. These are rejection tolerances, not devices for
    # silently changing policies or derivatives.
    ct_root_tol: float = 1.0e-12
    ct_foc_tol: float = 1.0e-6
    ct_min_marginal_value: float = 1.0e-14
    ct_drift_tolerance: float = 1.0e-12
    ct_zero_drift_tolerance: float = 1.0e-8
    ct_state_constraint_tolerance: float = 1.0e-8
    ct_generator_tolerance: float = 1.0e-9
    ct_max_abs_policy: float = 1.0e8
    ct_residual_growth_restart: float = 1.0e4
    ct_validate_generator: bool = True

    # The reference code solves a stationary problem and imposes rho above the
    # saving returns. The finite-horizon lifecycle model does not require that
    # restriction; set this flag only when reproducing the stationary problem.
    ct_enforce_stationary_conditions: bool = False

    # Retained only so older saved metadata can still be loaded. Updated code
    # never clips primitive policies or drifts independently. If supplied, this
    # value is treated as a hard rejection bound.
    ct_policy_drift_clip: Optional[float] = None

    # Data columns used only when estimating empirical policy functions or
    # predicting from a dataframe.
    id_col: str = "household_id"
    year_col: str = "year"
    age_col: str = "age"
    group_col: str = "group"
    education_col: str = "education"
    employed_col: str = "employed"
    income_state_col: str = "income_state"
    employment_state_col: str = "employment_state"
    liquid_col: str = "liquid_assets"
    illiquid_col: str = "illiquid_assets"
    current_income_col: str = "current_income"
    gross_income_col: str = "gross_income"
    labor_income_col: str = "labor_income"
    potential_labor_col: str = "potential_labor_income"
    unemployment_col: str = "unemployment_benefits"
    pension_col: str = "pension_income"
    taxes_col: str = "taxes"

    # Empirical estimation defaults.
    normalize_empirical_money: bool = True
    empirical_transition_smoothing: float = 1.0e-3
    empirical_min_cell_count: int = 25
    seed: int = 12345

    @property
    def ages(self) -> np.ndarray:
        return np.arange(self.min_age, self.max_age + 1, dtype=np.int64)

    @property
    def horizon(self) -> int:
        return self.max_age - self.min_age + 1

    def age_to_period(self, age: np.ndarray | int | float) -> np.ndarray | int:
        out = np.asarray(age) - self.min_age
        out = np.clip(out, 0, self.horizon - 1).astype(np.int64)
        return int(out) if np.isscalar(age) else out

    def period_to_age(self, period: np.ndarray | int | float) -> np.ndarray | int:
        out = self.min_age + np.asarray(period)
        return int(out) if np.isscalar(period) else out.astype(np.int64)

    def validate(self) -> None:
        if self.max_age < self.min_age:
            raise ValueError("max_age must be at least min_age")
        if self.retirement_age < self.min_age:
            raise ValueError("retirement_age must be at least min_age")
        if self.income_grid_size < 2:
            raise ValueError("income_grid_size must be at least 2")
        if self.liquid_grid_size < 5:
            raise ValueError("liquid_grid_size must be at least 5")
        if self.illiquid_grid_size < 5:
            raise ValueError("illiquid_grid_size must be at least 5")
        liquid_lower = self.borrowing_limit if self.allow_borrowing else 0.0
        if liquid_lower > 0.0:
            raise ValueError("borrowing_limit must be nonpositive")
        if self.liquid_max <= 0.0 or not liquid_lower < self.liquid_max:
            raise ValueError(
                "The liquid grid requires a positive upper bound above its lower bound"
            )
        if self.illiquid_min <= 0.0 or self.illiquid_max <= self.illiquid_min:
            raise ValueError("The illiquid grid must have 0 < illiquid_min < illiquid_max")
        if self.dense_liquid_upper <= 0.0:
            raise ValueError("dense_liquid_upper must be positive")
        if self.consumption_floor <= 0.0:
            raise ValueError("consumption_floor must be strictly positive")
        if self.ct_time_step <= 0.0:
            raise ValueError("ct_time_step must be strictly positive")
        if self.ct_convex_adjustment_cost <= 0.0:
            raise ValueError("ct_convex_adjustment_cost must be strictly positive")
        if self.ct_linear_adjustment_cost < 0.0 or self.ct_linear_adjustment_cost >= 1.0:
            raise ValueError("ct_linear_adjustment_cost must be in [0, 1)")
        if not 0.0 <= self.ct_automatic_illiquid_income_share <= 1.0:
            raise ValueError("ct_automatic_illiquid_income_share must be in [0,1]")
        if self.ct_illiquid_cost_floor <= 0.0:
            raise ValueError("ct_illiquid_cost_floor must be strictly positive")
        if self.illiquid_min + 10.0 * EPS < self.ct_illiquid_cost_floor:
            raise ValueError(
                "illiquid_min must be at least ct_illiquid_cost_floor so the grid "
                "matches the adjustment-cost formula used by the solver"
            )
        if self.discount_factor <= 0.0 or self.discount_factor >= 1.0:
            raise ValueError("discount_factor must be in (0, 1)")
        if self.risk_aversion <= 0.0:
            raise ValueError("risk_aversion must be positive")
        if self.ct_max_iterations <= self.ct_slow_iterations + 2:
            raise ValueError(
                "ct_max_iterations must exceed ct_slow_iterations by more than two; "
                "otherwise a Sabet-Schneider recovery block cannot finish"
            )
        if (
            self.ct_delta_min <= 0.0
            or self.ct_delta_base < self.ct_delta_min
            or self.ct_delta_max < self.ct_delta_base
        ):
            raise ValueError("Require 0 < ct_delta_min <= ct_delta_base <= ct_delta_max")
        if not 0.0 < self.ct_delta_backoff < 1.0:
            raise ValueError("ct_delta_backoff must lie strictly between zero and one")
        if self.ct_convergence_tol <= 0.0 or self.ct_relative_convergence_tol < 0.0:
            raise ValueError("HJB value-function convergence tolerances must be nonnegative")
        if self.ct_hjb_residual_tol <= 0.0:
            raise ValueError("ct_hjb_residual_tol must be strictly positive")
        if self.ct_max_restarts < 0 or self.ct_regime_stability_iterations < 0:
            raise ValueError("Restart and regime-stability counts must be nonnegative")
        for name in (
            "ct_root_tol",
            "ct_foc_tol",
            "ct_min_marginal_value",
            "ct_drift_tolerance",
            "ct_zero_drift_tolerance",
            "ct_state_constraint_tolerance",
            "ct_generator_tolerance",
            "ct_max_abs_policy",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be strictly positive")


@dataclass(frozen=True)
class IncomeProcessSpec:
    """Parametric no-data labor-income process.

    The log-income state is an AR(1), discretized by Rouwenhorst.

    When stationary_log_income_std is provided, innovation_std is derived as

        innovation_std
        = stationary_log_income_std * sqrt(1 - persistence**2).

    Otherwise, innovation_std is interpreted directly as the conditional
    innovation standard deviation.
    """

    name: str = "income_rho095_sd050"
    kind: str = "rouwenhorst"
    mean_income: float = 1.0
    persistence: float = 0.95

    # Retained for backward compatibility. This is used only when
    # stationary_log_income_std is None.
    innovation_std: Optional[float] = 0.20

    # Preferred parameterization for the model grid.
    stationary_log_income_std: Optional[float] = None

    age_profile_slope: float = 0.25
    age_profile_quadratic: float = -0.10
    group_log_income_shifters: tuple[float, ...] = (0.0,)


@dataclass(frozen=True)
class EmploymentPolicySpec:
    """Parametric no-data employment transition process.

    Employment state convention: 0 = employed, 1 = not employed/unemployed.
    """

    name: str = "emp_loss0035_find060"
    kind: str = "constant"
    job_loss_probability: float = 0.035
    job_finding_probability: float = 0.60
    loss_income_slope: float = 0.0
    finding_income_slope: float = 0.0
    retirement_absorbing: bool = True
    unemployment_replacement_rate: float = 0.50


@dataclass(frozen=True)
class TaxPolicySpec:
    """Parametric no-data tax and transfer policy."""

    name: str = "tax_progressive_baseline"
    kind: str = "progressive"  # none, flat, progressive
    flat_rate: float = 0.12
    payroll_rate: float = 0.08
    progressive_rate: float = 0.04
    progressive_exponent: float = 1.35
    deduction: float = 0.20
    tax_cap: float = 0.95

    # Government transfer guarantees this level of disposable income.
    # Zero disables the floor.
    minimum_after_tax_income: float = 0.0


@dataclass(frozen=True)
class PensionPolicySpec:
    """Parametric no-data pension policy."""

    name: str = "pension_rr045"
    kind: str = "replacement_rate"
    replacement_rate: float = 0.45
    pension_taxable: bool = True


@dataclass(frozen=True)
class PolicyFunctionSpec:
    """Complete exogenous policy-function specification.

    source can be:
    - "parametric": build from IncomeProcessSpec, EmploymentPolicySpec,
      TaxPolicySpec, and PensionPolicySpec.
    - "bundle": load arrays from ``bundle_path``.
    """

    name: str
    source: str = "parametric"
    income: IncomeProcessSpec = field(default_factory=IncomeProcessSpec)
    employment: EmploymentPolicySpec = field(default_factory=EmploymentPolicySpec)
    tax: TaxPolicySpec = field(default_factory=TaxPolicySpec)
    pension: PensionPolicySpec = field(default_factory=PensionPolicySpec)
    group_values: tuple[int, ...] = (0,)
    bundle_path: Optional[str] = None

    def short_name(self) -> str:
        if self.source == "bundle":
            return self.name
        return "__".join(
            [self.income.name, self.employment.name, self.tax.name, self.pension.name]
        )


@dataclass(frozen=True)
class SolverDiagnostics:
    model_id: str
    status: str
    max_iterations: int
    max_distance: float
    max_relative_distance: float = float("nan")
    max_hjb_residual: float = float("nan")
    max_regime_change: float = float("nan")
    total_restarts: int = 0
    total_root_solves: int = 0
    min_feasibility_slack: float = float("nan")
    max_zero_drift_error: float = float("nan")
    max_generator_row_sum: float = float("nan")
    message: str = ""


@dataclass(frozen=True)
class PolicyBundle:
    """Exogenous objects needed by the HA solver.

    Dimensions:
        G = number of groups
        H = number of ages
        K = number of income states
        E = 2 employment states

    Arrays:
        log_income_grid:        [G, K]
        income_transition:      [G, H, K, K]
        employment_transition:  [G, H, K, E, E]
        gross_income:           [G, H, K, E]
        after_tax_income:       [G, H, K, E]
    """

    name: str
    group_values: np.ndarray
    ages: np.ndarray
    log_income_grid: np.ndarray
    income_transition: np.ndarray
    employment_transition: np.ndarray
    gross_income: np.ndarray
    after_tax_income: np.ndarray
    money_scale: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def G(self) -> int:
        return int(len(self.group_values))

    @property
    def H(self) -> int:
        return int(len(self.ages))

    @property
    def K(self) -> int:
        return int(self.log_income_grid.shape[1])

    def validate(self) -> None:
        G, H, K = self.G, self.H, self.K
        if self.income_transition.shape != (G, H, K, K):
            raise ValueError("income_transition must have shape [G,H,K,K]")
        if self.employment_transition.shape != (G, H, K, 2, 2):
            raise ValueError("employment_transition must have shape [G,H,K,2,2]")
        if self.gross_income.shape != (G, H, K, 2):
            raise ValueError("gross_income must have shape [G,H,K,2]")
        if self.after_tax_income.shape != (G, H, K, 2):
            raise ValueError("after_tax_income must have shape [G,H,K,2]")
        if not np.all(np.isfinite(self.after_tax_income)):
            raise ValueError("after_tax_income contains non-finite values")
        if np.any(self.after_tax_income <= 0):
            raise ValueError("after_tax_income must be strictly positive")

    def fingerprint(self, n: int = 20) -> str:
        hasher = hashlib.sha1()
        hasher.update(self.name.encode("utf-8"))
        hasher.update(np.asarray([self.money_scale], dtype=np.float64).tobytes())
        hasher.update(json.dumps(dataclass_to_jsonable(self.metadata), sort_keys=True).encode("utf-8"))
        for arr in (
            self.group_values,
            self.ages,
            self.log_income_grid,
            self.income_transition,
            self.employment_transition,
            self.gross_income,
            self.after_tax_income,
        ):
            x = np.ascontiguousarray(arr)
            hasher.update(str(x.dtype).encode("ascii"))
            hasher.update(np.asarray(x.shape, dtype=np.int64).tobytes())
            hasher.update(x.tobytes())
        return hasher.hexdigest()[:n]

    def save(self, path: str | Path) -> Path:
        """Save a policy bundle as metadata.json + arrays.npz."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "money_scale": float(self.money_scale),
            "metadata": self.metadata,
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
        np.savez_compressed(
            path / "arrays.npz",
            group_values=self.group_values,
            ages=self.ages,
            log_income_grid=self.log_income_grid,
            income_transition=self.income_transition.astype(np.float32),
            employment_transition=self.employment_transition.astype(np.float32),
            gross_income=self.gross_income.astype(np.float32),
            after_tax_income=self.after_tax_income.astype(np.float32),
        )
        return path

    @staticmethod
    def load(path: str | Path) -> "PolicyBundle":
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        arr = np.load(path / "arrays.npz", allow_pickle=False)
        bundle = PolicyBundle(
            name=meta.get("name", path.name),
            group_values=arr["group_values"],
            ages=arr["ages"],
            log_income_grid=arr["log_income_grid"],
            income_transition=arr["income_transition"],
            employment_transition=arr["employment_transition"],
            gross_income=arr["gross_income"],
            after_tax_income=arr["after_tax_income"],
            money_scale=float(meta.get("money_scale", 1.0)),
            metadata=meta.get("metadata", {}),
        )
        bundle.validate()
        return bundle


# =============================================================================
# Small utilities
# =============================================================================


def dataclass_to_jsonable(x: Any) -> Any:
    if hasattr(x, "__dataclass_fields__"):
        return {k: dataclass_to_jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, dict):
        return {str(k): dataclass_to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [dataclass_to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def stable_hash(obj: Any, n: int = 12) -> str:
    payload = json.dumps(dataclass_to_jsonable(obj), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:n]


def expand_grid(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    values = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def update_dataclass(obj: Any, updates: dict[str, Any]) -> Any:
    valid = set(obj.__dataclass_fields__)
    bad = [k for k in updates if k not in valid]
    if bad:
        raise ValueError(f"Unknown fields for {type(obj).__name__}: {bad}")
    return replace(obj, **updates)


def safe_log(x: np.ndarray | pd.Series | float, floor: float = EPS) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=float), floor))


def nearest_indices(grid: np.ndarray, x: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    x = np.asarray(x, dtype=float)
    right = np.searchsorted(grid, x, side="left")
    right = np.clip(right, 0, len(grid) - 1)
    left = np.clip(right - 1, 0, len(grid) - 1)
    choose_right = np.abs(grid[right] - x) < np.abs(x - grid[left])
    return np.where(choose_right, right, left).astype(np.int64)


def two_sided_liquid_grid(lower: float, upper: float, size: int, dense_upper: float) -> np.ndarray:
    """Construct exactly ``size`` points, including the bounds and zero.

    The previous constructor duplicated ``dense_upper`` at the join between the
    dense and tail pieces and therefore returned ``size - 1`` points. Exact
    dimensions matter for saved-array shapes and grid-convergence exercises.
    """
    lower = float(lower)
    upper = float(upper)
    if size < 5:
        raise ValueError("liquid grid size must be at least 5")
    if lower > 0.0:
        raise ValueError("liquid grid lower bound must be nonpositive")
    if not lower < upper:
        raise ValueError("liquid grid requires lower < upper")
    if upper <= 0.0:
        raise ValueError("liquid grid upper bound must be positive")

    dense_upper = min(max(float(dense_upper), min(0.02, upper)), upper)
    n_neg = int(round(0.20 * size)) if lower < 0.0 else 0
    n_neg = min(n_neg, size - 3)
    n_nonnegative = size - n_neg

    pieces: list[np.ndarray] = []
    if n_neg > 0:
        # Exclude zero here; it is the first point of the nonnegative block.
        pieces.append(np.linspace(lower, 0.0, n_neg + 1)[:-1])

    if dense_upper >= upper - 10.0 * np.finfo(float).eps * max(1.0, abs(upper)):
        u = np.linspace(0.0, 1.0, n_nonnegative)
        pieces.append(upper * u**1.30)
    else:
        preferred_dense = int(round(0.60 * size))
        n_dense = min(max(preferred_dense, 2), n_nonnegative - 1)
        n_tail = n_nonnegative - n_dense
        u_dense = np.linspace(0.0, 1.0, n_dense)
        pieces.append(dense_upper * u_dense**1.30)

        # Exclude the zero of this local coordinate so ``dense_upper`` appears
        # only once and the final grid has the requested number of points.
        u_tail = np.linspace(0.0, 1.0, n_tail + 1)[1:]
        pieces.append(dense_upper + (upper - dense_upper) * u_tail**2.0)

    grid = np.concatenate(pieces)
    if len(grid) != size or np.any(np.diff(grid) <= 0.0):
        raise RuntimeError(
            f"Liquid-grid construction failed: requested {size} points, obtained {len(grid)}"
        )
    if lower < 0.0 and not np.any(grid == 0.0):
        raise RuntimeError("Borrowing grid must contain zero exactly")
    return grid


def positive_grid(lower: float, upper: float, size: int, curvature: float = 2.5) -> np.ndarray:
    u = np.linspace(0.0, 1.0, size)
    return float(lower) + (float(upper) - float(lower)) * u**curvature


def rouwenhorst(n: int, rho: float, sigma_eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Rouwenhorst discretization of y' = rho y + eps.

    Returns stationary-state log-deviation grid and transition matrix.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    rho = float(np.clip(rho, -0.999, 0.999))
    sigma_eps = max(float(sigma_eps), 1.0e-8)
    p = (1.0 + rho) / 2.0
    q = p
    P = np.array([[p, 1.0 - p], [1.0 - q, q]], dtype=float)
    for m in range(3, n + 1):
        P_old = P
        P = np.zeros((m, m), dtype=float)
        P[:-1, :-1] += p * P_old
        P[:-1, 1:] += (1.0 - p) * P_old
        P[1:, :-1] += (1.0 - q) * P_old
        P[1:, 1:] += q * P_old
        P[1:-1, :] /= 2.0
    sigma_y = sigma_eps / math.sqrt(max(1.0 - rho * rho, 1.0e-8))
    width = math.sqrt(n - 1.0) * sigma_y
    grid = np.linspace(-width, width, n)
    P = P / P.sum(axis=1, keepdims=True)
    return grid, P


def continuous_rate_from_annual_return(r: float) -> float:
    if r <= -0.999999:
        return math.log(1.0e-6)
    return math.log1p(float(r))


def discount_rate_from_beta(beta: float, dt: float, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    return -math.log(float(beta)) / max(float(dt), EPS)


def crra_utility(c: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    c = np.maximum(np.asarray(c, dtype=float), floor)
    if abs(gamma - 1.0) < 1.0e-10:
        return np.log(c)
    return c ** (1.0 - gamma) / (1.0 - gamma)


def inverse_marginal_utility(mu: np.ndarray, gamma: float, floor: float) -> np.ndarray:
    return np.maximum(np.asarray(mu, dtype=float), floor) ** (-1.0 / gamma)


def row_normalize(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.maximum(P, 0.0)
    return P / np.maximum(P.sum(axis=-1, keepdims=True), EPS)

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Return invariant probabilities pi satisfying pi @ P = pi."""
    P = row_normalize(np.asarray(P, dtype=float))
    n = P.shape[0]

    if P.shape != (n, n):
        raise ValueError("Transition matrix must be square")

    # Replace one redundant stationary equation with sum(pi) = 1.
    A = P.T - np.eye(n)
    A[-1, :] = 1.0

    rhs = np.zeros(n, dtype=float)
    rhs[-1] = 1.0

    try:
        pi = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        pi, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    if not np.all(np.isfinite(pi)):
        raise ValueError("Could not compute a finite stationary distribution")

    # Remove only negligible numerical negatives.
    if np.min(pi) < -1.0e-10:
        raise ValueError(
            "Computed stationary distribution contains materially negative probabilities"
        )

    pi = np.maximum(pi, 0.0)
    total = float(pi.sum())

    if total <= 0.0:
        raise ValueError("Computed stationary distribution has zero mass")

    return pi / total


# =============================================================================
# Policy-function construction
# =============================================================================


def make_asset_grids(cfg: HAConfig) -> tuple[np.ndarray, np.ndarray]:
    liquid_lower = cfg.borrowing_limit if cfg.allow_borrowing else 0.0
    b_grid = two_sided_liquid_grid(
        liquid_lower, cfg.liquid_max, cfg.liquid_grid_size, cfg.dense_liquid_upper
    )
    a_grid = positive_grid(cfg.illiquid_min, cfg.illiquid_max, cfg.illiquid_grid_size)
    return b_grid, a_grid


def age_profile(ages: np.ndarray, spec: IncomeProcessSpec) -> np.ndarray:
    x = (np.asarray(ages, dtype=float) - 45.0) / 20.0
    return np.exp(spec.age_profile_slope * x + spec.age_profile_quadratic * x**2)


def compute_taxes(
    gross: np.ndarray,
    labor: np.ndarray,
    pension: np.ndarray,
    tax_spec: TaxPolicySpec,
) -> np.ndarray:
    gross = np.maximum(np.asarray(gross, dtype=float), 0.0)
    labor = np.maximum(np.asarray(labor, dtype=float), 0.0)
    pension = np.maximum(np.asarray(pension, dtype=float), 0.0)

    if tax_spec.kind == "none":
        tax = np.zeros_like(gross)
    elif tax_spec.kind == "flat":
        tax = tax_spec.flat_rate * gross
    elif tax_spec.kind == "progressive":
        taxable = np.maximum(gross - tax_spec.deduction, 0.0)
        tax = tax_spec.payroll_rate * labor + tax_spec.flat_rate * taxable
        tax += tax_spec.progressive_rate * np.power(taxable, tax_spec.progressive_exponent)
    else:
        raise ValueError(f"Unsupported tax policy kind: {tax_spec.kind}")
    return np.minimum(np.maximum(tax, 0.0), tax_spec.tax_cap * np.maximum(gross, EPS))


def build_parametric_policy_bundle(cfg: HAConfig, spec: PolicyFunctionSpec) -> PolicyBundle:
    """Construct a policy bundle without using data."""
    cfg.validate()
    if spec.source != "parametric":
        raise ValueError("build_parametric_policy_bundle requires source='parametric'")

    ages = cfg.ages
    H = cfg.horizon
    K = cfg.income_grid_size
    E = 2
    group_values = np.asarray(spec.group_values, dtype=np.int64)
    G = len(group_values)
    if len(spec.income.group_log_income_shifters) not in {1, G}:
        raise ValueError("group_log_income_shifters must have length 1 or G")
    shifters = np.asarray(spec.income.group_log_income_shifters, dtype=float)
    if len(shifters) == 1:
        shifters = np.repeat(shifters, G)


    rho_z = float(spec.income.persistence)

    if spec.income.stationary_log_income_std is not None:
        stationary_sd = float(spec.income.stationary_log_income_std)

        if stationary_sd <= 0.0:
            raise ValueError("stationary_log_income_std must be positive")

        # Convert stationary dispersion into the conditional innovation SD.
        sigma_eps = stationary_sd * math.sqrt(
            max(1.0 - rho_z**2, EPS)
        )
    else:
        if spec.income.innovation_std is None:
            raise ValueError(
                "Specify either stationary_log_income_std or innovation_std"
            )

        sigma_eps = float(spec.income.innovation_std)

        if sigma_eps <= 0.0:
            raise ValueError("innovation_std must be positive")


    log_dev_grid, Pz_base = rouwenhorst(
        K,
        rho_z,
        sigma_eps,
    )

    # Normalize using the actual invariant probabilities of the Markov chain.
    pi_z = stationary_distribution(Pz_base)

    mean_income_before_shift = float(
        np.dot(pi_z, np.exp(log_dev_grid))
    )

    log_level_shift = (
        math.log(max(spec.income.mean_income, EPS))
        - math.log(max(mean_income_before_shift, EPS))
    )


    log_income_grid = np.zeros((G, K), dtype=float)
    income_transition = np.zeros((G, H, K, K), dtype=float)
    employment_transition = np.zeros((G, H, K, E, E), dtype=float)
    gross_income = np.zeros((G, H, K, E), dtype=float)
    after_tax_income = np.zeros((G, H, K, E), dtype=float)

    prof = age_profile(ages, spec.income)
    retirement_profile = float(np.interp(cfg.retirement_age, ages, prof))

    for g in range(G):
        log_income_grid[g] = log_dev_grid + log_level_shift + shifters[g]
        base_income = np.exp(log_income_grid[g])
        income_transition[g, :, :, :] = Pz_base[None, :, :]

        for h, age in enumerate(ages):
            income_rank = np.linspace(-0.5, 0.5, K)
            loss = spec.employment.job_loss_probability * np.exp(
                spec.employment.loss_income_slope * income_rank
            )
            find = spec.employment.job_finding_probability * np.exp(
                spec.employment.finding_income_slope * income_rank
            )
            loss = np.clip(loss, 0.0, 0.95)
            find = np.clip(find, 0.0, 0.99)

            if age >= cfg.retirement_age and spec.employment.retirement_absorbing:
                employment_transition[g, h, :, 0, :] = np.array([0.0, 1.0])
                employment_transition[g, h, :, 1, :] = np.array([0.0, 1.0])
            else:
                employment_transition[g, h, :, 0, 0] = 1.0 - loss
                employment_transition[g, h, :, 0, 1] = loss
                employment_transition[g, h, :, 1, 0] = find
                employment_transition[g, h, :, 1, 1] = 1.0 - find

            potential = base_income * prof[h]
            retirement_income_base = base_income * retirement_profile

            labor = np.zeros((K, E), dtype=float)
            ui = np.zeros((K, E), dtype=float)
            pension = np.zeros((K, E), dtype=float)

            if age < cfg.retirement_age:
                labor[:, 0] = potential
                ui[:, 1] = spec.employment.unemployment_replacement_rate * potential
            else:
                pension[:, :] = (
                    spec.pension.replacement_rate * retirement_income_base[:, None]
                )

            gross = labor + ui + pension
            tax = compute_taxes(gross, labor, pension, spec.tax)

            market_after_tax_income = gross - tax

            income_floor = max(
                float(spec.tax.minimum_after_tax_income),
                float(cfg.consumption_floor),
            )

            # Explicit transfer needed to reach the guaranteed disposable-income floor.
            minimum_income_transfer = np.maximum(
                income_floor - market_after_tax_income,
                0.0,
            )

            disposable_income = (
                market_after_tax_income
                + minimum_income_transfer
            )

            gross_income[g, h] = gross
            after_tax_income[g, h] = disposable_income

    bundle = PolicyBundle(
        name=spec.short_name(),
        group_values=group_values,
        ages=ages,
        log_income_grid=log_income_grid,
        income_transition=row_normalize(income_transition),
        employment_transition=row_normalize(employment_transition),
        gross_income=np.maximum(gross_income, cfg.consumption_floor),
        after_tax_income=np.maximum(after_tax_income, cfg.consumption_floor),
        money_scale=1.0,
        metadata={"source": "parametric", "spec": dataclass_to_jsonable(spec)},
    )
    bundle.validate()
    return bundle


def standardize_empirical_panel(panel: pd.DataFrame, cfg: HAConfig) -> pd.DataFrame:
    required = [cfg.id_col, cfg.year_col, cfg.age_col, cfg.liquid_col, cfg.illiquid_col]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Empirical panel is missing required columns: {missing}")
    out = panel.copy().sort_values([cfg.id_col, cfg.year_col]).reset_index(drop=True)
    if cfg.group_col not in out.columns:
        if cfg.education_col in out.columns:
            out[cfg.group_col] = out[cfg.education_col]
        else:
            out[cfg.group_col] = 0
    if cfg.gross_income_col not in out.columns:
        if cfg.current_income_col in out.columns:
            out[cfg.gross_income_col] = out[cfg.current_income_col]
        else:
            raise ValueError(
                f"Empirical panel needs {cfg.gross_income_col!r} or {cfg.current_income_col!r}."
            )
    if cfg.current_income_col not in out.columns:
        out[cfg.current_income_col] = out[cfg.gross_income_col]
    if cfg.taxes_col not in out.columns:
        out[cfg.taxes_col] = 0.0
    if cfg.pension_col not in out.columns:
        out[cfg.pension_col] = 0.0
    if cfg.unemployment_col not in out.columns:
        out[cfg.unemployment_col] = 0.0
    if cfg.employed_col not in out.columns:
        if cfg.labor_income_col in out.columns:
            out[cfg.employed_col] = out[cfg.labor_income_col].astype(float) > 0
        else:
            out[cfg.employed_col] = out[cfg.gross_income_col].astype(float) > 0
    if cfg.labor_income_col not in out.columns:
        out[cfg.labor_income_col] = np.where(
            (out[cfg.age_col].astype(int) < cfg.retirement_age)
            & out[cfg.employed_col].astype(bool),
            out[cfg.gross_income_col].astype(float)
            - out[cfg.pension_col].astype(float)
            - out[cfg.unemployment_col].astype(float),
            0.0,
        )
    out[cfg.age_col] = np.clip(out[cfg.age_col].astype(int), cfg.min_age, cfg.max_age)
    return out


def build_empirical_policy_bundle(
    panel: pd.DataFrame,
    cfg: HAConfig,
    *,
    name: str = "empirical_policy_bundle",
) -> PolicyBundle:
    """Estimate policy functions from a household-year panel.

    This intentionally uses transparent cell-count objects rather than the old
    ridge-regression infrastructure. Empty cells fall back to smooth parametric
    defaults based on the empirical income grid.
    """
    cfg.validate()
    df = standardize_empirical_panel(panel, cfg)

    positive_labor = df.loc[df[cfg.labor_income_col].astype(float) > 0, cfg.labor_income_col]
    if positive_labor.empty:
        positive_labor = df[cfg.gross_income_col].clip(lower=1.0)
    money_scale = float(positive_labor.mean()) if cfg.normalize_empirical_money else 1.0
    money_scale = max(money_scale, EPS)

    money_cols = [
        cfg.current_income_col,
        cfg.gross_income_col,
        cfg.labor_income_col,
        cfg.potential_labor_col,
        cfg.unemployment_col,
        cfg.pension_col,
        cfg.taxes_col,
        cfg.liquid_col,
        cfg.illiquid_col,
    ]
    for col in money_cols:
        if col in df.columns:
            df[col] = df[col].astype(float) / money_scale

    groups = np.asarray(sorted(pd.Series(df[cfg.group_col]).dropna().unique()), dtype=np.int64)
    if len(groups) == 0:
        groups = np.array([0], dtype=np.int64)
        df[cfg.group_col] = 0
    group_to_idx = {v: i for i, v in enumerate(groups)}
    df["_group_idx"] = df[cfg.group_col].map(group_to_idx).astype(int)

    # Build labor-income state. Carry positive labor income across zero-labor
    # observations within household, then fall back to the global median.
    log_labor = np.where(
        df[cfg.labor_income_col].astype(float) > 0,
        safe_log(df[cfg.labor_income_col]),
        np.nan,
    )
    df["_log_labor"] = log_labor
    df["_log_labor"] = df.groupby(cfg.id_col)["_log_labor"].transform(lambda s: s.ffill().bfill())
    df["_log_labor"] = df["_log_labor"].fillna(float(np.nanmedian(log_labor)))

    K = cfg.income_grid_size
    H = cfg.horizon
    G = len(groups)
    E = 2
    ages = cfg.ages

    log_income_grid = np.zeros((G, K), dtype=float)
    df["_income_state"] = 0
    for g in range(G):
        x = df.loc[df["_group_idx"] == g, "_log_labor"].to_numpy(dtype=float)
        if len(x) < K:
            x = df["_log_labor"].to_numpy(dtype=float)
        q = np.linspace(0.05, 0.95, K)
        grid = np.quantile(x, q)
        if len(np.unique(np.round(grid, 10))) < K:
            mu = float(np.mean(x))
            sd = float(np.std(x) + 1.0e-4)
            grid = np.linspace(mu - 1.5 * sd, mu + 1.5 * sd, K)
        log_income_grid[g] = grid
        mask = df["_group_idx"] == g
        df.loc[mask, "_income_state"] = nearest_indices(grid, df.loc[mask, "_log_labor"])

    df["_period"] = cfg.age_to_period(df[cfg.age_col].to_numpy(dtype=int))
    df["_emp_state"] = np.where(df[cfg.employed_col].astype(bool), 0, 1).astype(int)

    smoothing = cfg.empirical_transition_smoothing
    income_counts = np.full((G, H, K, K), smoothing, dtype=float)
    employment_counts = np.full((G, H, K, E, E), smoothing, dtype=float)
    global_income_counts = np.full((K, K), smoothing, dtype=float)
    global_employment_counts = np.full((K, E, E), smoothing, dtype=float)

    nxt = df[[cfg.id_col, cfg.year_col, "_income_state", "_emp_state"]].copy()
    nxt[cfg.year_col] = nxt[cfg.year_col] - 1
    nxt = nxt.rename(columns={"_income_state": "_next_income_state", "_emp_state": "_next_emp_state"})
    tr = df.merge(nxt, on=[cfg.id_col, cfg.year_col], how="inner")
    tr = tr[tr["_next_income_state"].notna()].copy()
    for _, row in tr.iterrows():
        g = int(row["_group_idx"])
        h = int(row["_period"])
        z = int(row["_income_state"])
        e = int(row["_emp_state"])
        zp = int(row["_next_income_state"])
        ep = int(row["_next_emp_state"])
        income_counts[g, h, z, zp] += 1.0
        employment_counts[g, h, z, e, ep] += 1.0
        global_income_counts[z, zp] += 1.0
        global_employment_counts[z, e, ep] += 1.0

    income_transition = row_normalize(income_counts)
    employment_transition = row_normalize(employment_counts)
    global_income_transition = row_normalize(global_income_counts)
    global_employment_transition = row_normalize(global_employment_counts)

    # Fallback transition matrices for very sparse cells.
    min_n = cfg.empirical_min_cell_count
    for g in range(G):
        for h in range(H):
            if income_counts[g, h].sum() < min_n:
                income_transition[g, h] = global_income_transition
            for z in range(K):
                for e in range(E):
                    if employment_counts[g, h, z, e].sum() < max(5, min_n // 4):
                        employment_transition[g, h, z, e] = global_employment_transition[z, e]

    # Cell means for gross income and taxes, with parametric fallback.
    fallback_spec = PolicyFunctionSpec(
        name="empirical_fallback",
        source="parametric",
        income=IncomeProcessSpec(name="empirical_income_grid", mean_income=1.0),
        employment=EmploymentPolicySpec(name="empirical_emp_fallback"),
        tax=TaxPolicySpec(name="empirical_tax_fallback"),
        pension=PensionPolicySpec(name="empirical_pension_fallback"),
        group_values=tuple(int(x) for x in groups),
    )
    fallback = build_parametric_policy_bundle(cfg, fallback_spec)
    fallback = PolicyBundle(
        name=fallback.name,
        group_values=groups,
        ages=ages,
        log_income_grid=log_income_grid,
        income_transition=fallback.income_transition,
        employment_transition=fallback.employment_transition,
        gross_income=fallback.gross_income,
        after_tax_income=fallback.after_tax_income,
        money_scale=money_scale,
    )
    gross_income = fallback.gross_income.copy()
    after_tax_income = fallback.after_tax_income.copy()

    df["_after_tax"] = np.maximum(
        df[cfg.gross_income_col].astype(float) - df[cfg.taxes_col].astype(float),
        cfg.consumption_floor,
    )
    grouped = df.groupby(["_group_idx", "_period", "_income_state", "_emp_state"])
    obs = grouped.agg(
        gross=(cfg.gross_income_col, "mean"),
        after_tax=("_after_tax", "mean"),
        n=("_after_tax", "size"),
    ).reset_index()
    for _, row in obs.iterrows():
        if int(row["n"]) >= max(3, cfg.empirical_min_cell_count // 5):
            g = int(row["_group_idx"])
            h = int(row["_period"])
            z = int(row["_income_state"])
            e = int(row["_emp_state"])
            gross_income[g, h, z, e] = max(float(row["gross"]), cfg.consumption_floor)
            after_tax_income[g, h, z, e] = max(float(row["after_tax"]), cfg.consumption_floor)

    bundle = PolicyBundle(
        name=name,
        group_values=groups,
        ages=ages,
        log_income_grid=log_income_grid,
        income_transition=income_transition,
        employment_transition=employment_transition,
        gross_income=gross_income,
        after_tax_income=after_tax_income,
        money_scale=money_scale,
        metadata={
            "source": "empirical_panel",
            "money_scale": money_scale,
            "n_rows": int(len(df)),
            "groups": groups.tolist(),
        },
    )
    bundle.validate()
    return bundle


def load_or_build_policy_bundle(
    cfg: HAConfig,
    spec: PolicyFunctionSpec,
    *,
    empirical_panel: Optional[pd.DataFrame] = None,
) -> PolicyBundle:
    if spec.source == "bundle":
        if spec.bundle_path is None:
            raise ValueError("PolicyFunctionSpec with source='bundle' needs bundle_path")
        return PolicyBundle.load(spec.bundle_path)
    if spec.source == "empirical_panel":
        if empirical_panel is None:
            raise ValueError("PolicyFunctionSpec with source='empirical_panel' needs empirical_panel")
        return build_empirical_policy_bundle(empirical_panel, cfg, name=spec.name)
    if spec.source == "parametric":
        return build_parametric_policy_bundle(cfg, spec)
    raise ValueError(f"Unknown policy source: {spec.source}")


# =============================================================================
# Sabet-Schneider continuous-time HJB solver
# =============================================================================


class HJBSolverError(RuntimeError):
    """Base class for failures in the continuous-time household solver."""


class PolicyUpdateError(HJBSolverError):
    """The current value-function iterate does not imply a valid nested-drift policy."""


class RootBracketError(PolicyUpdateError):
    """A zero-liquid-drift first-order condition was not properly bracketed."""


class StateFeasibilityError(HJBSolverError):
    """No admissible control satisfies the state constraints at one or more states."""


class HJBConvergenceError(HJBSolverError):
    """The nonlinear HJB iteration exhausted its iteration or restart budget."""


@dataclass(frozen=True)
class FeasibilityDiagnostics:
    min_slack: float
    group_index: int
    age_period: int
    income_state: int
    employment_state: int
    illiquid_index: int
    maximizing_deposit: float
    max_feasible_consumption: float


@dataclass(frozen=True)
class PolicyUpdateDiagnostics:
    min_vb: float
    min_va: float
    max_abs_deposit: float
    max_abs_liquid_drift: float
    max_abs_illiquid_drift: float
    max_zero_drift_error: float
    max_foc_residual: float
    root_solves: int
    zero_liquid_share: float
    lower_liquid_constraint_share: float
    upper_liquid_constraint_share: float
    lower_illiquid_constraint_share: float
    upper_illiquid_constraint_share: float


@dataclass(frozen=True)
class PolicyUpdateResult:
    consumption: np.ndarray
    deposit: np.ndarray
    liquid_drift: np.ndarray
    illiquid_drift: np.ndarray
    liquid_regime: np.ndarray
    illiquid_regime: np.ndarray
    diagnostics: PolicyUpdateDiagnostics


@dataclass(frozen=True)
class HJBEvaluation:
    policy: PolicyUpdateResult
    generator: sp.csr_matrix
    utility: np.ndarray
    residual: np.ndarray
    max_abs_residual: float
    normalized_residual: float
    max_generator_row_sum: float


@dataclass(frozen=True)
class HJBPeriodSolution:
    value: np.ndarray
    consumption: np.ndarray
    deposit: np.ndarray
    liquid_drift: np.ndarray
    illiquid_drift: np.ndarray
    iterations: int
    distance: float
    relative_distance: float
    hjb_residual: float
    regime_change: float
    restarts: int
    root_solves: int
    max_zero_drift_error: float
    max_generator_row_sum: float
    last_error: str = ""


def adjustment_cost(
    d: np.ndarray | float,
    a: np.ndarray | float,
    chi0: float,
    chi1: float,
    a_floor: float,
) -> np.ndarray:
    d_arr = np.asarray(d, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    return chi0 * np.abs(d_arr) + 0.5 * chi1 * d_arr**2 / np.maximum(a_arr, a_floor)


def service_flow(b: np.ndarray, a: np.ndarray, cfg: HAConfig) -> np.ndarray:
    return cfg.liquid_service_flow * np.maximum(b, 0.0) + cfg.illiquid_service_flow * np.maximum(a, 0.0)


def b_return_flow(b: np.ndarray, rb_pos: float, rb_neg: float) -> np.ndarray:
    b_arr = np.asarray(b, dtype=float)
    return np.where(b_arr >= 0.0, rb_pos * b_arr, rb_neg * b_arr)


def liquid_drift(
    c: np.ndarray,
    d: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    *,
    rb_pos: float,
    rb_neg: float,
    chi0: float,
    chi1: float,
    xi: float,
    a_floor: float,
) -> np.ndarray:
    return (
        (1.0 - xi) * y
        + b_return_flow(b, rb_pos, rb_neg)
        - d
        - adjustment_cost(d, a, chi0, chi1, a_floor)
        - c
    )


def illiquid_drift(
    d: np.ndarray,
    a: np.ndarray,
    y: np.ndarray,
    *,
    ra: float,
    xi: float,
) -> np.ndarray:
    return ra * a + xi * y + d


def _effective_policy_guard(cfg: HAConfig) -> float:
    """Return a hard validity bound; policies are rejected, never clipped."""
    guard = float(cfg.ct_max_abs_policy)
    legacy = cfg.ct_policy_drift_clip
    if legacy is not None and np.isfinite(float(legacy)) and float(legacy) > 0.0:
        guard = min(guard, float(legacy))
    return guard


def deposit_policy(
    Va: np.ndarray,
    MUc: np.ndarray,
    a: np.ndarray,
    chi0: float,
    chi1: float,
    a_floor: float,
) -> np.ndarray:
    """Sabet-Schneider conditional deposit rule.

    Invalid directional derivatives are represented by NaN and propagate only
    into candidates that are subsequently excluded by the nested-drift masks.
    No marginal value is silently floored.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = Va / MUc
        scale = np.maximum(a, a_floor) / chi1
        return (
            np.minimum(ratio - 1.0 + chi0, 0.0) * scale
            + np.maximum(ratio - 1.0 - chi0, 0.0) * scale
        )


def value_derivatives(
    V: np.ndarray,
    b_grid: np.ndarray,
    a_grid: np.ndarray,
    *,
    min_marginal_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw one-sided derivatives of ``V[K,E,B,A]``.

    Sabet-Schneider rely on positive marginal values. A nonpositive derivative
    indicates that the tentative value-function iterate lies outside the local
    region in which the policy update is valid. Such an iterate is rejected so
    that the tentative-guess algorithm can restart at a slower pseudo-time step.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 4:
        raise ValueError("V must have shape [income, employment, liquid, illiquid]")
    if not np.all(np.isfinite(V)):
        raise PolicyUpdateError("Value function contains non-finite entries")

    K, E, B, A = V.shape
    if B < 2 or A < 2:
        raise ValueError("The continuous-time solver needs at least two points in each asset dimension")

    db = np.diff(np.asarray(b_grid, dtype=float))
    da = np.diff(np.asarray(a_grid, dtype=float))
    if np.any(db <= 0.0) or np.any(da <= 0.0):
        raise ValueError("Asset grids must be strictly increasing")

    vb = (V[:, :, 1:, :] - V[:, :, :-1, :]) / db[None, None, :, None]
    va = (V[:, :, :, 1:] - V[:, :, :, :-1]) / da[None, None, None, :]

    for name, deriv in (("liquid", vb), ("illiquid", va)):
        if not np.all(np.isfinite(deriv)):
            raise PolicyUpdateError(f"{name.capitalize()} marginal values contain non-finite entries")
        bad = deriv <= float(min_marginal_value)
        if np.any(bad):
            idx = tuple(int(i) for i in np.argwhere(bad)[0])
            minimum = float(np.min(deriv))
            raise PolicyUpdateError(
                f"Nonpositive or numerically degenerate {name} marginal value: "
                f"min={minimum:.6e}, first_bad_index={idx}, "
                f"required>{float(min_marginal_value):.3e}"
            )

    VbF = np.full_like(V, np.nan)
    VbB = np.full_like(V, np.nan)
    VaF = np.full_like(V, np.nan)
    VaB = np.full_like(V, np.nan)
    VbF[:, :, :-1, :] = vb
    VbB[:, :, 1:, :] = vb
    VaF[:, :, :, :-1] = va
    VaB[:, :, :, 1:] = va
    return VbF, VbB, VaF, VaB


def _consumption_from_marginal_value(
    marginal_value: np.ndarray,
    service: np.ndarray,
    cfg: HAConfig,
) -> np.ndarray:
    out = np.full_like(marginal_value, np.nan, dtype=float)
    valid = np.isfinite(marginal_value)
    if np.any(valid):
        mu = marginal_value[valid]
        if np.any(mu <= cfg.ct_min_marginal_value):
            raise PolicyUpdateError("Consumption FOC received a nonpositive marginal value")
        with np.errstate(over="raise", invalid="raise"):
            effective_consumption = mu ** (-1.0 / cfg.risk_aversion)
        service_full = np.broadcast_to(service, marginal_value.shape)
        unconstrained = effective_consumption - service_full[valid]
        out[valid] = np.maximum(unconstrained, cfg.consumption_floor)
    return out


def positive_deposit_upper_bound(
    b: float,
    a: float,
    y: float,
    *,
    rb_pos: float,
    rb_neg: float,
    chi0: float,
    chi1: float,
    xi: float,
    a_floor: float,
    c_floor: float,
) -> float:
    """Largest positive deposit compatible with b-dot=0 and c>=c_floor."""
    liquid_flow = (1.0 - xi) * y + (rb_pos if b >= 0.0 else rb_neg) * b
    if liquid_flow <= c_floor:
        return 0.0
    aa = 0.5 * chi1 / max(a, a_floor)
    bb = 1.0 + chi0
    cc = c_floor - liquid_flow
    disc = bb * bb - 4.0 * aa * cc
    if not np.isfinite(disc) or disc <= 0.0:
        return 0.0
    root = (-bb + math.sqrt(disc)) / (2.0 * aa)
    return max(0.0, root * (1.0 - 1.0e-12))


def zero_liquid_consumption(
    d: np.ndarray | float,
    b: np.ndarray | float,
    a: np.ndarray | float,
    y: np.ndarray | float,
    *,
    rb_pos: float,
    rb_neg: float,
    chi0: float,
    chi1: float,
    xi: float,
    a_floor: float,
) -> np.ndarray:
    return (
        (1.0 - xi) * np.asarray(y, dtype=float)
        + b_return_flow(np.asarray(b, dtype=float), rb_pos, rb_neg)
        - np.asarray(d, dtype=float)
        - adjustment_cost(d, a, chi0, chi1, a_floor)
    )


def bdotzero_residual(
    d: float,
    b: float,
    a: float,
    y: float,
    Va: float,
    positive: bool,
    *,
    gamma: float,
    c_floor: float,
    rb_pos: float,
    rb_neg: float,
    chi0: float,
    chi1: float,
    xi: float,
    a_floor: float,
    service: float,
    feasibility_tolerance: float,
) -> float:
    c = float(
        zero_liquid_consumption(
            d,
            b,
            a,
            y,
            rb_pos=rb_pos,
            rb_neg=rb_neg,
            chi0=chi0,
            chi1=chi1,
            xi=xi,
            a_floor=a_floor,
        )
    )
    if c < c_floor - feasibility_tolerance:
        return float("nan")
    c = max(c, c_floor)
    effective_c = c + service
    if not np.isfinite(effective_c) or effective_c <= 0.0:
        return float("nan")
    mu = effective_c ** (-gamma)
    wedge = 1.0 + chi0 * (1.0 if positive else -1.0) + chi1 * d / max(a, a_floor)
    return float(mu * wedge - Va)


def strict_bracketed_root(
    fun,
    lo: float,
    hi: float,
    *,
    root_tol: float,
    foc_tol: float,
    residual_scale: float,
    context: str,
) -> tuple[float, float]:
    """Solve a bracketed scalar FOC or reject the tentative HJB update."""
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    width = hi - lo
    x_scale = max(1.0, abs(lo), abs(hi))
    machine_width = 4.0 * np.finfo(float).eps * x_scale
    if width <= machine_width:
        f0 = float(fun(0.5 * (lo + hi)))
        if not np.isfinite(f0):
            raise RootBracketError(f"Non-finite FOC at a degenerate bracket ({context})")
        scale = max(abs(float(residual_scale)), np.finfo(float).eps)
        scaled = abs(f0) / scale
        if scaled > foc_tol:
            raise RootBracketError(
                f"Degenerate root bracket does not satisfy the FOC ({context}); "
                f"scaled residual={scaled:.3e}"
            )
        return 0.5 * (lo + hi), scaled

    flo = float(fun(lo))
    fhi = float(fun(hi))
    if not np.isfinite(flo) or not np.isfinite(fhi):
        raise RootBracketError(
            f"Non-finite FOC endpoint in bracket [{lo:.6e}, {hi:.6e}] ({context})"
        )
    scale = max(abs(float(residual_scale)), np.finfo(float).eps)
    if abs(flo) / scale <= foc_tol:
        return lo, abs(flo) / scale
    if abs(fhi) / scale <= foc_tol:
        return hi, abs(fhi) / scale
    if np.signbit(flo) == np.signbit(fhi):
        raise RootBracketError(
            f"FOC is not bracketed ({context}): f(lo)={flo:.6e}, f(hi)={fhi:.6e}"
        )

    try:
        # The b-dot-zero brackets can be extremely narrow when a is close to
        # its lower bound. A fixed absolute tolerance (for example 1e-12) can
        # then stop Brent's method several economically meaningful digits away
        # from the root. Scale the absolute tolerance to the bracket width while
        # retaining the user tolerance on ordinary-sized brackets.
        effective_xtol = max(
            np.nextafter(0.0, 1.0),
            min(float(root_tol), max(width, machine_width) * 1.0e-10),
        )
        root = float(
            brentq(
                fun,
                lo,
                hi,
                xtol=effective_xtol,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=200,
            )
        )
    except Exception as exc:
        raise RootBracketError(f"Bracketed root solve failed ({context})") from exc

    fres = float(fun(root))
    scaled = abs(fres) / scale if np.isfinite(fres) else np.inf
    if not np.isfinite(scaled) or scaled > foc_tol:
        raise RootBracketError(
            f"Root solve did not satisfy the FOC ({context}); scaled residual={scaled:.3e}"
        )
    return root, scaled


def bdotzero_policy(
    bdrift_zero: np.ndarray,
    VaF: np.ndarray,
    VaB: np.ndarray,
    b_grid: np.ndarray,
    a_grid: np.ndarray,
    after_tax_income: np.ndarray,
    cfg: HAConfig,
    *,
    rb_pos: float,
    rb_neg: float,
    ra: float,
    chi0: float,
    chi1: float,
    xi: float,
) -> tuple[np.ndarray, int, float]:
    """Faithful Python port of Sabet-Schneider ``bdotzero.m``.

    Unlike the previous implementation, every numerical root must be genuinely
    bracketed and satisfy the FOC. A failed bracket rejects the tentative value
    update, which is how the reference tentative-guess algorithm is designed to
    recover from an iterate outside the local convergence region.
    """
    K, E, B, A = bdrift_zero.shape
    out = np.zeros((K, E, B, A), dtype=float)
    root_solves = 0
    max_foc_residual = 0.0

    for z in range(K):
        for e in range(E):
            y = float(after_tax_income[z, e])
            for ia, a_val in enumerate(a_grid):
                a = float(a_val)
                a_eff = max(a, cfg.ct_illiquid_cost_floor)
                dlower = (chi0 - 1.0) * a_eff / chi1
                for ib, b_val in enumerate(b_grid):
                    if not bool(bdrift_zero[z, e, ib, ia]):
                        continue
                    b = float(b_val)
                    dzerod = -(ra * a + xi * y)
                    serv = float(
                        cfg.liquid_service_flow * max(b, 0.0)
                        + cfg.illiquid_service_flow * max(a, 0.0)
                    )

                    def tester(x: float, Va: float, positive: bool) -> float:
                        if not np.isfinite(Va) or Va <= cfg.ct_min_marginal_value:
                            return float("nan")
                        return bdotzero_residual(
                            x,
                            b,
                            a,
                            y,
                            float(Va),
                            positive,
                            gamma=cfg.risk_aversion,
                            c_floor=cfg.consumption_floor,
                            rb_pos=rb_pos,
                            rb_neg=rb_neg,
                            chi0=chi0,
                            chi1=chi1,
                            xi=xi,
                            a_floor=cfg.ct_illiquid_cost_floor,
                            service=serv,
                            feasibility_tolerance=cfg.ct_state_constraint_tolerance,
                        )

                    def solve_root(lo: float, hi: float, Va: float, positive: bool, label: str) -> float:
                        nonlocal root_solves, max_foc_residual
                        context = f"z={z}, e={e}, ib={ib}, ia={ia}, branch={label}"
                        root, residual = strict_bracketed_root(
                            lambda x: tester(x, Va, positive),
                            lo,
                            hi,
                            root_tol=cfg.ct_root_tol,
                            foc_tol=cfg.ct_foc_tol,
                            residual_scale=max(
                                abs(float(Va)),
                                float(cfg.ct_min_marginal_value),
                            ),
                            context=context,
                        )
                        root_solves += 1
                        max_foc_residual = max(max_foc_residual, float(residual))
                        return root

                    if ia < A - 1:
                        # d > 0 and illiquid up-drift.
                        test_0_pos = tester(0.0, VaF[z, e, ib, ia], True)
                        if np.isfinite(test_0_pos) and test_0_pos < 0.0:
                            dmax = positive_deposit_upper_bound(
                                b,
                                a,
                                y,
                                rb_pos=rb_pos,
                                rb_neg=rb_neg,
                                chi0=chi0,
                                chi1=chi1,
                                xi=xi,
                                a_floor=cfg.ct_illiquid_cost_floor,
                                c_floor=cfg.consumption_floor,
                            )
                            if dmax <= 0.0:
                                raise StateFeasibilityError(
                                    "Zero-liquid-drift positive-deposit branch has no feasible positive deposit "
                                    f"at z={z}, e={e}, ib={ib}, ia={ia}"
                                )
                            out[z, e, ib, ia] = solve_root(
                                0.0,
                                dmax,
                                VaF[z, e, ib, ia],
                                True,
                                "positive_up",
                            )

                        # d <= 0 and illiquid up-drift.
                        elif dzerod < dlower:
                            test_0 = tester(0.0, VaF[z, e, ib, ia], False)
                            test_lower = tester(dlower, VaF[z, e, ib, ia], False)
                            if not np.isfinite(test_0) or not np.isfinite(test_lower):
                                raise PolicyUpdateError(
                                    f"Invalid zero-drift FOC at z={z}, e={e}, ib={ib}, ia={ia}"
                                )
                            if test_0 <= 0.0:
                                out[z, e, ib, ia] = 0.0
                            elif test_lower <= 0.0:
                                out[z, e, ib, ia] = solve_root(
                                    dlower,
                                    0.0,
                                    VaF[z, e, ib, ia],
                                    False,
                                    "negative_up",
                                )
                            else:
                                # With positive Va, the residual at dlower equals -Va.
                                raise PolicyUpdateError(
                                    "Nested-drift withdrawal lower bound has the wrong FOC sign; "
                                    f"z={z}, e={e}, ib={ib}, ia={ia}"
                                )

                        # d <= 0 and illiquid drift can be up, down, or zero.
                        else:
                            test_0 = tester(0.0, VaF[z, e, ib, ia], False)
                            test_zero_f = tester(dzerod, VaF[z, e, ib, ia], False)
                            if not np.isfinite(test_0) or not np.isfinite(test_zero_f):
                                raise PolicyUpdateError(
                                    f"Invalid zero-drift FOC at z={z}, e={e}, ib={ib}, ia={ia}"
                                )
                            if test_0 <= 0.0:
                                out[z, e, ib, ia] = 0.0
                            elif test_zero_f <= 0.0:
                                out[z, e, ib, ia] = solve_root(
                                    dzerod,
                                    0.0,
                                    VaF[z, e, ib, ia],
                                    False,
                                    "negative_up_to_zero",
                                )
                            elif ia == 0:
                                # Lower illiquid state constraint.
                                out[z, e, ib, ia] = dzerod
                            else:
                                test_zero_b = tester(dzerod, VaB[z, e, ib, ia], False)
                                if not np.isfinite(test_zero_b):
                                    raise PolicyUpdateError(
                                        f"Invalid backward illiquid FOC at z={z}, e={e}, ib={ib}, ia={ia}"
                                    )
                                if test_zero_b > 0.0:
                                    test_lower_b = tester(dlower, VaB[z, e, ib, ia], False)
                                    if not np.isfinite(test_lower_b):
                                        raise PolicyUpdateError(
                                            f"Invalid withdrawal-bound FOC at z={z}, e={e}, ib={ib}, ia={ia}"
                                        )
                                    if test_lower_b <= 0.0:
                                        out[z, e, ib, ia] = solve_root(
                                            dlower,
                                            dzerod,
                                            VaB[z, e, ib, ia],
                                            False,
                                            "negative_down",
                                        )
                                    else:
                                        raise PolicyUpdateError(
                                            "Nested-drift withdrawal lower bound has the wrong FOC sign; "
                                            f"z={z}, e={e}, ib={ib}, ia={ia}"
                                        )
                                else:
                                    out[z, e, ib, ia] = dzerod
                    else:
                        # Top illiquid boundary: only down-drift or zero drift.
                        if dzerod > dlower:
                            test_zero_b = tester(dzerod, VaB[z, e, ib, ia], False)
                        else:
                            test_zero_b = float("-inf")
                        if dzerod > dlower and np.isfinite(test_zero_b) and test_zero_b > 0.0:
                            test_lower_b = tester(dlower, VaB[z, e, ib, ia], False)
                            if not np.isfinite(test_lower_b):
                                raise PolicyUpdateError(
                                    f"Invalid top-boundary FOC at z={z}, e={e}, ib={ib}, ia={ia}"
                                )
                            if test_lower_b <= 0.0:
                                out[z, e, ib, ia] = solve_root(
                                    dlower,
                                    dzerod,
                                    VaB[z, e, ib, ia],
                                    False,
                                    "top_negative_down",
                                )
                            else:
                                raise PolicyUpdateError(
                                    "Top-boundary withdrawal lower bound has the wrong FOC sign; "
                                    f"z={z}, e={e}, ib={ib}, ia={ia}"
                                )
                        else:
                            # The state constraint forces zero illiquid drift.
                            out[z, e, ib, ia] = dzerod

    return out, root_solves, max_foc_residual


def _relative_error(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.abs(lhs - rhs) / np.maximum(1.0, np.maximum(np.abs(lhs), np.abs(rhs)))


def policy_update(
    V: np.ndarray,
    after_tax_income: np.ndarray,
    b_grid: np.ndarray,
    a_grid: np.ndarray,
    cfg: HAConfig,
    *,
    rb_pos: float,
    rb_neg: float,
    ra: float,
    chi0: float,
    chi1: float,
    xi: float,
) -> PolicyUpdateResult:
    """Nested-drift policy update, following ``updateHousehold.m``.

    The branch ordering, state-constraint treatment, and zero-liquid-drift
    root problem match the Sabet-Schneider reference implementation. Numerical
    inconsistencies reject the tentative value-function update instead of being
    hidden by derivative floors or independent policy/drift clipping.
    """
    VbF, VbB, VaF, VaB = value_derivatives(
        V,
        b_grid,
        a_grid,
        min_marginal_value=cfg.ct_min_marginal_value,
    )
    y = np.asarray(after_tax_income, dtype=float)[:, :, None, None]
    b = np.asarray(b_grid, dtype=float)[None, None, :, None]
    a = np.asarray(a_grid, dtype=float)[None, None, None, :]
    serv = service_flow(b, a, cfg)

    c_B = _consumption_from_marginal_value(VbB, serv, cfg)
    c_F = _consumption_from_marginal_value(VbF, serv, cfg)

    d_zero = -illiquid_drift(np.zeros_like(V), a, y, ra=ra, xi=xi)

    d_BB = deposit_policy(VaB, VbB, a, chi0, chi1, cfg.ct_illiquid_cost_floor)
    d_BF = deposit_policy(VaF, VbB, a, chi0, chi1, cfg.ct_illiquid_cost_floor)
    d_FB = deposit_policy(VaB, VbF, a, chi0, chi1, cfg.ct_illiquid_cost_floor)
    d_FF = deposit_policy(VaF, VbF, a, chi0, chi1, cfg.ct_illiquid_cost_floor)

    # Illiquid state constraints, exactly as in the reference implementation.
    d_FF[:, :, :, -1] = d_zero[:, :, :, -1]
    d_BF[:, :, :, -1] = d_zero[:, :, :, -1]
    d_FB[:, :, :, 0] = d_zero[:, :, :, 0]
    d_BB[:, :, :, 0] = d_zero[:, :, :, 0]

    drift_tol = float(cfg.ct_drift_tolerance)
    I_BB = d_BB < d_zero - drift_tol
    I_BF = d_BF > d_zero + drift_tol
    I_FF = d_FF > d_zero + drift_tol
    I_FB = d_FB < d_zero - drift_tol

    d_B = np.where(I_BF, d_BF, np.where(I_BB, d_BB, d_zero))
    d_B[:, :, 0, :] = 0.0  # Never selected at the lower liquid boundary.
    sb_B = liquid_drift(
        c_B,
        d_B,
        b,
        a,
        y,
        rb_pos=rb_pos,
        rb_neg=rb_neg,
        chi0=chi0,
        chi1=chi1,
        xi=xi,
        a_floor=cfg.ct_illiquid_cost_floor,
    )
    I_B = np.isfinite(sb_B) & (sb_B < -drift_tol)
    I_B[:, :, 0, :] = False

    d_F = np.where(I_FF, d_FF, np.where(I_FB, d_FB, d_zero))
    d_F[:, :, -1, :] = 0.0  # Never selected at the upper liquid boundary.
    sb_F = liquid_drift(
        c_F,
        d_F,
        b,
        a,
        y,
        rb_pos=rb_pos,
        rb_neg=rb_neg,
        chi0=chi0,
        chi1=chi1,
        xi=xi,
        a_floor=cfg.ct_illiquid_cost_floor,
    )
    I_F = np.isfinite(sb_F) & (sb_F > drift_tol) & (~I_B)
    I_F[:, :, -1, :] = False

    I_0 = ~(I_B | I_F)
    d_0, root_solves, max_foc_residual = bdotzero_policy(
        I_0,
        VaF,
        VaB,
        b_grid,
        a_grid,
        after_tax_income,
        cfg,
        rb_pos=rb_pos,
        rb_neg=rb_neg,
        ra=ra,
        chi0=chi0,
        chi1=chi1,
        xi=xi,
    )
    c_0 = zero_liquid_consumption(
        d_0,
        b,
        a,
        y,
        rb_pos=rb_pos,
        rb_neg=rb_neg,
        chi0=chi0,
        chi1=chi1,
        xi=xi,
        a_floor=cfg.ct_illiquid_cost_floor,
    )
    infeasible_zero = I_0 & (
        c_0 < cfg.consumption_floor - cfg.ct_state_constraint_tolerance
    )
    if np.any(infeasible_zero):
        idx = tuple(int(i) for i in np.argwhere(infeasible_zero)[0])
        raise StateFeasibilityError(
            "The zero-liquid-drift branch requires consumption below its lower bound; "
            f"first_bad_index={idx}, c={float(c_0[idx]):.6e}"
        )
    # Correct only roundoff-sized shortfalls. Drifts are recomputed below, so
    # accounting identities remain exact and any nonzero b drift is diagnosed.
    c_0 = np.where(I_0, np.maximum(c_0, cfg.consumption_floor), c_0)

    c = np.where(I_F, c_F, np.where(I_B, c_B, c_0))
    d = np.where(I_F, d_F, np.where(I_B, d_B, d_0))
    if not np.all(np.isfinite(c)) or not np.all(np.isfinite(d)):
        raise PolicyUpdateError("Nested-drift policy update produced non-finite controls")
    if np.any(c < cfg.consumption_floor - cfg.ct_state_constraint_tolerance):
        raise PolicyUpdateError("Nested-drift policy update violated the consumption lower bound")

    # Primitive controls determine both drifts. Nothing is independently clipped.
    sb = liquid_drift(
        c,
        d,
        b,
        a,
        y,
        rb_pos=rb_pos,
        rb_neg=rb_neg,
        chi0=chi0,
        chi1=chi1,
        xi=xi,
        a_floor=cfg.ct_illiquid_cost_floor,
    )
    sa = illiquid_drift(d, a, y, ra=ra, xi=xi)
    if not np.all(np.isfinite(sb)) or not np.all(np.isfinite(sa)):
        raise PolicyUpdateError("Nested-drift policy update produced non-finite asset drifts")

    guard = _effective_policy_guard(cfg)
    max_control = max(
        float(np.max(np.abs(c))),
        float(np.max(np.abs(d))),
        float(np.max(np.abs(sb))),
        float(np.max(np.abs(sa))),
    )
    if np.isfinite(guard) and max_control > guard:
        raise PolicyUpdateError(
            f"Policy magnitude {max_control:.6e} exceeds the hard validity guard {guard:.6e}; "
            "the policy is rejected rather than clipped"
        )

    state_tol = float(cfg.ct_state_constraint_tolerance)
    boundary_checks = (
        (float(np.min(sb[:, :, 0, :])), -state_tol, "lower liquid"),
        (-float(np.max(sb[:, :, -1, :])), -state_tol, "upper liquid"),
        (float(np.min(sa[:, :, :, 0])), -state_tol, "lower illiquid"),
        (-float(np.max(sa[:, :, :, -1])), -state_tol, "upper illiquid"),
    )
    for inward_measure, lower_bound, name in boundary_checks:
        if inward_measure < lower_bound:
            raise StateFeasibilityError(
                f"Policy points outside the {name} state boundary by {-inward_measure:.6e}"
            )

    zero_error = float(np.max(np.abs(sb[I_0]))) if np.any(I_0) else 0.0
    if zero_error > cfg.ct_zero_drift_tolerance:
        raise PolicyUpdateError(
            f"Zero-liquid-drift policy has max |b-dot|={zero_error:.6e}, "
            f"above tolerance {cfg.ct_zero_drift_tolerance:.3e}"
        )

    # Interior zero-drift consistency check from the Sabet-Schneider appendix:
    # VbF <= u'(c_0+s) <= VbB for a concave value function.
    if len(b_grid) > 2:
        interior = I_0[:, :, 1:-1, :]
        if np.any(interior):
            effective_c0 = c_0[:, :, 1:-1, :] + np.broadcast_to(serv, V.shape)[:, :, 1:-1, :]
            mu0 = effective_c0 ** (-cfg.risk_aversion)
            lower = VbF[:, :, 1:-1, :]
            upper = VbB[:, :, 1:-1, :]
            scale = np.maximum(
                float(cfg.ct_min_marginal_value),
                np.maximum(np.abs(lower), np.abs(upper)),
            )
            violation = np.maximum(lower - mu0, mu0 - upper) / scale
            max_violation = float(np.max(np.where(interior, violation, 0.0)))
            if max_violation > cfg.ct_foc_tol:
                raise PolicyUpdateError(
                    "Zero-liquid-drift marginal utility is not between the forward and backward "
                    f"liquid derivatives; max scaled violation={max_violation:.3e}"
                )

    liquid_regime = np.zeros(V.shape, dtype=np.int8)
    liquid_regime[I_B] = -1
    liquid_regime[I_F] = 1
    illiquid_regime = np.zeros(V.shape, dtype=np.int8)
    illiquid_regime[sa < -drift_tol] = -1
    illiquid_regime[sa > drift_tol] = 1

    diagnostics = PolicyUpdateDiagnostics(
        min_vb=float(min(np.nanmin(VbF), np.nanmin(VbB))),
        min_va=float(min(np.nanmin(VaF), np.nanmin(VaB))),
        max_abs_deposit=float(np.max(np.abs(d))),
        max_abs_liquid_drift=float(np.max(np.abs(sb))),
        max_abs_illiquid_drift=float(np.max(np.abs(sa))),
        max_zero_drift_error=zero_error,
        max_foc_residual=float(max_foc_residual),
        root_solves=int(root_solves),
        zero_liquid_share=float(np.mean(I_0)),
        lower_liquid_constraint_share=float(np.mean(I_0[:, :, 0, :])),
        upper_liquid_constraint_share=float(np.mean(I_0[:, :, -1, :])),
        lower_illiquid_constraint_share=float(np.mean(np.abs(sa[:, :, :, 0]) <= drift_tol)),
        upper_illiquid_constraint_share=float(np.mean(np.abs(sa[:, :, :, -1]) <= drift_tol)),
    )
    return PolicyUpdateResult(
        consumption=c,
        deposit=d,
        liquid_drift=sb,
        illiquid_drift=sa,
        liquid_regime=liquid_regime,
        illiquid_regime=illiquid_regime,
        diagnostics=diagnostics,
    )


def joint_exogenous_generator(Pz: np.ndarray, Pe: np.ndarray, dt: float) -> np.ndarray:
    """Convert annual transition matrices to a generator over ``(z,e)``."""
    Pz = row_normalize(np.asarray(Pz, dtype=float))
    Pe = row_normalize(np.asarray(Pe, dtype=float))
    K = Pz.shape[0]
    E = Pe.shape[1]
    n = K * E
    P = np.zeros((n, n), dtype=float)
    for z in range(K):
        for e in range(E):
            row = z * E + e
            for zp in range(K):
                for ep in range(E):
                    col = zp * E + ep
                    P[row, col] += float(Pz[z, zp]) * float(Pe[z, e, ep])
    P = row_normalize(P)
    return (P - np.eye(n)) / float(dt)


def build_hjb_generator(
    sb: np.ndarray,
    sa: np.ndarray,
    b_grid: np.ndarray,
    a_grid: np.ndarray,
    Q: np.ndarray,
) -> sp.csr_matrix:
    """Sparse upwind generator over ``[z,e,b,a]``."""
    K, E, B, A = sb.shape
    M = K * E * B * A
    ids = np.arange(M, dtype=np.int64).reshape(K, E, B, A)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    outflow = np.zeros((K, E, B, A), dtype=float)

    if B > 1:
        db = (np.asarray(b_grid)[1:] - np.asarray(b_grid)[:-1])[None, None, :, None]
        rate_up = np.maximum(sb[:, :, :-1, :], 0.0) / db
        rows.append(ids[:, :, :-1, :].ravel())
        cols.append(ids[:, :, 1:, :].ravel())
        vals.append(rate_up.ravel())
        outflow[:, :, :-1, :] += rate_up

        rate_down = -np.minimum(sb[:, :, 1:, :], 0.0) / db
        rows.append(ids[:, :, 1:, :].ravel())
        cols.append(ids[:, :, :-1, :].ravel())
        vals.append(rate_down.ravel())
        outflow[:, :, 1:, :] += rate_down

    if A > 1:
        da = (np.asarray(a_grid)[1:] - np.asarray(a_grid)[:-1])[None, None, None, :]
        rate_up = np.maximum(sa[:, :, :, :-1], 0.0) / da
        rows.append(ids[:, :, :, :-1].ravel())
        cols.append(ids[:, :, :, 1:].ravel())
        vals.append(rate_up.ravel())
        outflow[:, :, :, :-1] += rate_up

        rate_down = -np.minimum(sa[:, :, :, 1:], 0.0) / da
        rows.append(ids[:, :, :, 1:].ravel())
        cols.append(ids[:, :, :, :-1].ravel())
        vals.append(rate_down.ravel())
        outflow[:, :, :, 1:] += rate_down

    diag = -outflow.ravel()
    BA = B * A
    asset_block = np.arange(BA, dtype=np.int64)
    KE = K * E
    for k in range(KE):
        for kp in range(KE):
            q = float(Q[k, kp])
            if q == 0.0:
                continue
            if k == kp:
                diag[k * BA : (k + 1) * BA] += q
            else:
                rows.append(k * BA + asset_block)
                cols.append(kp * BA + asset_block)
                vals.append(np.full(BA, q, dtype=float))

    rows.append(np.arange(M, dtype=np.int64))
    cols.append(np.arange(M, dtype=np.int64))
    vals.append(diag)
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals)
    if not np.all(np.isfinite(val)):
        raise PolicyUpdateError("HJB generator contains non-finite transition rates")
    keep = np.abs(val) > 0.0
    return sp.csr_matrix((val[keep], (row[keep], col[keep])), shape=(M, M))


def validate_generator(Aop: sp.csr_matrix, tolerance: float) -> float:
    """Verify the monotone-generator conditions used by the implicit HJB step."""
    Aop = Aop.tocsr()
    if not np.all(np.isfinite(Aop.data)):
        raise PolicyUpdateError("HJB generator contains non-finite entries")
    coo = Aop.tocoo()
    off = coo.row != coo.col
    min_offdiag = float(np.min(coo.data[off])) if np.any(off) else 0.0
    max_diag = float(np.max(Aop.diagonal())) if Aop.shape[0] else 0.0
    row_sums = np.asarray(Aop.sum(axis=1)).ravel()
    max_row_sum = float(np.max(np.abs(row_sums))) if len(row_sums) else 0.0
    if min_offdiag < -tolerance:
        raise PolicyUpdateError(
            f"HJB generator has a negative off-diagonal entry {min_offdiag:.6e}"
        )
    if max_diag > tolerance:
        raise PolicyUpdateError(f"HJB generator has a positive diagonal entry {max_diag:.6e}")
    if max_row_sum > tolerance:
        raise PolicyUpdateError(
            f"HJB generator rows do not sum to zero; max error={max_row_sum:.6e}"
        )
    return max_row_sum


def initial_value_guess(
    terminal_value: np.ndarray,
    after_tax_income: np.ndarray,
    Pz: np.ndarray,
    Pe: np.ndarray,
    b_grid: np.ndarray,
    a_grid: np.ndarray,
    cfg: HAConfig,
    *,
    rho: float,
    rb_pos: float,
    rb_neg: float,
    ra: float,
    chi0: float,
    chi1: float,
) -> np.ndarray:
    """Finite-horizon counterpart of the Sabet-Schneider stationary guess.

    The reference stationary code uses ``u(c_guess)/rho``. With the lifecycle
    term, the correctly scaled no-asset-drift guess solves

        [(rho + 1/dt) I - Q] V0 = u(c_guess) + V_next/dt.
    """
    terminal_value = np.asarray(terminal_value, dtype=float)
    y = np.asarray(after_tax_income, dtype=float)[:, :, None, None]
    b = np.asarray(b_grid, dtype=float)[None, None, :, None]
    a = np.asarray(a_grid, dtype=float)[None, None, None, :]
    # A pure no-drift flow guess can be flat in an asset whenever its return is
    # zero. That gives V_b=0 or V_a=0 and prevents the nested-drift FOCs from
    # even producing the first tentative update. Add a conservative annuitized
    # liquidation flow so the launch value is strictly increasing in both
    # assets. This is an initial guess only; the solved policies remain fully
    # determined by the HJB.
    liquid_principal_flow = np.maximum(b - float(b_grid[0]), 0.0) / cfg.ct_time_step
    illiquid_liquidation_weight = max(1.0 - chi0, EPS) / (1.0 + chi1)
    illiquid_principal_flow = (
        illiquid_liquidation_weight
        * np.maximum(a - float(a_grid[0]), 0.0)
        / cfg.ct_time_step
    )
    c_guess = np.maximum(
        y
        + b_return_flow(b, rb_pos, rb_neg)
        + ra * a
        + liquid_principal_flow
        + illiquid_principal_flow,
        cfg.consumption_floor,
    )
    u = crra_utility(
        c_guess + service_flow(b, a, cfg),
        cfg.risk_aversion,
        cfg.consumption_floor,
    )

    K, E, B, A = terminal_value.shape
    KE = K * E
    BA = B * A
    Q = joint_exogenous_generator(Pz, Pe, cfg.ct_time_step)
    lhs = (rho + 1.0 / cfg.ct_time_step) * np.eye(KE) - Q
    rhs = u.reshape(KE, BA) + terminal_value.reshape(KE, BA) / cfg.ct_time_step
    try:
        guess = np.linalg.solve(lhs, rhs).reshape(terminal_value.shape)
    except np.linalg.LinAlgError as exc:
        raise HJBSolverError("Could not construct the finite-horizon initial value guess") from exc
    if not np.all(np.isfinite(guess)):
        raise HJBSolverError("Finite-horizon initial value guess contains non-finite entries")
    return guess


def evaluate_hjb(
    V: np.ndarray,
    terminal_value: np.ndarray,
    after_tax_income: np.ndarray,
    b_grid: np.ndarray,
    a_grid: np.ndarray,
    Q: np.ndarray,
    cfg: HAConfig,
    *,
    rho: float,
    rb_pos: float,
    rb_neg: float,
    ra: float,
    chi0: float,
    chi1: float,
    xi: float,
) -> HJBEvaluation:
    policy = policy_update(
        V,
        after_tax_income,
        b_grid,
        a_grid,
        cfg,
        rb_pos=rb_pos,
        rb_neg=rb_neg,
        ra=ra,
        chi0=chi0,
        chi1=chi1,
        xi=xi,
    )
    Aop = build_hjb_generator(
        policy.liquid_drift,
        policy.illiquid_drift,
        b_grid,
        a_grid,
        Q,
    )
    max_row_sum = 0.0
    if cfg.ct_validate_generator:
        max_row_sum = validate_generator(Aop, cfg.ct_generator_tolerance)

    b = np.asarray(b_grid, dtype=float)[None, None, :, None]
    a = np.asarray(a_grid, dtype=float)[None, None, None, :]
    u = crra_utility(
        policy.consumption + service_flow(b, a, cfg),
        cfg.risk_aversion,
        cfg.consumption_floor,
    ).ravel()
    v = np.asarray(V, dtype=float).ravel()
    terminal = np.asarray(terminal_value, dtype=float).ravel()
    Av = Aop @ v
    coefficient = rho + 1.0 / cfg.ct_time_step
    residual = u + Av + terminal / cfg.ct_time_step - coefficient * v
    denominator = (
        1.0
        + np.abs(u)
        + np.abs(Av)
        + np.abs(terminal / cfg.ct_time_step)
        + np.abs(coefficient * v)
    )
    normalized = float(np.max(np.abs(residual) / denominator))
    max_abs = float(np.max(np.abs(residual)))
    if not np.isfinite(normalized) or not np.isfinite(max_abs):
        raise PolicyUpdateError("HJB residual is non-finite")
    return HJBEvaluation(
        policy=policy,
        generator=Aop,
        utility=u,
        residual=residual,
        max_abs_residual=max_abs,
        normalized_residual=normalized,
        max_generator_row_sum=max_row_sum,
    )


def implicit_hjb_step(
    V: np.ndarray,
    terminal_value: np.ndarray,
    evaluation: HJBEvaluation,
    cfg: HAConfig,
    *,
    rho: float,
    delta: float,
) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    M = V.size
    inv_delta = 1.0 / float(delta)
    lhs = (
        (inv_delta + rho + 1.0 / cfg.ct_time_step) * sp.eye(M, format="csr")
        - evaluation.generator
    )
    rhs = (
        evaluation.utility
        + V.ravel() * inv_delta
        + np.asarray(terminal_value, dtype=float).ravel() / cfg.ct_time_step
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", spla.MatrixRankWarning)
            V_new = spla.spsolve(lhs, rhs).reshape(V.shape)
    except Exception as exc:
        raise PolicyUpdateError("Implicit sparse HJB solve failed") from exc
    if not np.all(np.isfinite(V_new)):
        raise PolicyUpdateError("Implicit HJB solve produced non-finite values")
    return V_new


def _regime_change(old: PolicyUpdateResult, new: PolicyUpdateResult) -> float:
    changed = (old.liquid_regime != new.liquid_regime) | (
        old.illiquid_regime != new.illiquid_regime
    )
    return float(np.mean(changed))


def solve_hjb_period(
    initial_value: np.ndarray,
    terminal_value: np.ndarray,
    after_tax_income: np.ndarray,
    Pz: np.ndarray,
    Pe: np.ndarray,
    b_grid: np.ndarray,
    a_grid: np.ndarray,
    cfg: HAConfig,
    *,
    rho: float,
    rb_pos: float,
    rb_neg: float,
    ra: float,
    chi0: float,
    chi1: float,
    xi: float,
) -> HJBPeriodSolution:
    """Solve one lifecycle HJB using the Sabet-Schneider tentative-guess loop."""
    V = np.asarray(initial_value, dtype=float).copy()
    terminal_value = np.asarray(terminal_value, dtype=float)
    Q = joint_exogenous_generator(Pz, Pe, cfg.ct_time_step)

    try:
        current = evaluate_hjb(
            V,
            terminal_value,
            after_tax_income,
            b_grid,
            a_grid,
            Q,
            cfg,
            rho=rho,
            rb_pos=rb_pos,
            rb_neg=rb_neg,
            ra=ra,
            chi0=chi0,
            chi1=chi1,
            xi=xi,
        )
    except Exception as exc:
        raise HJBConvergenceError(
            "The finite-horizon initial value guess does not imply a valid nested-drift policy"
        ) from exc

    # This mirrors solveHousehold.m: the safe launch point is not overwritten by
    # every aggressive update. It is updated only after a complete block of slow
    # successful iterations following a failed tentative update.
    safe_V = V.copy()
    safe_evaluation = current
    delta = float(cfg.ct_delta_max)
    slow_count = 0
    fast_count = 1
    restarts = 0
    stable_regime_count = 0
    last_distance = np.inf
    last_relative_distance = np.inf
    last_regime_change = np.inf
    total_root_solves = int(current.policy.diagnostics.root_solves)
    max_zero_drift_error = float(current.policy.diagnostics.max_zero_drift_error)
    max_generator_row_sum = float(current.max_generator_row_sum)
    last_error = ""

    for it in range(1, int(cfg.ct_max_iterations) + 1):
        try:
            V_new = implicit_hjb_step(
                V,
                terminal_value,
                current,
                cfg,
                rho=rho,
                delta=delta,
            )
            last_distance = float(np.max(np.abs(V_new - V)))
            scale = max(1.0, float(np.max(np.abs(V_new))))
            last_relative_distance = last_distance / scale

            candidate = evaluate_hjb(
                V_new,
                terminal_value,
                after_tax_income,
                b_grid,
                a_grid,
                Q,
                cfg,
                rho=rho,
                rb_pos=rb_pos,
                rb_neg=rb_neg,
                ra=ra,
                chi0=chi0,
                chi1=chi1,
                xi=xi,
            )
            last_regime_change = _regime_change(current.policy, candidate.policy)
            if last_regime_change <= cfg.ct_regime_change_tol:
                stable_regime_count += 1
            else:
                stable_regime_count = 0

            if (
                np.isfinite(cfg.ct_residual_growth_restart)
                and current.normalized_residual > 0.0
                and candidate.normalized_residual
                > cfg.ct_residual_growth_restart
                * max(current.normalized_residual, cfg.ct_hjb_residual_tol)
            ):
                raise PolicyUpdateError(
                    "Tentative update increased the normalized HJB residual from "
                    f"{current.normalized_residual:.3e} to {candidate.normalized_residual:.3e}"
                )

            V = V_new
            current = candidate
            total_root_solves += int(candidate.policy.diagnostics.root_solves)
            max_zero_drift_error = max(
                max_zero_drift_error,
                float(candidate.policy.diagnostics.max_zero_drift_error),
            )
            max_generator_row_sum = max(
                max_generator_row_sum,
                float(candidate.max_generator_row_sum),
            )

            value_threshold = (
                cfg.ct_convergence_tol
                + cfg.ct_relative_convergence_tol * max(1.0, float(np.max(np.abs(V))))
            )
            value_converged = last_distance <= value_threshold
            residual_converged = candidate.normalized_residual <= cfg.ct_hjb_residual_tol
            regimes_converged = (
                stable_regime_count >= cfg.ct_regime_stability_iterations
            )
            if value_converged and residual_converged and regimes_converged:
                pol = candidate.policy
                return HJBPeriodSolution(
                    value=V,
                    consumption=pol.consumption,
                    deposit=pol.deposit,
                    liquid_drift=pol.liquid_drift,
                    illiquid_drift=pol.illiquid_drift,
                    iterations=it,
                    distance=last_distance,
                    relative_distance=last_relative_distance,
                    hjb_residual=candidate.normalized_residual,
                    regime_change=last_regime_change,
                    restarts=restarts,
                    root_solves=total_root_solves,
                    max_zero_drift_error=max_zero_drift_error,
                    max_generator_row_sum=max_generator_row_sum,
                    last_error=last_error,
                )

        except Exception as exc:
            restarts += 1
            last_error = f"{type(exc).__name__}: {exc}"
            if restarts > int(cfg.ct_max_restarts):
                raise HJBConvergenceError(
                    "The Sabet-Schneider tentative-guess algorithm exceeded "
                    f"ct_max_restarts={cfg.ct_max_restarts}; last failure: {last_error}"
                ) from exc
            failed_during_slow_search = slow_count > 0
            V = safe_V.copy()
            current = safe_evaluation
            if failed_during_slow_search:
                # The reference code uses Delta=.5 after a failure. Across a
                # broad parameter grid, even .5 can be outside the local
                # convergence basin. Backtrack further only when the slow
                # launch itself fails; successful slow blocks still use a fixed
                # Delta exactly as in solveHousehold.m.
                delta = max(
                    float(cfg.ct_delta_min),
                    float(delta) * float(cfg.ct_delta_backoff),
                )
            else:
                delta = float(cfg.ct_delta_base)
            slow_count = 1
            fast_count = 1
            stable_regime_count = 0
            continue

        # Tentative-guess update-speed schedule from solveHousehold.m.
        if slow_count > 0:
            slow_count += 1
            if slow_count > int(cfg.ct_slow_iterations):
                safe_V = V.copy()
                safe_evaluation = current
                slow_count = 0
                fast_count = 1
        else:
            delta = min(
                float(cfg.ct_delta_max),
                float(cfg.ct_delta_base) * math.exp(fast_count),
            )
            fast_count += 1

    raise HJBConvergenceError(
        "HJB period did not converge within "
        f"{cfg.ct_max_iterations} iterations: distance={last_distance:.3e}, "
        f"relative_distance={last_relative_distance:.3e}, "
        f"normalized_residual={current.normalized_residual:.3e}, "
        f"regime_change={last_regime_change:.3e}, restarts={restarts}, "
        f"last_failure={last_error or 'none'}"
    )


def check_lower_liquid_boundary_feasibility(
    policy_bundle: PolicyBundle,
    b_grid: np.ndarray,
    a_grid: np.ndarray,
    cfg: HAConfig,
    *,
    rb_pos: float,
    rb_neg: float,
    ra: float,
    chi0: float,
    chi1: float,
    xi: float,
) -> FeasibilityDiagnostics:
    """Check whether every lower-liquid-boundary state admits an inward policy.

    For each exogenous state and illiquid grid point, this maximizes feasible
    consumption over deposits while respecting the illiquid state constraint at
    its lower and upper boundaries. A negative slack means no numerical method
    can solve the stated constrained problem at that state.
    """
    bmin = float(np.asarray(b_grid)[0])
    rb = rb_pos if bmin >= 0.0 else rb_neg
    min_slack = np.inf
    arg = (0, 0, 0, 0, 0, 0.0, np.inf)
    A = len(a_grid)

    for g in range(policy_bundle.G):
        for h in range(policy_bundle.H):
            income = np.asarray(policy_bundle.after_tax_income[g, h], dtype=float)
            for z in range(policy_bundle.K):
                for e in range(2):
                    y = float(income[z, e])
                    liquid_flow = (1.0 - xi) * y + rb * bmin
                    for ia, a_val in enumerate(a_grid):
                        a = float(a_val)
                        a_eff = max(a, cfg.ct_illiquid_cost_floor)
                        d_unconstrained = (chi0 - 1.0) * a_eff / chi1
                        d_zero = -(ra * a + xi * y)
                        if ia == 0:
                            d_star = max(d_unconstrained, d_zero)
                        elif ia == A - 1:
                            d_star = min(d_unconstrained, d_zero)
                        else:
                            d_star = d_unconstrained
                        c_max = float(
                            liquid_flow
                            - d_star
                            - adjustment_cost(
                                d_star,
                                a,
                                chi0,
                                chi1,
                                cfg.ct_illiquid_cost_floor,
                            )
                        )
                        slack = c_max - cfg.consumption_floor
                        if slack < min_slack:
                            min_slack = slack
                            arg = (g, h, z, e, ia, d_star, c_max)

    g, h, z, e, ia, d_star, c_max = arg
    diagnostics = FeasibilityDiagnostics(
        min_slack=float(min_slack),
        group_index=int(g),
        age_period=int(h),
        income_state=int(z),
        employment_state=int(e),
        illiquid_index=int(ia),
        maximizing_deposit=float(d_star),
        max_feasible_consumption=float(c_max),
    )
    if min_slack < -cfg.ct_state_constraint_tolerance:
        age = int(policy_bundle.ages[h])
        raise StateFeasibilityError(
            "No admissible policy exists at the lower liquid boundary: "
            f"group={g}, age={age}, z={z}, e={e}, a_index={ia}, "
            f"max feasible consumption={c_max:.6e}, floor={cfg.consumption_floor:.6e}, "
            f"slack={min_slack:.6e}. Move the borrowing limit toward zero, "
            "lower the borrowing rate/automatic illiquid contribution, or revise the income floor."
        )
    return diagnostics


def validate_solver_parameters(
    cfg: HAConfig,
    *,
    rho: float,
    rb_pos: float,
    ra: float,
    chi0: float,
    chi1: float,
) -> None:
    """Validate structural restrictions, distinguishing finite from infinite horizon."""
    values = {"rho": rho, "rb_pos": rb_pos, "ra": ra, "chi0": chi0, "chi1": chi1}
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError(f"Non-finite continuous-time solver parameter: {values}")
    if chi1 <= 0.0:
        raise ValueError("ct_convex_adjustment_cost must be strictly positive")
    if not 0.0 <= chi0 < 1.0:
        raise ValueError("ct_linear_adjustment_cost must lie in [0,1)")

    # The original stationary example imposes these transversality/no-explosion
    # restrictions. They are optional here because a bounded finite-horizon
    # lifecycle problem can be well defined when a return exceeds rho.
    if cfg.ct_enforce_stationary_conditions:
        if rho <= max(ra, rb_pos):
            raise ValueError(
                "Stationary Sabet-Schneider condition violated: rho must exceed "
                "the saving returns. Disable ct_enforce_stationary_conditions for "
                "the finite-horizon lifecycle problem."
            )
        if ra * chi1 >= 1.0 - chi0:
            raise ValueError(
                "Stationary Sabet-Schneider condition violated: ra*chi1 must be below 1-chi0"
            )


# =============================================================================
# Main solver and saved model format
# =============================================================================


@dataclass
class SolvedHAModel:
    config: HAConfig
    policy_bundle: PolicyBundle
    b_grid: np.ndarray
    a_grid: np.ndarray
    value: Optional[np.ndarray]
    consumption: np.ndarray
    deposit: np.ndarray
    liquid_drift: np.ndarray
    illiquid_drift: np.ndarray
    diagnostics: SolverDiagnostics
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(
        self,
        path: str | Path,
        *,
        save_value: bool = False,
        save_transitions: bool = False,
        dtype: str = "float32",
    ) -> Path:
        """Save compact model files.

        Only consumption and deposit are stored as full policy arrays. Liquid
        and illiquid drifts are reconstructed from these policies and the model
        accounting equations when a compact saved model is used for prediction.
        Transitions and value functions are optional and off by default.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
        }
        if dtype not in dtype_map:
            raise ValueError(
                f"Unsupported output dtype {dtype!r}; choose from {sorted(dtype_map)}."
            )
        dtype_np = dtype_map[dtype]
        meta = {
            "config": dataclass_to_jsonable(self.config),
            "policy_bundle_name": self.policy_bundle.name,
            # Store the full policy-function metadata so downstream scoring
            # tables can recover income/employment/tax/pension assumptions.
            # For parametric bundles this contains the original
            # PolicyFunctionSpec; for empirical/loaded bundles it contains
            # source and bundle metadata.
            "policy_bundle_metadata": dataclass_to_jsonable(self.policy_bundle.metadata),
            "money_scale": float(self.policy_bundle.money_scale),
            "diagnostics": dataclass_to_jsonable(self.diagnostics),
            "metadata": self.metadata,
            "saved_arrays": {
                "value": bool(save_value),
                "transitions": bool(save_transitions),
                "dtype": dtype,
                "policy_arrays": ["consumption", "deposit"],
                "drifts_saved": False,
            },
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))

        arrays: dict[str, np.ndarray] = {
            "group_values": self.policy_bundle.group_values,
            "ages": self.policy_bundle.ages,
            "log_income_grid": self.policy_bundle.log_income_grid.astype(dtype_np),
            "liquid_grid": self.b_grid.astype(dtype_np),
            "illiquid_grid": self.a_grid.astype(dtype_np),
            "consumption": self.consumption.astype(dtype_np),
            "deposit": self.deposit.astype(dtype_np),
            "gross_income": self.policy_bundle.gross_income.astype(dtype_np),
            "after_tax_income": self.policy_bundle.after_tax_income.astype(dtype_np),
        }
        if save_transitions:
            arrays["income_transition"] = self.policy_bundle.income_transition.astype(dtype_np)
            arrays["employment_transition"] = self.policy_bundle.employment_transition.astype(dtype_np)
        if save_value and self.value is not None:
            arrays["value"] = self.value.astype(dtype_np)
        np.savez_compressed(path / "arrays.npz", **arrays)
        return path


class HASolver:
    """Solve a single finite-horizon HA model for a supplied policy bundle."""

    def __init__(self, config: HAConfig, policy_bundle: PolicyBundle):
        config.validate()
        policy_bundle.validate()
        self.config = config
        self.policy_bundle = policy_bundle
        self.b_grid, self.a_grid = make_asset_grids(config)

    def solve(
        self,
        *,
        model_id: Optional[str] = None,
        keep_value: bool = False,
        warm_start_value: Optional[np.ndarray] = None,
    ) -> SolvedHAModel:
        cfg = self.config
        bundle = self.policy_bundle
        bundle_fingerprint = bundle.fingerprint()
        model_id = model_id or stable_hash(
            {"config": cfg, "policy_bundle_fingerprint": bundle_fingerprint},
            n=14,
        )

        G, H, K = bundle.G, bundle.H, bundle.K
        E = 2
        B = len(self.b_grid)
        A = len(self.a_grid)

        value = np.zeros((G, H + 1, K, E, B, A), dtype=float)
        consumption = np.zeros((G, H, K, E, B, A), dtype=float)
        deposit = np.zeros_like(consumption)
        liquid_pol = np.zeros_like(consumption)
        illiquid_pol = np.zeros_like(consumption)

        warm = None
        if warm_start_value is not None:
            warm = np.asarray(warm_start_value, dtype=float)
            allowed = {
                (G, H, K, E, B, A),
                (G, H + 1, K, E, B, A),
            }
            if warm.shape not in allowed:
                raise ValueError(
                    "warm_start_value must have shape [G,H,K,E,B,A] or [G,H+1,K,E,B,A]; "
                    f"received {warm.shape}"
                )
            if not np.all(np.isfinite(warm)):
                raise ValueError("warm_start_value contains non-finite entries")

        rb_pos = continuous_rate_from_annual_return(cfg.liquid_interest_rate)
        rb_neg = continuous_rate_from_annual_return(cfg.borrowing_interest_rate)
        ra = continuous_rate_from_annual_return(cfg.illiquid_interest_rate)
        rho = discount_rate_from_beta(
            cfg.discount_factor,
            cfg.ct_time_step,
            cfg.ct_discount_rate,
        )
        chi0 = float(cfg.ct_linear_adjustment_cost)
        chi1 = float(cfg.ct_convex_adjustment_cost)
        xi = float(cfg.ct_automatic_illiquid_income_share)
        validate_solver_parameters(
            cfg,
            rho=rho,
            rb_pos=rb_pos,
            ra=ra,
            chi0=chi0,
            chi1=chi1,
        )
        feasibility = check_lower_liquid_boundary_feasibility(
            bundle,
            self.b_grid,
            self.a_grid,
            cfg,
            rb_pos=rb_pos,
            rb_neg=rb_neg,
            ra=ra,
            chi0=chi0,
            chi1=chi1,
            xi=xi,
        )

        max_it = 0
        max_dist = 0.0
        max_relative_dist = 0.0
        max_residual = 0.0
        max_regime_change = 0.0
        total_restarts = 0
        total_root_solves = 0
        max_zero_drift_error = 0.0
        max_generator_row_sum = 0.0
        period_diagnostics: list[dict[str, Any]] = []

        for g in range(G):
            for h in range(H - 1, -1, -1):
                terminal = value[g, h + 1]
                if warm is not None:
                    init = warm[g, h].copy()
                else:
                    init = initial_value_guess(
                        terminal,
                        bundle.after_tax_income[g, h],
                        bundle.income_transition[g, h],
                        bundle.employment_transition[g, h],
                        self.b_grid,
                        self.a_grid,
                        cfg,
                        rho=rho,
                        rb_pos=rb_pos,
                        rb_neg=rb_neg,
                        ra=ra,
                        chi0=chi0,
                        chi1=chi1,
                    )

                try:
                    result = solve_hjb_period(
                        init,
                        terminal,
                        bundle.after_tax_income[g, h],
                        bundle.income_transition[g, h],
                        bundle.employment_transition[g, h],
                        self.b_grid,
                        self.a_grid,
                        cfg,
                        rho=rho,
                        rb_pos=rb_pos,
                        rb_neg=rb_neg,
                        ra=ra,
                        chi0=chi0,
                        chi1=chi1,
                        xi=xi,
                    )
                except Exception as exc:
                    age = int(bundle.ages[h])
                    raise HJBConvergenceError(
                        f"Household HJB failed for group_index={g}, age={age}: {exc}"
                    ) from exc

                value[g, h] = result.value
                consumption[g, h] = result.consumption
                deposit[g, h] = result.deposit
                liquid_pol[g, h] = result.liquid_drift
                illiquid_pol[g, h] = result.illiquid_drift

                max_it = max(max_it, int(result.iterations))
                max_dist = max(max_dist, float(result.distance))
                max_relative_dist = max(max_relative_dist, float(result.relative_distance))
                max_residual = max(max_residual, float(result.hjb_residual))
                max_regime_change = max(max_regime_change, float(result.regime_change))
                total_restarts += int(result.restarts)
                total_root_solves += int(result.root_solves)
                max_zero_drift_error = max(
                    max_zero_drift_error,
                    float(result.max_zero_drift_error),
                )
                max_generator_row_sum = max(
                    max_generator_row_sum,
                    float(result.max_generator_row_sum),
                )
                period_diagnostics.append(
                    {
                        "group_index": int(g),
                        "age_period": int(h),
                        "age": int(bundle.ages[h]),
                        "iterations": int(result.iterations),
                        "distance": float(result.distance),
                        "relative_distance": float(result.relative_distance),
                        "hjb_residual": float(result.hjb_residual),
                        "regime_change": float(result.regime_change),
                        "restarts": int(result.restarts),
                        "root_solves": int(result.root_solves),
                        "max_zero_drift_error": float(result.max_zero_drift_error),
                        "max_generator_row_sum": float(result.max_generator_row_sum),
                        "last_recovered_error": result.last_error,
                    }
                )

        diag = SolverDiagnostics(
            model_id=model_id,
            status="success",
            max_iterations=max_it,
            max_distance=max_dist,
            max_relative_distance=max_relative_dist,
            max_hjb_residual=max_residual,
            max_regime_change=max_regime_change,
            total_restarts=total_restarts,
            total_root_solves=total_root_solves,
            min_feasibility_slack=float(feasibility.min_slack),
            max_zero_drift_error=max_zero_drift_error,
            max_generator_row_sum=max_generator_row_sum,
        )
        return SolvedHAModel(
            config=cfg,
            policy_bundle=bundle,
            b_grid=self.b_grid,
            a_grid=self.a_grid,
            value=value if keep_value else None,
            consumption=consumption,
            deposit=deposit,
            liquid_drift=liquid_pol,
            illiquid_drift=illiquid_pol,
            diagnostics=diag,
            metadata={
                "rb_pos_ct": rb_pos,
                "rb_neg_ct": rb_neg,
                "ra_ct": ra,
                "rho_ct": rho,
                "chi0": chi0,
                "chi1": chi1,
                "xi": xi,
                "policy_bundle_fingerprint": bundle_fingerprint,
                "stationary_conditions_enforced": bool(
                    cfg.ct_enforce_stationary_conditions
                ),
                "lower_boundary_feasibility": dataclass_to_jsonable(feasibility),
                "period_diagnostics": period_diagnostics,
            },
        )


class SavedHAModel:
    """Lightweight prediction class for a saved HA model."""

    def __init__(self, metadata: dict[str, Any], arrays: dict[str, np.ndarray]):
        self.metadata = metadata
        self.config = HAConfig(**metadata["config"])
        self.money_scale = float(metadata.get("money_scale", 1.0))
        self.group_values = arrays["group_values"]
        self.ages = arrays["ages"]
        self.log_income_grid = arrays["log_income_grid"]
        self.liquid_grid = arrays["liquid_grid"]
        self.illiquid_grid = arrays["illiquid_grid"]
        self.consumption = arrays["consumption"]
        self.deposit = arrays["deposit"]
        # Older saved models may contain drift arrays. Compact models omit them
        # and reconstruct the drifts from consumption and deposit.
        self.liquid_drift = arrays.get("liquid_drift")
        self.illiquid_drift = arrays.get("illiquid_drift")
        self.after_tax_income = arrays.get("after_tax_income")

        solution_meta = metadata.get("metadata", {})
        self.rb_pos_ct = float(
            solution_meta.get(
                "rb_pos_ct",
                continuous_rate_from_annual_return(self.config.liquid_interest_rate),
            )
        )
        self.rb_neg_ct = float(
            solution_meta.get(
                "rb_neg_ct",
                continuous_rate_from_annual_return(self.config.borrowing_interest_rate),
            )
        )
        self.ra_ct = float(
            solution_meta.get(
                "ra_ct",
                continuous_rate_from_annual_return(self.config.illiquid_interest_rate),
            )
        )
        self.chi0 = float(
            solution_meta.get("chi0", self.config.ct_linear_adjustment_cost)
        )
        self.chi1 = float(
            solution_meta.get("chi1", self.config.ct_convex_adjustment_cost)
        )
        self.xi = float(
            solution_meta.get(
                "xi", self.config.ct_automatic_illiquid_income_share
            )
        )
        self.group_to_idx = {int(v): i for i, v in enumerate(self.group_values)}

    @staticmethod
    def load(path: str | Path) -> "SavedHAModel":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())
        npz = np.load(path / "arrays.npz", allow_pickle=False)
        arrays = {k: npz[k] for k in npz.files}
        return SavedHAModel(metadata, arrays)

    def _group_indices(self, df: pd.DataFrame) -> np.ndarray:
        cfg = self.config
        if "group_state" in df.columns:
            out = df["group_state"].astype(int).to_numpy()
            return np.clip(out, 0, len(self.group_values) - 1)
        if cfg.group_col in df.columns:
            vals = df[cfg.group_col].to_numpy()
        elif cfg.education_col in df.columns:
            vals = df[cfg.education_col].to_numpy()
        else:
            return np.zeros(len(df), dtype=np.int64)
        out = np.array([self.group_to_idx.get(int(v), 0) for v in vals], dtype=np.int64)
        return out

    def _income_indices(self, df: pd.DataFrame, g_idx: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.income_state_col in df.columns:
            return np.clip(df[cfg.income_state_col].astype(int).to_numpy(), 0, self.log_income_grid.shape[1] - 1)

        candidate_cols = [
            "labor_income_state",
            cfg.potential_labor_col,
            cfg.labor_income_col,
            cfg.gross_income_col,
            cfg.current_income_col,
            "income",
        ]
        income_col = next((c for c in candidate_cols if c in df.columns), None)
        if income_col is None:
            raise ValueError(
                "Prediction dataframe must contain income_state or one of: "
                + ", ".join(candidate_cols)
            )
        x = df[income_col].astype(float).to_numpy() / self.money_scale
        log_x = safe_log(x)
        out = np.empty(len(df), dtype=np.int64)
        for g in np.unique(g_idx):
            mask = g_idx == g
            out[mask] = nearest_indices(self.log_income_grid[int(g)], log_x[mask])
        return out

    def _employment_indices(self, df: pd.DataFrame) -> np.ndarray:
        cfg = self.config
        if cfg.employment_state_col in df.columns:
            return np.clip(df[cfg.employment_state_col].astype(int).to_numpy(), 0, 1)
        if cfg.employed_col in df.columns:
            return np.where(df[cfg.employed_col].astype(bool).to_numpy(), 0, 1).astype(np.int64)
        return np.zeros(len(df), dtype=np.int64)

    @staticmethod
    def _interp_grid_indices(grid: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid = np.asarray(grid, dtype=float)
        x = np.asarray(x, dtype=float)
        right = np.searchsorted(grid, x, side="left")
        right = np.clip(right, 0, len(grid) - 1)
        left = np.clip(right - 1, 0, len(grid) - 1)
        same = right == left
        denom = np.maximum(grid[right] - grid[left], EPS)
        w = np.where(same, 0.0, (x - grid[left]) / denom)
        w = np.clip(w, 0.0, 1.0)
        return left.astype(np.int64), right.astype(np.int64), w.astype(float)

    def _evaluate_policy(
        self,
        arr: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        z: np.ndarray,
        e: np.ndarray,
        b: np.ndarray,
        a: np.ndarray,
        *,
        interpolate_assets: bool,
    ) -> np.ndarray:
        if not interpolate_assets:
            ib = nearest_indices(self.liquid_grid, b)
            ia = nearest_indices(self.illiquid_grid, a)
            return arr[g, h, z, e, ib, ia].astype(float)

        ib0, ib1, wb = self._interp_grid_indices(self.liquid_grid, b)
        ia0, ia1, wa = self._interp_grid_indices(self.illiquid_grid, a)
        v00 = arr[g, h, z, e, ib0, ia0]
        v10 = arr[g, h, z, e, ib1, ia0]
        v01 = arr[g, h, z, e, ib0, ia1]
        v11 = arr[g, h, z, e, ib1, ia1]
        return (
            (1.0 - wb) * (1.0 - wa) * v00
            + wb * (1.0 - wa) * v10
            + (1.0 - wb) * wa * v01
            + wb * wa * v11
        ).astype(float)

    def _reconstruct_drifts(
        self,
        c: np.ndarray,
        d: np.ndarray,
        b: np.ndarray,
        a: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        z: np.ndarray,
        e: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct drifts from the saved primitive controls and accounting identities."""
        if self.after_tax_income is None:
            raise ValueError(
                "This compact saved model has no after_tax_income array, so "
                "liquid and illiquid drifts cannot be reconstructed."
            )

        y = self.after_tax_income[g, h, z, e].astype(float)
        sb = liquid_drift(
            c,
            d,
            b,
            a,
            y,
            rb_pos=self.rb_pos_ct,
            rb_neg=self.rb_neg_ct,
            chi0=self.chi0,
            chi1=self.chi1,
            xi=self.xi,
            a_floor=self.config.ct_illiquid_cost_floor,
        )
        sa = illiquid_drift(d, a, y, ra=self.ra_ct, xi=self.xi)
        if not np.all(np.isfinite(sb)) or not np.all(np.isfinite(sa)):
            raise FloatingPointError("Reconstructed saved-model drifts are non-finite")
        return np.asarray(sb, dtype=float), np.asarray(sa, dtype=float)

    def predict(
        self,
        df: pd.DataFrame,
        *,
        interpolate_assets: bool = True,
        include_transfers: bool = True,
        transfer_col: str = "transfer_income",
        return_state_indices: bool = False,
    ) -> pd.DataFrame:
        """Predict consumption and saving for household rows.

        Required columns:
            age, liquid_assets, illiquid_assets, and either income_state or an
            income column such as labor_income/current_income/gross_income.

        Optional columns:
            group/group_state, employed/employment_state, transfer_income.

        Unexpected transfers are treated as extra liquid assets before policy
        interpolation, which is the natural cash-on-hand interpretation.
        """
        cfg = self.config
        missing = [c for c in [cfg.age_col, cfg.liquid_col, cfg.illiquid_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Prediction dataframe is missing required columns: {missing}")
        out = df.copy()
        n = len(out)
        g = self._group_indices(out)
        h = cfg.age_to_period(out[cfg.age_col].to_numpy(dtype=float))
        z = self._income_indices(out, g)
        e = self._employment_indices(out)

        b = out[cfg.liquid_col].astype(float).to_numpy() / self.money_scale
        a = out[cfg.illiquid_col].astype(float).to_numpy() / self.money_scale
        if include_transfers and transfer_col in out.columns:
            b = b + out[transfer_col].astype(float).to_numpy() / self.money_scale
        b = np.clip(b, self.liquid_grid[0], self.liquid_grid[-1])
        a = np.clip(a, self.illiquid_grid[0], self.illiquid_grid[-1])

        c = self._evaluate_policy(self.consumption, g, h, z, e, b, a, interpolate_assets=interpolate_assets)
        d = self._evaluate_policy(self.deposit, g, h, z, e, b, a, interpolate_assets=interpolate_assets)
        if self.liquid_drift is not None and self.illiquid_drift is not None:
            # Preserve the prior prediction path for legacy saved models.
            sb = self._evaluate_policy(
                self.liquid_drift, g, h, z, e, b, a,
                interpolate_assets=interpolate_assets,
            )
            sa = self._evaluate_policy(
                self.illiquid_drift, g, h, z, e, b, a,
                interpolate_assets=interpolate_assets,
            )
        else:
            sb, sa = self._reconstruct_drifts(c, d, b, a, g, h, z, e)

        dt = float(cfg.ct_time_step)
        b_next = np.clip(b + dt * sb, self.liquid_grid[0], self.liquid_grid[-1])
        a_next = np.clip(a + dt * sa, self.illiquid_grid[0], self.illiquid_grid[-1])

        out["pred_consumption"] = c * self.money_scale
        out["pred_deposit"] = d * self.money_scale
        out["pred_liquid_drift"] = sb * self.money_scale
        out["pred_illiquid_drift"] = sa * self.money_scale
        out["pred_total_saving_flow"] = (sb + sa) * self.money_scale
        out["pred_next_liquid_assets"] = b_next * self.money_scale
        out["pred_next_illiquid_assets"] = a_next * self.money_scale
        if return_state_indices:
            out["ha_group_index"] = g
            out["ha_age_period"] = h
            out["ha_income_state"] = z
            out["ha_employment_state"] = e
        return out


# =============================================================================
# Batch jobs and parallelization
# =============================================================================


@dataclass(frozen=True)
class ModelJob:
    model_id: str
    config: HAConfig
    policy_spec: PolicyFunctionSpec
    output_dir: str
    keep_value: bool = False
    save_value: bool = False
    save_transitions: bool = False
    output_dtype: str = "float32"
    overwrite: bool = False


def _sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _policy_spec_identity(policy_spec: PolicyFunctionSpec) -> dict[str, Any]:
    identity = {"spec": dataclass_to_jsonable(policy_spec)}
    if policy_spec.source == "bundle" and policy_spec.bundle_path is not None:
        bundle_path = Path(policy_spec.bundle_path)
        digests: dict[str, str] = {}
        for name in ("metadata.json", "arrays.npz"):
            candidate = bundle_path / name
            if candidate.exists():
                digests[name] = _sha1_file(candidate)
        identity["bundle_file_digests"] = digests
    return identity


def make_model_id(config: HAConfig, policy_spec: PolicyFunctionSpec) -> str:
    return stable_hash(
        {"config": config, "policy_spec_identity": _policy_spec_identity(policy_spec)},
        n=14,
    )


def solve_and_save_job(job: ModelJob) -> dict[str, Any]:
    model_dir = Path(job.output_dir) / job.model_id
    if model_dir.exists() and not job.overwrite and (model_dir / "metadata.json").exists():
        return {
            "model_id": job.model_id,
            "status": "exists",
            "path": str(model_dir),
            "message": "already solved",
        }
    try:
        bundle = load_or_build_policy_bundle(job.config, job.policy_spec)
        solver = HASolver(job.config, bundle)
        solved = solver.solve(model_id=job.model_id, keep_value=job.keep_value or job.save_value)
        solved.save(
            model_dir,
            save_value=job.save_value,
            save_transitions=job.save_transitions,
            dtype=job.output_dtype,
        )
        return {
            "model_id": job.model_id,
            "status": "success",
            "path": str(model_dir),
            "policy": job.policy_spec.short_name(),
            "max_iterations": solved.diagnostics.max_iterations,
            "max_distance": solved.diagnostics.max_distance,
            "max_relative_distance": solved.diagnostics.max_relative_distance,
            "max_hjb_residual": solved.diagnostics.max_hjb_residual,
            "max_regime_change": solved.diagnostics.max_regime_change,
            "total_restarts": solved.diagnostics.total_restarts,
            "total_root_solves": solved.diagnostics.total_root_solves,
            "min_feasibility_slack": solved.diagnostics.min_feasibility_slack,
            "max_zero_drift_error": solved.diagnostics.max_zero_drift_error,
            "max_generator_row_sum": solved.diagnostics.max_generator_row_sum,
            "message": "",
        }
    except Exception as exc:
        model_dir.mkdir(parents=True, exist_ok=True)
        error = {
            "model_id": job.model_id,
            "status": "failed",
            "path": str(model_dir),
            "policy": job.policy_spec.short_name(),
            "message": repr(exc),
        }
        (model_dir / "error.json").write_text(json.dumps(error, indent=2, sort_keys=True))
        return error


def _materialize_empirical_policy_spec(
    *,
    cfg: HAConfig,
    spec: PolicyFunctionSpec,
    empirical_panel: pd.DataFrame,
    output_dir: Path,
    overwrite: bool,
) -> PolicyFunctionSpec:
    """Estimate an empirical policy bundle once and convert the spec to a bundle path.

    This keeps large dataframes out of parallel worker payloads. The bundle is
    specific to the HAConfig because the income grid size and lifecycle ages are
    config-dependent.
    """
    bundle_id = stable_hash({"config": cfg, "empirical_policy_spec": spec}, n=14)
    bundle_dir = output_dir / "_policy_bundles" / bundle_id
    if overwrite or not (bundle_dir / "metadata.json").exists():
        bundle = build_empirical_policy_bundle(empirical_panel, cfg, name=spec.name)
        bundle.save(bundle_dir)
    return PolicyFunctionSpec(
        name=spec.name,
        source="bundle",
        bundle_path=str(bundle_dir),
    )


def build_jobs(
    *,
    base_config: HAConfig,
    parameter_grid: dict[str, Sequence[Any]],
    policy_specs: Sequence[PolicyFunctionSpec],
    output_dir: str | Path,
    empirical_panel: Optional[pd.DataFrame] = None,
    keep_value: bool = False,
    save_value: bool = False,
    save_transitions: bool = False,
    output_dtype: str = "float32",
    overwrite: bool = False,
    max_models: Optional[int] = None,
) -> list[ModelJob]:
    jobs: list[ModelJob] = []
    output_dir = Path(output_dir)
    for params in expand_grid(parameter_grid):
        cfg = update_dataclass(base_config, params)
        for pol in policy_specs:
            pol_for_job = pol
            if pol.source == "empirical_panel":
                if empirical_panel is None:
                    raise ValueError(
                        "A policy spec uses source='empirical_panel', but no empirical_panel was passed."
                    )
                pol_for_job = _materialize_empirical_policy_spec(
                    cfg=cfg,
                    spec=pol,
                    empirical_panel=empirical_panel,
                    output_dir=output_dir,
                    overwrite=overwrite,
                )
            model_id = make_model_id(cfg, pol_for_job)
            jobs.append(
                ModelJob(
                    model_id=model_id,
                    config=cfg,
                    policy_spec=pol_for_job,
                    output_dir=str(output_dir),
                    keep_value=keep_value,
                    save_value=save_value,
                    save_transitions=save_transitions,
                    output_dtype=output_dtype,
                    overwrite=overwrite,
                )
            )
            if max_models is not None and len(jobs) >= max_models:
                return jobs
    return jobs


def solve_model_grid(
    *,
    base_config: HAConfig,
    parameter_grid: dict[str, Sequence[Any]],
    policy_specs: Sequence[PolicyFunctionSpec],
    output_dir: str | Path,
    n_jobs: int = 1,
    empirical_panel: Optional[pd.DataFrame] = None,
    keep_value: bool = False,
    save_value: bool = False,
    save_transitions: bool = False,
    output_dtype: str = "float32",
    overwrite: bool = False,
    max_models: Optional[int] = None,
) -> pd.DataFrame:
    """Solve and save a Cartesian grid of HA models.

    Set n_jobs to the number of cores allocated on a cluster. Each model is
    independent and is saved in its own subdirectory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(
        base_config=base_config,
        parameter_grid=parameter_grid,
        policy_specs=policy_specs,
        output_dir=output_dir,
        empirical_panel=empirical_panel,
        keep_value=keep_value,
        save_value=save_value,
        save_transitions=save_transitions,
        output_dtype=output_dtype,
        overwrite=overwrite,
        max_models=max_models,
    )
    (output_dir / "run_plan.json").write_text(
        json.dumps(
            {
                "base_config": dataclass_to_jsonable(base_config),
                "parameter_grid": dataclass_to_jsonable(parameter_grid),
                "n_policy_specs": len(policy_specs),
                "n_empirical_panel_rows": None if empirical_panel is None else int(len(empirical_panel)),
                "n_jobs_total": len(jobs),
                "output_dtype": output_dtype,
            },
            indent=2,
            sort_keys=True,
        )
    )

    rows: list[dict[str, Any]] = []
    if n_jobs <= 1:
        for job in tqdm(jobs, desc="Solving HA models"):
            rows.append(solve_and_save_job(job))
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            futures = [ex.submit(solve_and_save_job, job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Solving HA models"):
                rows.append(fut.result())

    manifest = pd.DataFrame(rows).sort_values(["status", "model_id"]).reset_index(drop=True)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return manifest


# =============================================================================
# Parameter and policy grids
# =============================================================================


def make_policy_specs_from_grid(
    *,
    income_grid: dict[str, Sequence[Any]],
    employment_grid: dict[str, Sequence[Any]],
    tax_grid: dict[str, Sequence[Any]],
    pension_grid: dict[str, Sequence[Any]],
    group_values: tuple[int, ...] = (0,),
    max_specs: Optional[int] = None,
) -> list[PolicyFunctionSpec]:
    """Create a list of plausible parametric policy-function specs.

    The four input dictionaries are dataclass field grids for the corresponding
    policy spec objects. Names are generated automatically unless supplied.
    """
    specs: list[PolicyFunctionSpec] = []
    for inc_u, emp_u, tax_u, pen_u in product(
        expand_grid(income_grid),
        expand_grid(employment_grid),
        expand_grid(tax_grid),
        expand_grid(pension_grid),
    ):
        inc = update_dataclass(IncomeProcessSpec(), inc_u)
        emp = update_dataclass(EmploymentPolicySpec(), emp_u)
        tax = update_dataclass(TaxPolicySpec(), tax_u)
        pen = update_dataclass(PensionPolicySpec(), pen_u)
        # Human-readable names.

        if inc.stationary_log_income_std is not None:
            income_risk_name = (
                f"sd{inc.stationary_log_income_std:.2f}"
            )
        else:
            income_risk_name = (
                f"eps{float(inc.innovation_std):.2f}"
            )

        inc = replace(
            inc,
            name=inc_u.get(
                "name",
                f"income_rho{inc.persistence:.2f}_{income_risk_name}",
            ),
        )

        emp = replace(
            emp,
            name=emp_u.get(
                "name",
                (
                    f"emp_loss{emp.job_loss_probability:.3f}"
                    f"_find{emp.job_finding_probability:.2f}"
                ),
            ),
        )

        tax = replace(
            tax,
            name=tax_u.get(
                "name",
                (
                    f"tax_{tax.kind}"
                    f"_flat{tax.flat_rate:.2f}"
                    f"_prog{tax.progressive_rate:.2f}"
                    f"_floor{tax.minimum_after_tax_income:.2f}"
                ),
            ),
        )


        pen = replace(
            pen,
            name=pen_u.get("name", f"pension_rr{pen.replacement_rate:.2f}"),
        )
        name = "__".join([inc.name, emp.name, tax.name, pen.name])
        specs.append(
            PolicyFunctionSpec(
                name=name,
                source="parametric",
                income=inc,
                employment=emp,
                tax=tax,
                pension=pen,
                group_values=group_values,
            )
        )
        if max_specs is not None and len(specs) >= max_specs:
            return specs
    return specs


def default_structural_parameter_grid() -> dict[str, Sequence[Any]]:
    """A broad but still feasible structural-parameter grid.

    For cluster runs, expand this further. For local testing, pass a smaller
    dictionary or use max_models in solve_model_grid.
    """
    return {
        "discount_factor": [0.92, 0.95, 0.97, 0.985],
        "risk_aversion": [2.0, 4.0, 6.0],
        "liquid_interest_rate": [0.0, 0.01, 0.02],
        "illiquid_interest_rate": [0.02, 0.04, 0.06],
        "borrowing_interest_rate": [0.06, 0.10, 0.15],
        "allow_borrowing": [True],
        "borrowing_limit": [0.0, -0.25, -0.50, -1.00],
        "ct_linear_adjustment_cost": [0.0, 0.02, 0.05, 0.10],
        "ct_convex_adjustment_cost": [0.5, 2.0, 8.0],
        "ct_automatic_illiquid_income_share": [0.0, 0.05, 0.10],
    }


def default_policy_function_specs(*, group_values: tuple[int, ...] = (0,), max_specs: Optional[int] = None) -> list[PolicyFunctionSpec]:
    """Wide no-data policy-function grid."""
    return make_policy_specs_from_grid(
        income_grid={
            "persistence": [0.90, 0.95, 0.98],
            "stationary_log_income_std": [0.10, 0.30, 0.50, 0.70],
            "age_profile_slope": [0.10, 0.25, 0.40],
        },
        employment_grid={
            "job_loss_probability": [0.02],
            "job_finding_probability": [0.85],
            "unemployment_replacement_rate": [0.30],
        },
        tax_grid={
            "kind": ["flat", "progressive"],
            "flat_rate": [0.08, 0.15, 0.25],
            "progressive_rate": [0.02, 0.05],
            "minimum_after_tax_income": [0.20],
        },
        pension_grid={"replacement_rate": [0.25, 0.45, 0.65]},
        group_values=group_values,
        max_specs=max_specs,
    )


def add_policy_bundle_specs(paths: Sequence[str | Path]) -> list[PolicyFunctionSpec]:
    """Create PolicyFunctionSpec objects from saved PolicyBundle directories."""
    out = []
    for p in paths:
        p = Path(p)
        out.append(
            PolicyFunctionSpec(
                name=p.name,
                source="bundle",
                bundle_path=str(p),
            )
        )
    return out

def _select_chunk(jobs: list[Any], *, chunk_id: int, n_chunks: int) -> list[Any]:
    if n_chunks <= 1:
        return jobs
    if chunk_id < 0 or chunk_id >= n_chunks:
        raise ValueError(f"chunk_id must be in [0, {n_chunks - 1}]")
    return [job for i, job in enumerate(jobs) if i % n_chunks == chunk_id]


def _sample_jobs(jobs: list[Any], *, sample_models: int | None, sample_seed: int) -> list[Any]:
    """Randomly sample models without replacement from an already-built job list.

    Sampling happens before SLURM chunking, so array jobs split the same global
    sample rather than each drawing a different local sample.
    """
    if sample_models is None or sample_models < 0 or sample_models >= len(jobs):
        return jobs
    if sample_models <= 0:
        return []
    rng = np.random.default_rng(int(sample_seed))
    idx = rng.choice(len(jobs), size=int(sample_models), replace=False)
    idx.sort()
    return [jobs[int(i)] for i in idx]

def build_sampled_jobs(
    *,
    base_config: HAConfig,
    parameter_grid: dict[str, Sequence[Any]],
    policy_specs: Sequence[PolicyFunctionSpec],
    output_dir: str | Path,
    sample_models: int,
    sample_seed: int,
    keep_value: bool = False,
    save_value: bool = False,
    save_transitions: bool = False,
    output_dtype: str = "float32",
    overwrite: bool = False,
) -> list[ModelJob]:
    """Randomly sample model jobs without materializing the full Cartesian product."""
    output_dir = Path(output_dir)

    param_list = expand_grid(parameter_grid)
    n_params = len(param_list)
    n_policies = len(policy_specs)
    total = n_params * n_policies

    n_sample = min(int(sample_models), total)
    rng = np.random.default_rng(int(sample_seed))
    flat_indices = rng.choice(total, size=n_sample, replace=False)

    jobs: list[ModelJob] = []
    seen: set[str] = set()

    for flat in tqdm(flat_indices, desc="Building sampled jobs"):
        i_param = int(flat // n_policies)
        i_policy = int(flat % n_policies)

        cfg = update_dataclass(base_config, param_list[i_param])
        pol = policy_specs[i_policy]

        model_id = make_model_id(cfg, pol)

        # Very unlikely, but keeps output robust if duplicate specs/hash collisions occur.
        if model_id in seen:
            continue
        seen.add(model_id)

        jobs.append(
            ModelJob(
                model_id=model_id,
                config=cfg,
                policy_spec=pol,
                output_dir=str(output_dir),
                keep_value=keep_value,
                save_value=save_value,
                save_transitions=save_transitions,
                output_dtype=output_dtype,
                overwrite=overwrite,
            )
        )

    return jobs

def solve_jobs(jobs: list[Any], *, n_jobs: int) -> pd.DataFrame:
    """Solve a list of prebuilt ModelJob objects."""
    rows: list[dict[str, Any]] = []
    if n_jobs <= 1:
        for job in tqdm(jobs, desc="Solving HA models"):
            rows.append(solve_and_save_job(job))
    else:
        with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
            futures = [ex.submit(solve_and_save_job, job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Solving HA models"):
                rows.append(fut.result())

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["status", "model_id"]).reset_index(drop=True)


def run_largest_grid(
    *,
    output_dir: str | Path,
    n_jobs: int = 16,
    max_policy_specs: int | None = None,
    max_models: int | None = None,
    chunk_id: int = 0,
    n_chunks: int = 1,
    overwrite: bool = False,
    save_value: bool = False,
    save_transitions: bool = False,
    output_dtype: str = "float32",
    resolution: str = "final",
    seed: int = 20260601,
    sample_models: int | None = None,
    sample_seed: int = 20260601,
) -> pd.DataFrame:
    """Main function for the large no-data HA model grid."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if resolution == "search":
        # Still substantially denser than the previous 15-point illiquid grid.
        # The larger upper bounds reduce artificial state-constraint regions.
        resolution_settings = dict(
            income_grid_size=10,
            liquid_grid_size=80,
            illiquid_grid_size=40,
            liquid_max=20.0,
            illiquid_max=50.0,
            ct_max_iterations=160,
            ct_convergence_tol=1.0e-6,
            ct_relative_convergence_tol=1.0e-7,
            ct_hjb_residual_tol=1.0e-6,
        )
    elif resolution == "final":
        # Close to the 80 x 70 asset-grid resolution in the reference code,
        # with additional liquid resolution for borrowing/hand-to-mouth states.
        resolution_settings = dict(
            income_grid_size=12,
            liquid_grid_size=120,
            illiquid_grid_size=70,
            liquid_max=50.0,
            illiquid_max=100.0,
            ct_max_iterations=300,
            ct_convergence_tol=1.0e-8,
            ct_relative_convergence_tol=1.0e-9,
            ct_hjb_residual_tol=1.0e-8,
        )
    else:
        raise ValueError("resolution must be 'search' or 'final'")

    cfg = HAConfig(
        min_age=25,
        retirement_age=65,
        max_age=80,
        allow_borrowing=True,
        borrowing_limit=0.0,
        **resolution_settings,
    )
    parameter_grid = default_structural_parameter_grid()
    policy_specs = default_policy_function_specs(
        group_values=(0,),
        max_specs=max_policy_specs,
    )

    print("Building jobs")
    if sample_models is not None:
        all_jobs = build_sampled_jobs(
            base_config=cfg,
            parameter_grid=parameter_grid,
            policy_specs=policy_specs,
            output_dir=output_dir,
            sample_models=sample_models,
            sample_seed=sample_seed,
            keep_value=False,
            save_value=save_value,
            save_transitions=save_transitions,
            output_dtype=output_dtype,
            overwrite=overwrite,
        )
    else:
        all_jobs = build_jobs(
            base_config=cfg,
            parameter_grid=parameter_grid,
            policy_specs=policy_specs,
            output_dir=output_dir,
            empirical_panel=None,
            keep_value=False,
            save_value=save_value,
            save_transitions=save_transitions,
            output_dtype=output_dtype,
            overwrite=overwrite,
            max_models=max_models,
        )

    number_of_jobs = len(all_jobs)
    print(f"Processing {number_of_jobs} total jobs, {n_jobs} parallel workers")

    chunk_jobs = _select_chunk(all_jobs, chunk_id=chunk_id, n_chunks=n_chunks)

    run_config = {
        "resolution": resolution,
        "base_config": dataclass_to_jsonable(cfg),
        "parameter_grid": dataclass_to_jsonable(parameter_grid),
        "max_policy_specs": max_policy_specs,
        "n_policy_specs_used": len(policy_specs),
        "n_jobs_total_before_chunking": len(all_jobs),
        "chunk_id": chunk_id,
        "n_chunks": n_chunks,
        "n_jobs_this_chunk": len(chunk_jobs),
        "warning": "This can still be a large run. Use --max-models for smoke tests or --chunk-id/--n-chunks for splitting.",
        "parallel_workers": n_jobs,
        "overwrite": overwrite,
        "save_value": save_value,
        "save_transitions": save_transitions,
        "output_dtype": output_dtype,
    }
    (output_dir / f"large_grid_run_config_chunk{chunk_id:04d}.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True)
    )

    print(json.dumps(run_config, indent=2, sort_keys=True))

    manifest = solve_jobs(chunk_jobs, n_jobs=n_jobs)
    manifest_path = output_dir / f"manifest_chunk{chunk_id:04d}_of_{n_chunks:04d}.csv"
    manifest.to_csv(manifest_path, index=False)

    # Also write a convenience manifest.csv for non-chunked runs.
    if n_chunks == 1:
        manifest.to_csv(output_dir / "manifest.csv", index=False)

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the large HA model grid.")
    parser.add_argument("--output-dir", type=str, default=Path("../data/ha_models"))
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    parser.add_argument("--max-policy-specs", type=int, default=None,
                        help="Number of policy-function specs to use. Use -1 for the full policy grid.")
    parser.add_argument("--max-models", type=int, default=None,
                        help="Optional hard cap for smoke tests.")
    parser.add_argument("--chunk-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--n-chunks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--save-value", action="store_true",
                        help="Save value functions. Usually false to keep artifacts compact.")
    parser.add_argument("--save-transitions", action="store_true", default=False,
                        help="Save transition matrices. Useful if a saved model will be used as a true HA DGP.")
    parser.add_argument(
        "--output-dtype",
        choices=["float16", "float32"],
        default="float32",
        help="Dtype for saved floating-point arrays. Default: float32.",
    )
    parser.add_argument("--resolution", choices=["search", "final"], default="search",
                        help="Use 'search' for large grids; use 'final' only for a small number of finalist models.")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--sample-models", type=int, default=None,
                        help="Randomly sample this many HA models from the bounded job list.")
    parser.add_argument("--sample-seed", type=int, default=20260601,
                        help="Seed for --sample-models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_policy_specs = args.max_policy_specs
    if max_policy_specs is not None:
        max_policy_specs = None if args.max_policy_specs < 0 else args.max_policy_specs
    manifest = run_largest_grid(
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        max_policy_specs=max_policy_specs,
        max_models=args.max_models,
        chunk_id=args.chunk_id,
        n_chunks=args.n_chunks,
        overwrite=args.overwrite,
        save_value=args.save_value,
        save_transitions=args.save_transitions,
        output_dtype=args.output_dtype,
        resolution=args.resolution,
        seed=args.seed,
        sample_models=args.sample_models,
        sample_seed=args.sample_seed,
    )
    print("\nCompleted HA grid run.")
    if not manifest.empty:
        print(manifest["status"].value_counts(dropna=False).to_string())
        print(f"\nManifest rows: {len(manifest):,}")


if __name__ == "__main__":
    main()
