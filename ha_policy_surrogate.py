"""Smooth, accounting-restricted surrogates for saved HA policy grids.

The module reads ``metadata.json`` and ``arrays.npz`` from many solved
heterogeneous-agent models, draws continuous household states from each model
with equal model weight, and estimates smooth policy maps over household states
and structural parameters. The default population design deliberately
oversamples poor and wealthy hand-to-mouth households while retaining a broad
state-space component.

The saved primitive policies are consumption and the net deposit from the
liquid account into the illiquid account.  The surrogate fits those two objects
and an auxiliary total-liquid-outflow target,

    consumption + deposit + adjustment_cost,

which directly identifies liquid saving.  Predictions are reconciled back to a
single accounting-consistent consumption/deposit pair.  Next assets, asset
changes, and drifts are then reconstructed from the HA budget equations.

The default feature map is a structured polynomial with ridge regularization. It
contains continuous main effects and powers, state-state interactions,
state-parameter interactions, one-hot categorical variables, and
category-specific slopes. Consumption coefficients can additionally be fitted
to exact finite-difference MPC equations from paired liquid-asset states. A
smooth random-Fourier RBF alternative is also available.

Expected policy-array layout
----------------------------
Saved policies normally use ``G,H,K,E,B,A`` = education/group, age, income
state, employment state, liquid assets, illiquid assets.  Five-dimensional
``H,K,E,B,A`` arrays are supported for a single group.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import t as student_t
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # tqdm is optional; stage messages still work without it.
    _tqdm = None

EPS = 1.0e-12


def _status(message: str, *, verbose: bool = True) -> None:
    """Print a timestamp-free status message immediately."""
    if verbose:
        print(message, flush=True)


def _progress_iter(
    iterable: Any,
    *,
    total: int,
    description: str,
    unit: str,
    enabled: bool,
) -> Any:
    """Wrap an iterable in tqdm when available and requested."""
    if enabled and _tqdm is not None:
        return _tqdm(
            iterable,
            total=total,
            desc=description,
            unit=unit,
            dynamic_ncols=True,
        )
    return iterable


# -----------------------------------------------------------------------------
# Saved-grid schema and discovery
# -----------------------------------------------------------------------------


ARRAY_ALIASES: dict[str, tuple[str, ...]] = {
    "ages": ("ages", "age_grid"),
    "liquid_grid": (
        "liquid_grid",
        "liquid_assets",
        "liquid_asset_grid",
        "b_grid",
        "bgrid",
    ),
    "illiquid_grid": (
        "illiquid_grid",
        "illiquid_assets",
        "illiquid_asset_grid",
        "a_grid",
        "agrid",
    ),
    "log_income_grid": (
        "log_income_grid",
        "log_income",
        "log_labor_income_grid",
        "z_grid",
    ),
    "income_grid": (
        "income_grid",
        "labor_income_grid",
        "current_income_grid",
    ),
    "after_tax_income": ("after_tax_income", "net_income"),
    "gross_income": ("gross_income", "current_income"),
    "consumption": (
        "consumption",
        "policy_consumption",
        "consumption_policy",
        "c_policy",
        "c_pol",
    ),
    "next_liquid_assets": (
        "next_liquid_assets",
        "policy_next_liquid_assets",
        "next_liquid_policy",
        "b_next",
        "b_next_policy",
    ),
    "next_illiquid_assets": (
        "next_illiquid_assets",
        "policy_next_illiquid_assets",
        "next_illiquid_policy",
        "a_next",
        "a_next_policy",
    ),
    "deposit": ("deposit", "illiquid_deposit", "d_policy", "d_pol"),
    "liquid_drift": ("liquid_drift", "sb_policy", "sb_pol"),
    "illiquid_drift": ("illiquid_drift", "sa_policy", "sa_pol"),
    "adjust_illiquid": (
        "adjust_illiquid",
        "adjusts_illiquid",
        "illiquid_adjustment",
    ),
    "hjb_distances": ("hjb_distances", "solver_distances"),
}


@dataclass(frozen=True)
class GridSchema:
    """How state and policy arrays are stored in ``arrays.npz``.

    ``policy_layout`` uses the letters:

    * G: education/group
    * H: age period
    * K: income state
    * E: employment state
    * B: current liquid assets
    * A: current illiquid assets

    The common layouts are ``GHKEBA`` and, for one group, ``HKEBA``.
    """

    policy_layout: str = "GHKEBA"
    key_overrides: Mapping[str, str] = field(default_factory=dict)
    money_units: str = "model"  # "model" or "data"
    include_derived_metadata_parameters: bool = False
    require_compact_policies: bool = False

    def __post_init__(self) -> None:
        layout = self.policy_layout.upper()
        expected = set("GHKEBA") if len(layout) == 6 else set("HKEBA")
        if (
            len(layout) not in {5, 6}
            or len(set(layout)) != len(layout)
            or set(layout) != expected
        ):
            raise ValueError(
                "policy_layout must contain each of GHKEBA exactly once, or "
                "HKEBA exactly once for a single education/group dimension."
            )
        if self.money_units not in {"model", "data"}:
            raise ValueError("money_units must be 'model' or 'data'.")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def discover_model_dirs(
    model_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    accepted_statuses: Sequence[str] = ("success", "exists", "converged"),
) -> list[Path]:
    """Discover directories containing ``metadata.json`` and ``arrays.npz``.

    Relative paths in a manifest are resolved first relative to the manifest's
    parent directory and then relative to ``model_root``.  This avoids a common
    current-working-directory bug in manifest readers.
    """

    root = Path(model_root).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser() if manifest_path else root / "manifest.csv"
    )
    dirs: list[Path] = []

    if manifest.exists():
        manifest = manifest.resolve()
        table = pd.read_csv(manifest)
        if "path" not in table.columns:
            raise ValueError(f"Manifest {manifest} is missing a 'path' column.")
        if "status" in table.columns:
            accepted = {str(s).lower() for s in accepted_statuses}
            table = table[table["status"].astype(str).str.lower().isin(accepted)].copy()
        for raw in table["path"].dropna().astype(str):
            p = Path(raw).expanduser()
            candidates = [p] if p.is_absolute() else [manifest.parent / p, root / p]
            chosen = next(
                (q.resolve() for q in candidates if q.exists()), candidates[0].resolve()
            )
            dirs.append(chosen)
    else:
        if not root.exists():
            raise FileNotFoundError(f"Model root does not exist: {root}")
        dirs = [
            p.resolve()
            for p in root.iterdir()
            if p.is_dir() and p.name != "_policy_bundles"
        ]

    out = [
        p
        for p in dirs
        if (p / "metadata.json").exists() and (p / "arrays.npz").exists()
    ]
    return sorted(dict.fromkeys(out), key=lambda p: str(p))


def inspect_saved_model(model_dir: str | Path) -> dict[str, Any]:
    """Return array keys/shapes and a compact metadata summary."""

    d = Path(model_dir)
    meta = _read_json(d / "metadata.json")
    with np.load(d / "arrays.npz", allow_pickle=False) as z:
        arrays = {
            k: {"shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files
        }
    return {
        "model_dir": str(d),
        "metadata_top_level_keys": sorted(meta),
        "diagnostics": meta.get("diagnostics", {}),
        "money_scale": meta.get("money_scale"),
        "arrays": arrays,
    }


def _find_array_key(
    arrays: Mapping[str, np.ndarray],
    canonical: str,
    schema: GridSchema,
    *,
    required: bool = True,
) -> str | None:
    override = schema.key_overrides.get(canonical)
    candidates = (override,) if override else ARRAY_ALIASES.get(canonical, (canonical,))
    for key in candidates:
        if key and key in arrays:
            return key
    if required:
        raise KeyError(
            f"Could not find {canonical!r}. Tried {list(candidates)}. "
            f"Available arrays: {sorted(arrays)}"
        )
    return None


def _flatten_numeric(prefix: str, value: Any, out: dict[str, float]) -> None:
    """Recursively flatten numeric metadata, including coefficient vectors."""

    if value is None or isinstance(value, (str, bytes, bool)):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            new_prefix = f"{prefix}__{key}" if prefix else str(key)
            _flatten_numeric(new_prefix, child, out)
        return
    if isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            _flatten_numeric(f"{prefix}__{i}", child, out)
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
        if np.isfinite(x):
            out[prefix] = x


def _flatten_categorical(prefix: str, value: Any, out: dict[str, str]) -> None:
    """Recursively flatten categorical metadata.

    String-valued policy labels such as ``tax.kind`` and economically relevant
    booleans are retained.  Free-form names, paths, and source labels are
    excluded below, because treating a model name as a category would let the
    surrogate memorize parameterizations rather than interpolate across them.
    """

    if value is None:
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            new_prefix = f"{prefix}__{key}" if prefix else str(key)
            _flatten_categorical(new_prefix, child, out)
        return
    if isinstance(value, (list, tuple)):
        # Lists of strings are usually names or labels rather than primitive
        # policy parameters.  Scalar list elements that are numeric continue to
        # be handled by _flatten_numeric.
        return
    if isinstance(value, (bool, np.bool_)):
        out[prefix] = "true" if bool(value) else "false"
        return
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        out[prefix] = str(text)


DEFAULT_PARAMETER_EXCLUDE = re.compile(
    r"(?:^|__)(?:"
    r"seed|show_progress|n_jobs|chunks?_per_worker|"
    r"income_grid_size|liquid_grid_size|illiquid_grid_size|"
    r"max_iterations|ct_max_iterations|solver_max_iterations|"
    r"convergence_tol|ct_convergence_tol|ct_relative_convergence_tol|"
    r"ct_hjb_residual_tol|ct_delta_min|ct_delta_max|ct_delta_base|"
    r"ct_delta_backoff|ct_slow_iterations|ct_max_restarts|"
    r"ct_regime_stability_iterations|ct_regime_change_tol|"
    r"ct_root_tol|ct_foc_tol|ct_min_marginal_value|"
    r"ct_drift_tolerance|ct_zero_drift_tolerance|"
    r"ct_state_constraint_tolerance|ct_generator_tolerance|"
    r"ct_max_abs_policy|ct_residual_growth_restart|derivative_floor|"
    r"policy_drift_clip|delta_max|delta_base|slow_iterations|"
    r"adjustment_tolerance|data_grid_quantile"
    r")(?:$|__)",
    flags=re.IGNORECASE,
)

DEFAULT_CATEGORICAL_PARAMETER_EXCLUDE = re.compile(
    r"(?:^|__)(?:"
    r"name|source|bundle_path|path|file|filename|policy_bundle_name|"
    r"id_col|year_col|age_col|group_col|education_col|"
    r"income_state_col|employment_state_col|employed_col|"
    r"current_income_col|gross_income_col|labor_income_col|"
    r"potential_labor_col|unemployment_col|pension_col|taxes_col|"
    r"liquid_col|illiquid_col|show_progress|ct_validate_generator|"
    r"ct_enforce_stationary_conditions"
    r")(?:$|__)",
    flags=re.IGNORECASE,
)


def extract_numeric_parameters(
    metadata: Mapping[str, Any],
    *,
    include_derived: bool = False,
    exclude_pattern: re.Pattern[str] | None = DEFAULT_PARAMETER_EXCLUDE,
) -> dict[str, float]:
    """Extract numeric structural and policy-function parameters.

    Nested dictionaries and coefficient lists are recursively expanded.
    Numerical solver controls and grid sizes are excluded by default because
    they are not economic inputs to the desired policy function.
    """

    raw: dict[str, float] = {}
    _flatten_numeric("param__config", metadata.get("config", {}), raw)
    policy_spec = (metadata.get("policy_bundle_metadata", {}) or {}).get("spec", {})
    _flatten_numeric("param__policy", policy_spec, raw)
    if include_derived:
        _flatten_numeric("param__derived", metadata.get("metadata", {}), raw)
    if exclude_pattern is None:
        return raw
    return {k: v for k, v in raw.items() if not exclude_pattern.search(k)}


def extract_categorical_parameters(
    metadata: Mapping[str, Any],
    *,
    include_derived: bool = False,
    exclude_pattern: re.Pattern[str] | None = DEFAULT_CATEGORICAL_PARAMETER_EXCLUDE,
) -> dict[str, str]:
    """Extract string and Boolean structural/policy parameters.

    The automatic selection stage retains only categories that vary across
    solved parameterizations.  Consequently constant values such as
    ``income.kind='rouwenhorst'`` disappear automatically, while a varying
    ``tax.kind`` is one-hot encoded in the fitted polynomial.
    """

    raw: dict[str, str] = {}
    _flatten_categorical("param__config", metadata.get("config", {}), raw)
    policy_spec = (metadata.get("policy_bundle_metadata", {}) or {}).get("spec", {})
    _flatten_categorical("param__policy", policy_spec, raw)
    if include_derived:
        _flatten_categorical("param__derived", metadata.get("metadata", {}), raw)
    if exclude_pattern is None:
        return raw
    return {k: v for k, v in raw.items() if not exclude_pattern.search(k)}


def _model_id(metadata: Mapping[str, Any], model_dir: Path) -> str:
    diag = metadata.get("diagnostics", {}) or {}
    return str(diag.get("model_id") or metadata.get("model_id") or model_dir.name)


def _model_is_usable(
    metadata: Mapping[str, Any],
    *,
    accepted_statuses: Sequence[str] = ("success", "exists", "converged"),
    max_solver_distance: float | None = None,
    max_relative_distance: float | None = None,
    max_hjb_residual: float | None = None,
) -> tuple[bool, str, str]:
    """Return usability, a human-readable reason, and a stable reason code.

    The updated HJB solver stores absolute and relative value changes plus a
    normalized HJB residual.  Optional cutoffs provide additional safeguards,
    although a new-format ``success`` model has already passed the solver's own
    convergence checks.
    """

    diag = metadata.get("diagnostics", {}) or {}
    status = diag.get("status")
    if status is not None:
        accepted = {str(s).lower() for s in accepted_statuses}
        if str(status).lower() not in accepted:
            return False, f"solver status={status!r}", "solver_status"

    if max_solver_distance is not None:
        distance = _finite_float(diag.get("max_distance"), np.nan)
        if np.isfinite(distance) and distance > float(max_solver_distance):
            return (
                False,
                f"max solver distance {distance:.8g} exceeds "
                f"threshold {float(max_solver_distance):.8g}",
                "solver_distance",
            )

    if max_relative_distance is not None:
        distance = _finite_float(diag.get("max_relative_distance"), np.nan)
        if np.isfinite(distance) and distance > float(max_relative_distance):
            return (
                False,
                f"max relative distance {distance:.8g} exceeds "
                f"threshold {float(max_relative_distance):.8g}",
                "solver_relative_distance",
            )

    if max_hjb_residual is not None:
        residual = _finite_float(diag.get("max_hjb_residual"), np.nan)
        if np.isfinite(residual) and residual > float(max_hjb_residual):
            return (
                False,
                f"max normalized HJB residual {residual:.8g} exceeds "
                f"threshold {float(max_hjb_residual):.8g}",
                "solver_hjb_residual",
            )

    return True, "", ""


def _inspect_policy_storage(
    model_dir: Path,
    metadata: Mapping[str, Any],
    schema: GridSchema,
) -> dict[str, Any]:
    """Inspect the saved policy representation without decompressing tensors.

    The current compact format stores only consumption and deposit as full
    GHKEBA policy arrays. Older formats may additionally contain drifts or
    next-asset arrays. Both are readable, but compact-only runs can request a
    strict check through ``GridSchema.require_compact_policies``.
    """

    arrays_path = model_dir / "arrays.npz"
    with np.load(arrays_path, allow_pickle=False) as z:
        keys = set(z.files)
        consumption_key = _find_array_key(z, "consumption", schema, required=False)
        deposit_key = _find_array_key(z, "deposit", schema, required=False)
        liquid_drift_key = _find_array_key(z, "liquid_drift", schema, required=False)
        illiquid_drift_key = _find_array_key(
            z, "illiquid_drift", schema, required=False
        )
        next_liquid_key = _find_array_key(
            z, "next_liquid_assets", schema, required=False
        )
        next_illiquid_key = _find_array_key(
            z, "next_illiquid_assets", schema, required=False
        )

        if consumption_key is None:
            policy_dtype = None
            policy_shape = None
        else:
            policy_dtype = str(z[consumption_key].dtype)
            policy_shape = "x".join(str(int(v)) for v in z[consumption_key].shape)

    has_drift = liquid_drift_key is not None or illiquid_drift_key is not None
    has_next = next_liquid_key is not None or next_illiquid_key is not None
    has_compact_primitives = consumption_key is not None and deposit_key is not None

    if has_compact_primitives and not has_drift and not has_next:
        storage_format = "compact_consumption_deposit"
    elif has_compact_primitives:
        storage_format = "consumption_deposit_with_legacy_arrays"
    elif consumption_key is not None and illiquid_drift_key is not None:
        storage_format = "legacy_deposit_recoverable_from_drift"
    else:
        storage_format = "incomplete"

    declared = metadata.get("saved_arrays", {}) or {}
    declared_policy_arrays = declared.get("policy_arrays")
    declared_drifts_saved = declared.get("drifts_saved")

    return {
        "policy_storage_format": storage_format,
        "policy_dtype": policy_dtype,
        "policy_shape": policy_shape,
        "has_consumption_policy": consumption_key is not None,
        "has_deposit_policy": deposit_key is not None,
        "has_saved_liquid_drift": liquid_drift_key is not None,
        "has_saved_illiquid_drift": illiquid_drift_key is not None,
        "has_saved_next_liquid_assets": next_liquid_key is not None,
        "has_saved_next_illiquid_assets": next_illiquid_key is not None,
        "metadata_policy_arrays": (
            json.dumps(declared_policy_arrays)
            if declared_policy_arrays is not None
            else None
        ),
        "metadata_drifts_saved": declared_drifts_saved,
        "array_key_count": len(keys),
    }


def _validate_policy_storage_for_training(
    storage: Mapping[str, Any],
    *,
    schema: GridSchema,
) -> tuple[bool, str]:
    """Validate that a saved model can supply consumption and deposit."""

    if not bool(storage.get("has_consumption_policy")):
        return False, "missing saved consumption policy"

    has_deposit = bool(storage.get("has_deposit_policy"))
    has_illiquid_drift = bool(storage.get("has_saved_illiquid_drift"))
    if not has_deposit and not has_illiquid_drift:
        return False, "missing deposit policy and legacy illiquid drift fallback"

    if (
        schema.require_compact_policies
        and storage.get("policy_storage_format") != "compact_consumption_deposit"
    ):
        return (
            False,
            "not compact consumption+deposit storage; rerun without "
            "--require-compact-policies to accept legacy arrays",
        )

    return True, ""


def build_model_catalog(
    model_dirs: Sequence[str | Path],
    *,
    schema: GridSchema = GridSchema(),
    max_solver_distance: float | None = None,
    max_relative_distance: float | None = None,
    max_hjb_residual: float | None = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build a transparent model catalog with explicit rejection codes."""

    rows: list[dict[str, Any]] = []
    iterator = _progress_iter(
        model_dirs,
        total=len(model_dirs),
        description="Cataloguing grids",
        unit="model",
        enabled=show_progress,
    )
    print_every = max(1, len(model_dirs) // 10)
    for position, raw in enumerate(iterator, start=1):
        if (
            show_progress
            and _tqdm is None
            and verbose
            and (
                position == 1
                or position == len(model_dirs)
                or position % print_every == 0
            )
        ):
            _status(
                f"  catalogued {position:,}/{len(model_dirs):,} models",
                verbose=verbose,
            )

        d = Path(raw)
        try:
            meta = _read_json(d / "metadata.json")
            diag = meta.get("diagnostics", {}) or {}
            usable, reason, rejection_code = _model_is_usable(
                meta,
                max_solver_distance=max_solver_distance,
                max_relative_distance=max_relative_distance,
                max_hjb_residual=max_hjb_residual,
            )

            storage = _inspect_policy_storage(d, meta, schema)
            storage_usable, storage_reason = _validate_policy_storage_for_training(
                storage,
                schema=schema,
            )
            if usable and not storage_usable:
                usable = False
                reason = storage_reason
                rejection_code = "policy_storage"
            elif not usable and storage_reason:
                reason = f"{reason}; {storage_reason}"

            numeric_params = extract_numeric_parameters(
                meta,
                include_derived=schema.include_derived_metadata_parameters,
            )
            categorical_params = extract_categorical_parameters(
                meta,
                include_derived=schema.include_derived_metadata_parameters,
            )

            distance_raw = diag.get("max_distance")
            try:
                distance = float(distance_raw)
            except (TypeError, ValueError):
                distance = np.nan

            row: dict[str, Any] = {
                "model_id": _model_id(meta, d),
                "model_dir": str(d),
                "usable": bool(usable),
                "rejection_code": rejection_code,
                "reason": reason,
                "solver_status": diag.get("status"),
                "solver_max_distance": distance,
                "solver_max_relative_distance": _finite_float(
                    diag.get("max_relative_distance"), np.nan
                ),
                "solver_max_hjb_residual": _finite_float(
                    diag.get("max_hjb_residual"), np.nan
                ),
                "solver_max_regime_change": _finite_float(
                    diag.get("max_regime_change"), np.nan
                ),
                "solver_total_restarts": _finite_float(
                    diag.get("total_restarts"), np.nan
                ),
                "solver_total_root_solves": _finite_float(
                    diag.get("total_root_solves"), np.nan
                ),
                "solver_min_feasibility_slack": _finite_float(
                    diag.get("min_feasibility_slack"), np.nan
                ),
                "solver_max_zero_drift_error": _finite_float(
                    diag.get("max_zero_drift_error"), np.nan
                ),
                "solver_max_generator_row_sum": _finite_float(
                    diag.get("max_generator_row_sum"), np.nan
                ),
                "solver_max_iterations": diag.get("max_iterations"),
                "money_scale": float(meta.get("money_scale", 1.0) or 1.0),
            }
            row.update(storage)
            row.update(numeric_params)
            row.update(categorical_params)
            rows.append(row)
        except Exception as exc:  # catalog all failures rather than aborting
            rows.append(
                {
                    "model_id": d.name,
                    "model_dir": str(d),
                    "usable": False,
                    "rejection_code": "metadata_or_array_error",
                    "reason": f"metadata/array error: {exc}",
                    "solver_status": None,
                    "solver_max_distance": np.nan,
                    "solver_max_iterations": np.nan,
                    "money_scale": np.nan,
                }
            )

    return pd.DataFrame(rows)


def choose_parameter_columns(
    catalog: pd.DataFrame,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    min_coverage: float = 0.80,
) -> list[str]:
    if include is not None:
        missing = [c for c in include if c not in catalog.columns]
        if missing:
            raise KeyError(f"Requested parameters are missing from metadata: {missing}")
        return [c for c in include if c not in set(exclude)]

    usable = catalog[catalog["usable"]].copy()
    selected: list[str] = []
    excluded = set(exclude)
    for col in catalog.columns:
        if not col.startswith("param__") or col in excluded:
            continue
        x = pd.to_numeric(usable[col], errors="coerce")
        if x.notna().mean() < min_coverage:
            continue
        if x.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return sorted(selected)


def choose_categorical_parameter_columns(
    catalog: pd.DataFrame,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    min_coverage: float = 0.80,
) -> list[str]:
    """Select varying string/Boolean parameters for one-hot encoding."""

    if include is not None:
        missing = [c for c in include if c not in catalog.columns]
        if missing:
            raise KeyError(
                f"Requested categorical parameters are missing from metadata: {missing}"
            )
        return [c for c in include if c not in set(exclude)]

    usable = catalog[catalog["usable"]].copy()
    selected: list[str] = []
    excluded = set(exclude)
    for col in catalog.columns:
        if not col.startswith("param__") or col in excluded:
            continue
        raw = usable[col] if col in usable else pd.Series(dtype=object)
        # Numeric parameters are handled by choose_parameter_columns.
        numeric = pd.to_numeric(raw, errors="coerce")
        nonmissing = raw.notna()
        if nonmissing.mean() < min_coverage:
            continue
        if numeric.notna().mean() >= min_coverage:
            continue
        values = raw[nonmissing].astype(str)
        if values.nunique(dropna=True) <= 1:
            continue
        selected.append(col)
    return sorted(selected)


# -----------------------------------------------------------------------------
# Grid adapter and balanced sampling
# -----------------------------------------------------------------------------


def _as_1d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {x.shape}.")
    return x


def _metadata_group_values(metadata: Mapping[str, Any], g_count: int) -> list[Any]:
    candidates = [
        metadata.get("group_values"),
        (metadata.get("metadata", {}) or {}).get("group_values"),
        ((metadata.get("policy_bundle_metadata", {}) or {}).get("spec", {}) or {}).get(
            "group_values"
        ),
    ]
    for values in candidates:
        if isinstance(values, (list, tuple)) and len(values) == g_count:
            return list(values)
    return list(range(g_count))


def _policy_shape_to_canonical(
    shape: Sequence[int], layout: str
) -> tuple[int, int, int, int, int, int]:
    layout = layout.upper()
    if len(shape) != len(layout):
        raise ValueError(
            f"Policy array has shape {tuple(shape)} but layout {layout!r} has "
            f"{len(layout)} axes."
        )
    sizes = dict(zip(layout, map(int, shape)))
    sizes.setdefault("G", 1)
    return tuple(sizes[k] for k in "GHKEBA")  # type: ignore[return-value]


def _index_policy_array(
    arr: np.ndarray,
    layout: str,
    coords: Mapping[str, np.ndarray],
) -> np.ndarray:
    idx = tuple(coords[letter] for letter in layout.upper())
    return np.asarray(arr[idx], dtype=float)


def _index_state_array(
    arr: np.ndarray,
    *,
    g: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    """Index common income-array layouts without silently permuting axes."""

    arr = np.asarray(arr)
    if arr.ndim == 4:
        return np.asarray(arr[g, h, k, e], dtype=float)
    if arr.ndim == 3:
        return np.asarray(arr[h, k, e], dtype=float)
    if arr.ndim == 2:
        # Usually H x K; employment-invariant income.
        return np.asarray(arr[h, k], dtype=float)
    if arr.ndim == 1:
        return np.asarray(arr[k], dtype=float)
    raise ValueError(f"Unsupported state-array shape {arr.shape}.")


def _stratified_coordinates(
    canonical_shape: tuple[int, int, int, int, int, int],
    n: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Marginally stratified state-grid sample with explicit boundary anchors."""

    if n <= 0:
        raise ValueError("n must be positive")
    letters = "GHKEBA"
    out: dict[str, np.ndarray] = {}
    for letter, size in zip(letters, canonical_shape):
        if size <= 0:
            raise ValueError(f"Axis {letter} has invalid size {size}.")
        if letter in {"G", "E"}:
            values = np.arange(n, dtype=np.int64) % size
            rng.shuffle(values)
        else:
            u = (np.arange(n, dtype=float) + rng.random(n)) / n
            values = np.minimum((u * size).astype(np.int64), size - 1)
            rng.shuffle(values)
        out[letter] = values

    # Guarantee that each non-categorical axis reaches both boundaries and its
    # midpoint, while retaining varied values on the other axes.
    cursor = 0
    for letter, size in zip(letters, canonical_shape):
        for value in dict.fromkeys([0, size // 2, size - 1]):
            if cursor >= n:
                break
            out[letter][cursor] = int(value)
            cursor += 1
    return out


def _mpc_aware_coordinates(
    canonical_shape: tuple[int, int, int, int, int, int],
    n: int,
    rng: np.random.Generator,
    *,
    liquid_grid: np.ndarray,
    pair_share: float,
    check_sizes: Sequence[float],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Sample broad grid points plus balanced liquid-wealth pairs.

    Earlier versions deliberately concentrated pair baselines very close to the
    borrowing constraint.  That was useful diagnostically, but it changed the
    effective training distribution and could make the fitted MPC function too
    steep near zero liquidity.  The revised sampler stratifies pair baselines
    over the full feasible liquid grid and then adds a small set of forced
    lower-bound anchors.  It therefore learns local slopes without letting
    low-liquid states dominate the level fit.

    Returns
    -------
    coords
        Coordinate arrays in canonical GHKEBA order.
    pair_id
        Integer pair identifier, or -1 for ordinary non-paired rows.
    pair_role
        0 for baseline, 1 for treated, and -1 for ordinary rows.
    pair_check
        Actual liquid-grid difference between treated and baseline rows.
    """

    if n <= 0:
        raise ValueError("n must be positive.")

    share = float(np.clip(pair_share, 0.0, 0.90))
    grid = np.asarray(liquid_grid, dtype=float)
    if grid.ndim != 1 or len(grid) < 2:
        raise ValueError("liquid_grid must contain at least two points.")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("liquid_grid must be strictly increasing.")

    checks = np.asarray(
        [float(x) for x in check_sizes if np.isfinite(float(x)) and float(x) > 0.0],
        dtype=float,
    )
    if checks.size == 0:
        checks = np.array([0.05, 0.10, 0.20], dtype=float)

    n_pairs = int(math.floor((share * n) / 2.0))
    n_pair_rows = 2 * n_pairs
    n_random = n - n_pair_rows

    letters = "GHKEBA"
    pieces: dict[str, list[np.ndarray]] = {letter: [] for letter in letters}
    pair_ids: list[np.ndarray] = []
    pair_roles: list[np.ndarray] = []
    pair_checks: list[np.ndarray] = []

    if n_random > 0:
        ordinary = _stratified_coordinates(canonical_shape, n_random, rng)
        for letter in letters:
            pieces[letter].append(ordinary[letter])
        pair_ids.append(np.full(n_random, -1, dtype=np.int64))
        pair_roles.append(np.full(n_random, -1, dtype=np.int8))
        pair_checks.append(np.full(n_random, np.nan, dtype=float))

    if n_pairs > 0:
        pair_states = _stratified_coordinates(canonical_shape, n_pairs, rng)
        base_b = np.empty(n_pairs, dtype=np.int64)
        treated_b = np.empty(n_pairs, dtype=np.int64)
        actual_check = np.empty(n_pairs, dtype=float)

        span = float(grid[-1] - grid[0])
        usable_checks = checks[checks < span - EPS]
        if usable_checks.size == 0:
            usable_checks = np.array([max(span / 10.0, EPS)], dtype=float)
        draw_checks = rng.choice(usable_checks, size=n_pairs, replace=True)

        # Stratification gives approximately equal coverage over feasible
        # baseline positions.  A few deterministic lower-bound anchors retain
        # precision exactly where the borrowing constraint begins to bind.
        baseline_quantiles = (
            np.arange(n_pairs, dtype=float) + rng.random(n_pairs)
        ) / n_pairs
        rng.shuffle(baseline_quantiles)

        for i, desired_check in enumerate(draw_checks):
            max_start_value = grid[-1] - float(desired_check)
            max_start = int(np.searchsorted(grid, max_start_value, side="right") - 1)
            max_start = int(np.clip(max_start, 0, len(grid) - 2))

            start = int(np.floor(baseline_quantiles[i] * (max_start + 1)))
            start = int(np.clip(start, 0, max_start))

            # Force a small number of exact low-slack pairs, without making
            # them half of the training sample as in the earlier specification.
            if i < min(8, n_pairs):
                start = min(i // 2, max_start)

            target_value = grid[start] + float(desired_check)
            right = int(np.searchsorted(grid, target_value, side="left"))
            right = int(np.clip(right, start + 1, len(grid) - 1))
            left = max(start + 1, right - 1)
            stop = min(
                (left, right),
                key=lambda j: abs(float(grid[j]) - target_value),
            )
            stop = int(np.clip(stop, start + 1, len(grid) - 1))

            base_b[i] = start
            treated_b[i] = stop
            actual_check[i] = float(grid[stop] - grid[start])

        for letter in letters:
            if letter == "B":
                values = np.empty(n_pair_rows, dtype=np.int64)
                values[0::2] = base_b
                values[1::2] = treated_b
            else:
                values = np.repeat(pair_states[letter], 2)
            pieces[letter].append(values)

        pair_ids.append(np.repeat(np.arange(n_pairs, dtype=np.int64), 2))
        roles = np.empty(n_pair_rows, dtype=np.int8)
        roles[0::2] = 0
        roles[1::2] = 1
        pair_roles.append(roles)
        pair_checks.append(np.repeat(actual_check, 2))

    coords = {
        letter: np.concatenate(pieces[letter]).astype(np.int64, copy=False)
        for letter in letters
    }
    pair_id = np.concatenate(pair_ids)
    pair_role = np.concatenate(pair_roles)
    pair_check = np.concatenate(pair_checks)

    if len(pair_id) != n:
        raise RuntimeError("MPC-aware sampler returned the wrong number of rows.")

    return coords, pair_id, pair_role, pair_check


def _resolve_ages(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    schema: GridSchema,
    h: int,
) -> np.ndarray:
    key = _find_array_key(arrays, "ages", schema, required=False)
    if key is not None:
        ages = _as_1d(arrays[key], key).astype(int)
    else:
        cfg = metadata.get("config", {}) or {}
        min_age = int(cfg.get("min_age", 0))
        ages = np.arange(min_age, min_age + h, dtype=int)
    if len(ages) != h:
        raise ValueError(f"Age grid length {len(ages)} does not match H={h}.")
    return ages


def _resolve_income_grid(
    arrays: Mapping[str, np.ndarray], schema: GridSchema, g_count: int, k_count: int
) -> tuple[np.ndarray, bool]:
    log_key = _find_array_key(arrays, "log_income_grid", schema, required=False)
    raw_key = _find_array_key(arrays, "income_grid", schema, required=False)
    if log_key is not None:
        grid = np.asarray(arrays[log_key], dtype=float)
        is_log = True
    elif raw_key is not None:
        grid = np.asarray(arrays[raw_key], dtype=float)
        is_log = False
    else:
        # Current income can still be taken from after-tax/gross arrays.  The
        # missing labor-income-state value is represented as NaN.
        return np.full((g_count, k_count), np.nan), False

    if grid.ndim == 1:
        grid = np.repeat(grid[None, :], g_count, axis=0)
    if grid.shape != (g_count, k_count):
        raise ValueError(
            f"Income grid has shape {grid.shape}; expected {(g_count, k_count)}."
        )
    return grid, is_log


def _money_factor(metadata: Mapping[str, Any], schema: GridSchema) -> float:
    if schema.money_units == "model":
        return 1.0
    scale = float(metadata.get("money_scale", 1.0) or 1.0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid money_scale={scale}.")
    return scale


def compute_taxes(
    gross: np.ndarray | Sequence[float],
    labor: np.ndarray | Sequence[float],
    pension: np.ndarray | Sequence[float],
    *,
    tax_kind: str | Sequence[str] | np.ndarray,
    flat_rate: float | np.ndarray,
    deduction: float | np.ndarray,
    payroll_rate: float | np.ndarray,
    progressive_rate: float | np.ndarray,
    progressive_exponent: float | np.ndarray,
    tax_cap: float | np.ndarray,
) -> np.ndarray:
    """Vectorized implementation of the model's three tax schedules.

    ``tax_kind`` may be a scalar or row-specific array containing ``none``,
    ``flat``, or ``progressive``.  All monetary inputs and ``deduction`` must be
    expressed in the same units.
    """

    gross_arr = np.maximum(np.asarray(gross, dtype=float), 0.0)
    labor_arr = np.maximum(np.asarray(labor, dtype=float), 0.0)
    pension_arr = np.maximum(np.asarray(pension, dtype=float), 0.0)
    shape = np.broadcast(gross_arr, labor_arr, pension_arr).shape
    gross_arr = np.broadcast_to(gross_arr, shape).astype(float, copy=False)
    labor_arr = np.broadcast_to(labor_arr, shape).astype(float, copy=False)
    pension_arr = np.broadcast_to(pension_arr, shape).astype(float, copy=False)

    kind = np.asarray(tax_kind, dtype=object)
    kind = np.broadcast_to(kind, shape).astype(str)
    flat = np.broadcast_to(np.asarray(flat_rate, dtype=float), shape)
    deduction_arr = np.broadcast_to(np.asarray(deduction, dtype=float), shape)
    payroll = np.broadcast_to(np.asarray(payroll_rate, dtype=float), shape)
    progressive = np.broadcast_to(np.asarray(progressive_rate, dtype=float), shape)
    exponent = np.broadcast_to(np.asarray(progressive_exponent, dtype=float), shape)
    cap = np.broadcast_to(np.asarray(tax_cap, dtype=float), shape)

    tax = np.zeros(shape, dtype=float)
    is_none = kind == "none"
    is_flat = kind == "flat"
    is_progressive = kind == "progressive"
    unsupported = ~(is_none | is_flat | is_progressive)
    if unsupported.any():
        bad = sorted(set(kind[unsupported].tolist()))
        raise ValueError(f"Unsupported tax policy kind(s): {bad}")

    tax[is_flat] = flat[is_flat] * gross_arr[is_flat]
    taxable = np.maximum(gross_arr - deduction_arr, 0.0)
    if is_progressive.any():
        tax[is_progressive] = (
            payroll[is_progressive] * labor_arr[is_progressive]
            + flat[is_progressive] * taxable[is_progressive]
            + progressive[is_progressive]
            * np.power(
                taxable[is_progressive],
                exponent[is_progressive],
            )
        )
    return np.minimum(
        np.maximum(tax, 0.0),
        cap * np.maximum(gross_arr, EPS),
    )


def _continuous_rate(annual_rate: Any, fallback: float = 0.0) -> float:
    """Convert a net annual return to the continuous rate used by the solver."""

    try:
        rate = float(annual_rate)
    except (TypeError, ValueError):
        return float(fallback)
    if not np.isfinite(rate) or rate <= -1.0:
        return float(fallback)
    return float(np.log1p(rate))


def _finite_float(value: Any, fallback: float = 0.0) -> float:
    """Convert optional metadata scalars to finite floats."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return out if np.isfinite(out) else float(fallback)


def _accounting_values_from_metadata(
    metadata: Mapping[str, Any],
    *,
    money_factor: float,
) -> dict[str, Any]:
    """Return row-level primitives needed to reconstruct both asset policies."""

    cfg = metadata.get("config", {}) or {}
    derived = metadata.get("metadata", {}) or {}
    policy_spec = (metadata.get("policy_bundle_metadata", {}) or {}).get(
        "spec", {}
    ) or {}
    tax = policy_spec.get("tax", {}) or {}

    rb_pos = derived.get(
        "rb_pos_ct",
        _continuous_rate(cfg.get("liquid_interest_rate", 0.0)),
    )
    rb_neg = derived.get(
        "rb_neg_ct",
        _continuous_rate(cfg.get("borrowing_interest_rate", 0.0)),
    )
    ra = derived.get(
        "ra_ct",
        _continuous_rate(cfg.get("illiquid_interest_rate", 0.0)),
    )
    chi0 = derived.get("chi0", cfg.get("ct_linear_adjustment_cost", 0.0))
    chi1 = derived.get("chi1", cfg.get("ct_convex_adjustment_cost", 0.0))
    xi = derived.get("xi", cfg.get("ct_automatic_illiquid_income_share", 0.0))

    values: dict[str, Any] = {
        "acct__ct_time_step": _finite_float(cfg.get("ct_time_step"), 1.0),
        "acct__rb_pos_ct": float(rb_pos),
        "acct__rb_neg_ct": float(rb_neg),
        "acct__ra_ct": float(ra),
        "acct__chi0": float(chi0),
        "acct__chi1": float(chi1),
        "acct__xi": float(xi),
        "acct__illiquid_cost_floor": _finite_float(
            cfg.get("ct_illiquid_cost_floor"), 1.0e-6
        )
        * float(money_factor),
        # The updated HJB solver rejects invalid policies instead of clipping
        # drifts.  Older files may store a numerical clip, while new files store
        # None.  A zero value means that no drift clipping is applied.
        "acct__policy_drift_clip": _finite_float(cfg.get("ct_policy_drift_clip"), 0.0),
        "acct__retirement_age": float(cfg.get("retirement_age", np.inf)),
        "acct__terminal_age": float(cfg.get("max_age", np.inf)),
        "acct__tax_kind": str(tax.get("kind", "none")),
        "acct__tax_flat_rate": float(tax.get("flat_rate", 0.0)),
        "acct__tax_deduction": float(tax.get("deduction", 0.0)) * float(money_factor),
        "acct__tax_payroll_rate": float(tax.get("payroll_rate", 0.0)),
        "acct__tax_progressive_rate": float(tax.get("progressive_rate", 0.0)),
        "acct__tax_progressive_exponent": float(tax.get("progressive_exponent", 1.0)),
        "acct__tax_cap": float(tax.get("tax_cap", 1.0)),
        "acct__minimum_after_tax_income": _finite_float(
            tax.get("minimum_after_tax_income"), 0.0
        )
        * float(money_factor),
    }
    for key, value in values.items():
        if key == "acct__tax_kind":
            continue
        if not np.isfinite(float(value)) and key not in {
            "acct__retirement_age",
            "acct__terminal_age",
            "acct__policy_drift_clip",
        }:
            raise ValueError(f"Invalid accounting metadata {key}={value!r}.")
    return values


def _reconstruct_saved_policy_accounting(
    *,
    consumption: np.ndarray,
    deposit: np.ndarray,
    current_liquid_assets: np.ndarray,
    current_illiquid_assets: np.ndarray,
    after_tax_income: np.ndarray,
    liquid_index: np.ndarray | None,
    illiquid_index: np.ndarray | None,
    liquid_grid: np.ndarray,
    illiquid_grid: np.ndarray,
    accounting: Mapping[str, Any],
    money_factor: float,
) -> dict[str, np.ndarray]:
    """Reconstruct all redundant policies from saved consumption and deposit.

    This mirrors the compact saved-model prediction path: apply the resource
    equations, the solver's optional drift clipping, the boundary state
    constraints, and finally the annual transition and grid projection.  The
    function works in the units selected by ``GridSchema.money_units``.
    """

    c = np.asarray(consumption, dtype=float)
    d = np.asarray(deposit, dtype=float)
    b = np.asarray(current_liquid_assets, dtype=float)
    a = np.asarray(current_illiquid_assets, dtype=float)
    y = np.asarray(after_tax_income, dtype=float)
    ib = None if liquid_index is None else np.asarray(liquid_index, dtype=np.int64)
    ia = None if illiquid_index is None else np.asarray(illiquid_index, dtype=np.int64)

    a_floor = max(float(accounting["acct__illiquid_cost_floor"]), EPS)
    chi0 = float(accounting["acct__chi0"])
    chi1 = float(accounting["acct__chi1"])
    xi = float(accounting["acct__xi"])
    rb_pos = float(accounting["acct__rb_pos_ct"])
    rb_neg = float(accounting["acct__rb_neg_ct"])
    ra = float(accounting["acct__ra_ct"])

    cost = chi0 * np.abs(d) + 0.5 * chi1 * d**2 / np.maximum(a, a_floor)
    liquid_return = np.where(b >= 0.0, rb_pos * b, rb_neg * b)
    sb = (1.0 - xi) * y + liquid_return - d - cost - c
    sa = ra * a + xi * y + d

    drift_clip = float(accounting.get("acct__policy_drift_clip", 0.0))
    if np.isfinite(drift_clip) and drift_clip > 0.0:
        sb = np.clip(sb, -drift_clip, drift_clip)
        sa = np.clip(sa, -drift_clip, drift_clip)

    # Match the solver's state constraints.  Population sampling evaluates
    # policies at continuous asset values, so boundary detection cannot rely
    # only on integer grid indices.
    b_lo = float(liquid_grid[0]) * float(money_factor)
    b_hi = float(liquid_grid[-1]) * float(money_factor)
    a_lo = float(illiquid_grid[0]) * float(money_factor)
    a_hi = float(illiquid_grid[-1]) * float(money_factor)
    b_tol = 1.0e-10 * max(1.0, abs(b_lo), abs(b_hi))
    a_tol = 1.0e-10 * max(1.0, abs(a_lo), abs(a_hi))
    at_b_lo = b <= b_lo + b_tol
    at_b_hi = b >= b_hi - b_tol
    at_a_lo = a <= a_lo + a_tol
    at_a_hi = a >= a_hi - a_tol
    if ib is not None:
        at_b_lo |= ib == 0
        at_b_hi |= ib == len(liquid_grid) - 1
    if ia is not None:
        at_a_lo |= ia == 0
        at_a_hi |= ia == len(illiquid_grid) - 1
    sb = np.where(at_b_lo, np.maximum(sb, 0.0), sb)
    sb = np.where(at_b_hi, np.minimum(sb, 0.0), sb)
    sa = np.where(at_a_lo, np.maximum(sa, 0.0), sa)
    sa = np.where(at_a_hi, np.minimum(sa, 0.0), sa)

    dt = float(accounting["acct__ct_time_step"])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Invalid ct_time_step={dt!r} in metadata.")

    next_b = np.clip(b + dt * sb, b_lo, b_hi)
    next_a = np.clip(a + dt * sa, a_lo, a_hi)

    return {
        "adjustment_cost": cost,
        "liquid_drift": sb,
        "illiquid_drift": sa,
        "next_liquid_assets": next_b,
        "next_illiquid_assets": next_a,
        "delta_liquid_assets": next_b - b,
        "delta_illiquid_assets": next_a - a,
    }


@dataclass(frozen=True)
class PopulationSamplingSpec:
    """Population mixture used to sample exact policy states for the surrogate.

    The defaults deliberately oversample low-liquidity households.  Shares are
    sampling weights, not estimates of population frequencies.  A small broad-
    support component protects global policy accuracy outside the main mixture.
    """

    poor_hand_to_mouth_share: float = 0.35
    wealthy_hand_to_mouth_share: float = 0.35
    broad_support_share: float = 0.10
    nonemployment_share: float = 0.20
    negative_liquid_share: float = 0.10
    borrowing_constraint_anchor_share: float = 0.25

    # The application is for working-age households. The solver treats
    # age >= retirement_age as retired, so working-age sampling uses
    # age < retirement_age. This avoids spending rows and feature capacity on
    # pension-state mass points that are outside the target application.
    working_age_only: bool = True

    # Fraction of explicit MPC pairs forced to start in debt and finish with
    # positive liquid wealth after the transfer. These pairs identify the kink
    # at zero liquid assets rather than relying on chance draws near zero.
    mpc_cross_zero_share: float = 0.30

    poor_liquid_ratio_median: float = 0.012
    poor_illiquid_ratio_median: float = 0.08
    wealthy_liquid_ratio_median: float = 0.010
    wealthy_illiquid_ratio_median: float = 2.50
    regular_liquid_ratio_median: float = 0.30
    regular_illiquid_ratio_median: float = 1.40
    liquid_ratio_log_sd: float = 0.75
    illiquid_ratio_log_sd: float = 0.85

    age_beta_a: float = 2.2
    age_beta_b: float = 2.0

    @property
    def regular_saver_share(self) -> float:
        return (
            1.0
            - float(self.poor_hand_to_mouth_share)
            - float(self.wealthy_hand_to_mouth_share)
            - float(self.broad_support_share)
        )

    def validate(self) -> None:
        for name in (
            "poor_hand_to_mouth_share",
            "wealthy_hand_to_mouth_share",
            "broad_support_share",
            "nonemployment_share",
            "negative_liquid_share",
            "borrowing_constraint_anchor_share",
            "mpc_cross_zero_share",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0,1].")
        if self.regular_saver_share < -1.0e-12:
            raise ValueError("Population sampling shares sum to more than one.")
        for name in (
            "poor_liquid_ratio_median",
            "poor_illiquid_ratio_median",
            "wealthy_liquid_ratio_median",
            "wealthy_illiquid_ratio_median",
            "regular_liquid_ratio_median",
            "regular_illiquid_ratio_median",
            "age_beta_a",
            "age_beta_b",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name in ("liquid_ratio_log_sd", "illiquid_ratio_log_sd"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative.")


def _rouwenhorst_stationary_weights(n: int) -> np.ndarray:
    """Stationary weights for the symmetric Rouwenhorst chain."""

    if n <= 0:
        raise ValueError("n must be positive.")
    if n == 1:
        return np.ones(1, dtype=float)
    weights = np.asarray(
        [math.comb(n - 1, j) for j in range(n)],
        dtype=float,
    )
    return weights / weights.sum()


def _canonical_policy_array(array: np.ndarray, layout: str) -> np.ndarray:
    """Return a policy tensor in canonical GHKEBA order."""

    arr = np.asarray(array)
    layout = str(layout).upper()
    if len(layout) == 5:
        arr = arr[None, ...]
        layout = "G" + layout
    if len(layout) != 6 or set(layout) != set("GHKEBA"):
        raise ValueError(f"Unsupported policy layout {layout!r}.")
    permutation = [layout.index(axis) for axis in "GHKEBA"]
    return np.transpose(arr, permutation)


def _continuous_grid_brackets(
    grid: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.asarray(grid, dtype=float)
    x = np.clip(np.asarray(values, dtype=float), grid[0], grid[-1])
    right = np.searchsorted(grid, x, side="left")
    right = np.clip(right, 0, len(grid) - 1)
    left = np.clip(right - 1, 0, len(grid) - 1)
    same = left == right
    denominator = np.maximum(grid[right] - grid[left], EPS)
    weight = np.where(same, 0.0, (x - grid[left]) / denominator)
    return left.astype(np.int64), right.astype(np.int64), weight.astype(float)


def _bilinear_policy_values(
    canonical_array: np.ndarray,
    *,
    g: np.ndarray,
    h: np.ndarray,
    k: np.ndarray,
    e: np.ndarray,
    liquid_values: np.ndarray,
    illiquid_values: np.ndarray,
    liquid_grid: np.ndarray,
    illiquid_grid: np.ndarray,
) -> np.ndarray:
    ib0, ib1, wb = _continuous_grid_brackets(liquid_grid, liquid_values)
    ia0, ia1, wa = _continuous_grid_brackets(illiquid_grid, illiquid_values)
    v00 = canonical_array[g, h, k, e, ib0, ia0]
    v10 = canonical_array[g, h, k, e, ib1, ia0]
    v01 = canonical_array[g, h, k, e, ib0, ia1]
    v11 = canonical_array[g, h, k, e, ib1, ia1]
    return np.asarray(
        (1.0 - wb) * (1.0 - wa) * v00
        + wb * (1.0 - wa) * v10
        + (1.0 - wb) * wa * v01
        + wb * wa * v11,
        dtype=float,
    )


def _draw_population_states(
    *,
    n: int,
    rng: np.random.Generator,
    ages: np.ndarray,
    log_income_grid: np.ndarray,
    gross_income: np.ndarray,
    liquid_grid: np.ndarray,
    illiquid_grid: np.ndarray,
    retirement_age: float,
    spec: PopulationSamplingSpec,
    maximum_check: float = 0.0,
) -> dict[str, np.ndarray]:
    """Draw continuous asset states and discrete exogenous states."""

    spec.validate()
    if n <= 0:
        return {
            "g": np.empty(0, dtype=np.int64),
            "h": np.empty(0, dtype=np.int64),
            "k": np.empty(0, dtype=np.int64),
            "e": np.empty(0, dtype=np.int64),
            "b": np.empty(0, dtype=float),
            "a": np.empty(0, dtype=float),
            "population_type": np.empty(0, dtype=object),
        }

    G, H, K, E = gross_income.shape
    g = rng.integers(0, G, size=n, dtype=np.int64)

    age_array = np.asarray(ages, dtype=float)
    if bool(spec.working_age_only):
        # In the solved model, age == retirement_age is already a retired state.
        eligible_h = np.flatnonzero(age_array < float(retirement_age))
        if len(eligible_h) == 0:
            raise ValueError("No working-age grid points satisfy age < retirement_age.")
    else:
        eligible_h = np.arange(H, dtype=np.int64)
    age_draw = rng.beta(float(spec.age_beta_a), float(spec.age_beta_b), size=n)
    age_position = np.minimum(
        (age_draw * len(eligible_h)).astype(np.int64), len(eligible_h) - 1
    )
    h = eligible_h[age_position].astype(np.int64)
    income_weights = _rouwenhorst_stationary_weights(K)
    k = rng.choice(K, size=n, replace=True, p=income_weights).astype(np.int64)
    age_values = age_array[h]
    retired = age_values >= float(retirement_age)
    e = np.where(
        retired,
        min(1, E - 1),
        (rng.random(n) < float(spec.nonemployment_share)).astype(np.int64),
    )
    e = np.clip(e, 0, E - 1).astype(np.int64)

    labor_state = np.exp(np.asarray(log_income_grid, dtype=float)[g, k])
    income_scale = np.maximum(labor_state, 1.0e-6)

    draw = rng.random(n)
    poor_cut = float(spec.poor_hand_to_mouth_share)
    wealthy_cut = poor_cut + float(spec.wealthy_hand_to_mouth_share)
    broad_cut = wealthy_cut + float(spec.broad_support_share)
    poor = draw < poor_cut
    wealthy = (draw >= poor_cut) & (draw < wealthy_cut)
    broad = (draw >= wealthy_cut) & (draw < broad_cut)
    regular = ~(poor | wealthy | broad)

    liquid_ratio = np.zeros(n, dtype=float)
    illiquid_ratio = np.zeros(n, dtype=float)
    for mask, liquid_median, illiquid_median in (
        (
            poor,
            spec.poor_liquid_ratio_median,
            spec.poor_illiquid_ratio_median,
        ),
        (
            wealthy,
            spec.wealthy_liquid_ratio_median,
            spec.wealthy_illiquid_ratio_median,
        ),
        (
            regular,
            spec.regular_liquid_ratio_median,
            spec.regular_illiquid_ratio_median,
        ),
    ):
        count = int(mask.sum())
        if count == 0:
            continue
        liquid_ratio[mask] = rng.lognormal(
            mean=np.log(float(liquid_median)),
            sigma=float(spec.liquid_ratio_log_sd),
            size=count,
        )
        illiquid_ratio[mask] = rng.lognormal(
            mean=np.log(float(illiquid_median)),
            sigma=float(spec.illiquid_ratio_log_sd),
            size=count,
        )

    b = liquid_ratio * income_scale
    a = illiquid_ratio * income_scale

    # A fraction of low-liquidity draws are placed very close to the borrowing
    # constraint.  This is the region in which finite-difference MPC curvature
    # is most important and where uniform grid sampling is least efficient.
    htm = poor | wealthy
    anchor = htm & (rng.random(n) < float(spec.borrowing_constraint_anchor_share))
    if anchor.any():
        span = max(float(liquid_grid[-1] - liquid_grid[0]), EPS)
        anchor_slack = rng.lognormal(
            mean=np.log(max(1.0e-4, 0.003 * span)),
            sigma=0.75,
            size=int(anchor.sum()),
        )
        b[anchor] = float(liquid_grid[0]) + anchor_slack

    if float(liquid_grid[0]) < 0.0:
        debt = htm & (rng.random(n) < float(spec.negative_liquid_share))
        if debt.any():
            debt_ratio = rng.lognormal(
                mean=np.log(0.04), sigma=0.65, size=int(debt.sum())
            )
            b[debt] = -debt_ratio * income_scale[debt]

    # The broad-support component is sampled continuously over the numerical
    # asset grids.  It prevents the population objective from forgetting the
    # rest of the policy surface.
    if broad.any():
        count = int(broad.sum())
        ub = rng.random(count)
        ua = rng.random(count)
        b[broad] = np.interp(
            ub,
            np.linspace(0.0, 1.0, len(liquid_grid)),
            liquid_grid,
        )
        a[broad] = np.interp(
            ua,
            np.linspace(0.0, 1.0, len(illiquid_grid)),
            illiquid_grid,
        )

    b_upper = float(liquid_grid[-1]) - max(float(maximum_check), 0.0)
    b_upper = max(float(liquid_grid[0]), b_upper)
    b = np.clip(b, float(liquid_grid[0]), b_upper)
    a = np.clip(a, float(illiquid_grid[0]), float(illiquid_grid[-1]))

    population_type = np.select(
        [poor, wealthy, broad],
        ["poor hand-to-mouth", "wealthy hand-to-mouth", "broad support"],
        default="regular saver",
    ).astype(object)
    return {
        "g": g,
        "h": h,
        "k": k,
        "e": e,
        "b": b.astype(float),
        "a": a.astype(float),
        "population_type": population_type,
    }


def _population_sample_coordinates(
    *,
    n_rows: int,
    rng: np.random.Generator,
    ages: np.ndarray,
    log_income_grid: np.ndarray,
    gross_income: np.ndarray,
    liquid_grid: np.ndarray,
    illiquid_grid: np.ndarray,
    retirement_age: float,
    population_spec: PopulationSamplingSpec,
    pair_share: float,
    check_sizes: Sequence[float],
) -> dict[str, np.ndarray]:
    share = float(np.clip(pair_share, 0.0, 0.90))
    n_pairs = int(math.floor(share * n_rows / 2.0))
    n_pair_rows = 2 * n_pairs
    n_level = int(n_rows - n_pair_rows)

    checks = np.asarray(
        [float(x) for x in check_sizes if np.isfinite(float(x)) and float(x) > 0.0],
        dtype=float,
    )
    span = float(liquid_grid[-1] - liquid_grid[0])
    checks = checks[checks < span - EPS]
    if checks.size == 0:
        checks = np.asarray([max(span / 10.0, EPS)], dtype=float)

    pieces: list[dict[str, np.ndarray]] = []
    pair_id_parts: list[np.ndarray] = []
    pair_role_parts: list[np.ndarray] = []
    pair_check_parts: list[np.ndarray] = []
    pair_cross_zero_parts: list[np.ndarray] = []

    if n_level > 0:
        level = _draw_population_states(
            n=n_level,
            rng=rng,
            ages=ages,
            log_income_grid=log_income_grid,
            gross_income=gross_income,
            liquid_grid=liquid_grid,
            illiquid_grid=illiquid_grid,
            retirement_age=retirement_age,
            spec=population_spec,
        )
        pieces.append(level)
        pair_id_parts.append(np.full(n_level, -1, dtype=np.int64))
        pair_role_parts.append(np.full(n_level, -1, dtype=np.int8))
        pair_check_parts.append(np.full(n_level, np.nan, dtype=float))
        pair_cross_zero_parts.append(np.zeros(n_level, dtype=np.int8))

    if n_pairs > 0:
        selected_checks = rng.choice(checks, size=n_pairs, replace=True)
        baseline = _draw_population_states(
            n=n_pairs,
            rng=rng,
            ages=ages,
            log_income_grid=log_income_grid,
            gross_income=gross_income,
            liquid_grid=liquid_grid,
            illiquid_grid=illiquid_grid,
            retirement_age=retirement_age,
            spec=population_spec,
            maximum_check=float(np.max(selected_checks)),
        )

        # Force a controlled share of MPC pairs to cross b=0. A smooth random
        # population otherwise places too few baselines in the narrow interval
        # (-check, 0), even though those finite differences are especially
        # important when borrowing and saving rates differ.
        crosses_zero = np.zeros(n_pairs, dtype=np.int8)
        if float(liquid_grid[0]) < 0.0 and population_spec.mpc_cross_zero_share > 0:
            n_cross = min(
                n_pairs,
                int(round(float(population_spec.mpc_cross_zero_share) * n_pairs)),
            )
            if n_cross > 0:
                cross_indices = rng.choice(n_pairs, size=n_cross, replace=False)
                for idx in np.asarray(cross_indices, dtype=np.int64):
                    check = float(selected_checks[idx])
                    # Leave a small margin on both sides so floating-point
                    # clipping cannot turn the pair into a same-side transfer.
                    lower = max(float(liquid_grid[0]), -0.95 * check)
                    upper = min(-0.05 * check, -10.0 * EPS)
                    if lower < upper:
                        # A symmetric beta draw covers both deep and shallow
                        # debt while putting useful mass away from endpoints.
                        u = float(rng.beta(1.5, 1.5))
                        baseline["b"][idx] = lower + u * (upper - lower)
                        crosses_zero[idx] = 1

        pair_piece: dict[str, np.ndarray] = {}
        for key in ("g", "h", "k", "e", "a", "population_type"):
            pair_piece[key] = np.repeat(baseline[key], 2)
        pair_b = np.empty(n_pair_rows, dtype=float)
        pair_b[0::2] = baseline["b"]
        pair_b[1::2] = baseline["b"] + selected_checks
        pair_piece["b"] = pair_b
        pieces.append(pair_piece)
        pair_id_parts.append(np.repeat(np.arange(n_pairs, dtype=np.int64), 2))
        roles = np.empty(n_pair_rows, dtype=np.int8)
        roles[0::2] = 0
        roles[1::2] = 1
        pair_role_parts.append(roles)
        pair_check_parts.append(np.repeat(selected_checks, 2))
        pair_cross_zero_parts.append(np.repeat(crosses_zero, 2))

    result: dict[str, np.ndarray] = {}
    for key in ("g", "h", "k", "e", "b", "a", "population_type"):
        result[key] = np.concatenate([piece[key] for piece in pieces])
    result["mpc_pair_id"] = np.concatenate(pair_id_parts)
    result["mpc_pair_role"] = np.concatenate(pair_role_parts)
    result["mpc_pair_check"] = np.concatenate(pair_check_parts)
    result["mpc_crosses_zero"] = np.concatenate(pair_cross_zero_parts)
    return result


def sample_one_saved_policy_population(
    model_dir: str | Path,
    *,
    n_rows: int,
    parameter_values: Mapping[str, Any],
    schema: GridSchema = GridSchema(),
    random_state: int = 0,
    mpc_pair_share: float = 0.60,
    mpc_check_sizes: Sequence[float] = (0.05, 0.10, 0.188679, 0.25),
    population_spec: PopulationSamplingSpec = PopulationSamplingSpec(),
) -> pd.DataFrame:
    """Sample continuous population states and interpolate the exact policies."""

    d = Path(model_dir)
    metadata = _read_json(d / "metadata.json")
    model_id = _model_id(metadata, d)
    rng = np.random.default_rng(int(random_state))
    with np.load(d / "arrays.npz", allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}

    storage = _inspect_policy_storage(d, metadata, schema)
    usable, reason = _validate_policy_storage_for_training(storage, schema=schema)
    if not usable:
        raise ValueError(f"Unsupported saved-policy representation: {reason}.")

    consumption_key = _find_array_key(arrays, "consumption", schema, required=True)
    deposit_key = _find_array_key(arrays, "deposit", schema, required=True)
    consumption_array = _canonical_policy_array(
        arrays[consumption_key], schema.policy_layout
    )
    deposit_array = _canonical_policy_array(arrays[deposit_key], schema.policy_layout)
    if consumption_array.shape != deposit_array.shape:
        raise ValueError("Consumption and deposit policies have different shapes.")
    G, H, K, E, B, A = consumption_array.shape

    liquid_key = _find_array_key(arrays, "liquid_grid", schema, required=True)
    illiquid_key = _find_array_key(arrays, "illiquid_grid", schema, required=True)
    liquid_grid = _as_1d(arrays[liquid_key], liquid_key).astype(float)
    illiquid_grid = _as_1d(arrays[illiquid_key], illiquid_key).astype(float)
    if len(liquid_grid) != B or len(illiquid_grid) != A:
        raise ValueError("Saved policy dimensions do not match the asset grids.")

    ages = _resolve_ages(arrays, metadata, schema, H)
    income_grid, income_is_log = _resolve_income_grid(arrays, schema, G, K)
    if not income_is_log:
        income_grid = np.log(np.maximum(income_grid, EPS))
    gross_key = _find_array_key(arrays, "gross_income", schema, required=True)
    after_tax_key = _find_array_key(arrays, "after_tax_income", schema, required=True)
    gross_array = np.asarray(arrays[gross_key], dtype=float)
    after_tax_array = np.asarray(arrays[after_tax_key], dtype=float)
    if gross_array.ndim == 3 and G == 1:
        gross_array = gross_array[None, ...]
    if after_tax_array.ndim == 3 and G == 1:
        after_tax_array = after_tax_array[None, ...]
    if gross_array.shape != (G, H, K, E) or after_tax_array.shape != (G, H, K, E):
        raise ValueError(
            "Income arrays must have shape [G,H,K,E], or [H,K,E] for a "
            "single-group model."
        )

    factor = _money_factor(metadata, schema)
    accounting = _accounting_values_from_metadata(metadata, money_factor=factor)
    accounting["acct__terminal_age"] = float(np.max(ages))
    retirement_age = float(accounting["acct__retirement_age"])
    group_values = _metadata_group_values(metadata, G)

    draw = _population_sample_coordinates(
        n_rows=n_rows,
        rng=rng,
        ages=ages,
        log_income_grid=income_grid,
        gross_income=gross_array,
        liquid_grid=liquid_grid,
        illiquid_grid=illiquid_grid,
        retirement_age=retirement_age,
        population_spec=population_spec,
        pair_share=mpc_pair_share,
        check_sizes=mpc_check_sizes,
    )
    g = draw["g"].astype(np.int64)
    h = draw["h"].astype(np.int64)
    k = draw["k"].astype(np.int64)
    e = draw["e"].astype(np.int64)
    b_model = draw["b"].astype(float)
    a_model = draw["a"].astype(float)

    gross_model = gross_array[g, h, k, e]
    after_tax_model = after_tax_array[g, h, k, e]
    labor_state_model = np.exp(income_grid[g, k])
    age_values = np.asarray(ages, dtype=float)[h]
    retired = age_values >= retirement_age
    employed = (e == 0) & (~retired)
    unemployed = (~employed) & (~retired)
    labor_model = np.where(employed, gross_model, 0.0)
    pension_model = np.where(retired, gross_model, 0.0)
    unemployment_model = np.where(unemployed, gross_model, 0.0)
    taxes_model = compute_taxes(
        gross_model,
        labor_model,
        pension_model,
        tax_kind=accounting["acct__tax_kind"],
        flat_rate=accounting["acct__tax_flat_rate"],
        deduction=float(accounting["acct__tax_deduction"]) / factor,
        payroll_rate=accounting["acct__tax_payroll_rate"],
        progressive_rate=accounting["acct__tax_progressive_rate"],
        progressive_exponent=accounting["acct__tax_progressive_exponent"],
        tax_cap=accounting["acct__tax_cap"],
    )
    minimum_transfer_model = np.maximum(
        after_tax_model - (gross_model - taxes_model), 0.0
    )

    consumption_model = _bilinear_policy_values(
        consumption_array,
        g=g,
        h=h,
        k=k,
        e=e,
        liquid_values=b_model,
        illiquid_values=a_model,
        liquid_grid=liquid_grid,
        illiquid_grid=illiquid_grid,
    )
    deposit_model = _bilinear_policy_values(
        deposit_array,
        g=g,
        h=h,
        k=k,
        e=e,
        liquid_values=b_model,
        illiquid_values=a_model,
        liquid_grid=liquid_grid,
        illiquid_grid=illiquid_grid,
    )

    current_b = b_model * factor
    current_a = a_model * factor
    gross = gross_model * factor
    after_tax = after_tax_model * factor
    labor_state = labor_state_model * factor
    consumption = consumption_model * factor
    deposit = deposit_model * factor
    reconstructed = _reconstruct_saved_policy_accounting(
        consumption=consumption,
        deposit=deposit,
        current_liquid_assets=current_b,
        current_illiquid_assets=current_a,
        after_tax_income=after_tax,
        liquid_index=None,
        illiquid_index=None,
        liquid_grid=liquid_grid,
        illiquid_grid=illiquid_grid,
        accounting=accounting,
        money_factor=factor,
    )

    ib0, _, _ = _continuous_grid_brackets(liquid_grid, b_model)
    ia0, _, _ = _continuous_grid_brackets(illiquid_grid, a_model)
    n = len(g)
    data: dict[str, Any] = {
        "model_id": np.repeat(model_id, n),
        "model_dir": np.repeat(str(d), n),
        "education_state": g,
        "education": np.asarray([group_values[index] for index in g], dtype=object),
        "age": age_values,
        "terminal_age": np.repeat(float(np.max(ages)), n),
        "years_to_retirement": retirement_age - age_values,
        "years_to_retirement_positive": np.maximum(retirement_age - age_values, 0.0),
        "years_since_retirement": np.maximum(age_values - retirement_age, 0.0),
        "years_to_terminal": np.maximum(float(np.max(ages)) - age_values, 0.0),
        "is_retired": retired.astype(int),
        "income_state": k,
        "employment_state": e,
        "employed": employed.astype(int),
        "current_income": gross,
        "gross_income": gross,
        "after_tax_income": after_tax,
        "taxes": taxes_model * factor,
        "minimum_income_transfer": minimum_transfer_model * factor,
        "labor_income": labor_model * factor,
        "pension_income": pension_model * factor,
        "unemployment_benefits": unemployment_model * factor,
        "labor_income_state": labor_state,
        "liquid_assets": current_b,
        "illiquid_assets": current_a,
        "total_assets": current_b + current_a,
        "cash_on_hand": current_b + after_tax,
        "liquid_grid_min": np.repeat(float(liquid_grid[0] * factor), n),
        "liquid_grid_max": np.repeat(float(liquid_grid[-1] * factor), n),
        "illiquid_grid_min": np.repeat(float(illiquid_grid[0] * factor), n),
        "illiquid_grid_max": np.repeat(float(illiquid_grid[-1] * factor), n),
        "money_scale": np.repeat(float(metadata.get("money_scale", 1.0) or 1.0), n),
        "population_type": draw["population_type"],
        "consumption": consumption,
        "deposit": deposit,
        "adjustment_cost": reconstructed["adjustment_cost"],
        "liquid_outflow": consumption + deposit + reconstructed["adjustment_cost"],
        "liquid_drift": reconstructed["liquid_drift"],
        "illiquid_drift": reconstructed["illiquid_drift"],
        "next_liquid_assets": reconstructed["next_liquid_assets"],
        "next_illiquid_assets": reconstructed["next_illiquid_assets"],
        "delta_liquid_assets": reconstructed["delta_liquid_assets"],
        "delta_illiquid_assets": reconstructed["delta_illiquid_assets"],
        "mpc_pair_id": draw["mpc_pair_id"],
        "mpc_pair_role": draw["mpc_pair_role"],
        "mpc_check_units": draw["mpc_pair_check"] * factor,
        "mpc_crosses_zero": draw["mpc_crosses_zero"].astype(np.int8),
        "grid_g": g,
        "grid_h": h,
        "grid_k": k,
        "grid_e": e,
        "grid_b": ib0,
        "grid_a": ia0,
    }
    for key, value in accounting.items():
        data[key] = np.repeat(value, n)
    for key, value in parameter_values.items():
        data[key] = np.repeat(value, n)
    out = pd.DataFrame(data)
    required = [
        "current_income",
        "after_tax_income",
        "liquid_assets",
        "illiquid_assets",
        "consumption",
        "deposit",
        "liquid_outflow",
    ]
    finite = np.ones(len(out), dtype=bool)
    for column in required:
        finite &= np.isfinite(
            pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
        )
    return out.loc[finite].reset_index(drop=True)


def _sample_model_worker(
    payload: tuple[Any, ...],
) -> tuple[str, pd.DataFrame | None, str | None]:
    """Process-pool worker with one BLAS thread per sampled model."""

    (
        model_id,
        model_dir,
        n_rows,
        parameters,
        schema,
        random_state,
        sampling_mode,
        mpc_pair_share,
        mpc_check_sizes,
        population_spec,
    ) = payload
    try:
        # Multiple model workers run concurrently. Prevent each process from
        # creating a full BLAS thread pool and oversubscribing the node.
        with threadpool_limits(limits=1):
            if sampling_mode == "population":
                frame = sample_one_saved_policy_population(
                    model_dir,
                    n_rows=n_rows,
                    parameter_values=parameters,
                    schema=schema,
                    random_state=random_state,
                    mpc_pair_share=mpc_pair_share,
                    mpc_check_sizes=mpc_check_sizes,
                    population_spec=population_spec,
                )
            else:
                frame = sample_one_saved_policy_grid(
                    model_dir,
                    n_rows=n_rows,
                    parameter_values=parameters,
                    schema=schema,
                    random_state=random_state,
                    mpc_pair_share=mpc_pair_share,
                    mpc_check_sizes=mpc_check_sizes,
                )
        return str(model_id), frame, None
    except Exception as exc:
        return str(model_id), None, f"{type(exc).__name__}: {exc}"


def sample_one_saved_policy_grid(
    model_dir: str | Path,
    *,
    n_rows: int,
    parameter_values: Mapping[str, Any],
    schema: GridSchema = GridSchema(),
    random_state: int = 0,
    include_optional_policies: bool = True,
    mpc_pair_share: float = 0.60,
    mpc_check_sizes: Sequence[float] = (0.02, 0.05, 0.10, 0.188679, 0.20, 0.50),
) -> pd.DataFrame:
    """Sample one saved policy grid into a tidy state-policy table.

    The primitive policy targets are consumption and the net deposit flow from
    the liquid account into the illiquid account.  Both next-asset policies are
    retained as validation targets, but they are reconstructed from the model's
    accounting identities rather than estimated independently.
    """

    d = Path(model_dir)
    metadata = _read_json(d / "metadata.json")
    model_id = _model_id(metadata, d)
    rng = np.random.default_rng(random_state)

    with np.load(d / "arrays.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}

    storage = _inspect_policy_storage(d, metadata, schema)
    storage_usable, storage_reason = _validate_policy_storage_for_training(
        storage, schema=schema
    )
    if not storage_usable:
        raise ValueError(f"Unsupported saved-policy representation: {storage_reason}.")

    consumption_key = _find_array_key(arrays, "consumption", schema, required=True)
    deposit_key = _find_array_key(arrays, "deposit", schema, required=False)
    next_liquid_key = _find_array_key(
        arrays, "next_liquid_assets", schema, required=False
    )
    next_illiquid_key = _find_array_key(
        arrays, "next_illiquid_assets", schema, required=False
    )
    liquid_drift_key = _find_array_key(arrays, "liquid_drift", schema, required=False)
    illiquid_drift_key = _find_array_key(
        arrays, "illiquid_drift", schema, required=False
    )

    if deposit_key is None and illiquid_drift_key is None:
        raise KeyError(
            "The accounting-restricted surrogate requires either a saved deposit "
            "policy or illiquid_drift from which deposit can be recovered. "
            f"Available arrays: {sorted(arrays)}"
        )
    c_arr = arrays[consumption_key]  # type: ignore[index]
    canonical_shape = _policy_shape_to_canonical(c_arr.shape, schema.policy_layout)
    g_count, h_count, k_count, e_count, b_count, a_count = canonical_shape

    policy_keys = {
        "consumption": consumption_key,
        "deposit": deposit_key,
        "next_liquid_assets": next_liquid_key,
        "next_illiquid_assets": next_illiquid_key,
        "liquid_drift": liquid_drift_key,
        "illiquid_drift": illiquid_drift_key,
    }
    for canonical, key in policy_keys.items():
        if key is None:
            continue
        shape = _policy_shape_to_canonical(arrays[key].shape, schema.policy_layout)
        if shape != canonical_shape:
            raise ValueError(
                f"{canonical} has canonical shape {shape}, not {canonical_shape}."
            )

    liquid_key = _find_array_key(arrays, "liquid_grid", schema, required=True)
    illiquid_key = _find_array_key(arrays, "illiquid_grid", schema, required=True)
    liquid_grid = _as_1d(arrays[liquid_key], liquid_key).astype(float)  # type: ignore[index]
    illiquid_grid = _as_1d(arrays[illiquid_key], illiquid_key).astype(float)  # type: ignore[index]
    if len(liquid_grid) != b_count or len(illiquid_grid) != a_count:
        raise ValueError(
            "Policy-array asset axes do not match saved asset-grid lengths: "
            f"B={b_count} vs {len(liquid_grid)}, A={a_count} vs {len(illiquid_grid)}."
        )

    ages = _resolve_ages(arrays, metadata, schema, h_count)
    income_grid, income_grid_is_log = _resolve_income_grid(
        arrays, schema, g_count, k_count
    )
    group_values = _metadata_group_values(metadata, g_count)
    factor = _money_factor(metadata, schema)
    accounting = _accounting_values_from_metadata(metadata, money_factor=factor)
    # The saved age grid is authoritative.  Store its last age explicitly so
    # prediction-time lifecycle features match this exact model.
    accounting["acct__terminal_age"] = float(np.max(ages))

    coords, mpc_pair_id, mpc_pair_role, mpc_pair_check = _mpc_aware_coordinates(
        canonical_shape,
        n_rows,
        rng,
        liquid_grid=liquid_grid,
        pair_share=mpc_pair_share,
        check_sizes=mpc_check_sizes,
    )
    g, h, k, e, b, a = (coords[x] for x in "GHKEBA")

    after_tax_key = _find_array_key(arrays, "after_tax_income", schema, required=False)
    gross_key = _find_array_key(arrays, "gross_income", schema, required=False)
    if after_tax_key is None and gross_key is None:
        raise KeyError(
            "At least one of gross_income/current_income or after_tax_income must "
            "be saved in arrays.npz."
        )
    after_tax_model = (
        _index_state_array(arrays[after_tax_key], g=g, h=h, k=k, e=e)  # type: ignore[index]
        if after_tax_key is not None
        else np.full(n_rows, np.nan)
    )
    gross_model = (
        _index_state_array(arrays[gross_key], g=g, h=h, k=k, e=e)  # type: ignore[index]
        if gross_key is not None
        else after_tax_model.copy()
    )

    labor_state_model = income_grid[g, k]
    if income_grid_is_log:
        labor_state_model = np.exp(labor_state_model)

    cfg = metadata.get("config", {}) or {}
    retirement_age = float(accounting["acct__retirement_age"])
    age_values = ages[h].astype(float)
    retired = age_values >= retirement_age
    employed = (e == 0) & (~retired)
    unemployed = (~employed) & (~retired)

    # In the parametric bundle, gross income is labor income for employed
    # working-age households, unemployment benefits for non-employed workers,
    # and pension income after retirement.
    labor_model = np.where(employed, gross_model, 0.0)
    pension_model = np.where(retired, gross_model, 0.0)
    unemployment_model = np.where(unemployed, gross_model, 0.0)

    if after_tax_key is None:
        taxes_model = compute_taxes(
            gross_model,
            labor_model,
            pension_model,
            tax_kind=accounting["acct__tax_kind"],
            flat_rate=accounting["acct__tax_flat_rate"],
            deduction=float(accounting["acct__tax_deduction"]) / factor,
            payroll_rate=accounting["acct__tax_payroll_rate"],
            progressive_rate=accounting["acct__tax_progressive_rate"],
            progressive_exponent=accounting["acct__tax_progressive_exponent"],
            tax_cap=accounting["acct__tax_cap"],
        )
        after_tax_model = gross_model - taxes_model
    else:
        taxes_model = np.maximum(gross_model - after_tax_model, 0.0)

    current_b = liquid_grid[b] * factor
    current_a = illiquid_grid[a] * factor
    gross = gross_model * factor
    after_tax = after_tax_model * factor
    labor_state = labor_state_model * factor
    labor = labor_model * factor
    pension = pension_model * factor
    unemployment = unemployment_model * factor
    taxes = taxes_model * factor

    consumption = (
        _index_policy_array(arrays[consumption_key], schema.policy_layout, coords)
        * factor
    )

    if deposit_key is not None:
        deposit = (
            _index_policy_array(arrays[deposit_key], schema.policy_layout, coords)
            * factor
        )
    else:
        saved_illiquid_drift = (
            _index_policy_array(
                arrays[illiquid_drift_key], schema.policy_layout, coords  # type: ignore[index]
            )
            * factor
        )
        deposit = (
            saved_illiquid_drift
            - float(accounting["acct__ra_ct"]) * current_a
            - float(accounting["acct__xi"]) * after_tax
        )

    reconstructed = _reconstruct_saved_policy_accounting(
        consumption=consumption,
        deposit=deposit,
        current_liquid_assets=current_b,
        current_illiquid_assets=current_a,
        after_tax_income=after_tax,
        liquid_index=b,
        illiquid_index=a,
        liquid_grid=liquid_grid,
        illiquid_grid=illiquid_grid,
        accounting=accounting,
        money_factor=factor,
    )
    adjustment_cost = reconstructed["adjustment_cost"]
    liquid_drift_identity = reconstructed["liquid_drift"]
    illiquid_drift_identity = reconstructed["illiquid_drift"]
    next_b = reconstructed["next_liquid_assets"]
    next_a = reconstructed["next_illiquid_assets"]

    data: dict[str, Any] = {
        "model_id": np.repeat(model_id, n_rows),
        "model_dir": np.repeat(str(d), n_rows),
        "education_state": g.astype(int),
        "education": np.asarray([group_values[i] for i in g], dtype=object),
        "age": age_values,
        "terminal_age": np.repeat(float(np.max(ages)), n_rows),
        "years_to_retirement": retirement_age - age_values,
        "years_to_retirement_positive": np.maximum(retirement_age - age_values, 0.0),
        "years_since_retirement": np.maximum(age_values - retirement_age, 0.0),
        "years_to_terminal": np.maximum(float(np.max(ages)) - age_values, 0.0),
        "is_retired": retired.astype(int),
        "income_state": k.astype(int),
        "employment_state": e.astype(int),
        "employed": employed.astype(int),
        "current_income": gross,
        "gross_income": gross,
        "after_tax_income": after_tax,
        "taxes": taxes,
        "labor_income": labor,
        "pension_income": pension,
        "unemployment_benefits": unemployment,
        "labor_income_state": labor_state,
        "liquid_assets": current_b,
        "illiquid_assets": current_a,
        "total_assets": current_b + current_a,
        "cash_on_hand": current_b + after_tax,
        "liquid_grid_min": np.repeat(float(liquid_grid[0] * factor), n_rows),
        "liquid_grid_max": np.repeat(float(liquid_grid[-1] * factor), n_rows),
        "illiquid_grid_min": np.repeat(float(illiquid_grid[0] * factor), n_rows),
        "illiquid_grid_max": np.repeat(float(illiquid_grid[-1] * factor), n_rows),
        "money_scale": np.repeat(
            float(metadata.get("money_scale", 1.0) or 1.0), n_rows
        ),
        "consumption": consumption,
        "deposit": deposit,
        "adjustment_cost": adjustment_cost,
        # Total use of liquid resources.  This is the natural target for
        # accurately predicting liquid saving because liquid_drift equals
        # liquid resources minus this object.
        "liquid_outflow": consumption + deposit + adjustment_cost,
        "liquid_drift": liquid_drift_identity,
        "illiquid_drift": illiquid_drift_identity,
        "next_liquid_assets": next_b,
        "next_illiquid_assets": next_a,
        "delta_liquid_assets": next_b - current_b,
        "delta_illiquid_assets": next_a - current_a,
        "mpc_pair_id": mpc_pair_id,
        "mpc_pair_role": mpc_pair_role,
        "mpc_check_units": mpc_pair_check * factor,
        "grid_g": g,
        "grid_h": h,
        "grid_k": k,
        "grid_e": e,
        "grid_b": b,
        "grid_a": a,
    }

    # Retain saved drifts as diagnostics when present; the identity-based values
    # above are the authoritative accounting objects used by the surrogate.
    if include_optional_policies:
        if liquid_drift_key is not None:
            data["saved_liquid_drift"] = (
                _index_policy_array(
                    arrays[liquid_drift_key], schema.policy_layout, coords
                )
                * factor
            )
        if illiquid_drift_key is not None:
            data["saved_illiquid_drift"] = (
                _index_policy_array(
                    arrays[illiquid_drift_key], schema.policy_layout, coords
                )
                * factor
            )
        if next_liquid_key is not None:
            data["saved_next_liquid_assets"] = (
                _index_policy_array(
                    arrays[next_liquid_key], schema.policy_layout, coords
                )
                * factor
            )
        if next_illiquid_key is not None:
            data["saved_next_illiquid_assets"] = (
                _index_policy_array(
                    arrays[next_illiquid_key], schema.policy_layout, coords
                )
                * factor
            )
        adjust_key = _find_array_key(arrays, "adjust_illiquid", schema, required=False)
        if adjust_key is not None:
            try:
                data["adjust_illiquid"] = _index_policy_array(
                    arrays[adjust_key], schema.policy_layout, coords
                )
            except ValueError:
                pass

    for key, value in accounting.items():
        data[key] = np.repeat(value, n_rows)
    for key, value in parameter_values.items():
        data[key] = np.repeat(value, n_rows)

    out = pd.DataFrame(data)
    required_finite = [
        "age",
        "current_income",
        "after_tax_income",
        "liquid_assets",
        "illiquid_assets",
        "consumption",
        "deposit",
        "liquid_outflow",
        "next_liquid_assets",
        "next_illiquid_assets",
    ]
    mask = np.ones(len(out), dtype=bool)
    for col in required_finite:
        mask &= np.isfinite(pd.to_numeric(out[col], errors="coerce").to_numpy())
    return out.loc[mask].reset_index(drop=True)


@dataclass
class PolicyDatasetResult:
    data: pd.DataFrame
    catalog: pd.DataFrame
    parameter_columns: list[str]
    categorical_parameter_columns: list[str]
    rows_per_model: int
    failures: pd.DataFrame


def _compact_policy_sample_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Reduce the in-memory footprint of a large sampled policy dataset."""

    out = frame.copy()
    category_columns = [
        column
        for column in out.columns
        if (
            column
            in {
                "model_id",
                "model_dir",
                "education",
                "population_type",
                "acct__tax_kind",
            }
            or column.startswith("param__")
            and not pd.api.types.is_numeric_dtype(out[column])
        )
    ]
    for column in category_columns:
        out[column] = out[column].astype("category")

    int8_columns = [
        "education_state",
        "income_state",
        "employment_state",
        "employed",
        "is_retired",
        "mpc_pair_role",
        "grid_g",
        "grid_e",
    ]
    int16_columns = ["grid_h", "grid_k", "grid_b", "grid_a"]
    for column in int8_columns:
        if column in out:
            out[column] = (
                pd.to_numeric(out[column], errors="coerce").fillna(-1).astype(np.int8)
            )
    for column in int16_columns:
        if column in out:
            out[column] = (
                pd.to_numeric(out[column], errors="coerce").fillna(-1).astype(np.int16)
            )
    if "mpc_pair_id" in out:
        out["mpc_pair_id"] = (
            pd.to_numeric(out["mpc_pair_id"], errors="coerce")
            .fillna(-1)
            .astype(np.int32)
        )

    for column in out.select_dtypes(include=["float64"]).columns:
        out[column] = out[column].astype(np.float32)
    return out


def build_policy_dataset(
    model_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    schema: GridSchema = GridSchema(),
    rows_per_model: int = 1_000,
    max_total_rows: int = 3_000_000,
    max_models: int | None = None,
    sampling_mode: str = "population",
    population_spec: PopulationSamplingSpec = PopulationSamplingSpec(),
    n_jobs: int = 1,
    mpc_pair_share: float = 0.60,
    mpc_check_sizes: Sequence[float] = (0.02, 0.05, 0.10, 0.188679, 0.20, 0.50),
    parameter_include: Sequence[str] | None = None,
    parameter_exclude: Sequence[str] = (),
    categorical_parameter_include: Sequence[str] | None = None,
    categorical_parameter_exclude: Sequence[str] = (),
    max_solver_distance: float | None = None,
    max_relative_distance: float | None = None,
    max_hjb_residual: float | None = None,
    random_state: int = 123,
    output_path: str | Path | None = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> PolicyDatasetResult:
    """Construct a balanced state-policy sample from many solved models.

    Every parameterization contributes the same number of rows.  Population
    mode samples continuous asset states from a hand-to-mouth-heavy mixture and
    bilinearly interpolates the exact policies.  Grid mode retains the older
    marginally stratified numerical-grid sampler.  Model-level sampling can be
    parallelized across CPU workers.
    """

    sampling_mode = str(sampling_mode).lower().strip()
    if sampling_mode not in {"population", "grid"}:
        raise ValueError("sampling_mode must be 'population' or 'grid'.")
    population_spec.validate()
    n_jobs = max(1, int(n_jobs))

    stage_start = time.perf_counter()
    dirs = discover_model_dirs(model_root, manifest_path=manifest_path)
    if max_models is not None and len(dirs) > int(max_models):
        # Random rather than lexicographic truncation, which can accidentally
        # select a nonrepresentative block when model IDs are ordered by run.
        selection_rng = np.random.default_rng(random_state)
        chosen = np.sort(
            selection_rng.choice(len(dirs), size=int(max_models), replace=False)
        )
        dirs = [dirs[int(i)] for i in chosen]
    if not dirs:
        raise FileNotFoundError(f"No saved models found under {model_root}.")
    _status(
        f"[data 1/3] Found {len(dirs):,} saved model directories.",
        verbose=verbose,
    )
    _status("[data 2/3] Reading metadata and solver diagnostics...", verbose=verbose)

    catalog = build_model_catalog(
        dirs,
        schema=schema,
        max_solver_distance=max_solver_distance,
        max_relative_distance=max_relative_distance,
        max_hjb_residual=max_hjb_residual,
        verbose=verbose,
        show_progress=show_progress,
    )
    usable = catalog[catalog["usable"]].copy().reset_index(drop=True)
    if usable.empty:
        reasons = catalog[["model_dir", "reason"]].to_string(index=False)
        raise RuntimeError(f"No usable saved models.\n{reasons}")

    parameter_columns = choose_parameter_columns(
        catalog,
        include=parameter_include,
        exclude=parameter_exclude,
    )
    categorical_parameter_columns = choose_categorical_parameter_columns(
        catalog,
        include=categorical_parameter_include,
        exclude=categorical_parameter_exclude,
    )

    n_models = len(usable)
    _status(
        f"  usable models: {n_models:,}/{len(catalog):,}; "
        f"continuous parameters: {len(parameter_columns):,}; "
        f"categorical parameters: {len(categorical_parameter_columns):,}",
        verbose=verbose,
    )
    if "policy_storage_format" in catalog:
        storage_counts_all = (
            catalog["policy_storage_format"]
            .fillna("unreadable")
            .value_counts(dropna=False)
            .to_dict()
        )
        storage_text = ", ".join(
            f"{name}={int(count):,}" for name, count in storage_counts_all.items()
        )
        _status(
            f"  policy storage across all discovered models: {storage_text}",
            verbose=verbose,
        )

    rejected = catalog[~catalog["usable"]].copy()
    if not rejected.empty:
        if "rejection_code" in rejected:
            reason_counts = (
                rejected["rejection_code"]
                .replace("", "unspecified")
                .fillna("unspecified")
                .value_counts()
            )
            reason_text = ", ".join(
                f"{name}={int(count):,}" for name, count in reason_counts.items()
            )
            _status(f"  rejected models by reason: {reason_text}", verbose=verbose)
        if "solver_max_distance" in rejected and max_solver_distance is not None:
            distances = pd.to_numeric(
                rejected.loc[
                    rejected.get("rejection_code", "") == "solver_distance",
                    "solver_max_distance",
                ],
                errors="coerce",
            ).dropna()
            if len(distances):
                _status(
                    "  excluded solver distances: "
                    f"min={distances.min():.3g}, median={distances.median():.3g}, "
                    f"max={distances.max():.3g}; cutoff={float(max_solver_distance):.3g}",
                    verbose=verbose,
                )
    effective_rows = min(int(rows_per_model), max(1, int(max_total_rows) // n_models))
    if effective_rows < 250:
        warnings.warn(
            f"Only {effective_rows} rows/model fit under max_total_rows. "
            "That is usually too sparse for a two-asset lifecycle surface and "
            "MPC finite differences. Increase max_total_rows, reduce max_models, "
            "or both; 250-500 rows/model is a practical minimum for initial fits.",
            stacklevel=2,
        )

    _status(
        f"[data 3/3] Sampling {effective_rows:,} {sampling_mode} policy points "
        f"from each of {n_models:,} models (cap: {max_total_rows:,} rows; "
        f"workers: {n_jobs:,})...",
        verbose=verbose,
    )
    rng = np.random.default_rng(random_state)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    payloads: list[tuple[Any, ...]] = []
    for _, row in usable.iterrows():
        params: dict[str, Any] = {
            col: float(row[col])
            for col in parameter_columns
            if col in row and pd.notna(row[col])
        }
        params.update(
            {
                col: str(row[col])
                for col in categorical_parameter_columns
                if col in row and pd.notna(row[col])
            }
        )
        payloads.append(
            (
                str(row["model_id"]),
                str(row["model_dir"]),
                int(effective_rows),
                params,
                schema,
                int(rng.integers(0, np.iinfo(np.int32).max)),
                sampling_mode,
                float(mpc_pair_share),
                tuple(float(x) for x in mpc_check_sizes),
                population_spec,
            )
        )

    if n_jobs == 1:
        iterator = _progress_iter(
            payloads,
            total=len(payloads),
            description="Sampling models",
            unit="model",
            enabled=show_progress,
        )
        results = (_sample_model_worker(payload) for payload in iterator)
        for model_id, frame, error in results:
            if frame is not None:
                frames.append(frame)
            else:
                row = usable.loc[usable["model_id"].astype(str) == str(model_id)].iloc[
                    0
                ]
                failures.append(
                    {
                        "model_id": str(model_id),
                        "model_dir": str(row["model_dir"]),
                        "reason": str(error),
                    }
                )
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_map = {
                executor.submit(_sample_model_worker, payload): payload
                for payload in payloads
            }
            iterator = _progress_iter(
                as_completed(future_map),
                total=len(future_map),
                description="Sampling models",
                unit="model",
                enabled=show_progress,
            )
            for future in iterator:
                payload = future_map[future]
                model_id, model_dir = str(payload[0]), str(payload[1])
                try:
                    returned_id, frame, error = future.result()
                except Exception as exc:
                    returned_id, frame, error = (
                        model_id,
                        None,
                        (f"{type(exc).__name__}: {exc}"),
                    )
                if frame is not None:
                    frames.append(frame)
                else:
                    failures.append(
                        {
                            "model_id": str(returned_id),
                            "model_dir": model_dir,
                            "reason": str(error),
                        }
                    )

    if not frames:
        failure_table = pd.DataFrame(failures)
        details = failure_table[["model_dir", "reason"]].to_string(index=False)
        raise RuntimeError(
            "Every saved grid failed to load. The loader errors were:\n" + details
        )
    data = pd.concat(frames, ignore_index=True, sort=False)
    data = _compact_policy_sample_dtypes(data)
    _status(
        f"  sampled dataset: {len(data):,} rows from {len(frames):,} models; "
        f"failures: {len(failures):,}; elapsed: "
        f"{time.perf_counter() - stage_start:,.1f}s",
        verbose=verbose,
    )

    # Continuous parameters missing in a subset of files are median-imputed by
    # the feature encoder.  Explicit indicators keep missing metadata distinct
    # from an ordinary median value.  Categorical parameters use a dedicated
    # missing category and are one-hot encoded.
    for col in parameter_columns:
        if col not in data:
            data[col] = np.nan
        if data[col].isna().any():
            data[f"{col}__missing"] = data[col].isna().astype(int)
    for col in categorical_parameter_columns:
        if col not in data:
            data[col] = "__missing__"
        data[col] = data[col].where(data[col].notna(), "__missing__").astype(str)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_dataframe(data, path)
        catalog.to_csv(path.with_name(path.stem + "_catalog.csv"), index=False)
        pd.DataFrame(failures).to_csv(
            path.with_name(path.stem + "_failures.csv"), index=False
        )

    return PolicyDatasetResult(
        data=data,
        catalog=catalog,
        parameter_columns=parameter_columns,
        categorical_parameter_columns=categorical_parameter_columns,
        rows_per_model=effective_rows,
        failures=pd.DataFrame(failures),
    )


def _write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except ImportError:
            fallback = path.with_suffix(".pkl.gz")
            df.to_pickle(fallback, compression="gzip")
            return fallback
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return path
    if suffix == ".gz" and path.name.endswith(".pkl.gz"):
        df.to_pickle(path, compression="gzip")
        return path
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    fallback = path.with_suffix(".pkl.gz")
    df.to_pickle(fallback, compression="gzip")
    return fallback


def read_policy_dataset(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    name = p.name.lower()
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if name.endswith(".pkl.gz"):
        return pd.read_pickle(p, compression="gzip")
    if p.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported policy-dataset format: {p}")


# -----------------------------------------------------------------------------
# Smooth feature maps
# -----------------------------------------------------------------------------


# Primitive household states exposed to downstream visualization code.
DEFAULT_STATE_COLUMNS = (
    "age",
    "current_income",
    "after_tax_income",
    "labor_income_state",
    "liquid_assets",
    "illiquid_assets",
)

# Internally generated state features that align policy functions across income
# scales and borrowing limits.  These are regression inputs but are not exposed
# as independently adjustable household states in the visualization page.
# The liquidity spline uses an economically scaled log transform.  The scale
# is expressed as a fraction of permanent income: 0.02 means that the transform
# devotes substantial resolution to the first two percent of annual permanent
# income above the borrowing limit, where the holdout diagnostics show the
# sharpest MPC curvature.
LIQUID_SLACK_LOG_SCALE = 0.02

DEFAULT_ENGINEERED_STATE_COLUMNS = (
    # Ratios use the household's persistent labor-income state as the scale.
    # They are not redundant with model-wide normalization because permanent
    # income varies across households.
    "liquid_assets_to_income",
    # Exact continuous hinge at b=0. Together with raw liquid assets, this
    # permits different borrowing-side and saving-side slopes without a jump
    # in the consumption policy level.
    "liquid_assets_positive_part",
    # Retained as an ordinary low-order state and for diagnostics.
    "liquid_slack_to_income",
    # High-resolution liquidity coordinate used by every flexible liquid spline.
    "log1p_liquid_slack_to_income",
    "illiquid_assets_to_income",
    # Current gross and disposable income remain in the low-order feature map.
    # The flexible current-resource spline below uses after-tax income through
    # liquid_resources_to_income, so neither ratio needs its own spline basis.
    "pre_tax_income_to_income",
    "after_tax_income_to_income",
    # Cash-on-hand measured relative to the borrowing limit rather than zero:
    # (liquid assets - borrowing limit + after-tax income) / permanent income.
    "liquid_resources_to_income",
)

# Current gross income is retained as an input/display variable, but is redundant
# in the regression once after-tax income, the persistent labor-income state,
# and the tax parameters are present.  Its normalized counterpart is likewise
# kept for diagnostics but excluded from the fitted feature map.
DEFAULT_REDUNDANT_STATE_COLUMNS = (
    "current_income",
    "pre_tax_income_to_income",
)

DEFAULT_SPLINE_COLUMNS = (
    # The liquid-slack spline is evaluated on the scaled log coordinate and
    # receives custom low-tail knots rather than sample quantile knots.
    "log1p_liquid_slack_to_income",
    # Disposable cash-on-hand above the borrowing limit is the current liquid-
    # resource state entering the consumption problem.
    "liquid_resources_to_income",
    # Flexible portfolio composition and lifecycle coordinates.
    "illiquid_assets_to_income",
    "age",
)

# Per-column knot counts and placement.  Ordinary splines retain quantile knots.
# The two liquid-resource splines instead use deterministic low-tail spacing
# between robust support endpoints.  This avoids allowing a long high-wealth
# tail to consume most of the spline resolution.
DEFAULT_SPLINE_COLUMN_N_KNOTS: tuple[tuple[str, int], ...] = (
    ("log1p_liquid_slack_to_income", 12),
    ("liquid_resources_to_income", 10),
    # Additional resolution for the model-specific hump in illiquid wealth.
    ("illiquid_assets_to_income", 10),
)
DEFAULT_SPLINE_KNOT_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("log1p_liquid_slack_to_income", "low_tail"),
    ("liquid_resources_to_income", "low_tail"),
)
DEFAULT_SPLINE_KNOT_POWERS: tuple[tuple[str, float], ...] = (
    # The log transform already compresses the upper tail, so a moderate power
    # creates knots around very small liquid slack without collapsing the rest.
    ("log1p_liquid_slack_to_income", 1.8),
    ("liquid_resources_to_income", 2.0),
)

# State interactions retained in every specification.  Every pair includes a
# liquid-resource coordinate and therefore can affect finite-difference MPCs.
DEFAULT_SPLINE_TENSOR_PAIRS = (
    ("log1p_liquid_slack_to_income", "illiquid_assets_to_income"),
    ("log1p_liquid_slack_to_income", "age"),
    ("age", "illiquid_assets_to_income"),
    ("liquid_resources_to_income", "age"),
)

# Let the location and height of the illiquid-wealth MPC hump vary jointly with
# current liquid resources and lifecycle position.
DEFAULT_SPLINE_TENSOR_TRIPLES: tuple[tuple[str, str, str], ...] = (
    (
        "liquid_resources_to_income",
        "illiquid_assets_to_income",
        "age",
    ),
)

# Employment-specific portfolio/liquidity surfaces.
DEFAULT_SPLINE_CATEGORICAL_TENSOR_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "employment_state",
        "log1p_liquid_slack_to_income",
        "liquid_resources_to_income",
    ),
    (
        "employment_state",
        "log1p_liquid_slack_to_income",
        "illiquid_assets_to_income",
    ),
)

# Terms that repeatedly emerged as the dominant parameter nonlinearities in
# model-level holdout diagnostics. They are mandatory so the automatic search
# evaluates remaining candidates conditional on this economically important
# baseline rather than rediscovering them one at a time.
_RISK_AVERSION = "param__config__risk_aversion"
_DISCOUNT_FACTOR = "param__config__discount_factor"
_CONVEX_ADJUSTMENT_COST = "param__config__ct_convex_adjustment_cost"
_LINEAR_ADJUSTMENT_COST = "param__config__ct_linear_adjustment_cost"

DEFAULT_PARAMETER_POWER_TERMS: tuple[tuple[str, int], ...] = ()
DEFAULT_SPLINE_PARAMETER_POWER_TERMS: tuple[tuple[str, str, int], ...] = (
    ("log1p_liquid_slack_to_income", _RISK_AVERSION, 2),
    ("liquid_resources_to_income", _RISK_AVERSION, 2),
)
DEFAULT_SPLINE_PARAMETER_PAIR_TERMS: tuple[tuple[str, str, str], ...] = (
    (
        "log1p_liquid_slack_to_income",
        _DISCOUNT_FACTOR,
        _RISK_AVERSION,
    ),
    (
        "liquid_resources_to_income",
        _DISCOUNT_FACTOR,
        _RISK_AVERSION,
    ),
    (
        "log1p_liquid_slack_to_income",
        _CONVEX_ADJUSTMENT_COST,
        _RISK_AVERSION,
    ),
)

# Parameter modulation of an entire liquid-by-illiquid spline surface.  The old
# basis allowed one-dimensional spline slopes to vary with parameters, but the
# shape of the joint portfolio surface was common across all solved models.
DEFAULT_SPLINE_TENSOR_PARAMETER_TERMS: tuple[tuple[str, str, str], ...] = (
    (
        "log1p_liquid_slack_to_income",
        "illiquid_assets_to_income",
        _CONVEX_ADJUSTMENT_COST,
    ),
    (
        "log1p_liquid_slack_to_income",
        "illiquid_assets_to_income",
        _LINEAR_ADJUSTMENT_COST,
    ),
    (
        "log1p_liquid_slack_to_income",
        "illiquid_assets_to_income",
        _RISK_AVERSION,
    ),
    (
        "liquid_resources_to_income",
        "illiquid_assets_to_income",
        _CONVEX_ADJUSTMENT_COST,
    ),
)

# Automatic higher-order parameter selection remains one-dimensional for
# computational tractability, but it now searches against the economically
# relevant liquid coordinates.
DEFAULT_PARAMETER_SELECTION_SPLINE_COLUMNS: tuple[str, ...] = (
    "log1p_liquid_slack_to_income",
    "liquid_resources_to_income",
)
DEFAULT_PARAMETER_SELECTION_POWERS: tuple[int, ...] = (2, 3)

# General category-specific slopes already include employment_state x b and
# employment_state x max(b,0), which identify an employment-specific kink at
# zero. These targeted second-order terms additionally let the curvature on
# each side of the kink differ for employed and unemployed households.
DEFAULT_CATEGORICAL_STATE_POWER_TERMS: tuple[tuple[str, str, int], ...] = (
    ("employment_state", "liquid_assets", 2),
    ("employment_state", "liquid_assets_positive_part", 2),
)

# Working-age-only training makes the old retirement exclusions mostly
# irrelevant, but retaining them preserves compatibility with optional runs
# that include retirees.
DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS = ("is_retired",)
DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS = ("is_retired",)

# One-dimensional spline-by-parameter terms remain useful. The targeted power
# and pair interactions above provide the nonlinear parameter modulation that
# the previous basis lacked.
DEFAULT_SPLINE_PARAMETER_PATTERNS = (
    "discount_factor",
    "risk_aversion",
    "borrowing_limit",
    "borrowing_interest_rate",
    "liquid_interest_rate",
    "illiquid_interest_rate",
    "innovation_std",
    "stationary_log_income_std",
    "age_profile_slope",
    "persistence",
    "job_loss_probability",
    "job_finding_probability",
    "unemployment_replacement_rate",
    "minimum_after_tax_income",
    "replacement_rate",
    "flat_rate",
    "progressive_rate",
    "ct_linear_adjustment_cost",
    "ct_convex_adjustment_cost",
    "ct_automatic_illiquid_income_share",
)

DEFAULT_CATEGORICAL_STATE_COLUMNS = (
    "education",
    "employment_state",
    "is_retired",
)
# Backwards-compatible import used by earlier scripts.
DEFAULT_CATEGORICAL_COLUMNS = DEFAULT_CATEGORICAL_STATE_COLUMNS
DEFAULT_MONETARY_COLUMNS = (
    "current_income",
    "gross_income",
    "after_tax_income",
    "labor_income",
    "pension_income",
    "unemployment_benefits",
    "taxes",
    "labor_income_state",
    "illiquid_assets",
    "total_assets",
    "cash_on_hand",
)


@dataclass(frozen=True)
class FeatureSpec:
    state_columns: tuple[str, ...] = DEFAULT_STATE_COLUMNS
    engineered_state_columns: tuple[str, ...] = DEFAULT_ENGINEERED_STATE_COLUMNS
    redundant_state_columns: tuple[str, ...] = DEFAULT_REDUNDANT_STATE_COLUMNS
    parameter_columns: tuple[str, ...] = ()
    categorical_state_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_STATE_COLUMNS
    categorical_parameter_columns: tuple[str, ...] = ()
    signed_log_columns: tuple[str, ...] = DEFAULT_MONETARY_COLUMNS

    @property
    def continuous_state_columns(self) -> tuple[str, ...]:
        """All primitive and engineered continuous states, including display-only."""

        return tuple(dict.fromkeys(self.state_columns + self.engineered_state_columns))

    @property
    def model_state_columns(self) -> tuple[str, ...]:
        """Continuous states actually supplied to the regression feature map."""

        redundant = set(self.redundant_state_columns)
        return tuple(c for c in self.continuous_state_columns if c not in redundant)

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.categorical_state_columns + self.categorical_parameter_columns
            )
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        # These are the columns required by the encoder.  Primitive display-only
        # states such as current_income are completed earlier by prepare_inputs.
        return tuple(
            dict.fromkeys(
                self.model_state_columns
                + self.parameter_columns
                + self.categorical_columns
            )
        )


class StandardizedFeatureEncoder:
    """Median imputation, selective asinh transforms, and one-hot coding."""

    def __init__(self, spec: FeatureSpec):
        self.spec = spec
        self.medians_: dict[str, float] = {}
        self.asinh_scales_: dict[str, float] = {}
        self.means_: dict[str, float] = {}
        self.stds_: dict[str, float] = {}
        self.categories_: dict[str, list[str]] = {}
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame) -> "StandardizedFeatureEncoder":
        missing = [c for c in self.spec.required_columns if c not in X.columns]
        if missing:
            raise KeyError(f"Feature columns missing from training data: {missing}")

        for col in self.spec.model_state_columns + self.spec.parameter_columns:
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            values = np.where(np.isfinite(values), values, median)
            self.medians_[col] = median

            if col in self.spec.signed_log_columns:
                nonzero = np.abs(values[np.abs(values) > EPS])
                scale = float(np.median(nonzero)) if len(nonzero) else 1.0
                scale = max(scale, EPS)
                self.asinh_scales_[col] = scale
                values = np.arcsinh(values / scale)

            mean = float(np.mean(values))
            std = float(np.std(values))
            self.means_[col] = mean
            self.stds_[col] = max(std, 1.0e-8)

        for col in self.spec.categorical_columns:
            values = X[col].where(X[col].notna(), "__missing__").astype(str)
            self.categories_[col] = sorted(values.unique().tolist())

        self.fitted_ = True
        return self

    def _numeric_block(self, X: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for col in cols:
            values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            values = np.where(np.isfinite(values), values, self.medians_[col])
            if col in self.asinh_scales_:
                values = np.arcsinh(values / self.asinh_scales_[col])
            values = (values - self.means_[col]) / self.stds_[col]
            blocks.append(values[:, None])
        if not blocks:
            return np.empty((len(X), 0), dtype=np.float32)
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    def transform_parts(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        if not self.fitted_:
            raise RuntimeError("Feature encoder is not fitted.")
        missing = [c for c in self.spec.required_columns if c not in X.columns]
        if missing:
            raise KeyError(f"Feature columns missing at prediction time: {missing}")

        state = self._numeric_block(X, self.spec.model_state_columns)
        params = self._numeric_block(X, self.spec.parameter_columns)

        cat_blocks: list[np.ndarray] = []
        cat_names: list[str] = []
        for col in self.spec.categorical_columns:
            values = X[col].where(X[col].notna(), "__missing__").astype(str).to_numpy()
            known = self.categories_[col]
            # Treatment coding: first observed level is the baseline.
            encoded = known[1:]
            block = np.zeros((len(X), len(encoded)), dtype=np.float32)
            lookup = {value: j for j, value in enumerate(encoded)}
            for i, value in enumerate(values):
                j = lookup.get(value)
                if j is not None:
                    block[i, j] = 1.0
            cat_blocks.append(block)
            cat_names.extend([f"{col}={value}" for value in encoded])

        cats = (
            np.concatenate(cat_blocks, axis=1)
            if cat_blocks
            else np.empty((len(X), 0), dtype=np.float32)
        )
        return state, params, cats, cat_names

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        state, params, cats, _ = self.transform_parts(X)
        return np.concatenate([state, params, cats], axis=1)


def _matches_any_pattern(name: str, patterns: Sequence[str]) -> bool:
    lower = str(name).lower()
    return any(str(pattern).lower() in lower for pattern in patterns)


def _explicit_spline_knots(
    values: np.ndarray,
    *,
    n_knots: int,
    strategy: str,
    power: float,
) -> np.ndarray | str:
    """Return sklearn spline knots with optional low-tail concentration.

    ``quantile`` reproduces the previous behavior. ``uniform`` places knots at
    equal distances in the supplied feature coordinate. ``low_tail`` uses
    power-spaced distances from the lower support endpoint, placing many more
    knots where liquidity is low.  The upper endpoint is the 99.5th percentile;
    rarer observations use the spline transformer's linear extrapolation rather
    than consuming most of the interior knot resolution.
    """

    strategy = str(strategy).strip().lower()
    if strategy == "quantile":
        return "quantile"
    if strategy not in {"uniform", "low_tail"}:
        raise ValueError(
            "Spline knot strategy must be 'quantile', 'uniform', or 'low_tail'; "
            f"received {strategy!r}."
        )
    if int(n_knots) < 4:
        raise ValueError("Every spline needs at least four knots.")
    if not np.isfinite(float(power)) or float(power) <= 0.0:
        raise ValueError("Spline low-tail powers must be finite and positive.")

    finite = np.asarray(values, dtype=float).ravel()
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("Cannot construct spline knots from an empty column.")

    lower = float(np.min(finite))
    upper = float(np.quantile(finite, 0.995))
    if upper <= lower + 1.0e-12:
        upper = float(np.max(finite))
    if upper <= lower + 1.0e-12:
        raise ValueError("Spline column has no usable continuous support.")

    u = np.linspace(0.0, 1.0, int(n_knots), dtype=float)
    if strategy == "low_tail":
        u = u ** float(power)
    knots = lower + (upper - lower) * u
    if np.any(np.diff(knots) <= 0.0):
        raise ValueError("Constructed spline knots are not strictly increasing.")
    return knots[:, None]


class StructuredPolynomialMap:
    """Standardized polynomial, spline, and targeted tensor-product basis.

    The map combines smooth state curvature with a small set of economically
    motivated interactions. It supports parameter nonlinearities that can alter
    liquid-wealth slopes and category-specific spline tensors, especially for
    the employed/unemployed distinction in the working-age application.
    """

    def __init__(
        self,
        spec: FeatureSpec,
        *,
        degree: int = 2,
        state_state_interactions: bool = True,
        state_parameter_interactions: bool = True,
        parameter_parameter_interactions: bool = False,
        categorical_slopes: bool = True,
        categorical_slope_exclude_patterns: Sequence[str] = (
            DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS
        ),
        spline_columns: Sequence[str] = DEFAULT_SPLINE_COLUMNS,
        spline_n_knots: int = 8,
        spline_column_n_knots: Sequence[
            tuple[str, int]
        ] = DEFAULT_SPLINE_COLUMN_N_KNOTS,
        spline_knot_strategies: Sequence[
            tuple[str, str]
        ] = DEFAULT_SPLINE_KNOT_STRATEGIES,
        spline_knot_powers: Sequence[tuple[str, float]] = DEFAULT_SPLINE_KNOT_POWERS,
        spline_degree: int = 3,
        spline_parameter_interactions: bool = True,
        spline_categorical_interactions: bool = True,
        spline_categorical_exclude_patterns: Sequence[str] = (
            DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS
        ),
        spline_parameter_patterns: Sequence[str] = DEFAULT_SPLINE_PARAMETER_PATTERNS,
        spline_tensor_pairs: Sequence[tuple[str, str]] = DEFAULT_SPLINE_TENSOR_PAIRS,
        spline_tensor_triples: Sequence[
            tuple[str, str, str]
        ] = DEFAULT_SPLINE_TENSOR_TRIPLES,
        spline_categorical_tensor_pairs: Sequence[
            tuple[str, str, str]
        ] = DEFAULT_SPLINE_CATEGORICAL_TENSOR_PAIRS,
        spline_tensor_parameter_terms: Sequence[
            tuple[str, str, str]
        ] = DEFAULT_SPLINE_TENSOR_PARAMETER_TERMS,
        parameter_power_terms: Sequence[
            tuple[str, int]
        ] = DEFAULT_PARAMETER_POWER_TERMS,
        spline_parameter_power_terms: Sequence[
            tuple[str, str, int]
        ] = DEFAULT_SPLINE_PARAMETER_POWER_TERMS,
        spline_parameter_pair_terms: Sequence[
            tuple[str, str, str]
        ] = DEFAULT_SPLINE_PARAMETER_PAIR_TERMS,
        categorical_state_power_terms: Sequence[
            tuple[str, str, int]
        ] = DEFAULT_CATEGORICAL_STATE_POWER_TERMS,
        final_standardize: bool = True,
        scaler_sample_rows: int = 100_000,
        random_state: int = 123,
    ):
        if degree not in {1, 2, 3, 4}:
            raise ValueError("degree must be 1, 2, 3, or 4")
        if spline_n_knots < 4:
            raise ValueError("spline_n_knots must be at least 4.")
        if spline_degree < 1:
            raise ValueError("spline_degree must be positive.")

        self.spec = spec
        self.degree = int(degree)
        self.state_state_interactions = bool(state_state_interactions)
        self.state_parameter_interactions = bool(state_parameter_interactions)
        self.parameter_parameter_interactions = bool(parameter_parameter_interactions)
        self.categorical_slopes = bool(categorical_slopes)
        self.categorical_slope_exclude_patterns = tuple(
            categorical_slope_exclude_patterns
        )
        self.spline_columns = tuple(
            col for col in spline_columns if col in spec.model_state_columns
        )
        self.spline_n_knots = int(spline_n_knots)
        self.spline_column_n_knots = {
            str(column): int(count) for column, count in spline_column_n_knots
        }
        self.spline_knot_strategies = {
            str(column): str(strategy).strip().lower()
            for column, strategy in spline_knot_strategies
        }
        self.spline_knot_powers = {
            str(column): float(power) for column, power in spline_knot_powers
        }
        for column, count in self.spline_column_n_knots.items():
            if count < 4:
                raise ValueError(f"Spline column {column!r} has fewer than four knots.")
        for column, strategy in self.spline_knot_strategies.items():
            if strategy not in {"quantile", "uniform", "low_tail"}:
                raise ValueError(
                    f"Unsupported knot strategy {strategy!r} for {column!r}."
                )
        self.spline_degree = int(spline_degree)
        self.spline_parameter_interactions = bool(spline_parameter_interactions)
        self.spline_categorical_interactions = bool(spline_categorical_interactions)
        self.spline_categorical_exclude_patterns = tuple(
            spline_categorical_exclude_patterns
        )
        self.spline_parameter_patterns = tuple(spline_parameter_patterns)
        self.spline_tensor_pairs = tuple(
            (str(left), str(right)) for left, right in spline_tensor_pairs
        )
        self.spline_tensor_triples = tuple(
            (str(first), str(second), str(third))
            for first, second, third in spline_tensor_triples
        )
        self.spline_categorical_tensor_pairs = tuple(
            (str(category), str(left), str(right))
            for category, left, right in spline_categorical_tensor_pairs
        )
        self.spline_tensor_parameter_terms = tuple(
            (str(left), str(right), str(parameter))
            for left, right, parameter in spline_tensor_parameter_terms
        )
        self.parameter_power_terms = tuple(
            (str(parameter), int(power))
            for parameter, power in parameter_power_terms
            if int(power) >= 2
        )
        self.spline_parameter_power_terms = tuple(
            (str(spline), str(parameter), int(power))
            for spline, parameter, power in spline_parameter_power_terms
            if int(power) >= 2
        )
        self.spline_parameter_pair_terms = tuple(
            (str(spline), str(parameter_1), str(parameter_2))
            for spline, parameter_1, parameter_2 in spline_parameter_pair_terms
        )
        self.categorical_state_power_terms = tuple(
            (str(category), str(state), int(power))
            for category, state, power in categorical_state_power_terms
            if int(power) >= 1
        )
        self.final_standardize = bool(final_standardize)
        self.scaler_sample_rows = int(max(1, scaler_sample_rows))
        self.random_state = int(random_state)

        self.encoder = StandardizedFeatureEncoder(spec)
        self.spline_transformers_: dict[str, SplineTransformer] = {}
        self.spline_feature_names_: list[str] = []
        self.spline_slices_: dict[str, slice] = {}
        self.spline_knots_: dict[str, list[float]] = {}
        self.active_spline_tensor_pairs_: list[tuple[str, str]] = []
        self.active_spline_tensor_triples_: list[tuple[str, str, str]] = []
        self.active_spline_categorical_tensor_pairs_: list[
            tuple[np.ndarray, list[str], str, str]
        ] = []
        self.active_spline_tensor_parameter_terms_: list[tuple[str, str, int, str]] = []
        self.active_parameter_power_terms_: list[tuple[int, str, int]] = []
        self.active_spline_parameter_power_terms_: list[tuple[str, int, str, int]] = []
        self.active_spline_parameter_pair_terms_: list[
            tuple[str, int, str, int, str]
        ] = []
        self.active_categorical_state_power_terms_: list[
            tuple[np.ndarray, list[str], int, str, int]
        ] = []
        self.spline_parameter_indices_: np.ndarray = np.empty(0, dtype=np.int64)
        self.categorical_slope_indices_: np.ndarray = np.empty(0, dtype=np.int64)
        self.spline_categorical_indices_: np.ndarray = np.empty(0, dtype=np.int64)
        self.raw_feature_names_: list[str] = []
        self.feature_names_: list[str] = []
        self.keep_mask_: np.ndarray | None = None
        self.design_means_: np.ndarray | None = None
        self.design_stds_: np.ndarray | None = None
        self.fitted_: bool = False

    def _raw_filled_column(self, X: pd.DataFrame, col: str) -> np.ndarray:
        values = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
        median = float(self.encoder.medians_.get(col, 0.0))
        return np.where(np.isfinite(values), values, median)[:, None]

    def _fit_components(self, X: pd.DataFrame) -> None:
        self.encoder.fit(X)
        _, _, _, cat_names = self.encoder.transform_parts(X.iloc[:1])
        state_names = list(self.spec.model_state_columns)
        parameter_names = list(self.spec.parameter_columns)

        self.categorical_slope_indices_ = np.asarray(
            [
                i
                for i, name in enumerate(cat_names)
                if not _matches_any_pattern(
                    name, self.categorical_slope_exclude_patterns
                )
            ],
            dtype=np.int64,
        )
        categorical_slope_names = [
            cat_names[i] for i in self.categorical_slope_indices_
        ]
        self.spline_categorical_indices_ = np.asarray(
            [
                i
                for i, name in enumerate(cat_names)
                if not _matches_any_pattern(
                    name, self.spline_categorical_exclude_patterns
                )
            ],
            dtype=np.int64,
        )
        spline_categorical_names = [
            cat_names[i] for i in self.spline_categorical_indices_
        ]

        self.spline_transformers_ = {}
        self.spline_feature_names_ = []
        self.spline_slices_ = {}
        self.spline_knots_ = {}
        for col in self.spline_columns:
            values = self._raw_filled_column(X, col)
            if np.unique(np.round(values.ravel(), 12)).size < 4:
                continue

            n_knots = int(self.spline_column_n_knots.get(col, self.spline_n_knots))
            strategy = self.spline_knot_strategies.get(col, "quantile")
            power = float(self.spline_knot_powers.get(col, 2.0))
            knot_values = values
            if col == "log1p_liquid_slack_to_income":
                # The borrowing constraint is an economically meaningful knot
                # even when continuous population draws do not land exactly on it.
                knot_values = np.vstack([values, np.zeros((1, 1), dtype=float)])
            knots = _explicit_spline_knots(
                knot_values,
                n_knots=n_knots,
                strategy=strategy,
                power=power,
            )
            transformer = SplineTransformer(
                n_knots=n_knots,
                degree=self.spline_degree,
                knots=knots,
                extrapolation="linear",
                include_bias=False,
            )
            transformer.fit(values)

            if isinstance(knots, str):
                finite_values = values.ravel()
                finite_values = finite_values[np.isfinite(finite_values)]
                stored_knots = np.quantile(
                    finite_values,
                    np.linspace(0.0, 1.0, n_knots),
                )
            else:
                stored_knots = np.asarray(knots, dtype=float).ravel()
            self.spline_knots_[col] = [float(value) for value in stored_knots]

            names = transformer.get_feature_names_out([col]).tolist()
            start = len(self.spline_feature_names_)
            self.spline_transformers_[col] = transformer
            self.spline_feature_names_.extend(names)
            self.spline_slices_[col] = slice(start, start + len(names))

        self.active_spline_tensor_pairs_ = [
            (left, right)
            for left, right in self.spline_tensor_pairs
            if left in self.spline_slices_ and right in self.spline_slices_
        ]
        self.active_spline_tensor_triples_ = [
            (first, second, third)
            for first, second, third in self.spline_tensor_triples
            if (
                first in self.spline_slices_
                and second in self.spline_slices_
                and third in self.spline_slices_
            )
        ]
        self.active_spline_categorical_tensor_pairs_ = []
        for category, left, right in self.spline_categorical_tensor_pairs:
            if left not in self.spline_slices_ or right not in self.spline_slices_:
                continue
            category_indices = np.asarray(
                [
                    i
                    for i, feature_name in enumerate(cat_names)
                    if feature_name.startswith(f"{category}=")
                    and not _matches_any_pattern(
                        feature_name, self.spline_categorical_exclude_patterns
                    )
                ],
                dtype=np.int64,
            )
            if category_indices.size == 0:
                continue
            self.active_spline_categorical_tensor_pairs_.append(
                (
                    category_indices,
                    [cat_names[i] for i in category_indices],
                    left,
                    right,
                )
            )

        selected_parameter_indices = [
            i
            for i, name in enumerate(parameter_names)
            if _matches_any_pattern(name, self.spline_parameter_patterns)
            and not name.endswith("__missing")
        ]
        self.spline_parameter_indices_ = np.asarray(
            selected_parameter_indices, dtype=np.int64
        )
        selected_parameter_names = [
            parameter_names[i] for i in selected_parameter_indices
        ]

        parameter_lookup = {name: i for i, name in enumerate(parameter_names)}
        state_lookup = {name: i for i, name in enumerate(state_names)}
        self.active_spline_tensor_parameter_terms_ = [
            (
                left,
                right,
                parameter_lookup[parameter],
                parameter,
            )
            for left, right, parameter in self.spline_tensor_parameter_terms
            if (
                left in self.spline_slices_
                and right in self.spline_slices_
                and parameter in parameter_lookup
            )
        ]
        self.active_parameter_power_terms_ = [
            (parameter_lookup[parameter], parameter, int(power))
            for parameter, power in self.parameter_power_terms
            if parameter in parameter_lookup and int(power) > self.degree
        ]
        self.active_spline_parameter_power_terms_ = [
            (spline, parameter_lookup[parameter], parameter, int(power))
            for spline, parameter, power in self.spline_parameter_power_terms
            if spline in self.spline_slices_ and parameter in parameter_lookup
        ]
        self.active_spline_parameter_pair_terms_ = [
            (
                spline,
                parameter_lookup[parameter_1],
                parameter_1,
                parameter_lookup[parameter_2],
                parameter_2,
            )
            for spline, parameter_1, parameter_2 in self.spline_parameter_pair_terms
            if (
                spline in self.spline_slices_
                and parameter_1 in parameter_lookup
                and parameter_2 in parameter_lookup
                and parameter_1 != parameter_2
            )
        ]
        self.active_categorical_state_power_terms_ = []
        for category, state_name, power in self.categorical_state_power_terms:
            if state_name not in state_lookup:
                continue
            category_indices = np.asarray(
                [
                    i
                    for i, feature_name in enumerate(cat_names)
                    if feature_name.startswith(f"{category}=")
                ],
                dtype=np.int64,
            )
            if category_indices.size == 0:
                continue
            category_names = [cat_names[i] for i in category_indices]
            # The ordinary category-specific slope block already contains
            # category x state for power one. Keep only genuinely additional
            # powers here to avoid duplicate columns.
            if int(power) == 1 and self.categorical_slopes:
                continue
            self.active_categorical_state_power_terms_.append(
                (
                    category_indices,
                    category_names,
                    state_lookup[state_name],
                    state_name,
                    int(power),
                )
            )

        names = state_names + parameter_names + cat_names
        numeric_names = state_names + parameter_names
        if self.degree >= 2:
            names += [f"{x}^2" for x in numeric_names]
        if self.degree >= 3:
            names += [f"{x}^3" for x in numeric_names]
        if self.degree >= 4:
            names += [f"{x}^4" for x in numeric_names]
        names += [
            f"{parameter}^{power}"
            for _, parameter, power in self.active_parameter_power_terms_
        ]
        if self.state_state_interactions:
            names += [
                f"{state_names[i]}*{state_names[j]}"
                for i in range(len(state_names))
                for j in range(i + 1, len(state_names))
            ]
        if self.state_parameter_interactions:
            names += [
                f"{state}*{parameter}"
                for state in state_names
                for parameter in parameter_names
            ]
        if self.parameter_parameter_interactions:
            names += [
                f"{parameter_names[i]}*{parameter_names[j]}"
                for i in range(len(parameter_names))
                for j in range(i + 1, len(parameter_names))
            ]
        if self.categorical_slopes:
            names += [
                f"{category}*{state}"
                for category in categorical_slope_names
                for state in state_names
            ]
        for (
            _,
            category_names,
            _,
            state_name,
            power,
        ) in self.active_categorical_state_power_terms_:
            names += [f"{category}*{state_name}^{power}" for category in category_names]

        names += self.spline_feature_names_
        if self.spline_parameter_interactions:
            names += [
                f"{spline}*{parameter}"
                for spline in self.spline_feature_names_
                for parameter in selected_parameter_names
            ]
        if self.spline_categorical_interactions:
            names += [
                f"{spline}*{category}"
                for spline in self.spline_feature_names_
                for category in spline_categorical_names
            ]
        for (
            spline_name,
            _,
            parameter_name,
            power,
        ) in self.active_spline_parameter_power_terms_:
            basis_names = self.spline_feature_names_[self.spline_slices_[spline_name]]
            names += [f"{basis}*{parameter_name}^{power}" for basis in basis_names]
        for (
            spline_name,
            _,
            parameter_1,
            _,
            parameter_2,
        ) in self.active_spline_parameter_pair_terms_:
            basis_names = self.spline_feature_names_[self.spline_slices_[spline_name]]
            names += [f"{basis}*{parameter_1}*{parameter_2}" for basis in basis_names]
        for left, right in self.active_spline_tensor_pairs_:
            left_names = self.spline_feature_names_[self.spline_slices_[left]]
            right_names = self.spline_feature_names_[self.spline_slices_[right]]
            names += [
                f"tensor({left_name},{right_name})"
                for left_name in left_names
                for right_name in right_names
            ]
        for left, right, _, parameter in self.active_spline_tensor_parameter_terms_:
            left_names = self.spline_feature_names_[self.spline_slices_[left]]
            right_names = self.spline_feature_names_[self.spline_slices_[right]]
            names += [
                f"{parameter}*tensor({left_name},{right_name})"
                for left_name in left_names
                for right_name in right_names
            ]
        for (
            _,
            category_names,
            left,
            right,
        ) in self.active_spline_categorical_tensor_pairs_:
            left_names = self.spline_feature_names_[self.spline_slices_[left]]
            right_names = self.spline_feature_names_[self.spline_slices_[right]]
            names += [
                f"{category}*tensor({left_name},{right_name})"
                for category in category_names
                for left_name in left_names
                for right_name in right_names
            ]
        for first, second, third in self.active_spline_tensor_triples_:
            first_names = self.spline_feature_names_[self.spline_slices_[first]]
            second_names = self.spline_feature_names_[self.spline_slices_[second]]
            third_names = self.spline_feature_names_[self.spline_slices_[third]]
            names += [
                f"tensor3({first_name},{second_name},{third_name})"
                for first_name in first_names
                for second_name in second_names
                for third_name in third_names
            ]

        self.raw_feature_names_ = names

    def _spline_block(self, X: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for col, transformer in self.spline_transformers_.items():
            blocks.append(
                transformer.transform(self._raw_filled_column(X, col)).astype(
                    np.float32, copy=False
                )
            )
        if not blocks:
            return np.empty((len(X), 0), dtype=np.float32)
        return np.concatenate(blocks, axis=1)

    def _raw_transform(self, X: pd.DataFrame) -> np.ndarray:
        state, params, cats, _ = self.encoder.transform_parts(X)
        numeric = np.concatenate([state, params], axis=1)
        blocks: list[np.ndarray] = [state, params, cats]

        if self.degree >= 2:
            blocks.append(numeric**2)
        if self.degree >= 3:
            blocks.append(numeric**3)
        if self.degree >= 4:
            blocks.append(numeric**4)
        for parameter_index, _, power in self.active_parameter_power_terms_:
            blocks.append(params[:, parameter_index : parameter_index + 1] ** power)
        if self.state_state_interactions and state.shape[1] > 1:
            blocks.extend(
                state[:, i : i + 1] * state[:, j : j + 1]
                for i in range(state.shape[1])
                for j in range(i + 1, state.shape[1])
            )
        if self.state_parameter_interactions and params.shape[1] > 0:
            blocks.append((state[:, :, None] * params[:, None, :]).reshape(len(X), -1))
        if self.parameter_parameter_interactions and params.shape[1] > 1:
            blocks.extend(
                params[:, i : i + 1] * params[:, j : j + 1]
                for i in range(params.shape[1])
                for j in range(i + 1, params.shape[1])
            )
        if (
            self.categorical_slopes
            and self.categorical_slope_indices_.size > 0
            and state.shape[1] > 0
        ):
            selected_cats = cats[:, self.categorical_slope_indices_]
            blocks.append(
                (selected_cats[:, :, None] * state[:, None, :]).reshape(len(X), -1)
            )
        for (
            category_indices,
            _,
            state_index,
            _,
            power,
        ) in self.active_categorical_state_power_terms_:
            selected_cats = cats[:, category_indices]
            state_power = state[:, state_index : state_index + 1] ** power
            blocks.append(selected_cats * state_power)

        spline = self._spline_block(X)
        if spline.shape[1] > 0:
            blocks.append(spline)
            if (
                self.spline_parameter_interactions
                and self.spline_parameter_indices_.size > 0
            ):
                selected_params = params[:, self.spline_parameter_indices_]
                blocks.append(
                    (spline[:, :, None] * selected_params[:, None, :]).reshape(
                        len(X), -1
                    )
                )
            if (
                self.spline_categorical_interactions
                and self.spline_categorical_indices_.size > 0
            ):
                selected_cats = cats[:, self.spline_categorical_indices_]
                blocks.append(
                    (spline[:, :, None] * selected_cats[:, None, :]).reshape(len(X), -1)
                )
            for (
                spline_name,
                parameter_index,
                _,
                power,
            ) in self.active_spline_parameter_power_terms_:
                spline_block = spline[:, self.spline_slices_[spline_name]]
                parameter_power = (
                    params[:, parameter_index : parameter_index + 1] ** power
                )
                blocks.append(spline_block * parameter_power)
            for (
                spline_name,
                parameter_1_index,
                _,
                parameter_2_index,
                _,
            ) in self.active_spline_parameter_pair_terms_:
                spline_block = spline[:, self.spline_slices_[spline_name]]
                parameter_product = (
                    params[:, parameter_1_index : parameter_1_index + 1]
                    * params[:, parameter_2_index : parameter_2_index + 1]
                )
                blocks.append(spline_block * parameter_product)
            for left, right in self.active_spline_tensor_pairs_:
                left_block = spline[:, self.spline_slices_[left]]
                right_block = spline[:, self.spline_slices_[right]]
                blocks.append(
                    (left_block[:, :, None] * right_block[:, None, :]).reshape(
                        len(X), -1
                    )
                )
            for left, right, parameter_index, _ in getattr(
                self,
                "active_spline_tensor_parameter_terms_",
                [],
            ):
                left_block = spline[:, self.spline_slices_[left]]
                right_block = spline[:, self.spline_slices_[right]]
                tensor_block = (
                    left_block[:, :, None] * right_block[:, None, :]
                ).reshape(len(X), -1)
                blocks.append(
                    tensor_block * params[:, parameter_index : parameter_index + 1]
                )
            for (
                category_indices,
                _,
                left,
                right,
            ) in self.active_spline_categorical_tensor_pairs_:
                selected_cats = cats[:, category_indices]
                left_block = spline[:, self.spline_slices_[left]]
                right_block = spline[:, self.spline_slices_[right]]
                blocks.append(
                    (
                        selected_cats[:, :, None, None]
                        * left_block[:, None, :, None]
                        * right_block[:, None, None, :]
                    ).reshape(len(X), -1)
                )
            for first, second, third in self.active_spline_tensor_triples_:
                first_block = spline[:, self.spline_slices_[first]]
                second_block = spline[:, self.spline_slices_[second]]
                third_block = spline[:, self.spline_slices_[third]]
                blocks.append(
                    (
                        first_block[:, :, None, None]
                        * second_block[:, None, :, None]
                        * third_block[:, None, None, :]
                    ).reshape(len(X), -1)
                )

        raw = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        if raw.shape[1] != len(self.raw_feature_names_):
            raise RuntimeError(
                "Feature-name/design mismatch: "
                f"{len(self.raw_feature_names_)} names for {raw.shape[1]} columns."
            )
        return raw

    def fit(self, X: pd.DataFrame) -> "StructuredPolynomialMap":
        self._fit_components(X)
        if len(X) > self.scaler_sample_rows:
            rng = np.random.default_rng(self.random_state)
            positions = np.sort(
                rng.choice(len(X), size=self.scaler_sample_rows, replace=False)
            )
            sample = X.iloc[positions]
        else:
            sample = X

        raw = self._raw_transform(sample).astype(np.float64, copy=False)
        finite = np.all(np.isfinite(raw), axis=0)
        std = np.nanstd(raw, axis=0)
        keep = finite & (std > 1.0e-10)
        if not np.any(keep):
            raise ValueError("The constructed design matrix has no varying columns.")

        self.keep_mask_ = keep
        kept = raw[:, keep]
        self.design_means_ = np.mean(kept, axis=0)
        self.design_stds_ = np.maximum(np.std(kept, axis=0), 1.0e-8)
        self.feature_names_ = [
            name for name, use in zip(self.raw_feature_names_, keep) if use
        ]
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_ or self.keep_mask_ is None:
            raise RuntimeError("Feature map is not fitted.")
        raw = self._raw_transform(X)[:, self.keep_mask_].astype(np.float64, copy=False)
        if self.final_standardize:
            if self.design_means_ is None or self.design_stds_ is None:
                raise RuntimeError("Final design scaler is not fitted.")
            raw = (raw - self.design_means_) / self.design_stds_
        return raw.astype(np.float32, copy=False)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


class SmoothRFFMap:
    """Smooth nonlinear feature map using random Fourier RBF features."""

    def __init__(
        self,
        spec: FeatureSpec,
        *,
        n_components: int = 512,
        gamma: float | None = None,
        include_linear: bool = True,
        random_state: int = 123,
    ):
        self.spec = spec
        self.n_components = int(n_components)
        self.gamma = gamma
        self.include_linear = bool(include_linear)
        self.random_state = int(random_state)
        self.encoder = StandardizedFeatureEncoder(spec)
        self.random_weights_: np.ndarray | None = None
        self.random_offset_: np.ndarray | None = None
        self.gamma_: float | None = None
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "SmoothRFFMap":
        self.encoder.fit(X)
        Z = self.encoder.transform(X.iloc[: min(len(X), 5_000)])
        d = max(Z.shape[1], 1)
        gamma = float(self.gamma) if self.gamma is not None else 1.0 / d
        rng = np.random.default_rng(self.random_state)
        self.random_weights_ = rng.normal(
            loc=0.0,
            scale=math.sqrt(2.0 * gamma),
            size=(d, self.n_components),
        ).astype(np.float32)
        self.random_offset_ = rng.uniform(
            0.0, 2.0 * math.pi, size=self.n_components
        ).astype(np.float32)
        self.gamma_ = gamma
        self.feature_names_ = [f"rbf_{i}" for i in range(self.n_components)]
        if self.include_linear:
            self.feature_names_ += [f"linear_{i}" for i in range(d)]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.random_weights_ is None or self.random_offset_ is None:
            raise RuntimeError("RFF map is not fitted.")
        Z = self.encoder.transform(X).astype(np.float32, copy=False)
        rff = np.sqrt(2.0 / self.n_components) * np.cos(
            Z @ self.random_weights_ + self.random_offset_
        )
        if self.include_linear:
            return np.concatenate([rff, Z], axis=1).astype(np.float32, copy=False)
        return rff.astype(np.float32, copy=False)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)


# -----------------------------------------------------------------------------
# Accounting-restricted targets and fitted bundle
# -----------------------------------------------------------------------------


class TargetTransform:
    """Stable transformation for smooth regression targets."""

    def __init__(self, kind: str):
        if kind not in {"identity", "asinh", "log1p"}:
            raise ValueError(f"Unknown target transform {kind!r}")
        self.kind = kind
        self.scale_: float = 1.0

    def fit(self, y: np.ndarray) -> "TargetTransform":
        y = np.asarray(y, dtype=float)
        finite = y[np.isfinite(y)]
        nonzero = np.abs(finite[np.abs(finite) > EPS])
        self.scale_ = max(
            float(np.median(nonzero)) if len(nonzero) else 1.0,
            EPS,
        )
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.kind == "identity":
            return y
        if self.kind == "asinh":
            return np.arcsinh(y / self.scale_)
        return np.log1p(np.maximum(y, 0.0) / self.scale_)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.kind == "identity":
            return y
        if self.kind == "asinh":
            return np.sinh(np.clip(y, -30.0, 30.0)) * self.scale_
        return np.expm1(np.clip(y, -30.0, 30.0)) * self.scale_


PRIMITIVE_POLICY_TARGETS = ("consumption", "deposit")
AUXILIARY_POLICY_TARGETS = ("liquid_outflow",)
ALL_FIT_TARGETS = PRIMITIVE_POLICY_TARGETS + AUXILIARY_POLICY_TARGETS
VALIDATION_TARGETS = (
    "consumption",
    "deposit",
    "next_liquid_assets",
    "delta_liquid_assets",
    "next_illiquid_assets",
    "delta_illiquid_assets",
)


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float,
) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), float(default), dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return np.where(np.isfinite(values), values, float(default))


def _categorical_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: str,
) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), str(default), dtype=object)
    values = frame[column].where(frame[column].notna(), str(default)).astype(str)
    return values.to_numpy(dtype=object)


def _resolve_numeric_input(
    frame: pd.DataFrame,
    *,
    candidates: Sequence[str],
    accounting_column: str,
    accounting_defaults: Mapping[str, Any],
) -> np.ndarray:
    fallback = float(accounting_defaults.get(accounting_column, 0.0))
    for column in candidates:
        if column in frame:
            return _numeric_column(frame, column, default=fallback)
    if accounting_column in frame:
        return _numeric_column(frame, accounting_column, default=fallback)
    return np.full(len(frame), fallback, dtype=float)


def _resolve_categorical_input(
    frame: pd.DataFrame,
    *,
    candidates: Sequence[str],
    accounting_column: str,
    accounting_defaults: Mapping[str, Any],
) -> np.ndarray:
    fallback = str(accounting_defaults.get(accounting_column, "__missing__"))
    for column in candidates:
        if column in frame:
            return _categorical_column(frame, column, default=fallback)
    if accounting_column in frame:
        return _categorical_column(frame, accounting_column, default=fallback)
    return np.full(len(frame), fallback, dtype=object)


def _prediction_money_factor(
    frame: pd.DataFrame,
    *,
    money_units: str,
    accounting_defaults: Mapping[str, Any],
) -> np.ndarray:
    if money_units == "model":
        return np.ones(len(frame), dtype=float)
    default_scale = float(accounting_defaults.get("money_scale", 1.0) or 1.0)
    scale = _numeric_column(frame, "money_scale", default=default_scale)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("money_scale must be finite and positive in data-unit mode.")
    return scale


def _resolve_continuous_rate(
    frame: pd.DataFrame,
    *,
    derived_candidates: Sequence[str],
    annual_candidates: Sequence[str],
    accounting_column: str,
    accounting_defaults: Mapping[str, Any],
) -> np.ndarray:
    fallback = float(accounting_defaults.get(accounting_column, 0.0))
    for column in derived_candidates:
        if column in frame:
            return _numeric_column(frame, column, default=fallback)
    for column in annual_candidates:
        if column in frame:
            annual = _numeric_column(frame, column, default=np.expm1(fallback))
            if np.any(annual <= -1.0):
                raise ValueError(f"Annual return in {column} must exceed -1.")
            return np.log1p(annual)
    if accounting_column in frame:
        return _numeric_column(frame, accounting_column, default=fallback)
    return np.full(len(frame), fallback, dtype=float)


def _accounting_defaults_from_data(data: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for column in data.columns:
        if not column.startswith("acct__") and column != "money_scale":
            continue
        if column == "acct__tax_kind":
            mode = data[column].dropna().astype(str).mode()
            out[column] = mode.iloc[0] if len(mode) else "none"
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        finite = values[np.isfinite(values)]
        out[column] = float(finite.median()) if len(finite) else 0.0
    out.setdefault("acct__ct_time_step", 1.0)
    out.setdefault("acct__rb_pos_ct", 0.0)
    out.setdefault("acct__rb_neg_ct", 0.0)
    out.setdefault("acct__ra_ct", 0.0)
    out.setdefault("acct__chi0", 0.0)
    out.setdefault("acct__chi1", 0.0)
    out.setdefault("acct__xi", 0.0)
    out.setdefault("acct__illiquid_cost_floor", 1.0e-6)
    out.setdefault("acct__policy_drift_clip", 0.0)
    out.setdefault("acct__retirement_age", np.inf)
    out.setdefault("acct__terminal_age", np.inf)
    out.setdefault("acct__tax_kind", "none")
    out.setdefault("acct__tax_flat_rate", 0.0)
    out.setdefault("acct__tax_deduction", 0.0)
    out.setdefault("acct__tax_payroll_rate", 0.0)
    out.setdefault("acct__tax_progressive_rate", 0.0)
    out.setdefault("acct__tax_progressive_exponent", 1.0)
    out.setdefault("acct__tax_cap", 1.0)
    out.setdefault("acct__minimum_after_tax_income", 0.0)
    out.setdefault("money_scale", 1.0)
    return out


def prepare_policy_inputs(
    X: pd.DataFrame,
    *,
    feature_spec: FeatureSpec,
    default_values: Mapping[str, Any],
    accounting_defaults: Mapping[str, Any],
    money_units: str,
) -> pd.DataFrame:
    """Complete and reconcile state/parameter inputs before prediction.

    The function recomputes labor, pension, taxes, and after-tax income using
    the currently selected tax category and tax parameters.  It also refreshes
    deterministic state variables such as retirement status, total assets, and
    cash on hand.  This prevents the interactive page from feeding the
    surrogate combinations that violate the model's own tax accounting.
    """

    out = X.copy().reset_index(drop=True)
    n = len(out)
    if n == 0:
        return out

    # Fill all direct regression inputs first.  Derived accounting variables
    # are overwritten below with internally consistent values.
    for column in feature_spec.state_columns + feature_spec.parameter_columns:
        if column not in out:
            out[column] = default_values.get(column, 0.0)
    for column in feature_spec.categorical_columns:
        if column not in out:
            out[column] = default_values.get(column, "__missing__")

    for column in (
        "liquid_grid_min",
        "liquid_grid_max",
        "illiquid_grid_min",
        "illiquid_grid_max",
        "money_scale",
    ):
        if column not in out and column in default_values:
            out[column] = default_values[column]
    for column, value in accounting_defaults.items():
        if column.startswith("acct__") and column not in out:
            out[column] = value

    factor = _prediction_money_factor(
        out,
        money_units=money_units,
        accounting_defaults=accounting_defaults,
    )

    # The economically relevant borrowing-constraint location moves with the
    # selected borrowing limit.  Keep the reconstruction and engineered liquid
    # slack aligned with that parameter rather than a stale sample median.
    if "param__config__borrowing_limit" in out:
        borrowing_limit = _numeric_column(
            out,
            "param__config__borrowing_limit",
            default=float(default_values.get("liquid_grid_min", 0.0)),
        )
        out["liquid_grid_min"] = borrowing_limit

    # current_income is the gross income state used by the saved policy grids.
    if "current_income" not in out and "gross_income" in out:
        out["current_income"] = out["gross_income"]
    gross = _numeric_column(
        out,
        "current_income",
        default=float(default_values.get("current_income", 0.0)),
    )
    gross = np.maximum(gross, 0.0)
    out["current_income"] = gross
    out["gross_income"] = gross

    age = _numeric_column(out, "age", default=float(default_values.get("age", 0.0)))
    retirement_age = _resolve_numeric_input(
        out,
        candidates=("param__config__retirement_age",),
        accounting_column="acct__retirement_age",
        accounting_defaults=accounting_defaults,
    )
    retired = age >= retirement_age
    out["is_retired"] = retired.astype(int)
    out["years_to_retirement"] = retirement_age - age
    out["years_to_retirement_positive"] = np.maximum(retirement_age - age, 0.0)
    out["years_since_retirement"] = np.maximum(age - retirement_age, 0.0)

    terminal_age = _resolve_numeric_input(
        out,
        candidates=("param__config__max_age", "terminal_age"),
        accounting_column="acct__terminal_age",
        accounting_defaults=accounting_defaults,
    )
    # Older inputs may not carry max_age.  The fitted bundle does, but this
    # fallback keeps diagnostics on hand-built rows finite.
    fallback_terminal = float(default_values.get("terminal_age", np.nan))
    if not np.isfinite(fallback_terminal):
        fallback_terminal = float(default_values.get("age", 0.0))
    terminal_age = np.where(np.isfinite(terminal_age), terminal_age, fallback_terminal)
    terminal_age = np.maximum(terminal_age, age)
    out["terminal_age"] = terminal_age
    out["years_to_terminal"] = np.maximum(terminal_age - age, 0.0)

    employment_state = _numeric_column(
        out,
        "employment_state",
        default=float(default_values.get("employment_state", 0.0)),
    )
    employed = (employment_state == 0.0) & (~retired)
    unemployed = (~employed) & (~retired)
    labor = np.where(employed, gross, 0.0)
    pension = np.where(retired, gross, 0.0)
    unemployment = np.where(unemployed, gross, 0.0)

    tax_kind = _resolve_categorical_input(
        out,
        candidates=("param__policy__tax__kind",),
        accounting_column="acct__tax_kind",
        accounting_defaults=accounting_defaults,
    )
    flat_rate = _resolve_numeric_input(
        out,
        candidates=("param__policy__tax__flat_rate",),
        accounting_column="acct__tax_flat_rate",
        accounting_defaults=accounting_defaults,
    )
    raw_deduction = None
    if "param__policy__tax__deduction" in out:
        raw_deduction = (
            _numeric_column(
                out,
                "param__policy__tax__deduction",
                default=float(accounting_defaults.get("acct__tax_deduction", 0.0)),
            )
            * factor
        )
    deduction = (
        raw_deduction
        if raw_deduction is not None
        else _resolve_numeric_input(
            out,
            candidates=(),
            accounting_column="acct__tax_deduction",
            accounting_defaults=accounting_defaults,
        )
    )
    payroll_rate = _resolve_numeric_input(
        out,
        candidates=("param__policy__tax__payroll_rate",),
        accounting_column="acct__tax_payroll_rate",
        accounting_defaults=accounting_defaults,
    )
    progressive_rate = _resolve_numeric_input(
        out,
        candidates=("param__policy__tax__progressive_rate",),
        accounting_column="acct__tax_progressive_rate",
        accounting_defaults=accounting_defaults,
    )
    progressive_exponent = _resolve_numeric_input(
        out,
        candidates=("param__policy__tax__progressive_exponent",),
        accounting_column="acct__tax_progressive_exponent",
        accounting_defaults=accounting_defaults,
    )
    tax_cap = _resolve_numeric_input(
        out,
        candidates=("param__policy__tax__tax_cap",),
        accounting_column="acct__tax_cap",
        accounting_defaults=accounting_defaults,
    )
    taxes = compute_taxes(
        gross,
        labor,
        pension,
        tax_kind=tax_kind,
        flat_rate=flat_rate,
        deduction=deduction,
        payroll_rate=payroll_rate,
        progressive_rate=progressive_rate,
        progressive_exponent=progressive_exponent,
        tax_cap=tax_cap,
    )
    market_after_tax = gross - taxes
    if "param__policy__tax__minimum_after_tax_income" in out:
        minimum_after_tax = (
            _numeric_column(
                out,
                "param__policy__tax__minimum_after_tax_income",
                default=float(
                    accounting_defaults.get("acct__minimum_after_tax_income", 0.0)
                ),
            )
            * factor
        )
    else:
        minimum_after_tax = _resolve_numeric_input(
            out,
            candidates=(),
            accounting_column="acct__minimum_after_tax_income",
            accounting_defaults=accounting_defaults,
        )
    # Match the updated HA grids: the government transfer guarantees a minimum
    # level of disposable income after market taxes.
    minimum_after_tax = np.maximum(minimum_after_tax, 0.0)
    minimum_income_transfer = np.maximum(minimum_after_tax - market_after_tax, 0.0)
    after_tax = market_after_tax + minimum_income_transfer

    # The interactive explorer can request a proportional income slice while
    # still exposing only pre-tax income. This internal override is never used
    # during training or ordinary prediction; it simply lets a plotted change
    # in pre-tax income move disposable income in the same proportion as the
    # selected baseline state.
    if "__after_tax_income_override" in out:
        override = pd.to_numeric(
            out["__after_tax_income_override"], errors="coerce"
        ).to_numpy(dtype=float)
        use_override = np.isfinite(override)
        after_tax = np.where(use_override, np.maximum(override, 0.0), after_tax)
        minimum_income_transfer = np.where(
            use_override,
            np.maximum(after_tax - market_after_tax, 0.0),
            minimum_income_transfer,
        )

    out["labor_income"] = labor
    out["pension_income"] = pension
    out["unemployment_benefits"] = unemployment
    out["taxes"] = taxes
    out["minimum_income_transfer"] = minimum_income_transfer
    out["after_tax_income"] = after_tax
    out["employed"] = employed.astype(int)

    if {"liquid_assets", "illiquid_assets"}.issubset(out.columns):
        liquid = pd.to_numeric(out["liquid_assets"], errors="coerce")
        illiquid = pd.to_numeric(out["illiquid_assets"], errors="coerce")
        out["total_assets"] = liquid + illiquid
    if "liquid_assets" in out:
        liquid = pd.to_numeric(out["liquid_assets"], errors="coerce")
        out["cash_on_hand"] = liquid + after_tax

    # MPC-oriented engineered states.  Permanent labor-income capacity is the
    # natural scale variable in the parametric HA model.  Liquid slack aligns
    # borrowing constraints across models with different debt limits.
    liquid = _numeric_column(
        out,
        "liquid_assets",
        default=float(default_values.get("liquid_assets", 0.0)),
    )
    illiquid = _numeric_column(
        out,
        "illiquid_assets",
        default=float(default_values.get("illiquid_assets", 0.0)),
    )
    liquid_min = _numeric_column(
        out,
        "liquid_grid_min",
        default=float(default_values.get("liquid_grid_min", 0.0)),
    )
    permanent_income = _numeric_column(
        out,
        "labor_income_state",
        default=float(default_values.get("labor_income_state", 1.0)),
    )
    default_income = abs(float(default_values.get("labor_income_state", 1.0)))
    income_floor = max(1.0e-6, 1.0e-3 * default_income)
    income_scale = np.maximum(np.abs(permanent_income), income_floor)

    liquid_slack = liquid - liquid_min
    out["liquid_slack"] = liquid_slack
    out["liquid_assets_to_income"] = liquid / income_scale

    # Continuous hinge at b=0. The raw liquid-assets feature supplies the
    # slope on the debt side; this positive-part feature allows the slope and
    # quadratic curvature to change on the saving side without introducing a
    # discontinuity in the policy level. With a degree-two polynomial basis,
    # the resulting specification can fit separate quadratics for b<0 and b>0.
    out["liquid_assets_positive_part"] = np.maximum(liquid, 0.0)

    liquid_slack_to_income = liquid_slack / income_scale
    out["liquid_slack_to_income"] = liquid_slack_to_income
    out["log1p_liquid_slack_to_income"] = np.log1p(
        np.maximum(liquid_slack_to_income, 0.0) / LIQUID_SLACK_LOG_SCALE
    )
    out["illiquid_assets_to_income"] = illiquid / income_scale
    out["pre_tax_income_to_income"] = gross / income_scale
    out["after_tax_income_to_income"] = after_tax / income_scale

    # Current liquid resources measured above the borrowing limit.  This is the
    # cash-on-hand state that is comparable across models with different debt
    # limits and directly combines liquid slack with disposable income.
    out["liquid_resources_to_income"] = (liquid_slack + after_tax) / income_scale
    # Retain the conventional zero-based cash-on-hand ratio as a diagnostic.
    out["cash_on_hand_to_income"] = (liquid + after_tax) / income_scale

    # Restore any still-missing feature columns after the accounting update.
    for column in feature_spec.required_columns:
        if column not in out:
            out[column] = default_values.get(column, "__missing__")
    return out


def _transfer_use_from_deposit(
    deposit: np.ndarray,
    illiquid_assets: np.ndarray,
    *,
    chi0: np.ndarray,
    chi1: np.ndarray,
    a_floor: np.ndarray,
) -> np.ndarray:
    """Return d + adjustment_cost(d,a), the liquid-resource use of a transfer."""

    d = np.asarray(deposit, dtype=float)
    a = np.asarray(illiquid_assets, dtype=float)
    floor = np.maximum(np.asarray(a_floor, dtype=float), EPS)
    return (
        d
        + np.asarray(chi0, dtype=float) * np.abs(d)
        + 0.5 * np.asarray(chi1, dtype=float) * d**2 / np.maximum(a, floor)
    )


def _deposit_from_transfer_use(
    transfer_use: np.ndarray,
    illiquid_assets: np.ndarray,
    *,
    chi0: np.ndarray,
    chi1: np.ndarray,
    a_floor: np.ndarray,
) -> np.ndarray:
    """Invert d + chi(d,a) on the solver's economically admissible branch.

    The map is monotone on the deposit interval selected by the HJB first-order
    condition.  Positive resource use maps to deposits and negative resource use
    maps to withdrawals.  Predictions below the feasible withdrawal minimum are
    projected to that minimum rather than generating complex roots.
    """

    u = np.asarray(transfer_use, dtype=float)
    a = np.asarray(illiquid_assets, dtype=float)
    chi0_arr = np.asarray(chi0, dtype=float)
    chi1_arr = np.maximum(np.asarray(chi1, dtype=float), EPS)
    floor = np.maximum(np.asarray(a_floor, dtype=float), EPS)
    a_eff = np.maximum(a, floor)

    positive = u >= 0.0
    B = np.where(positive, 1.0 + chi0_arr, 1.0 - chi0_arr)
    B = np.maximum(B, EPS)
    A = 0.5 * chi1_arr / a_eff

    # On the withdrawal branch the minimum feasible transfer use is attained
    # where 1-chi0 + chi1*d/a = 0.
    u_min = -(B**2) / np.maximum(4.0 * A, EPS)
    u_feasible = np.where(positive, u, np.maximum(u, u_min))
    discriminant = np.maximum(B**2 + 4.0 * A * u_feasible, 0.0)

    quadratic = (-B + np.sqrt(discriminant)) / np.maximum(2.0 * A, EPS)
    linear = u_feasible / B
    d = np.where(A > 1.0e-12, quadratic, linear)

    lower = -(1.0 - chi0_arr) * a_eff / chi1_arr
    d = np.where(positive, np.maximum(d, 0.0), np.minimum(d, 0.0))
    d = np.maximum(d, lower)
    return d


def reconstruct_policy_outputs(
    X: pd.DataFrame,
    primitive_predictions: pd.DataFrame,
    *,
    accounting_defaults: Mapping[str, Any],
    money_units: str,
    output_bounds: Mapping[str, tuple[float, float]] | None = None,
    liquid_outflow_weight: float = 0.0,
    project: bool = True,
) -> pd.DataFrame:
    """Reconcile smooth policy predictions and enforce the HA accounting system.

    Consumption and deposit are fitted directly.  A third auxiliary regression
    predicts total liquid outflow,

        q = c + d + adjustment_cost(d,a),

    because q determines liquid saving one-for-one.  The selected reconciliation
    weight blends direct q with the q implied by the direct deposit prediction,
    then analytically recovers an accounting-consistent deposit.  A weight of
    zero reproduces the old consumption/deposit system; a weight of one uses the
    direct liquid-outflow fit.
    """

    if "liquid_assets" not in X or "illiquid_assets" not in X:
        raise KeyError(
            "Accounting reconstruction requires liquid_assets and illiquid_assets."
        )
    if "after_tax_income" not in X:
        raise KeyError("Accounting reconstruction requires after_tax_income.")
    if not set(PRIMITIVE_POLICY_TARGETS).issubset(primitive_predictions.columns):
        raise KeyError("Predictions must contain consumption and deposit.")

    b = pd.to_numeric(X["liquid_assets"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(X["illiquid_assets"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(X["after_tax_income"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(primitive_predictions["consumption"], errors="coerce").to_numpy(
        dtype=float
    )
    d_direct = pd.to_numeric(
        primitive_predictions["deposit"], errors="coerce"
    ).to_numpy(dtype=float)
    if project:
        c = np.maximum(c, 0.0)

    dt = _resolve_numeric_input(
        X,
        candidates=("param__config__ct_time_step",),
        accounting_column="acct__ct_time_step",
        accounting_defaults=accounting_defaults,
    )
    rb_pos = _resolve_continuous_rate(
        X,
        derived_candidates=("param__derived__rb_pos_ct",),
        annual_candidates=("param__config__liquid_interest_rate",),
        accounting_column="acct__rb_pos_ct",
        accounting_defaults=accounting_defaults,
    )
    rb_neg = _resolve_continuous_rate(
        X,
        derived_candidates=("param__derived__rb_neg_ct",),
        annual_candidates=("param__config__borrowing_interest_rate",),
        accounting_column="acct__rb_neg_ct",
        accounting_defaults=accounting_defaults,
    )
    ra = _resolve_continuous_rate(
        X,
        derived_candidates=("param__derived__ra_ct",),
        annual_candidates=("param__config__illiquid_interest_rate",),
        accounting_column="acct__ra_ct",
        accounting_defaults=accounting_defaults,
    )
    chi0 = _resolve_numeric_input(
        X,
        candidates=(
            "param__derived__chi0",
            "param__config__ct_linear_adjustment_cost",
        ),
        accounting_column="acct__chi0",
        accounting_defaults=accounting_defaults,
    )
    chi1 = _resolve_numeric_input(
        X,
        candidates=(
            "param__derived__chi1",
            "param__config__ct_convex_adjustment_cost",
        ),
        accounting_column="acct__chi1",
        accounting_defaults=accounting_defaults,
    )
    xi = _resolve_numeric_input(
        X,
        candidates=(
            "param__derived__xi",
            "param__config__ct_automatic_illiquid_income_share",
        ),
        accounting_column="acct__xi",
        accounting_defaults=accounting_defaults,
    )

    factor = _prediction_money_factor(
        X,
        money_units=money_units,
        accounting_defaults=accounting_defaults,
    )
    if "param__config__ct_illiquid_cost_floor" in X:
        a_floor = (
            _numeric_column(
                X,
                "param__config__ct_illiquid_cost_floor",
                default=float(
                    accounting_defaults.get("acct__illiquid_cost_floor", 1e-6)
                ),
            )
            * factor
        )
    else:
        a_floor = _resolve_numeric_input(
            X,
            candidates=(),
            accounting_column="acct__illiquid_cost_floor",
            accounting_defaults=accounting_defaults,
        )
    a_floor = np.maximum(a_floor, EPS)

    transfer_use_direct = _transfer_use_from_deposit(
        d_direct,
        a,
        chi0=chi0,
        chi1=chi1,
        a_floor=a_floor,
    )
    q_from_direct_deposit = c + transfer_use_direct

    weight = float(np.clip(liquid_outflow_weight, 0.0, 1.0))
    if "liquid_outflow" in primitive_predictions and weight > 0.0:
        q_direct = pd.to_numeric(
            primitive_predictions["liquid_outflow"], errors="coerce"
        ).to_numpy(dtype=float)
        q_target = weight * q_direct + (1.0 - weight) * q_from_direct_deposit
    else:
        q_direct = q_from_direct_deposit.copy()
        q_target = q_from_direct_deposit

    d = _deposit_from_transfer_use(
        q_target - c,
        a,
        chi0=chi0,
        chi1=chi1,
        a_floor=a_floor,
    )
    transfer_use = _transfer_use_from_deposit(
        d,
        a,
        chi0=chi0,
        chi1=chi1,
        a_floor=a_floor,
    )
    q = c + transfer_use
    adjustment_cost = transfer_use - d

    liquid_return = np.where(b >= 0.0, rb_pos * b, rb_neg * b)
    liquid_resources = (1.0 - xi) * y + liquid_return
    liquid_drift = liquid_resources - q
    illiquid_drift = ra * a + xi * y + d

    drift_clip = _resolve_numeric_input(
        X,
        candidates=("param__config__ct_policy_drift_clip",),
        accounting_column="acct__policy_drift_clip",
        accounting_defaults=accounting_defaults,
    )
    finite_clip = np.isfinite(drift_clip) & (drift_clip > 0.0)
    liquid_drift = np.where(
        finite_clip,
        np.minimum(np.maximum(liquid_drift, -drift_clip), drift_clip),
        liquid_drift,
    )
    illiquid_drift = np.where(
        finite_clip,
        np.minimum(np.maximum(illiquid_drift, -drift_clip), drift_clip),
        illiquid_drift,
    )

    if output_bounds is None:
        output_bounds = {}
    liq_default = output_bounds.get(
        "next_liquid_assets",
        (float(np.nanmin(b)), float(np.nanmax(b))),
    )
    ill_default = output_bounds.get(
        "next_illiquid_assets",
        (float(np.nanmin(a)), float(np.nanmax(a))),
    )
    liq_lo = _numeric_column(X, "liquid_grid_min", default=liq_default[0])
    liq_hi = _numeric_column(X, "liquid_grid_max", default=liq_default[1])
    ill_lo = _numeric_column(X, "illiquid_grid_min", default=ill_default[0])
    ill_hi = _numeric_column(X, "illiquid_grid_max", default=ill_default[1])

    # Match the original solver's state constraints before forming annual
    # transitions.  This also makes the reported drifts agree with exact grids.
    b_tol = 1.0e-10 * np.maximum(1.0, np.maximum(np.abs(liq_lo), np.abs(liq_hi)))
    a_tol = 1.0e-10 * np.maximum(1.0, np.maximum(np.abs(ill_lo), np.abs(ill_hi)))
    liquid_drift = np.where(
        b <= liq_lo + b_tol, np.maximum(liquid_drift, 0.0), liquid_drift
    )
    liquid_drift = np.where(
        b >= liq_hi - b_tol, np.minimum(liquid_drift, 0.0), liquid_drift
    )
    illiquid_drift = np.where(
        a <= ill_lo + a_tol, np.maximum(illiquid_drift, 0.0), illiquid_drift
    )
    illiquid_drift = np.where(
        a >= ill_hi - a_tol, np.minimum(illiquid_drift, 0.0), illiquid_drift
    )

    next_b_unclipped = b + dt * liquid_drift
    next_a_unclipped = a + dt * illiquid_drift
    if project:
        next_b = np.minimum(np.maximum(next_b_unclipped, liq_lo), liq_hi)
        next_a = np.minimum(np.maximum(next_a_unclipped, ill_lo), ill_hi)
    else:
        next_b = next_b_unclipped
        next_a = next_a_unclipped

    out = pd.DataFrame(index=X.index)
    out["consumption"] = c
    out["deposit"] = d
    out["deposit_direct"] = d_direct
    out["liquid_outflow"] = q
    out["liquid_outflow_direct"] = q_direct
    out["adjustment_cost"] = adjustment_cost
    out["liquid_drift"] = liquid_drift
    out["illiquid_drift"] = illiquid_drift
    out["next_liquid_assets_unclipped"] = next_b_unclipped
    out["next_illiquid_assets_unclipped"] = next_a_unclipped
    out["next_liquid_assets"] = next_b
    out["next_illiquid_assets"] = next_a
    out["delta_liquid_assets"] = next_b - b
    out["delta_illiquid_assets"] = next_a - a
    out["liquid_asset_projection_binding"] = np.abs(next_b - next_b_unclipped) > 1.0e-10
    out["illiquid_asset_projection_binding"] = (
        np.abs(next_a - next_a_unclipped) > 1.0e-10
    )
    return out


@dataclass
class PolicySurrogateBundle:
    feature_map: Any
    regressions: dict[str, Any]
    feature_spec: FeatureSpec
    internal_targets: tuple[str, ...]
    target_transforms: dict[str, TargetTransform]
    target_parameterization: str
    output_bounds: dict[str, tuple[float, float]]
    feature_ranges: dict[str, dict[str, Any]]
    default_values: dict[str, Any]
    accounting_defaults: dict[str, Any]
    validation_metrics: pd.DataFrame
    validation_by_model: pd.DataFrame
    training_metadata: dict[str, Any]
    liquid_outflow_weight: float = 0.0

    def prepare_inputs(self, X: pd.DataFrame) -> pd.DataFrame:
        money_units = str(self.training_metadata.get("money_units", "model"))
        return prepare_policy_inputs(
            X,
            feature_spec=self.feature_spec,
            default_values=self.default_values,
            accounting_defaults=self.accounting_defaults,
            money_units=money_units,
        )

    def _internal_prediction(
        self, X: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        prepared = self.prepare_inputs(X)
        Z = self.feature_map.transform(prepared)
        out: dict[str, np.ndarray] = {}
        for target in self.internal_targets:
            if target not in self.regressions:
                raise KeyError(f"No fitted regression is stored for {target!r}.")
            pred_t = np.asarray(self.regressions[target].predict(Z), dtype=float)
            out[target] = self.target_transforms[target].inverse(pred_t)
        return prepared, pd.DataFrame(out, index=prepared.index)

    def predict(self, X: pd.DataFrame, *, project: bool = True) -> pd.DataFrame:
        prepared, primitive = self._internal_prediction(X)
        money_units = str(self.training_metadata.get("money_units", "model"))
        return reconstruct_policy_outputs(
            prepared,
            primitive,
            accounting_defaults=self.accounting_defaults,
            money_units=money_units,
            output_bounds=self.output_bounds,
            liquid_outflow_weight=float(self.liquid_outflow_weight),
            project=project,
        )

    def make_default_row(self) -> pd.DataFrame:
        return pd.DataFrame([self.default_values])

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p, compress=3)
        return p

    @staticmethod
    def load(path: str | Path) -> "PolicySurrogateBundle":
        value = joblib.load(path)
        if not isinstance(value, PolicySurrogateBundle):
            raise TypeError(f"{path} does not contain a PolicySurrogateBundle.")
        if not hasattr(value, "regressions"):
            raise TypeError(
                "This bundle was created by the older independent-asset-target "
                "surrogate. Retrain it with the accounting-restricted code."
            )
        version = int(value.training_metadata.get("bundle_format_version", 0))
        if version != 10:
            raise TypeError(
                "This bundle predates the true-holdout, forward-selected "
                "surrogate format. Retrain it with the current code."
            )
        if not hasattr(value, "liquid_outflow_weight"):
            value.liquid_outflow_weight = 0.0
        return value


def _feature_range_summary(
    data: pd.DataFrame,
    spec: FeatureSpec,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in spec.continuous_state_columns + spec.parameter_columns:
        x = pd.to_numeric(data[col], errors="coerce")
        finite = x[np.isfinite(x)]
        if finite.empty:
            out[col] = {
                "kind": "continuous",
                "min": 0.0,
                "max": 0.0,
                "q01": 0.0,
                "q99": 0.0,
                "median": 0.0,
            }
        else:
            out[col] = {
                "kind": "continuous",
                "min": float(finite.min()),
                "max": float(finite.max()),
                "q01": float(finite.quantile(0.01)),
                "q99": float(finite.quantile(0.99)),
                "median": float(finite.median()),
            }
    for col in spec.categorical_columns:
        values = data[col].where(data[col].notna(), "__missing__")
        counts = values.astype(str).value_counts()
        out[col] = {
            "kind": "categorical",
            "values": counts.index.tolist(),
            "mode": str(counts.index[0]) if len(counts) else "__missing__",
        }
    return out


def _default_row(
    data: pd.DataFrame,
    spec: FeatureSpec,
    accounting_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for col in spec.continuous_state_columns + spec.parameter_columns:
        x = pd.to_numeric(data[col], errors="coerce")
        row[col] = float(x.median()) if x.notna().any() else 0.0
    for col in spec.categorical_columns:
        mode = data[col].dropna().astype(str).mode()
        row[col] = mode.iloc[0] if len(mode) else "__missing__"
    for col in (
        "liquid_grid_min",
        "liquid_grid_max",
        "illiquid_grid_min",
        "illiquid_grid_max",
        "money_scale",
    ):
        if col in data:
            row[col] = float(pd.to_numeric(data[col], errors="coerce").median())
    row.update(accounting_defaults)
    return row


def _target_transform_kind(
    target: str,
    *,
    consumption_transform: str,
    deposit_transform: str,
    liquid_outflow_transform: str,
) -> str:
    if target == "consumption":
        return consumption_transform
    if target == "deposit":
        return deposit_transform
    if target == "liquid_outflow":
        return liquid_outflow_transform
    raise KeyError(f"Unknown fitted target {target!r}.")


def _primitive_predictions_from_models(
    feature_map: Any,
    regressions: Mapping[str, Any],
    transforms: Mapping[str, TargetTransform],
    X: pd.DataFrame,
) -> pd.DataFrame:
    Z = feature_map.transform(X)
    out: dict[str, np.ndarray] = {}
    for target in regressions:
        if target not in transforms:
            continue
        pred_t = np.asarray(regressions[target].predict(Z), dtype=float)
        out[target] = transforms[target].inverse(pred_t)
    return pd.DataFrame(out, index=X.index)


def _paired_mpc_table(
    data: pd.DataFrame,
    predicted_consumption: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """Return exact/predicted MPCs and baseline lifecycle states for each pair."""

    required = {
        "model_id",
        "mpc_pair_id",
        "mpc_pair_role",
        "mpc_check_units",
        "consumption",
    }
    if not required.issubset(data.columns):
        return pd.DataFrame()

    frame = data.reset_index(drop=True)
    pair_id = pd.to_numeric(frame["mpc_pair_id"], errors="coerce").to_numpy(dtype=float)
    role = pd.to_numeric(frame["mpc_pair_role"], errors="coerce").to_numpy(dtype=float)
    check = pd.to_numeric(frame["mpc_check_units"], errors="coerce").to_numpy(
        dtype=float
    )
    exact = pd.to_numeric(frame["consumption"], errors="coerce").to_numpy(dtype=float)

    valid = (
        np.isfinite(pair_id)
        & (pair_id >= 0)
        & np.isin(role, [0.0, 1.0])
        & np.isfinite(check)
        & (check > EPS)
        & np.isfinite(exact)
    )
    if not valid.any():
        return pd.DataFrame()

    if predicted_consumption is None:
        predicted = np.full(len(frame), np.nan, dtype=float)
    else:
        predicted = pd.to_numeric(
            pd.Series(predicted_consumption), errors="coerce"
        ).to_numpy(dtype=float)
        if len(predicted) != len(frame):
            raise ValueError("predicted_consumption has the wrong length.")

    temp_data: dict[str, Any] = {
        "row_position": np.arange(len(frame), dtype=np.int64),
        "model_id": frame["model_id"].astype(str),
        "pair_id": pair_id.astype(np.int64),
        "role": role.astype(np.int8),
        "check": check,
        "exact_consumption": exact,
        "predicted_consumption": predicted,
    }
    for column in (
        "age",
        "is_retired",
        "employment_state",
        "years_to_terminal",
        "years_since_retirement",
        "employment_state",
        "current_income",
        "pre_tax_income_to_income",
        "liquid_assets",
        "liquid_slack_to_income",
        "illiquid_assets_to_income",
        "mpc_crosses_zero",
        "population_type",
    ):
        if column in frame:
            if column == "population_type":
                temp_data[column] = frame[column].astype(str).to_numpy(dtype=object)
            else:
                temp_data[column] = pd.to_numeric(
                    frame[column], errors="coerce"
                ).to_numpy(dtype=float)

    temp = pd.DataFrame(temp_data).loc[valid]
    keys = ["model_id", "pair_id"]
    baseline = temp[temp["role"] == 0].set_index(keys)
    treated = temp[temp["role"] == 1].set_index(keys)
    common = baseline.index.intersection(treated.index)
    if len(common) == 0:
        return pd.DataFrame()

    baseline = baseline.loc[common]
    treated = treated.loc[common]
    pair_check = 0.5 * (
        baseline["check"].to_numpy(dtype=float) + treated["check"].to_numpy(dtype=float)
    )
    good = np.isfinite(pair_check) & (pair_check > EPS)

    out_data: dict[str, Any] = {
        "model_id": [idx[0] for idx in common],
        "pair_id": [idx[1] for idx in common],
        "baseline_position": baseline["row_position"].to_numpy(dtype=np.int64),
        "treated_position": treated["row_position"].to_numpy(dtype=np.int64),
        "check_units": pair_check,
        "exact_mpc": (
            treated["exact_consumption"].to_numpy(dtype=float)
            - baseline["exact_consumption"].to_numpy(dtype=float)
        )
        / pair_check,
        "predicted_mpc": (
            treated["predicted_consumption"].to_numpy(dtype=float)
            - baseline["predicted_consumption"].to_numpy(dtype=float)
        )
        / pair_check,
    }
    for column in (
        "age",
        "is_retired",
        "years_to_terminal",
        "years_since_retirement",
        "employment_state",
        "current_income",
        "pre_tax_income_to_income",
        "liquid_assets",
        "liquid_slack_to_income",
        "illiquid_assets_to_income",
        "mpc_crosses_zero",
        "population_type",
    ):
        if column in baseline:
            out_data[column] = (
                baseline[column].astype(str).to_numpy(dtype=object)
                if column == "population_type"
                else baseline[column].to_numpy(dtype=float)
            )

    out = pd.DataFrame(out_data).loc[good].reset_index(drop=True)
    if "mpc_crosses_zero" not in out and "liquid_assets" in out:
        baseline_b = pd.to_numeric(out["liquid_assets"], errors="coerce").to_numpy(
            dtype=float
        )
        out["mpc_crosses_zero"] = (
            (baseline_b < 0.0)
            & (
                baseline_b
                + pd.to_numeric(out["check_units"], errors="coerce").to_numpy(
                    dtype=float
                )
                > 0.0
            )
        ).astype(np.int8)
    elif "mpc_crosses_zero" in out:
        out["mpc_crosses_zero"] = (
            pd.to_numeric(out["mpc_crosses_zero"], errors="coerce")
            .fillna(0)
            .astype(np.int8)
        )
    return out


@dataclass
class MPCPairDesign:
    """Finite-difference feature rows and exact MPC targets.

    ``difference`` contains

        (Z_treated - Z_baseline) / check

    for baseline/treated states that differ only in liquid assets.  The design
    is constructed after the final polynomial/spline basis has been
    standardized, so its columns are on comparable scales.
    """

    difference: np.ndarray
    target: np.ndarray
    checks: np.ndarray
    model_ids: np.ndarray
    baseline_positions: np.ndarray
    treated_positions: np.ndarray
    n_available: int

    @property
    def n_pairs(self) -> int:
        return int(len(self.target))


def _balanced_indices_within_block(
    block: pd.DataFrame,
    *,
    n_take: int,
    rng: np.random.Generator,
) -> list[int]:
    """Select evenly across retirement, age, and liquid-wealth cells."""

    if n_take >= len(block):
        return block.index.astype(int).tolist()
    if n_take <= 0:
        return []

    strata = [
        np.asarray(values, dtype=np.int64).copy()
        for values in block.groupby("__mpc_stratum", sort=False).groups.values()
    ]
    for values in strata:
        rng.shuffle(values)

    selected: list[int] = []
    cursor = 0
    # Round-robin sampling prevents a common lifecycle cell from using the
    # entire model quota before terminal-age or retiree cells are represented.
    while len(selected) < n_take and strata:
        next_strata = []
        for values in strata:
            if cursor < len(values):
                selected.append(int(values[cursor]))
                if len(selected) == n_take:
                    break
            if cursor + 1 < len(values):
                next_strata.append(values)
        cursor += 1
        if not next_strata and len(selected) < n_take:
            remaining = np.asarray(
                [i for i in block.index if int(i) not in set(selected)],
                dtype=np.int64,
            )
            if len(remaining):
                extra = rng.choice(
                    remaining,
                    size=min(n_take - len(selected), len(remaining)),
                    replace=False,
                )
                selected.extend(int(i) for i in extra)
            break
        strata = next_strata
    return selected[:n_take]


def _stratified_pair_subsample(
    pairs: pd.DataFrame,
    *,
    max_pairs: int | None,
    random_state: int,
    age_bins: Sequence[float] = (35.0, 45.0, 55.0, 60.0),
    liquid_bins: int = 4,
    illiquid_bins: int = 4,
    target_bins: int = 0,
) -> pd.DataFrame:
    """Subsample pairs by model, lifecycle, assets, employment, and MPC level."""

    if pairs.empty:
        return pairs.copy()

    work = pairs.reset_index(drop=True).copy()
    age_edges = [-np.inf, *sorted(float(x) for x in age_bins), np.inf]
    age_values = pd.to_numeric(
        work["age"] if "age" in work else pd.Series(np.nan, index=work.index),
        errors="coerce",
    )
    work["__age_bin"] = (
        pd.cut(
            age_values,
            bins=age_edges,
            labels=False,
            include_lowest=True,
        )
        .fillna(-1)
        .astype(int)
    )
    retired = (
        pd.to_numeric(
            (
                work["is_retired"]
                if "is_retired" in work
                else pd.Series(np.nan, index=work.index)
            ),
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )
    work["__retired"] = retired
    employment_state = (
        pd.to_numeric(
            (
                work["employment_state"]
                if "employment_state" in work
                else pd.Series(np.nan, index=work.index)
            ),
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )
    work["__employment_state"] = employment_state

    liquid = pd.to_numeric(
        (
            work["liquid_slack_to_income"]
            if "liquid_slack_to_income" in work
            else pd.Series(np.nan, index=work.index)
        ),
        errors="coerce",
    )
    finite_liquid = liquid[np.isfinite(liquid)]
    if int(liquid_bins) > 1 and len(finite_liquid) >= int(liquid_bins):
        quantiles = np.unique(
            np.quantile(finite_liquid, np.linspace(0.0, 1.0, int(liquid_bins) + 1))
        )
        if len(quantiles) >= 3:
            quantiles[0] = -np.inf
            quantiles[-1] = np.inf
            work["__liquid_bin"] = (
                pd.cut(
                    liquid,
                    bins=quantiles,
                    labels=False,
                    include_lowest=True,
                    duplicates="drop",
                )
                .fillna(-1)
                .astype(int)
            )
        else:
            work["__liquid_bin"] = 0
    else:
        work["__liquid_bin"] = 0

    illiquid = pd.to_numeric(
        (
            work["illiquid_assets_to_income"]
            if "illiquid_assets_to_income" in work
            else pd.Series(np.nan, index=work.index)
        ),
        errors="coerce",
    )
    finite_illiquid = illiquid[np.isfinite(illiquid)]
    if int(illiquid_bins) > 1 and len(finite_illiquid) >= int(illiquid_bins):
        quantiles = np.unique(
            np.quantile(
                finite_illiquid,
                np.linspace(0.0, 1.0, int(illiquid_bins) + 1),
            )
        )
        if len(quantiles) >= 3:
            quantiles[0] = -np.inf
            quantiles[-1] = np.inf
            work["__illiquid_bin"] = (
                pd.cut(
                    illiquid,
                    bins=quantiles,
                    labels=False,
                    include_lowest=True,
                    duplicates="drop",
                )
                .fillna(-1)
                .astype(int)
            )
        else:
            work["__illiquid_bin"] = 0
    else:
        work["__illiquid_bin"] = 0

    exact_mpc = pd.to_numeric(
        (
            work["exact_mpc"]
            if "exact_mpc" in work
            else pd.Series(np.nan, index=work.index)
        ),
        errors="coerce",
    )
    finite_mpc = exact_mpc[np.isfinite(exact_mpc)]
    if int(target_bins) > 1 and len(finite_mpc) >= int(target_bins):
        quantiles = np.unique(
            np.quantile(finite_mpc, np.linspace(0.0, 1.0, int(target_bins) + 1))
        )
        if len(quantiles) >= 3:
            quantiles[0] = -np.inf
            quantiles[-1] = np.inf
            work["__target_bin"] = (
                pd.cut(
                    exact_mpc,
                    bins=quantiles,
                    labels=False,
                    include_lowest=True,
                    duplicates="drop",
                )
                .fillna(-1)
                .astype(int)
            )
        else:
            work["__target_bin"] = 0
    else:
        work["__target_bin"] = 0

    population_type = (
        work["population_type"].astype(str)
        if "population_type" in work
        else pd.Series("unknown", index=work.index)
    )
    work["__population_type"] = population_type
    work["__crosses_zero"] = (
        pd.to_numeric(
            (
                work["mpc_crosses_zero"]
                if "mpc_crosses_zero" in work
                else pd.Series(0, index=work.index)
            ),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    work["__mpc_stratum"] = list(
        zip(
            work["__retired"],
            work["__employment_state"],
            work["__age_bin"],
            work["__liquid_bin"],
            work["__illiquid_bin"],
            work["__target_bin"],
            work["__crosses_zero"],
            work["__population_type"],
        )
    )

    if max_pairs is None or max_pairs <= 0 or len(work) <= int(max_pairs):
        return work.drop(columns=[c for c in work if c.startswith("__")]).reset_index(
            drop=True
        )

    limit = int(max_pairs)
    rng = np.random.default_rng(int(random_state))
    model_groups = list(work.groupby("model_id", sort=False))
    n_models = len(model_groups)
    if n_models == 0:
        return work.iloc[:0].copy()

    if n_models >= limit:
        chosen = rng.choice(np.arange(n_models), size=limit, replace=False)
        selected = []
        for position in chosen:
            _, block = model_groups[int(position)]
            selected.extend(_balanced_indices_within_block(block, n_take=1, rng=rng))
    else:
        base_quota = limit // n_models
        remainder = limit % n_models
        order = rng.permutation(n_models)
        bonus = set(order[:remainder].tolist())
        selected = []
        for position, (_, block) in enumerate(model_groups):
            quota = base_quota + (1 if position in bonus else 0)
            selected.extend(
                _balanced_indices_within_block(
                    block,
                    n_take=min(quota, len(block)),
                    rng=rng,
                )
            )

        if len(selected) < limit:
            selected_set = set(selected)
            remaining = np.asarray(
                [i for i in work.index if int(i) not in selected_set],
                dtype=np.int64,
            )
            if len(remaining):
                extra = rng.choice(
                    remaining,
                    size=min(limit - len(selected), len(remaining)),
                    replace=False,
                )
                selected.extend(int(i) for i in extra)

    selected = sorted(set(selected))[:limit]
    return (
        work.loc[selected]
        .drop(columns=[c for c in work if c.startswith("__")])
        .reset_index(drop=True)
    )


def _build_mpc_pair_design(
    data: pd.DataFrame,
    Z: np.ndarray,
    *,
    max_pairs: int | None,
    random_state: int,
    trim_quantile: float = 0.0,
    age_bins: Sequence[float] = (35.0, 45.0, 55.0, 60.0),
    liquid_bins: int = 4,
    illiquid_bins: int = 4,
    target_bins: int = 0,
) -> MPCPairDesign | None:
    """Construct standardized finite-difference rows for the MPC objective."""

    pairs = _paired_mpc_table(data)
    if pairs.empty:
        return None

    n_available = int(len(pairs))
    pairs = _stratified_pair_subsample(
        pairs,
        max_pairs=max_pairs,
        random_state=random_state,
        age_bins=age_bins,
        liquid_bins=liquid_bins,
        illiquid_bins=illiquid_bins,
        target_bins=target_bins,
    )

    baseline = pairs["baseline_position"].to_numpy(dtype=np.int64)
    treated = pairs["treated_position"].to_numpy(dtype=np.int64)
    checks = pairs["check_units"].to_numpy(dtype=float)
    targets = pairs["exact_mpc"].to_numpy(dtype=float)

    valid = (
        (baseline >= 0)
        & (treated >= 0)
        & (baseline < len(Z))
        & (treated < len(Z))
        & np.isfinite(checks)
        & (checks > EPS)
        & np.isfinite(targets)
    )
    baseline = baseline[valid]
    treated = treated[valid]
    checks = checks[valid]
    targets = targets[valid]
    model_ids = pairs.loc[valid, "model_id"].astype(str).to_numpy(dtype=object)

    q = float(trim_quantile)
    if q < 0.0 or q >= 0.5:
        raise ValueError("mpc_objective_trim_quantile must be in [0, 0.5).")
    if q > 0.0 and len(targets) >= 100:
        lo, hi = np.quantile(targets, [q, 1.0 - q])
        keep = (targets >= lo) & (targets <= hi)
        baseline = baseline[keep]
        treated = treated[keep]
        checks = checks[keep]
        targets = targets[keep]
        model_ids = model_ids[keep]

    if len(targets) == 0:
        return None

    difference = (
        Z[treated].astype(np.float64, copy=False)
        - Z[baseline].astype(np.float64, copy=False)
    ) / checks[:, None]
    finite = np.isfinite(targets) & np.all(np.isfinite(difference), axis=1)
    if not finite.any():
        return None

    return MPCPairDesign(
        difference=difference[finite].astype(np.float32, copy=False),
        target=targets[finite].astype(np.float64, copy=False),
        checks=checks[finite].astype(np.float64, copy=False),
        model_ids=model_ids[finite],
        baseline_positions=baseline[finite].astype(np.int64, copy=False),
        treated_positions=treated[finite].astype(np.int64, copy=False),
        n_available=n_available,
    )


@dataclass(frozen=True)
class RidgeSufficientStatistics:
    """Centered design moments shared by every target and ridge candidate."""

    gram: np.ndarray
    mean_z: np.ndarray
    target_crosses: dict[str, np.ndarray]
    target_means: dict[str, float]
    n_rows: int


def _compute_ridge_statistics(
    Z: np.ndarray,
    targets: Mapping[str, np.ndarray],
    *,
    chunk_size: int = 100_000,
) -> RidgeSufficientStatistics:
    """Accumulate ``Z'Z`` and all ``Z'y`` products in one pass."""

    Z_arr = np.asarray(Z)
    if Z_arr.ndim != 2 or len(Z_arr) == 0:
        raise ValueError("Z must be a nonempty two-dimensional design matrix.")
    target_arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in targets.items()
    }
    for name, values in target_arrays.items():
        if len(values) != len(Z_arr):
            raise ValueError(f"Target {name!r} has the wrong number of rows.")

    n, p = Z_arr.shape
    sum_z = np.zeros(p, dtype=np.float64)
    sum_zz = np.zeros((p, p), dtype=np.float64)
    sum_y = {name: 0.0 for name in target_arrays}
    sum_zy = {name: np.zeros(p, dtype=np.float64) for name in target_arrays}
    step = max(1, int(chunk_size))
    for start in range(0, n, step):
        stop = min(start + step, n)
        z = np.asarray(Z_arr[start:stop], dtype=np.float64)
        sum_z += z.sum(axis=0)
        sum_zz += z.T @ z
        for name, values in target_arrays.items():
            yy = values[start:stop]
            sum_y[name] += float(yy.sum())
            sum_zy[name] += z.T @ yy

    mean_z = sum_z / n
    gram = sum_zz / n - np.outer(mean_z, mean_z)
    gram = 0.5 * (gram + gram.T)
    means = {name: total / n for name, total in sum_y.items()}
    crosses = {name: sum_zy[name] / n - mean_z * means[name] for name in target_arrays}
    return RidgeSufficientStatistics(
        gram=gram,
        mean_z=mean_z,
        target_crosses=crosses,
        target_means=means,
        n_rows=int(n),
    )


@dataclass(frozen=True)
class MPCSufficientStatistics:
    gram: np.ndarray
    cross: np.ndarray
    n_pairs: int
    n_available: int


def _compute_mpc_statistics(
    pair_design: MPCPairDesign | None,
    *,
    chunk_size: int = 100_000,
) -> MPCSufficientStatistics | None:
    if pair_design is None or pair_design.n_pairs == 0:
        return None
    D = np.asarray(pair_design.difference)
    m = np.asarray(pair_design.target, dtype=np.float64)
    n, p = D.shape
    gram_sum = np.zeros((p, p), dtype=np.float64)
    cross_sum = np.zeros(p, dtype=np.float64)
    step = max(1, int(chunk_size))
    for start in range(0, n, step):
        stop = min(start + step, n)
        d = np.asarray(D[start:stop], dtype=np.float64)
        mm = m[start:stop]
        gram_sum += d.T @ d
        cross_sum += d.T @ mm
    return MPCSufficientStatistics(
        gram=gram_sum / n,
        cross=cross_sum / n,
        n_pairs=int(n),
        n_available=int(pair_design.n_available),
    )


@dataclass
class ClosedFormRidgeRegressor:
    """Sample-size-invariant ridge regression from sufficient statistics.

    The objective is mean squared error plus ``alpha * ||beta||^2``.  The
    intercept is unpenalized.  Moment accumulation is chunked so the full design
    never has to be copied to float64.
    """

    alpha: float
    moment_chunk_size: int = 100_000
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    solver_diagnostics_: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _moments(
        Z: np.ndarray,
        y: np.ndarray,
        *,
        chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int]:
        Z_arr = np.asarray(Z)
        y_arr = np.asarray(y, dtype=np.float64)
        if Z_arr.ndim != 2 or len(y_arr) != len(Z_arr):
            raise ValueError("Z and y have incompatible dimensions.")
        n, p = Z_arr.shape
        sum_z = np.zeros(p, dtype=np.float64)
        sum_zz = np.zeros((p, p), dtype=np.float64)
        sum_zy = np.zeros(p, dtype=np.float64)
        sum_y = 0.0
        for start in range(0, n, max(1, int(chunk_size))):
            stop = min(start + max(1, int(chunk_size)), n)
            z = np.asarray(Z_arr[start:stop], dtype=np.float64)
            yy = y_arr[start:stop]
            sum_z += z.sum(axis=0)
            sum_zz += z.T @ z
            sum_zy += z.T @ yy
            sum_y += float(yy.sum())
        return sum_z, sum_zz, sum_y, sum_zy, int(n)

    @classmethod
    def _centered_system(
        cls,
        Z: np.ndarray,
        y: np.ndarray,
        *,
        chunk_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        sum_z, sum_zz, sum_y, sum_zy, n = cls._moments(Z, y, chunk_size=chunk_size)
        if n <= 0:
            raise ValueError("Cannot fit ridge on an empty sample.")
        mean_z = sum_z / n
        mean_y = sum_y / n
        gram = sum_zz / n - np.outer(mean_z, mean_z)
        cross = sum_zy / n - mean_z * mean_y
        gram = 0.5 * (gram + gram.T)
        return gram, cross, mean_z, float(mean_y)

    @staticmethod
    def _solve_system(
        gram: np.ndarray,
        cross: np.ndarray,
        *,
        alpha: float,
    ) -> tuple[np.ndarray, float]:
        if alpha < 0.0:
            raise ValueError("alpha must be nonnegative.")
        p = gram.shape[0]
        system = np.asarray(gram, dtype=np.float64).copy()
        system.flat[:: p + 1] += float(alpha) + 1.0e-10
        # A full SVD-based condition number is as expensive as another solve.
        # Record it only for modest feature sets used in diagnostics.
        condition = float(np.linalg.cond(system)) if p <= 500 else float("nan")
        try:
            coef = np.linalg.solve(system, np.asarray(cross, dtype=np.float64))
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(
                system, np.asarray(cross, dtype=np.float64), rcond=None
            )
        return np.asarray(coef, dtype=np.float64), condition

    def fit_from_statistics(
        self,
        statistics: RidgeSufficientStatistics,
        target: str,
    ) -> "ClosedFormRidgeRegressor":
        if target not in statistics.target_crosses:
            raise KeyError(f"No sufficient statistics were stored for {target!r}.")
        coef, condition = self._solve_system(
            statistics.gram,
            statistics.target_crosses[target],
            alpha=self.alpha,
        )
        self.coef_ = coef
        self.intercept_ = float(
            statistics.target_means[target] - statistics.mean_z @ coef
        )
        self.solver_diagnostics_ = {
            "method": "closed_form_cached_moments",
            "condition_number": condition,
            "alpha": float(self.alpha),
            "n_level_rows": int(statistics.n_rows),
        }
        return self

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "ClosedFormRidgeRegressor":
        gram, cross, mean_z, mean_y = self._centered_system(
            Z, y, chunk_size=self.moment_chunk_size
        )
        coef, condition = self._solve_system(gram, cross, alpha=self.alpha)
        self.coef_ = coef
        self.intercept_ = float(mean_y - mean_z @ coef)
        self.solver_diagnostics_ = {
            "method": "closed_form_moments",
            "condition_number": condition,
            "alpha": float(self.alpha),
            "n_level_rows": int(len(Z)),
        }
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("The ridge regressor is not fitted.")
        return float(self.intercept_) + np.asarray(Z, dtype=np.float64) @ self.coef_


@dataclass
class JointConsumptionMPCRegressor(ClosedFormRidgeRegressor):
    """Ridge regression fitted jointly to consumption levels and exact MPCs.

    It minimizes average consumption-level squared error plus
    ``mpc_weight`` times average finite-difference MPC squared error, with a
    conventional ridge penalty.  The MPC is always implied by the same fitted
    consumption function; no separate MPC head is estimated.
    """

    mpc_weight: float = 1.0

    def fit_from_statistics(
        self,
        statistics: RidgeSufficientStatistics,
        target: str,
        pair_statistics: MPCSufficientStatistics,
    ) -> "JointConsumptionMPCRegressor":
        if target not in statistics.target_crosses:
            raise KeyError(f"No sufficient statistics were stored for {target!r}.")
        if pair_statistics is None or pair_statistics.n_pairs == 0:
            raise ValueError("Joint MPC fitting requires valid MPC moments.")
        if self.mpc_weight <= 0.0:
            raise ValueError("mpc_weight must be positive.")
        gram = statistics.gram + float(self.mpc_weight) * pair_statistics.gram
        cross = (
            statistics.target_crosses[target]
            + float(self.mpc_weight) * pair_statistics.cross
        )
        coef, condition = self._solve_system(gram, cross, alpha=self.alpha)
        self.coef_ = coef
        self.intercept_ = float(
            statistics.target_means[target] - statistics.mean_z @ coef
        )
        self.solver_diagnostics_ = {
            "method": "closed_form_cached_level_plus_mpc_moments",
            "condition_number": condition,
            "alpha": float(self.alpha),
            "mpc_weight": float(self.mpc_weight),
            "n_level_rows": int(statistics.n_rows),
            "n_mpc_pairs": int(pair_statistics.n_pairs),
            "mpc_pairs_available": int(pair_statistics.n_available),
        }
        return self

    def fit(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        pair_design: MPCPairDesign,
    ) -> "JointConsumptionMPCRegressor":
        if pair_design is None or pair_design.n_pairs == 0:
            raise ValueError("Joint MPC fitting requires at least one valid pair.")
        if self.mpc_weight <= 0.0:
            raise ValueError("mpc_weight must be positive.")

        level_gram, level_cross, mean_z, mean_y = self._centered_system(
            Z, y, chunk_size=self.moment_chunk_size
        )
        D = np.asarray(pair_design.difference)
        m = np.asarray(pair_design.target, dtype=np.float64)
        if D.ndim != 2 or D.shape[1] != Z.shape[1] or len(D) != len(m):
            raise ValueError("MPC pair design has incompatible dimensions.")
        _, pair_gram_sum, _, pair_cross_sum, n_pairs = self._moments(
            D,
            m,
            chunk_size=self.moment_chunk_size,
        )
        pair_gram = pair_gram_sum / n_pairs
        pair_cross = pair_cross_sum / n_pairs

        gram = level_gram + float(self.mpc_weight) * pair_gram
        cross = level_cross + float(self.mpc_weight) * pair_cross
        coef, condition = self._solve_system(gram, cross, alpha=self.alpha)
        self.coef_ = coef
        self.intercept_ = float(mean_y - mean_z @ coef)
        self.solver_diagnostics_ = {
            "method": "closed_form_level_plus_mpc_moments",
            "condition_number": condition,
            "alpha": float(self.alpha),
            "mpc_weight": float(self.mpc_weight),
            "n_level_rows": int(len(Z)),
            "n_mpc_pairs": int(n_pairs),
            "mpc_pairs_available": int(pair_design.n_available),
        }
        return self

    def predict_mpc(
        self,
        Z_baseline: np.ndarray,
        Z_treated: np.ndarray,
        check_sizes: np.ndarray,
    ) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("The joint consumption/MPC regressor is not fitted.")
        checks = np.asarray(check_sizes, dtype=np.float64)
        if np.any(~np.isfinite(checks)) or np.any(checks <= 0.0):
            raise ValueError("check_sizes must be finite and positive.")
        difference = (
            np.asarray(Z_treated, dtype=np.float64)
            - np.asarray(Z_baseline, dtype=np.float64)
        ) / checks[:, None]
        return difference @ self.coef_


def _fit_consumption_regressor(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    mpc_weight: float,
    pair_design: MPCPairDesign | None,
    max_iter: int,
    tol: float,
) -> Any:
    """Fit ordinary ridge or the joint level/MPC estimator."""

    if float(mpc_weight) <= 0.0 or pair_design is None or pair_design.n_pairs == 0:
        model = ClosedFormRidgeRegressor(alpha=float(alpha))
        model.fit(Z, y)
        return model

    model = JointConsumptionMPCRegressor(
        alpha=float(alpha),
        mpc_weight=float(mpc_weight),
    )
    model.fit(Z, y, pair_design)
    return model


def _fit_regressor_from_statistics(
    statistics: RidgeSufficientStatistics,
    *,
    target: str,
    alpha: float,
    mpc_weight: float = 0.0,
    pair_statistics: MPCSufficientStatistics | None = None,
) -> Any:
    if target == "consumption" and float(mpc_weight) > 0.0:
        if pair_statistics is None:
            raise ValueError(
                "A positive MPC weight requires MPC sufficient statistics."
            )
        model = JointConsumptionMPCRegressor(
            alpha=float(alpha),
            mpc_weight=float(mpc_weight),
        )
        return model.fit_from_statistics(statistics, target, pair_statistics)
    model = ClosedFormRidgeRegressor(alpha=float(alpha))
    return model.fit_from_statistics(statistics, target)


REFERENCE_POLICY_LABELS: dict[str, str] = {
    "reference_70_20_10": "70% consumption / 20% liquid / 10% illiquid",
    "reference_consume_all": "Consume all disposable income",
}


def _clip_reference_asset(
    values: np.ndarray,
    truth: pd.DataFrame,
    *,
    lower_column: str,
    upper_column: str,
) -> np.ndarray:
    """Project a simple reference rule onto row-specific asset bounds."""

    out = np.asarray(values, dtype=float)
    if lower_column in truth:
        lower = pd.to_numeric(truth[lower_column], errors="coerce").to_numpy(
            dtype=float
        )
        out = np.where(np.isfinite(lower), np.maximum(out, lower), out)
    if upper_column in truth:
        upper = pd.to_numeric(truth[upper_column], errors="coerce").to_numpy(
            dtype=float
        )
        out = np.where(np.isfinite(upper), np.minimum(out, upper), out)
    return out


def _reference_policy_prediction(
    truth: pd.DataFrame,
    target: str,
    *,
    rule: str,
) -> np.ndarray | None:
    """Predictions from the two deliberately simple validation references.

    The 70/20/10 rule consumes 70 percent of disposable income, adds 20
    percent to liquid wealth, and adds 10 percent to illiquid wealth. The
    consume-all rule consumes all disposable income and leaves both assets
    unchanged. These are descriptive references, not accounting solutions to
    the HA model.
    """

    if rule not in REFERENCE_POLICY_LABELS:
        raise KeyError(f"Unknown validation reference rule {rule!r}.")
    if "after_tax_income" not in truth:
        return None

    income = pd.to_numeric(truth["after_tax_income"], errors="coerce").to_numpy(
        dtype=float
    )
    liquid = pd.to_numeric(truth["liquid_assets"], errors="coerce").to_numpy(
        dtype=float
    )
    illiquid = pd.to_numeric(truth["illiquid_assets"], errors="coerce").to_numpy(
        dtype=float
    )

    if rule == "reference_70_20_10":
        consumption = 0.70 * income
        deposit = 0.10 * income
        next_liquid = liquid + 0.20 * income
        next_illiquid = illiquid + 0.10 * income
    else:
        consumption = income
        deposit = np.zeros(len(truth), dtype=float)
        next_liquid = liquid.copy()
        next_illiquid = illiquid.copy()

    next_liquid = _clip_reference_asset(
        next_liquid,
        truth,
        lower_column="liquid_grid_min",
        upper_column="liquid_grid_max",
    )
    next_illiquid = _clip_reference_asset(
        next_illiquid,
        truth,
        lower_column="illiquid_grid_min",
        upper_column="illiquid_grid_max",
    )

    predictions = {
        "consumption": consumption,
        "deposit": deposit,
        "next_liquid_assets": next_liquid,
        "delta_liquid_assets": next_liquid - liquid,
        "next_illiquid_assets": next_illiquid,
        "delta_illiquid_assets": next_illiquid - illiquid,
    }
    return predictions.get(target)


def _reference_metric_fields(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float]:
    """Return RMSE, R-squared, bias, and surrogate skill field names."""

    y = np.asarray(truth, dtype=float)
    p = np.asarray(prediction, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_r2": np.nan,
            f"{prefix}_bias": np.nan,
        }
    y = y[ok]
    p = p[ok]
    error = p - y
    return {
        f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        f"{prefix}_bias": float(np.mean(error)),
    }


def _mpc_metric_row_from_pairs(
    data: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    split: str,
    model_id: str | None = None,
) -> dict[str, Any] | None:
    pairs = _paired_mpc_table(data, pred["consumption"].to_numpy(dtype=float))
    if pairs.empty:
        return None
    if model_id is not None:
        pairs = pairs[pairs["model_id"].astype(str) == str(model_id)]
        if pairs.empty:
            return None

    exact = pairs["exact_mpc"].to_numpy(dtype=float)
    predicted = pairs["predicted_mpc"].to_numpy(dtype=float)
    ok = np.isfinite(exact) & np.isfinite(predicted)
    exact, predicted = exact[ok], predicted[ok]
    if len(exact) == 0:
        return None

    mse = float(np.mean((predicted - exact) ** 2))
    rmse = math.sqrt(mse)
    sd = float(np.std(exact))
    row: dict[str, Any] = {
        "split": split,
        "target": "mpc",
        "n": int(len(exact)),
        "rmse": rmse,
        "nrmse_sd": float(rmse / max(sd, EPS)),
        "mae": float(np.mean(np.abs(predicted - exact))),
        "r2": float(r2_score(exact, predicted)) if len(exact) > 1 else np.nan,
        "bias": float(np.mean(predicted - exact)),
        "benchmark": None,
        "benchmark_rmse": np.nan,
        "benchmark_r2": np.nan,
        "rmse_relative_to_benchmark": np.nan,
        "skill_vs_benchmark": np.nan,
    }
    row.update(
        _reference_metric_fields(
            exact,
            np.full(len(exact), 0.70, dtype=float),
            prefix="reference_70_20_10",
        )
    )
    row.update(
        _reference_metric_fields(
            exact,
            np.full(len(exact), 1.00, dtype=float),
            prefix="reference_consume_all",
        )
    )

    if model_id is not None:
        row["model_id"] = str(model_id)
    return row


def _benchmark_prediction(
    truth: pd.DataFrame,
    target: str,
) -> tuple[str | None, np.ndarray | None]:
    if target == "deposit":
        return "zero_deposit", np.zeros(len(truth), dtype=float)
    if target == "next_liquid_assets":
        return (
            "persistence",
            pd.to_numeric(truth["liquid_assets"], errors="coerce").to_numpy(
                dtype=float
            ),
        )
    if target == "delta_liquid_assets":
        return "zero_change", np.zeros(len(truth), dtype=float)
    if target == "next_illiquid_assets":
        return (
            "persistence",
            pd.to_numeric(truth["illiquid_assets"], errors="coerce").to_numpy(
                dtype=float
            ),
        )
    if target == "delta_illiquid_assets":
        return "zero_change", np.zeros(len(truth), dtype=float)
    return None, None


def _single_metric_row(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    target: str,
    split: str,
    model_id: str | None = None,
) -> dict[str, Any] | None:
    y = pd.to_numeric(truth[target], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pred[target], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if not len(y):
        return None
    mse = float(np.mean((p - y) ** 2))
    rmse = math.sqrt(mse)
    sd = float(np.std(y))
    row: dict[str, Any] = {
        "split": split,
        "target": target,
        "n": int(len(y)),
        "rmse": rmse,
        "nrmse_sd": float(rmse / max(sd, EPS)),
        "mae": float(mean_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "bias": float(np.mean(p - y)),
        "benchmark": None,
        "benchmark_rmse": np.nan,
        "benchmark_r2": np.nan,
        "rmse_relative_to_benchmark": np.nan,
        "skill_vs_benchmark": np.nan,
    }
    if model_id is not None:
        row["model_id"] = model_id

    benchmark_name, benchmark_full = _benchmark_prediction(truth, target)
    if benchmark_full is not None:
        benchmark = np.asarray(benchmark_full, dtype=float)[ok]
        benchmark_mse = float(np.mean((benchmark - y) ** 2))
        benchmark_rmse = math.sqrt(benchmark_mse)
        row.update(
            {
                "benchmark": benchmark_name,
                "benchmark_rmse": benchmark_rmse,
                "benchmark_r2": (
                    float(r2_score(y, benchmark)) if len(y) > 1 else np.nan
                ),
                "rmse_relative_to_benchmark": float(rmse / max(benchmark_rmse, EPS)),
                "skill_vs_benchmark": float(1.0 - mse / max(benchmark_mse, EPS)),
            }
        )

    for rule in REFERENCE_POLICY_LABELS:
        reference_full = _reference_policy_prediction(truth, target, rule=rule)
        if reference_full is not None:
            row.update(
                _reference_metric_fields(
                    y,
                    np.asarray(reference_full, dtype=float)[ok],
                    prefix=rule,
                )
            )

    return row


def _metrics_table(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in VALIDATION_TARGETS:
        row = _single_metric_row(truth, pred, target=target, split=split)
        if row is not None:
            rows.append(row)
    mpc_row = _mpc_metric_row_from_pairs(truth, pred, split=split)
    if mpc_row is not None:
        rows.append(mpc_row)
    return pd.DataFrame(rows)


def _metrics_by_model(
    data: pd.DataFrame,
    pred: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, indices in data.groupby("model_id", observed=True).groups.items():
        truth = data.loc[indices]
        predicted = pred.loc[indices]
        for target in VALIDATION_TARGETS:
            row = _single_metric_row(
                truth,
                predicted,
                target=target,
                split=split,
                model_id=str(model_id),
            )
            if row is not None:
                rows.append(row)
        mpc_row = _mpc_metric_row_from_pairs(
            truth,
            predicted,
            split=split,
            model_id=str(model_id),
        )
        if mpc_row is not None:
            rows.append(mpc_row)
    return pd.DataFrame(rows)


def _stratified_model_holdout(
    model_table: pd.DataFrame,
    *,
    share: float,
    category_columns: Sequence[str],
    random_state: int,
    label: str,
) -> tuple[set[str], set[str]]:
    """Split model IDs while keeping every observed category in estimation.

    For each joint categorical-parameter cell, at least one model is retained in
    the estimation side.  Singleton cells therefore remain in estimation rather
    than becoming impossible-to-predict category levels in the held-out set.
    """

    if not 0.0 < float(share) < 1.0:
        raise ValueError(f"{label} share must be strictly between zero and one.")
    table = model_table.copy()
    if category_columns:
        table["__category_signature"] = (
            table[list(category_columns)].astype(str).agg(" | ".join, axis=1)
        )
    else:
        table["__category_signature"] = "__all__"
    rng = np.random.default_rng(random_state)
    estimation: set[str] = set()
    holdout: set[str] = set()
    for _, block in table.groupby("__category_signature", sort=True):
        ids = np.asarray(
            sorted(block["model_id"].astype(str).unique().tolist()),
            dtype=object,
        )
        rng.shuffle(ids)
        n = len(ids)
        n_holdout = 0 if n <= 1 else min(max(1, int(round(n * share))), n - 1)
        holdout.update(ids[:n_holdout].tolist())
        estimation.update(ids[n_holdout:].tolist())
    if not holdout:
        raise ValueError(
            f"Could not construct a nonempty {label} set while retaining every "
            "categorical-parameter level in estimation. More solved models per "
            "categorical policy combination are required."
        )
    return estimation, holdout


def _normalize_target_parameterization(value: str) -> str:
    if value == "accounting":
        return value
    if value in {"changes", "levels"}:
        warnings.warn(
            f"target_parameterization={value!r} is a legacy option. The updated "
            "surrogate always estimates consumption and deposit, then derives "
            "both asset choices from the accounting identities.",
            stacklevel=3,
        )
        return "accounting"
    raise ValueError(
        "target_parameterization must be 'accounting' (legacy 'changes' and "
        "'levels' are accepted as aliases)."
    )


def _relative_rmse(
    truth: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    *,
    benchmark: np.ndarray | pd.Series | None = None,
) -> float:
    """RMSE relative to a natural benchmark scale."""

    y = pd.to_numeric(pd.Series(truth), errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(pd.Series(predicted), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if not ok.any():
        return np.inf
    y = y[ok]
    p = p[ok]
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    if benchmark is None:
        scale = float(np.std(y))
    else:
        b = pd.to_numeric(pd.Series(benchmark), errors="coerce").to_numpy(dtype=float)
        b = b[ok]
        scale = float(np.sqrt(np.mean((b - y) ** 2)))
    return rmse / max(scale, EPS)


@dataclass(frozen=True)
class ParameterTermCandidate:
    """One higher-order parameter term considered by tuning-set screening."""

    kind: str
    parameters: tuple[str, ...]
    spline: str | None = None
    power: int | None = None

    @property
    def label(self) -> str:
        if self.kind == "parameter_power":
            return f"{self.parameters[0]}^{int(self.power or 0)}"
        if self.kind == "spline_parameter_power":
            return (
                f"spline({self.spline})*{self.parameters[0]}^" f"{int(self.power or 0)}"
            )
        if self.kind == "spline_parameter_pair":
            return (
                f"spline({self.spline})*{self.parameters[0]}*" f"{self.parameters[1]}"
            )
        raise KeyError(f"Unknown parameter-term candidate kind {self.kind!r}.")

    def metadata(self) -> dict[str, Any]:
        return {
            "candidate": self.label,
            "kind": self.kind,
            "spline": self.spline,
            "parameter_1": self.parameters[0] if self.parameters else None,
            "parameter_2": self.parameters[1] if len(self.parameters) > 1 else None,
            "power": self.power,
        }


@dataclass(frozen=True)
class ParameterTermSelectionResult:
    parameter_power_terms: tuple[tuple[str, int], ...]
    spline_parameter_power_terms: tuple[tuple[str, str, int], ...]
    spline_parameter_pair_terms: tuple[tuple[str, str, str], ...]
    search_rows: tuple[dict[str, Any], ...]
    baseline_metrics: dict[str, float]
    n_candidates: int
    n_selected: int


def _dedupe_tuples(values: Sequence[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
    return tuple(dict.fromkeys(tuple(value) for value in values))


def _model_subsample_to_row_cap(
    data: pd.DataFrame,
    *,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Take a model-balanced row sample while preserving complete MPC pairs.

    Earlier code selected a subset of complete models to meet the row cap. That
    made parameter screening depend on only a small portion of the parameter
    grid. The revised sampler keeps every model and assigns an approximately
    equal row quota to each one, retaining both members of any selected MPC pair.
    """

    if max_rows is None or int(max_rows) <= 0 or len(data) <= int(max_rows):
        return data.sort_values("model_id", kind="mergesort").reset_index(drop=True)

    model_ids = sorted(data["model_id"].astype(str).unique().tolist())
    if int(max_rows) < 4 * len(model_ids):
        raise ValueError(
            "Parameter-search row cap is too small to retain useful rows from "
            f"all {len(model_ids):,} models. Use at least {4 * len(model_ids):,}."
        )
    quota = max(4, int(max_rows) // len(model_ids))
    rng = np.random.default_rng(int(random_state))
    selected_positions: list[np.ndarray] = []

    model_as_text = data["model_id"].astype(str)
    for model_id in model_ids:
        positions = np.flatnonzero(model_as_text.to_numpy() == model_id)
        if len(positions) <= quota:
            selected_positions.append(positions)
            continue

        block = data.iloc[positions]
        pair_id = pd.to_numeric(
            block.get("mpc_pair_id", pd.Series(-1, index=block.index)),
            errors="coerce",
        )
        pair_role = pd.to_numeric(
            block.get("mpc_pair_role", pd.Series(-1, index=block.index)),
            errors="coerce",
        )
        paired = pair_id.ge(0) & pair_role.isin([0, 1])

        chosen_local: list[int] = []
        if paired.any():
            pair_table = (
                pd.DataFrame(
                    {
                        "position": np.arange(len(block), dtype=np.int64),
                        "pair_id": pair_id.to_numpy(),
                        "paired": paired.to_numpy(),
                    }
                )
                .loc[lambda frame: frame["paired"]]
                .groupby("pair_id", sort=True)["position"]
                .apply(list)
            )
            complete_pairs = [
                values for values in pair_table.tolist() if len(values) == 2
            ]
            max_pairs = min(len(complete_pairs), max(1, quota // 4))
            if max_pairs > 0:
                pair_choices = rng.choice(
                    len(complete_pairs), size=max_pairs, replace=False
                )
                for choice in np.sort(pair_choices):
                    chosen_local.extend(complete_pairs[int(choice)])

        remaining_budget = quota - len(chosen_local)
        if remaining_budget > 0:
            available = np.setdiff1d(
                np.arange(len(block), dtype=np.int64),
                np.asarray(chosen_local, dtype=np.int64),
                assume_unique=False,
            )
            count = min(remaining_budget, len(available))
            if count > 0:
                chosen_local.extend(
                    rng.choice(available, size=count, replace=False).tolist()
                )

        selected_positions.append(positions[np.sort(np.asarray(chosen_local))])

    chosen = np.sort(np.concatenate(selected_positions))
    out = data.iloc[chosen].copy()
    return out.sort_values(
        ["model_id", "mpc_pair_id", "mpc_pair_role"], kind="mergesort"
    ).reset_index(drop=True)


def _candidate_parameter_names(
    train: pd.DataFrame,
    spec: FeatureSpec,
    *,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> tuple[list[str], dict[str, int]]:
    names: list[str] = []
    unique_counts: dict[str, int] = {}
    for name in spec.parameter_columns:
        if name.endswith("__missing") or name not in train:
            continue
        if include_patterns and not _matches_any_pattern(name, include_patterns):
            continue
        if exclude_patterns and _matches_any_pattern(name, exclude_patterns):
            continue
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        n_unique = int(np.unique(np.round(finite, 12)).size)
        if n_unique < 2:
            continue
        names.append(name)
        unique_counts[name] = n_unique
    return sorted(names), unique_counts


def _generate_parameter_term_candidates(
    train: pd.DataFrame,
    feature_map: StructuredPolynomialMap,
    *,
    polynomial_degree: int,
    selection_spline_columns: Sequence[str],
    powers: Sequence[int],
    include_parameter_powers: bool,
    parameter_include_patterns: Sequence[str],
    parameter_exclude_patterns: Sequence[str],
    mandatory_parameter_power_terms: Sequence[tuple[str, int]],
    mandatory_spline_parameter_power_terms: Sequence[tuple[str, str, int]],
    mandatory_spline_parameter_pair_terms: Sequence[tuple[str, str, str]],
) -> list[ParameterTermCandidate]:
    parameter_names, unique_counts = _candidate_parameter_names(
        train,
        feature_map.spec,
        include_patterns=parameter_include_patterns,
        exclude_patterns=parameter_exclude_patterns,
    )
    active_splines = [
        str(name)
        for name in selection_spline_columns
        if str(name) in feature_map.spline_slices_
    ]
    mandatory_parameter_powers = {
        (str(parameter), int(power))
        for parameter, power in mandatory_parameter_power_terms
    }
    mandatory_spline_powers = {
        (str(spline), str(parameter), int(power))
        for spline, parameter, power in mandatory_spline_parameter_power_terms
    }
    mandatory_spline_pairs = {
        (str(spline), *sorted((str(parameter_1), str(parameter_2))))
        for spline, parameter_1, parameter_2 in mandatory_spline_parameter_pair_terms
    }

    candidates: list[ParameterTermCandidate] = []
    candidate_powers = sorted({int(power) for power in powers if int(power) >= 2})
    for parameter in parameter_names:
        n_unique = unique_counts[parameter]
        for power in candidate_powers:
            # A standalone power is useful only if it is not already generated
            # by the global polynomial basis and the parameter has enough support
            # to distinguish that power from lower-order terms.
            if (
                include_parameter_powers
                and power > int(polynomial_degree)
                and n_unique >= power + 1
                and (parameter, power) not in mandatory_parameter_powers
            ):
                candidates.append(
                    ParameterTermCandidate(
                        kind="parameter_power",
                        parameters=(parameter,),
                        power=power,
                    )
                )
            # Spline x parameter powers alter the liquid-wealth slope and can
            # therefore improve MPCs even when parameter^2 already enters as a
            # standalone global polynomial term.
            if n_unique >= min(power + 1, 4):
                for spline in active_splines:
                    key = (spline, parameter, power)
                    if key in mandatory_spline_powers:
                        continue
                    candidates.append(
                        ParameterTermCandidate(
                            kind="spline_parameter_power",
                            spline=spline,
                            parameters=(parameter,),
                            power=power,
                        )
                    )

    for parameter_1, parameter_2 in combinations(parameter_names, 2):
        for spline in active_splines:
            key = (spline, *sorted((parameter_1, parameter_2)))
            if key in mandatory_spline_pairs:
                continue
            candidates.append(
                ParameterTermCandidate(
                    kind="spline_parameter_pair",
                    spline=spline,
                    parameters=(parameter_1, parameter_2),
                )
            )
    return candidates


def _parameter_search_components(
    feature_map: StructuredPolynomialMap,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    _, parameters, _, _ = feature_map.encoder.transform_parts(X)
    splines = feature_map._spline_block(X)
    return (
        np.asarray(parameters, dtype=np.float32),
        np.asarray(splines, dtype=np.float32),
    )


def _candidate_raw_block(
    candidate: ParameterTermCandidate,
    *,
    parameters: np.ndarray,
    parameter_lookup: Mapping[str, int],
    splines: np.ndarray,
    spline_slices: Mapping[str, slice],
) -> np.ndarray:
    if candidate.kind == "parameter_power":
        index = int(parameter_lookup[candidate.parameters[0]])
        return parameters[:, index : index + 1] ** int(candidate.power or 1)

    if candidate.spline is None or candidate.spline not in spline_slices:
        return np.empty((len(parameters), 0), dtype=np.float32)
    spline_block = splines[:, spline_slices[candidate.spline]]
    if candidate.kind == "spline_parameter_power":
        index = int(parameter_lookup[candidate.parameters[0]])
        multiplier = parameters[:, index : index + 1] ** int(candidate.power or 1)
        return spline_block * multiplier
    if candidate.kind == "spline_parameter_pair":
        index_1 = int(parameter_lookup[candidate.parameters[0]])
        index_2 = int(parameter_lookup[candidate.parameters[1]])
        multiplier = (
            parameters[:, index_1 : index_1 + 1] * parameters[:, index_2 : index_2 + 1]
        )
        return spline_block * multiplier
    raise KeyError(f"Unknown parameter-term candidate kind {candidate.kind!r}.")


def _standardize_candidate_blocks(
    train_raw: np.ndarray,
    tune_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    train_values = np.asarray(train_raw, dtype=np.float64)
    tune_values = np.asarray(tune_raw, dtype=np.float64)
    if train_values.ndim != 2 or tune_values.ndim != 2:
        raise ValueError("Candidate feature blocks must be two-dimensional.")
    finite_columns = np.all(np.isfinite(train_values), axis=0) & np.all(
        np.isfinite(tune_values), axis=0
    )
    means = np.mean(train_values, axis=0)
    stds = np.std(train_values, axis=0)
    keep = finite_columns & np.isfinite(means) & np.isfinite(stds) & (stds > 1.0e-10)
    if not np.any(keep):
        return None
    means = means[keep]
    stds = stds[keep]
    train_standardized = (train_values[:, keep] - means) / stds
    tune_standardized = (tune_values[:, keep] - means) / stds
    return (
        train_standardized.astype(np.float32, copy=False),
        tune_standardized.astype(np.float32, copy=False),
    )


def _centered_cross_moment(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_mean: np.ndarray,
    right_mean: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Return E[(left-E left)'(right-E right)] in bounded memory."""

    x = np.asarray(left)
    c = np.asarray(right)
    if x.ndim != 2 or c.ndim != 2 or len(x) != len(c):
        raise ValueError("Cross-moment arrays have incompatible dimensions.")
    total = np.zeros((x.shape[1], c.shape[1]), dtype=np.float64)
    step = max(1, int(chunk_size))
    for start in range(0, len(x), step):
        stop = min(start + step, len(x))
        xx = np.asarray(x[start:stop], dtype=np.float64)
        cc = np.asarray(c[start:stop], dtype=np.float64)
        total += xx.T @ cc
    return total / len(x) - np.outer(left_mean, right_mean)


def _uncentered_cross_moment(
    left: np.ndarray,
    right: np.ndarray,
    *,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Return E[left' right] for finite-difference designs."""

    x = np.asarray(left)
    c = np.asarray(right)
    if x.ndim != 2 or c.ndim != 2 or len(x) != len(c):
        raise ValueError("Cross-moment arrays have incompatible dimensions.")
    total = np.zeros((x.shape[1], c.shape[1]), dtype=np.float64)
    step = max(1, int(chunk_size))
    for start in range(0, len(x), step):
        stop = min(start + step, len(x))
        xx = np.asarray(x[start:stop], dtype=np.float64)
        cc = np.asarray(c[start:stop], dtype=np.float64)
        total += xx.T @ cc
    return total / len(x)


def _ridge_system_factor(
    statistics: RidgeSufficientStatistics,
    *,
    target: str,
    alpha: float,
    mpc_weight: float,
    pair_statistics: MPCSufficientStatistics | None,
) -> tuple[tuple[np.ndarray, bool], np.ndarray]:
    """Factor the fixed baseline ridge/MPC normal equations once."""

    gram = np.asarray(statistics.gram, dtype=np.float64).copy()
    rhs = np.asarray(statistics.target_crosses[target], dtype=np.float64).copy()
    if float(mpc_weight) > 0.0:
        if pair_statistics is None:
            raise ValueError("A positive MPC weight requires pair statistics.")
        gram += float(mpc_weight) * np.asarray(pair_statistics.gram, dtype=np.float64)
        rhs += float(mpc_weight) * np.asarray(pair_statistics.cross, dtype=np.float64)
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    # Rare nearly singular feature blocks receive a tiny deterministic jitter.
    jitter = max(1.0e-12, float(alpha) * 1.0e-10)
    for _ in range(6):
        try:
            factor = cho_factor(
                gram,
                lower=True,
                overwrite_a=False,
                check_finite=False,
            )
            beta = cho_solve(factor, rhs, check_finite=False)
            return factor, np.asarray(beta, dtype=np.float64)
        except np.linalg.LinAlgError:
            gram.flat[:: gram.shape[0] + 1] += jitter
            jitter *= 10.0
    raise np.linalg.LinAlgError("Could not factor the baseline ridge system.")


def _exact_augmented_candidate_prediction(
    *,
    Z_train: np.ndarray,
    Z_tune: np.ndarray,
    candidate_train: np.ndarray,
    candidate_tune: np.ndarray,
    y_train: np.ndarray,
    baseline_statistics: RidgeSufficientStatistics,
    baseline_pair_design: MPCPairDesign | None,
    baseline_pair_statistics: MPCSufficientStatistics | None,
    baseline_factor: tuple[np.ndarray, bool],
    baseline_beta: np.ndarray,
    alpha: float,
    mpc_weight: float,
) -> np.ndarray:
    """Exact one-candidate refit using a Schur complement.

    The common baseline coefficients are reoptimized jointly with the candidate
    block at the baseline-selected alpha and direct-MPC weight. This is exactly
    the result from fitting [Z, C] but avoids refactorizing the large Z block for
    every candidate.
    """

    candidate_statistics = _compute_ridge_statistics(
        candidate_train,
        {"consumption": np.asarray(y_train, dtype=np.float64)},
    )
    cross_level = _centered_cross_moment(
        Z_train,
        candidate_train,
        left_mean=baseline_statistics.mean_z,
        right_mean=candidate_statistics.mean_z,
    )

    candidate_pair_statistics: MPCSufficientStatistics | None = None
    cross_mpc = np.zeros_like(cross_level)
    if baseline_pair_design is not None:
        candidate_difference = (
            candidate_train[baseline_pair_design.treated_positions]
            - candidate_train[baseline_pair_design.baseline_positions]
        ) / baseline_pair_design.checks[:, None]
        candidate_pair_design = MPCPairDesign(
            difference=candidate_difference.astype(np.float32, copy=False),
            target=baseline_pair_design.target,
            checks=baseline_pair_design.checks,
            model_ids=baseline_pair_design.model_ids,
            baseline_positions=baseline_pair_design.baseline_positions,
            treated_positions=baseline_pair_design.treated_positions,
            n_available=baseline_pair_design.n_available,
        )
        candidate_pair_statistics = _compute_mpc_statistics(candidate_pair_design)
        cross_mpc = _uncentered_cross_moment(
            baseline_pair_design.difference,
            candidate_difference,
        )

    B = cross_level.copy()
    C = np.asarray(candidate_statistics.gram, dtype=np.float64).copy()
    h = np.asarray(
        candidate_statistics.target_crosses["consumption"], dtype=np.float64
    ).copy()
    if float(mpc_weight) > 0.0:
        if baseline_pair_statistics is None or candidate_pair_statistics is None:
            raise ValueError("A positive MPC weight requires candidate pair moments.")
        B += float(mpc_weight) * cross_mpc
        C += float(mpc_weight) * np.asarray(
            candidate_pair_statistics.gram, dtype=np.float64
        )
        h += float(mpc_weight) * np.asarray(
            candidate_pair_statistics.cross, dtype=np.float64
        )
    C.flat[:: C.shape[0] + 1] += float(alpha)

    A_inv_B = cho_solve(
        baseline_factor,
        B,
        check_finite=False,
    )
    schur = C - B.T @ A_inv_B
    schur = 0.5 * (schur + schur.T)
    rhs = h - B.T @ baseline_beta
    jitter = max(1.0e-12, float(alpha) * 1.0e-10)
    delta: np.ndarray | None = None
    for _ in range(6):
        try:
            delta = np.linalg.solve(schur, rhs)
            break
        except np.linalg.LinAlgError:
            schur.flat[:: schur.shape[0] + 1] += jitter
            jitter *= 10.0
    if delta is None:
        raise np.linalg.LinAlgError("Could not solve the candidate Schur system.")

    beta = baseline_beta - A_inv_B @ delta
    intercept = float(
        baseline_statistics.target_means["consumption"]
        - baseline_statistics.mean_z @ beta
        - candidate_statistics.mean_z @ delta
    )
    return (
        np.asarray(Z_tune, dtype=np.float64) @ beta
        + np.asarray(candidate_tune, dtype=np.float64) @ delta
        + intercept
    )


def _consumption_screen_metrics(
    data: pd.DataFrame,
    prediction: np.ndarray,
    *,
    mpc_loss_weight: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    truth = pd.to_numeric(data["consumption"], errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(prediction, dtype=float)
    finite = np.isfinite(truth) & np.isfinite(pred)
    level_r2 = (
        float(r2_score(truth[finite], pred[finite]))
        if int(finite.sum()) > 1
        else np.nan
    )
    level_rmse = (
        float(np.sqrt(np.mean((pred[finite] - truth[finite]) ** 2)))
        if finite.any()
        else np.nan
    )
    level_nrmse = _relative_rmse(truth, pred)

    pairs = _paired_mpc_table(data, pred)
    mpc_r2 = np.nan
    mpc_rmse = np.nan
    mpc_nrmse = np.nan
    mpc_bias = np.nan
    n_pairs = 0
    combined_score = level_nrmse
    if not pairs.empty:
        exact = pairs["exact_mpc"].to_numpy(dtype=float)
        predicted = pairs["predicted_mpc"].to_numpy(dtype=float)
        ok = np.isfinite(exact) & np.isfinite(predicted)
        if ok.any():
            errors = predicted[ok] - exact[ok]
            mpc_rmse = float(np.sqrt(np.mean(errors**2)))
            mpc_sd = float(np.std(exact[ok]))
            mpc_nrmse = mpc_rmse / max(mpc_sd, EPS)
            mpc_bias = float(np.mean(errors))
            mpc_r2 = (
                float(r2_score(exact[ok], predicted[ok]))
                if int(ok.sum()) > 1
                else np.nan
            )
            n_pairs = int(ok.sum())
            combined_score = (
                level_nrmse
                + float(mpc_loss_weight) * mpc_nrmse
                + 0.10 * abs(mpc_bias) / max(mpc_sd, EPS)
            )
    return (
        {
            "level_r2": level_r2,
            "level_rmse": level_rmse,
            "level_nrmse": level_nrmse,
            "mpc_r2": mpc_r2,
            "mpc_rmse": mpc_rmse,
            "mpc_nrmse": mpc_nrmse,
            "mpc_bias": mpc_bias,
            "combined_score": float(combined_score),
            "n_mpc_pairs": int(n_pairs),
        },
        pairs,
    )


def _model_level_squared_error_improvements(
    *,
    model_ids: np.ndarray,
    truth: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "model_id": np.asarray(model_ids, dtype=object).astype(str),
            "truth": np.asarray(truth, dtype=float),
            "before": np.asarray(before, dtype=float),
            "after": np.asarray(after, dtype=float),
        }
    )
    finite = np.isfinite(frame[["truth", "before", "after"]]).all(axis=1)
    frame = frame.loc[finite]
    if frame.empty:
        return np.empty(0, dtype=float)
    frame["before_sq"] = (frame["before"] - frame["truth"]) ** 2
    frame["after_sq"] = (frame["after"] - frame["truth"]) ** 2
    grouped = frame.groupby("model_id", observed=True).agg(
        before_mse=("before_sq", "mean"),
        after_mse=("after_sq", "mean"),
    )
    return (grouped["before_mse"] - grouped["after_mse"]).to_numpy(dtype=float)


def _bootstrap_mean_improvement(
    values: np.ndarray,
    *,
    reps: int,
    confidence: float,
    random_state: int,
) -> tuple[float, float, float]:
    """One-sided model-level significance and a bootstrap lower bound."""

    improvements = np.asarray(values, dtype=float)
    improvements = improvements[np.isfinite(improvements)]
    if len(improvements) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(improvements))
    win_share = float(np.mean(improvements > 0.0))
    if len(improvements) > 1:
        standard_error = float(
            np.std(improvements, ddof=1) / np.sqrt(len(improvements))
        )
        if standard_error > 0.0 and np.isfinite(standard_error):
            statistic = mean / standard_error
            p_value = float(student_t.sf(statistic, df=len(improvements) - 1))
        else:
            p_value = 0.0 if mean > 0.0 else 1.0
    else:
        p_value = 0.0 if mean > 0.0 else 1.0

    if int(reps) <= 0 or len(improvements) == 1:
        return mean, p_value, win_share
    rng = np.random.default_rng(int(random_state))
    positions = rng.integers(
        0,
        len(improvements),
        size=(int(reps), len(improvements)),
    )
    bootstrap_means = improvements[positions].mean(axis=1)
    lower = float(np.quantile(bootstrap_means, 1.0 - float(confidence)))
    return lower, p_value, win_share


def _benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if len(finite_positions) == 0:
        return out
    finite_values = values[finite_positions]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    inverse = np.empty(m, dtype=np.int64)
    inverse[order] = np.arange(m)
    out[finite_positions] = adjusted[inverse]
    return out


def _screen_higher_order_parameter_terms_once(
    train: pd.DataFrame,
    tune: pd.DataFrame,
    *,
    spec: FeatureSpec,
    polynomial_degree: int,
    spline_columns: Sequence[str],
    spline_n_knots: int,
    spline_column_n_knots: Sequence[tuple[str, int]],
    spline_knot_strategies: Sequence[tuple[str, str]],
    spline_knot_powers: Sequence[tuple[str, float]],
    spline_tensor_pairs: Sequence[tuple[str, str]],
    spline_tensor_triples: Sequence[tuple[str, str, str]],
    spline_categorical_tensor_pairs: Sequence[tuple[str, str, str]],
    spline_tensor_parameter_terms: Sequence[tuple[str, str, str]],
    mandatory_parameter_power_terms: Sequence[tuple[str, int]],
    mandatory_spline_parameter_power_terms: Sequence[tuple[str, str, int]],
    mandatory_spline_parameter_pair_terms: Sequence[tuple[str, str, str]],
    categorical_state_power_terms: Sequence[tuple[str, str, int]],
    categorical_slope_exclude_patterns: Sequence[str],
    spline_categorical_exclude_patterns: Sequence[str],
    ridge_alphas: Sequence[float],
    candidate_mpc_weights: Sequence[float],
    mpc_loss_weight: float,
    mpc_objective_max_pairs: int | None,
    mpc_objective_trim_quantile: float,
    mpc_stratify_age_bins: Sequence[float],
    mpc_stratify_liquid_bins: int,
    mpc_stratify_illiquid_bins: int,
    mpc_stratify_target_bins: int,
    selection_spline_columns: Sequence[str],
    selection_powers: Sequence[int],
    parameter_include_patterns: Sequence[str],
    parameter_exclude_patterns: Sequence[str],
    include_parameter_powers: bool,
    max_train_rows: int | None,
    max_tune_rows: int | None,
    min_mpc_r2_gain: float,
    min_level_r2_gain: float,
    max_mpc_r2_loss_for_level_term: float,
    min_model_win_share: float,
    fdr: float,
    bootstrap_reps: int,
    bootstrap_confidence: float,
    max_selected_terms: int,
    random_state: int,
    verbose: bool,
    show_progress: bool,
) -> ParameterTermSelectionResult:
    """Screen higher-order parameter terms on tuning models only.

    Each candidate is added to the common baseline design and fitted exactly at
    the baseline-selected ridge penalty and direct-MPC weight. A Schur-complement
    solve reoptimizes the common baseline coefficients without rebuilding or
    refactorizing the large common feature block for every candidate. The final
    selected terms are then jointly retuned and refitted in the ordinary full
    surrogate.
    """

    screen_train = _model_subsample_to_row_cap(
        train,
        max_rows=max_train_rows,
        random_state=random_state + 101,
    )
    screen_tune = _model_subsample_to_row_cap(
        tune,
        max_rows=max_tune_rows,
        random_state=random_state + 102,
    )
    _status(
        "[parameter search] screening higher-order terms on "
        f"{len(screen_train):,} training rows / "
        f"{len(screen_tune):,} tuning rows...",
        verbose=verbose,
    )

    baseline_map = StructuredPolynomialMap(
        spec,
        degree=int(polynomial_degree),
        state_state_interactions=True,
        state_parameter_interactions=True,
        parameter_parameter_interactions=False,
        categorical_slopes=True,
        categorical_slope_exclude_patterns=tuple(categorical_slope_exclude_patterns),
        spline_columns=tuple(spline_columns),
        spline_n_knots=int(spline_n_knots),
        spline_column_n_knots=tuple(spline_column_n_knots),
        spline_knot_strategies=tuple(spline_knot_strategies),
        spline_knot_powers=tuple(spline_knot_powers),
        spline_degree=3,
        spline_parameter_interactions=True,
        spline_categorical_interactions=True,
        spline_categorical_exclude_patterns=tuple(spline_categorical_exclude_patterns),
        spline_tensor_pairs=tuple(spline_tensor_pairs),
        spline_tensor_triples=tuple(spline_tensor_triples),
        spline_categorical_tensor_pairs=tuple(spline_categorical_tensor_pairs),
        spline_tensor_parameter_terms=tuple(spline_tensor_parameter_terms),
        parameter_power_terms=tuple(mandatory_parameter_power_terms),
        spline_parameter_power_terms=tuple(mandatory_spline_parameter_power_terms),
        spline_parameter_pair_terms=tuple(mandatory_spline_parameter_pair_terms),
        categorical_state_power_terms=tuple(categorical_state_power_terms),
        random_state=int(random_state),
    )
    Z_train = baseline_map.fit_transform(screen_train)
    Z_tune = baseline_map.transform(screen_tune)
    y_train = pd.to_numeric(screen_train["consumption"], errors="coerce").to_numpy(
        dtype=float
    )

    train_pair_design = _build_mpc_pair_design(
        screen_train,
        Z_train,
        max_pairs=mpc_objective_max_pairs,
        random_state=random_state + 111,
        trim_quantile=mpc_objective_trim_quantile,
        age_bins=mpc_stratify_age_bins,
        liquid_bins=mpc_stratify_liquid_bins,
        illiquid_bins=mpc_stratify_illiquid_bins,
        target_bins=mpc_stratify_target_bins,
    )
    baseline_statistics = _compute_ridge_statistics(
        Z_train,
        {"consumption": y_train},
    )
    baseline_pair_statistics = _compute_mpc_statistics(train_pair_design)

    best_baseline_model: Any | None = None
    best_baseline_metrics: dict[str, float] | None = None
    best_baseline_pairs = pd.DataFrame()
    best_baseline_score = np.inf
    best_baseline_alpha = float(ridge_alphas[0])
    best_baseline_weight = 0.0
    for alpha in ridge_alphas:
        for weight in candidate_mpc_weights:
            if float(weight) > 0.0 and baseline_pair_statistics is None:
                continue
            model = _fit_regressor_from_statistics(
                baseline_statistics,
                target="consumption",
                alpha=float(alpha),
                mpc_weight=float(weight),
                pair_statistics=baseline_pair_statistics,
            )
            tune_prediction = np.asarray(model.predict(Z_tune), dtype=float)
            metrics, pairs = _consumption_screen_metrics(
                screen_tune,
                tune_prediction,
                mpc_loss_weight=mpc_loss_weight,
            )
            if metrics["combined_score"] < best_baseline_score:
                best_baseline_score = float(metrics["combined_score"])
                best_baseline_model = model
                best_baseline_metrics = metrics
                best_baseline_pairs = pairs
                best_baseline_alpha = float(alpha)
                best_baseline_weight = float(weight)
    if best_baseline_model is None or best_baseline_metrics is None:
        raise RuntimeError("Could not fit the parameter-search baseline model.")

    baseline_tune_prediction = np.asarray(
        best_baseline_model.predict(Z_tune), dtype=float
    )
    baseline_factor, baseline_beta = _ridge_system_factor(
        baseline_statistics,
        target="consumption",
        alpha=best_baseline_alpha,
        mpc_weight=best_baseline_weight,
        pair_statistics=baseline_pair_statistics,
    )

    train_parameters, train_splines = _parameter_search_components(
        baseline_map, screen_train
    )
    tune_parameters, tune_splines = _parameter_search_components(
        baseline_map, screen_tune
    )
    parameter_lookup = {
        name: i for i, name in enumerate(baseline_map.spec.parameter_columns)
    }
    candidates = _generate_parameter_term_candidates(
        screen_train,
        baseline_map,
        polynomial_degree=polynomial_degree,
        selection_spline_columns=selection_spline_columns,
        powers=selection_powers,
        include_parameter_powers=include_parameter_powers,
        parameter_include_patterns=parameter_include_patterns,
        parameter_exclude_patterns=parameter_exclude_patterns,
        mandatory_parameter_power_terms=mandatory_parameter_power_terms,
        mandatory_spline_parameter_power_terms=mandatory_spline_parameter_power_terms,
        mandatory_spline_parameter_pair_terms=mandatory_spline_parameter_pair_terms,
    )
    _status(
        f"  baseline tuning MPC R2={best_baseline_metrics['mpc_r2']:.4f}; "
        f"consumption R2={best_baseline_metrics['level_r2']:.4f}; "
        f"screening {len(candidates):,} candidates...",
        verbose=verbose,
    )

    rows: list[dict[str, Any]] = []
    iterator = _progress_iter(
        candidates,
        total=len(candidates),
        description="Higher-order parameter search",
        unit="term",
        enabled=show_progress,
    )
    for candidate_position, candidate in enumerate(iterator):
        train_raw = _candidate_raw_block(
            candidate,
            parameters=train_parameters,
            parameter_lookup=parameter_lookup,
            splines=train_splines,
            spline_slices=baseline_map.spline_slices_,
        )
        tune_raw = _candidate_raw_block(
            candidate,
            parameters=tune_parameters,
            parameter_lookup=parameter_lookup,
            splines=tune_splines,
            spline_slices=baseline_map.spline_slices_,
        )
        standardized = _standardize_candidate_blocks(train_raw, tune_raw)
        row: dict[str, Any] = candidate.metadata()
        row.update(
            {
                "candidate_index": int(candidate_position),
                "n_features": 0,
                "baseline_alpha": best_baseline_alpha,
                "baseline_mpc_weight": best_baseline_weight,
                "baseline_consumption_r2": best_baseline_metrics["level_r2"],
                "baseline_mpc_r2": best_baseline_metrics["mpc_r2"],
                "baseline_score": best_baseline_metrics["combined_score"],
                "selected": False,
                "screen_status": "ok",
            }
        )
        if standardized is None:
            row["screen_status"] = "constant_or_nonfinite"
            rows.append(row)
            continue
        candidate_train, candidate_tune = standardized
        row["n_features"] = int(candidate_train.shape[1])

        try:
            best_prediction = _exact_augmented_candidate_prediction(
                Z_train=Z_train,
                Z_tune=Z_tune,
                candidate_train=candidate_train,
                candidate_tune=candidate_tune,
                y_train=y_train,
                baseline_statistics=baseline_statistics,
                baseline_pair_design=train_pair_design,
                baseline_pair_statistics=baseline_pair_statistics,
                baseline_factor=baseline_factor,
                baseline_beta=baseline_beta,
                alpha=best_baseline_alpha,
                mpc_weight=best_baseline_weight,
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            row["screen_status"] = f"fit_failed:{type(exc).__name__}"
            rows.append(row)
            continue
        best_metrics, best_pairs = _consumption_screen_metrics(
            screen_tune,
            best_prediction,
            mpc_loss_weight=mpc_loss_weight,
        )
        best_alpha = best_baseline_alpha

        delta_level_r2 = float(
            best_metrics["level_r2"] - best_baseline_metrics["level_r2"]
        )
        delta_mpc_r2 = float(best_metrics["mpc_r2"] - best_baseline_metrics["mpc_r2"])
        score_gain = float(
            best_baseline_metrics["combined_score"] - best_metrics["combined_score"]
        )

        truth_level = pd.to_numeric(
            screen_tune["consumption"], errors="coerce"
        ).to_numpy(dtype=float)
        level_improvements = _model_level_squared_error_improvements(
            model_ids=screen_tune["model_id"].astype(str).to_numpy(dtype=object),
            truth=truth_level,
            before=baseline_tune_prediction,
            after=best_prediction,
        )
        level_lower, level_p, level_win_share = _bootstrap_mean_improvement(
            level_improvements,
            reps=bootstrap_reps,
            confidence=bootstrap_confidence,
            random_state=random_state + 2000 + candidate_position,
        )

        mpc_lower = np.nan
        mpc_p = np.nan
        mpc_win_share = np.nan
        if not best_baseline_pairs.empty and not best_pairs.empty:
            common_keys = ["model_id", "pair_id"]
            before_pairs = best_baseline_pairs[
                common_keys + ["exact_mpc", "predicted_mpc"]
            ].rename(columns={"predicted_mpc": "before_mpc"})
            after_pairs = best_pairs[common_keys + ["predicted_mpc"]].rename(
                columns={"predicted_mpc": "after_mpc"}
            )
            comparison = before_pairs.merge(
                after_pairs,
                on=common_keys,
                how="inner",
                validate="one_to_one",
            )
            mpc_improvements = _model_level_squared_error_improvements(
                model_ids=comparison["model_id"].to_numpy(dtype=object),
                truth=comparison["exact_mpc"].to_numpy(dtype=float),
                before=comparison["before_mpc"].to_numpy(dtype=float),
                after=comparison["after_mpc"].to_numpy(dtype=float),
            )
            mpc_lower, mpc_p, mpc_win_share = _bootstrap_mean_improvement(
                mpc_improvements,
                reps=bootstrap_reps,
                confidence=bootstrap_confidence,
                random_state=random_state + 4000 + candidate_position,
            )

        row.update(
            {
                "candidate_alpha": best_alpha,
                "consumption_r2": best_metrics["level_r2"],
                "mpc_r2": best_metrics["mpc_r2"],
                "consumption_r2_gain": delta_level_r2,
                "mpc_r2_gain": delta_mpc_r2,
                "score": best_metrics["combined_score"],
                "score_gain": score_gain,
                "mpc_rmse": best_metrics["mpc_rmse"],
                "mpc_bias": best_metrics["mpc_bias"],
                "level_mse_improvement_lower": level_lower,
                "level_p_value": level_p,
                "level_model_win_share": level_win_share,
                "mpc_mse_improvement_lower": mpc_lower,
                "mpc_p_value": mpc_p,
                "mpc_model_win_share": mpc_win_share,
            }
        )
        rows.append(row)

    if not rows:
        return ParameterTermSelectionResult(
            parameter_power_terms=tuple(mandatory_parameter_power_terms),
            spline_parameter_power_terms=tuple(mandatory_spline_parameter_power_terms),
            spline_parameter_pair_terms=tuple(mandatory_spline_parameter_pair_terms),
            search_rows=(),
            baseline_metrics=best_baseline_metrics,
            n_candidates=0,
            n_selected=0,
        )

    search = pd.DataFrame(rows)
    search["mpc_q_value"] = _benjamini_hochberg(
        pd.to_numeric(search.get("mpc_p_value"), errors="coerce")
    )
    search["level_q_value"] = _benjamini_hochberg(
        pd.to_numeric(search.get("level_p_value"), errors="coerce")
    )
    is_level_only = search["kind"].eq("parameter_power")
    apply_fdr = float(fdr) < 1.0
    mpc_fdr_ok = (
        pd.to_numeric(search["mpc_q_value"], errors="coerce").le(float(fdr))
        if apply_fdr
        else pd.Series(True, index=search.index)
    )
    level_fdr_ok = (
        pd.to_numeric(search["level_q_value"], errors="coerce").le(float(fdr))
        if apply_fdr
        else pd.Series(True, index=search.index)
    )
    mpc_significant = (
        ~is_level_only
        & pd.to_numeric(search["mpc_r2_gain"], errors="coerce").ge(
            float(min_mpc_r2_gain)
        )
        & pd.to_numeric(search["mpc_mse_improvement_lower"], errors="coerce").gt(0.0)
        & mpc_fdr_ok
        & pd.to_numeric(search["mpc_model_win_share"], errors="coerce").ge(
            float(min_model_win_share)
        )
        & pd.to_numeric(search["score_gain"], errors="coerce").gt(0.0)
    )
    level_significant = (
        is_level_only
        & pd.to_numeric(search["consumption_r2_gain"], errors="coerce").ge(
            float(min_level_r2_gain)
        )
        & pd.to_numeric(search["level_mse_improvement_lower"], errors="coerce").gt(0.0)
        & level_fdr_ok
        & pd.to_numeric(search["level_model_win_share"], errors="coerce").ge(
            float(min_model_win_share)
        )
        & pd.to_numeric(search["mpc_r2_gain"], errors="coerce").ge(
            -float(max_mpc_r2_loss_for_level_term)
        )
        & pd.to_numeric(search["score_gain"], errors="coerce").gt(0.0)
    )
    search["passes_significance"] = mpc_significant | level_significant
    search["selection_priority"] = np.where(
        is_level_only,
        pd.to_numeric(search["consumption_r2_gain"], errors="coerce"),
        pd.to_numeric(search["mpc_r2_gain"], errors="coerce"),
    )
    eligible = search[search["passes_significance"]].sort_values(
        ["selection_priority", "score_gain", "candidate"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if int(max_selected_terms) > 0:
        eligible = eligible.head(int(max_selected_terms))
    selected_labels = set(eligible["candidate"].astype(str))
    search["selected"] = search["candidate"].astype(str).isin(selected_labels)

    selected_parameter_powers = list(mandatory_parameter_power_terms)
    selected_spline_powers = list(mandatory_spline_parameter_power_terms)
    selected_spline_pairs = list(mandatory_spline_parameter_pair_terms)
    candidate_by_label = {candidate.label: candidate for candidate in candidates}
    for label in eligible["candidate"].astype(str):
        candidate = candidate_by_label[label]
        if candidate.kind == "parameter_power":
            selected_parameter_powers.append(
                (candidate.parameters[0], int(candidate.power or 0))
            )
        elif candidate.kind == "spline_parameter_power":
            selected_spline_powers.append(
                (
                    str(candidate.spline),
                    candidate.parameters[0],
                    int(candidate.power or 0),
                )
            )
        elif candidate.kind == "spline_parameter_pair":
            selected_spline_pairs.append(
                (
                    str(candidate.spline),
                    candidate.parameters[0],
                    candidate.parameters[1],
                )
            )

    search = search.sort_values(
        ["selected", "selection_priority", "candidate"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    _status(
        f"  selected {len(eligible):,} of {len(candidates):,} higher-order "
        "parameter terms for the full fit.",
        verbose=verbose,
    )
    if verbose and len(eligible):
        for row in eligible.itertuples(index=False):
            _status(
                f"    {row.candidate}: MPC R2 gain={row.mpc_r2_gain:+.4f}; "
                f"consumption R2 gain={row.consumption_r2_gain:+.4f}; "
                f"q={row.mpc_q_value if row.kind != 'parameter_power' else row.level_q_value:.4g}",
                verbose=verbose,
            )

    return ParameterTermSelectionResult(
        parameter_power_terms=_dedupe_tuples(selected_parameter_powers),
        spline_parameter_power_terms=_dedupe_tuples(selected_spline_powers),
        spline_parameter_pair_terms=_dedupe_tuples(selected_spline_pairs),
        search_rows=tuple(search.to_dict(orient="records")),
        baseline_metrics={
            **best_baseline_metrics,
            "alpha": best_baseline_alpha,
            "mpc_weight": best_baseline_weight,
            "screen_train_rows": int(len(screen_train)),
            "screen_tune_rows": int(len(screen_tune)),
            "screen_train_models": int(screen_train["model_id"].nunique()),
            "screen_tune_models": int(screen_tune["model_id"].nunique()),
        },
        n_candidates=int(len(candidates)),
        n_selected=int(len(eligible)),
    )


def _select_higher_order_parameter_terms(
    train: pd.DataFrame,
    tune: pd.DataFrame,
    **kwargs: Any,
) -> ParameterTermSelectionResult:
    """Greedy forward selection of higher-order parameter terms.

    The one-term screening routine is rerun after every accepted term, so each
    remaining candidate is evaluated conditional on the crucial terms and all
    terms selected in earlier rounds. This avoids the old redundancy problem in
    which many correlated candidates passed separate one-at-a-time screens and
    were then included jointly without checking their incremental contribution.
    """

    requested_limit = int(kwargs.get("max_selected_terms", 0))
    # Zero means genuinely uncapped forward selection. The loop is still finite:
    # every successful round adds a previously absent candidate, and it stops as
    # soon as no remaining candidate passes the tuning criteria.
    max_rounds: int | None = requested_limit if requested_limit > 0 else None

    current_parameter_powers = _dedupe_tuples(
        kwargs.get("mandatory_parameter_power_terms", ())
    )
    current_spline_powers = _dedupe_tuples(
        kwargs.get("mandatory_spline_parameter_power_terms", ())
    )
    current_spline_pairs = _dedupe_tuples(
        kwargs.get("mandatory_spline_parameter_pair_terms", ())
    )

    all_rows: list[dict[str, Any]] = []
    first_baseline_metrics: dict[str, float] = {}
    total_candidates_screened = 0
    selected_count = 0

    round_index = 0
    while max_rounds is None or round_index < max_rounds:
        round_index += 1
        round_kwargs = dict(kwargs)
        round_kwargs["mandatory_parameter_power_terms"] = current_parameter_powers
        round_kwargs["mandatory_spline_parameter_power_terms"] = current_spline_powers
        round_kwargs["mandatory_spline_parameter_pair_terms"] = current_spline_pairs
        round_kwargs["max_selected_terms"] = 1
        # Keep the same model-preserving screen samples in every round so
        # incremental gains are comparable as the baseline expands.
        round_kwargs["random_state"] = int(kwargs.get("random_state", 123))

        result = _screen_higher_order_parameter_terms_once(
            train,
            tune,
            **round_kwargs,
        )
        if round_index == 1:
            first_baseline_metrics = dict(result.baseline_metrics)
        total_candidates_screened += int(result.n_candidates)

        round_rows = [dict(row) for row in result.search_rows]
        for row in round_rows:
            row["forward_round"] = round_index
            row["selected_in_round"] = bool(row.get("selected", False))
        all_rows.extend(round_rows)

        new_parameter_powers = tuple(
            term
            for term in result.parameter_power_terms
            if term not in current_parameter_powers
        )
        new_spline_powers = tuple(
            term
            for term in result.spline_parameter_power_terms
            if term not in current_spline_powers
        )
        new_spline_pairs = tuple(
            term
            for term in result.spline_parameter_pair_terms
            if term not in current_spline_pairs
        )
        n_new = (
            len(new_parameter_powers) + len(new_spline_powers) + len(new_spline_pairs)
        )
        if n_new == 0:
            break

        current_parameter_powers = _dedupe_tuples(
            current_parameter_powers + new_parameter_powers
        )
        current_spline_powers = _dedupe_tuples(
            current_spline_powers + new_spline_powers
        )
        current_spline_pairs = _dedupe_tuples(current_spline_pairs + new_spline_pairs)
        selected_count += n_new

        limit_text = "unlimited" if max_rounds is None else f"{max_rounds:,}"
        _status(
            f"[parameter search] accepted forward term {selected_count:,}/"
            f"{limit_text}; rebuilding the baseline before screening the "
            "remaining candidates.",
            verbose=bool(kwargs.get("verbose", True)),
        )

    return ParameterTermSelectionResult(
        parameter_power_terms=current_parameter_powers,
        spline_parameter_power_terms=current_spline_powers,
        spline_parameter_pair_terms=current_spline_pairs,
        search_rows=tuple(all_rows),
        baseline_metrics={
            **first_baseline_metrics,
            "selection_method": "greedy_forward",
            "forward_terms_selected": selected_count,
            "forward_term_limit": max_rounds,
        },
        n_candidates=total_candidates_screened,
        n_selected=selected_count,
    )


def _load_or_create_model_split(
    fit_data: pd.DataFrame,
    *,
    categorical_parameters: Sequence[str],
    validation_share: float,
    tuning_share: float,
    random_state: int,
    split_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create or reuse a deterministic model-level train/tune/test split.

    When a split file already exists, its tuning and test model IDs are frozen.
    Newly added solved models are assigned to training, while missing frozen IDs
    raise an error. This gives stable holdout comparisons across retraining runs.
    """

    available_ids = set(fit_data["model_id"].astype(str).unique().tolist())
    path = None if split_path is None else Path(split_path)

    if path is not None and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        train_ids = set(map(str, payload.get("training_models", [])))
        tune_ids = set(map(str, payload.get("tuning_models", [])))
        test_ids = set(map(str, payload.get("test_models", [])))
        frozen_ids = train_ids | tune_ids | test_ids
        missing = sorted((tune_ids | test_ids) - available_ids)
        if missing:
            raise ValueError(
                f"Frozen split {path} contains tuning/test models that are no "
                "longer available:\n" + "\n".join(missing)
            )
        if (train_ids & tune_ids) or (train_ids & test_ids) or (tune_ids & test_ids):
            raise ValueError(f"Frozen split {path} contains overlapping model sets.")
        # Preserve frozen tuning/test sets. Any newly solved models augment the
        # training set rather than changing the evaluation sample.
        train_ids = (train_ids & available_ids) | (available_ids - frozen_ids)
    else:
        if categorical_parameters:
            model_table = (
                fit_data[["model_id", *categorical_parameters]]
                .drop_duplicates(subset=["model_id"])
                .sort_values("model_id", kind="mergesort")
                .reset_index(drop=True)
            )
            development_ids, test_ids = _stratified_model_holdout(
                model_table,
                share=validation_share,
                category_columns=categorical_parameters,
                random_state=random_state,
                label="test",
            )
            development_table = model_table[
                model_table["model_id"].astype(str).isin(development_ids)
            ].copy()
            train_ids, tune_ids = _stratified_model_holdout(
                development_table,
                share=tuning_share,
                category_columns=categorical_parameters,
                random_state=random_state + 1,
                label="tuning",
            )
        else:
            model_ids = np.asarray(sorted(available_ids), dtype=object)
            rng = np.random.default_rng(int(random_state))
            rng.shuffle(model_ids)
            n_models = len(model_ids)
            n_test = min(max(1, int(round(n_models * validation_share))), n_models - 2)
            test_ids = set(map(str, model_ids[:n_test]))
            development_ids = set(map(str, model_ids[n_test:]))
            development_array = np.asarray(sorted(development_ids), dtype=object)
            rng_tune = np.random.default_rng(int(random_state) + 1)
            rng_tune.shuffle(development_array)
            n_tune = min(
                max(1, int(round(len(development_array) * tuning_share))),
                len(development_array) - 1,
            )
            tune_ids = set(map(str, development_array[:n_tune]))
            train_ids = set(map(str, development_array[n_tune:]))

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "random_state": int(random_state),
                        "validation_share": float(validation_share),
                        "tuning_share_of_development": float(tuning_share),
                        "training_models": sorted(train_ids),
                        "tuning_models": sorted(tune_ids),
                        "test_models": sorted(test_ids),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

    if not train_ids or not tune_ids or not test_ids:
        raise ValueError(
            "Model split must contain nonempty train, tuning, and test sets."
        )

    train = fit_data[fit_data["model_id"].astype(str).isin(train_ids)].copy()
    tune = fit_data[fit_data["model_id"].astype(str).isin(tune_ids)].copy()
    test = fit_data[fit_data["model_id"].astype(str).isin(test_ids)].copy()
    if train.empty or tune.empty or test.empty:
        raise ValueError("Frozen model split produced an empty row-level partition.")
    return (
        train.reset_index(drop=True),
        tune.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def fit_policy_surrogate(
    data: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    categorical_parameter_columns: Sequence[str] = (),
    state_columns: Sequence[str] = DEFAULT_STATE_COLUMNS,
    engineered_state_columns: Sequence[str] = DEFAULT_ENGINEERED_STATE_COLUMNS,
    redundant_state_columns: Sequence[str] = DEFAULT_REDUNDANT_STATE_COLUMNS,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_STATE_COLUMNS,
    model_type: str = "polynomial",
    polynomial_degree: int = 2,
    spline_columns: Sequence[str] = DEFAULT_SPLINE_COLUMNS,
    spline_n_knots: int = 8,
    spline_column_n_knots: Sequence[tuple[str, int]] = DEFAULT_SPLINE_COLUMN_N_KNOTS,
    spline_knot_strategies: Sequence[tuple[str, str]] = DEFAULT_SPLINE_KNOT_STRATEGIES,
    spline_knot_powers: Sequence[tuple[str, float]] = DEFAULT_SPLINE_KNOT_POWERS,
    spline_tensor_pairs: Sequence[tuple[str, str]] = DEFAULT_SPLINE_TENSOR_PAIRS,
    spline_tensor_triples: Sequence[
        tuple[str, str, str]
    ] = DEFAULT_SPLINE_TENSOR_TRIPLES,
    spline_categorical_tensor_pairs: Sequence[
        tuple[str, str, str]
    ] = DEFAULT_SPLINE_CATEGORICAL_TENSOR_PAIRS,
    spline_tensor_parameter_terms: Sequence[
        tuple[str, str, str]
    ] = DEFAULT_SPLINE_TENSOR_PARAMETER_TERMS,
    parameter_power_terms: Sequence[tuple[str, int]] = DEFAULT_PARAMETER_POWER_TERMS,
    spline_parameter_power_terms: Sequence[
        tuple[str, str, int]
    ] = DEFAULT_SPLINE_PARAMETER_POWER_TERMS,
    spline_parameter_pair_terms: Sequence[
        tuple[str, str, str]
    ] = DEFAULT_SPLINE_PARAMETER_PAIR_TERMS,
    auto_select_parameter_terms: bool = True,
    parameter_selection_spline_columns: Sequence[
        str
    ] = DEFAULT_PARAMETER_SELECTION_SPLINE_COLUMNS,
    parameter_selection_powers: Sequence[int] = DEFAULT_PARAMETER_SELECTION_POWERS,
    parameter_selection_parameter_patterns: Sequence[str] = (),
    parameter_selection_exclude_patterns: Sequence[str] = (),
    parameter_selection_include_parameter_powers: bool = True,
    parameter_selection_max_train_rows: int | None = 300_000,
    parameter_selection_max_tune_rows: int | None = 150_000,
    parameter_selection_min_mpc_r2_gain: float = 0.005,
    parameter_selection_min_level_r2_gain: float = 0.001,
    parameter_selection_max_mpc_r2_loss_for_level_term: float = 0.001,
    parameter_selection_min_model_win_share: float = 0.55,
    parameter_selection_fdr: float = 0.10,
    parameter_selection_bootstrap_reps: int = 200,
    parameter_selection_bootstrap_confidence: float = 0.95,
    parameter_selection_max_terms: int = 0,
    categorical_state_power_terms: Sequence[
        tuple[str, str, int]
    ] = DEFAULT_CATEGORICAL_STATE_POWER_TERMS,
    categorical_slope_exclude_patterns: Sequence[
        str
    ] = DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS,
    spline_categorical_exclude_patterns: Sequence[
        str
    ] = DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS,
    rff_components: int = 512,
    ridge_alphas: Sequence[float] = (
        1.0e-4,
        1.0e-3,
        1.0e-2,
        0.1,
        1.0,
        10.0,
        100.0,
    ),
    target_parameterization: str = "accounting",
    consumption_transform: str = "identity",
    deposit_transform: str = "asinh",
    liquid_outflow_transform: str = "identity",
    mpc_loss_weight: float = 2.0,
    mpc_difference_weight: float | None = None,
    mpc_difference_weight_grid: Sequence[float] = (
        0.0,
        0.25,
        0.50,
        1.0,
        2.0,
        4.0,
        8.0,
    ),
    mpc_objective_max_pairs: int | None = 500_000,
    mpc_objective_max_iter: int = 200,
    mpc_objective_tol: float = 1.0e-6,
    mpc_objective_trim_quantile: float = 0.0,
    mpc_stratify_age_bins: Sequence[float] = (35.0, 45.0, 55.0, 60.0),
    mpc_stratify_liquid_bins: int = 4,
    mpc_stratify_illiquid_bins: int = 10,
    mpc_stratify_target_bins: int = 5,
    liquid_change_loss_weight: float = 1.0,
    deposit_reconciliation_loss_weight: float = 0.50,
    liquid_outflow_weight: float | None = None,
    money_units: str = "model",
    working_age_only: bool = True,
    validation_share: float = 0.20,
    tuning_share: float = 0.20,
    model_split_path: str | Path | None = None,
    evaluation_bundle_path: str | Path | None = None,
    random_state: int = 123,
    verbose: bool = True,
    show_progress: bool = True,
) -> PolicySurrogateBundle:
    """Fit and validate a smooth, accounting-restricted policy surrogate.

    Consumption is estimated from a joint objective containing ordinary
    consumption-level errors and exact finite-difference MPC errors.  The MPC
    weight is either fixed by ``mpc_difference_weight`` or selected on tuning
    models from ``mpc_difference_weight_grid``.  Deposit and total liquid
    outflow retain ordinary ridge fits, and all reported asset policies remain
    accounting-consistent.
    """

    target_parameterization = _normalize_target_parameterization(
        target_parameterization
    )
    valid_transforms = {"identity", "asinh", "log1p"}
    for name, value in {
        "consumption_transform": consumption_transform,
        "deposit_transform": deposit_transform,
        "liquid_outflow_transform": liquid_outflow_transform,
    }.items():
        if value not in valid_transforms:
            raise ValueError(f"Unsupported {name}={value!r}.")

    if mpc_loss_weight < 0.0:
        raise ValueError("mpc_loss_weight must be nonnegative.")
    if liquid_change_loss_weight < 0.0 or deposit_reconciliation_loss_weight < 0.0:
        raise ValueError("Reconciliation loss weights must be nonnegative.")
    if mpc_difference_weight is not None and float(mpc_difference_weight) < 0.0:
        raise ValueError("mpc_difference_weight must be nonnegative.")
    candidate_mpc_weights = (
        [float(mpc_difference_weight)]
        if mpc_difference_weight is not None
        else sorted({float(x) for x in mpc_difference_weight_grid})
    )
    if not candidate_mpc_weights or any(x < 0.0 for x in candidate_mpc_weights):
        raise ValueError("MPC difference weights must be a nonempty nonnegative grid.")
    if consumption_transform != "identity" and any(
        x > 0.0 for x in candidate_mpc_weights
    ):
        raise ValueError(
            "Direct MPC targeting requires --consumption-transform identity."
        )
    if mpc_objective_max_pairs is not None and int(mpc_objective_max_pairs) <= 0:
        raise ValueError("mpc_objective_max_pairs must be positive or None.")
    if int(mpc_objective_max_iter) <= 0 or float(mpc_objective_tol) <= 0.0:
        raise ValueError("MPC objective solver controls must be positive.")
    if not 0.0 <= float(mpc_objective_trim_quantile) < 0.5:
        raise ValueError("mpc_objective_trim_quantile must be in [0, 0.5).")
    if int(mpc_stratify_liquid_bins) < 1:
        raise ValueError("mpc_stratify_liquid_bins must be at least one.")
    if int(mpc_stratify_illiquid_bins) < 1:
        raise ValueError("mpc_stratify_illiquid_bins must be at least one.")
    if int(mpc_stratify_target_bins) < 0:
        raise ValueError("mpc_stratify_target_bins must be nonnegative.")
    if bool(auto_select_parameter_terms):
        if model_type != "polynomial":
            raise ValueError(
                "Automatic higher-order parameter selection is available only "
                "for model_type='polynomial'."
            )
        if consumption_transform != "identity":
            raise ValueError(
                "Automatic parameter-term selection currently requires "
                "consumption_transform='identity' so tuning MPCs are exact "
                "finite differences in consumption levels."
            )
        if not 0.0 <= float(parameter_selection_fdr) <= 1.0:
            raise ValueError("parameter_selection_fdr must be in [0,1].")
        if not 0.0 < float(parameter_selection_bootstrap_confidence) < 1.0:
            raise ValueError(
                "parameter_selection_bootstrap_confidence must lie in (0,1)."
            )
        if not 0.0 <= float(parameter_selection_min_model_win_share) <= 1.0:
            raise ValueError(
                "parameter_selection_min_model_win_share must be in [0,1]."
            )
        if float(parameter_selection_min_mpc_r2_gain) < 0.0:
            raise ValueError("parameter_selection_min_mpc_r2_gain must be nonnegative.")
        if float(parameter_selection_min_level_r2_gain) < 0.0:
            raise ValueError(
                "parameter_selection_min_level_r2_gain must be nonnegative."
            )
        if int(parameter_selection_bootstrap_reps) < 0:
            raise ValueError("parameter_selection_bootstrap_reps must be nonnegative.")
        if int(parameter_selection_max_terms) < 0:
            raise ValueError("parameter_selection_max_terms must be nonnegative.")
        if not any(int(power) >= 2 for power in parameter_selection_powers):
            raise ValueError(
                "parameter_selection_powers must contain at least one integer >= 2."
            )

    required_targets = {
        "consumption",
        "deposit",
        "liquid_outflow",
        "next_liquid_assets",
        "next_illiquid_assets",
        "delta_liquid_assets",
        "delta_illiquid_assets",
        "model_id",
    }
    missing = sorted(required_targets - set(data.columns))
    if missing:
        raise KeyError(f"Training data are missing required columns: {missing}")

    fit_start = time.perf_counter()
    _status(
        f"[fit 1/6] Preparing {len(data):,} sampled rows from "
        f"{data['model_id'].nunique():,} solved parameterizations...",
        verbose=verbose,
    )

    parameters = tuple(parameter_columns)
    categorical_parameters = tuple(categorical_parameter_columns)
    missing_params = [c for c in parameters + categorical_parameters if c not in data]
    if missing_params:
        raise KeyError(f"Parameter columns missing from data: {missing_params}")

    missing_indicators = tuple(
        c for c in data.columns if c.endswith("__missing") and c[:-9] in parameters
    )
    spec = FeatureSpec(
        state_columns=tuple(state_columns),
        engineered_state_columns=tuple(engineered_state_columns),
        redundant_state_columns=tuple(redundant_state_columns),
        parameter_columns=parameters + missing_indicators,
        categorical_state_columns=tuple(categorical_columns),
        categorical_parameter_columns=categorical_parameters,
        # Liquid assets and normalized slack enter in levels.  Other monetary
        # stocks/flows retain robust asinh scaling.
        signed_log_columns=tuple(
            c
            for c in DEFAULT_MONETARY_COLUMNS
            if c in set(state_columns) and c != "liquid_assets"
        ),
    )

    accounting_defaults = _accounting_defaults_from_data(data)
    initial_defaults: dict[str, Any] = {}
    for column in spec.continuous_state_columns + spec.parameter_columns:
        if column not in data:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        initial_defaults[column] = (
            float(values.median()) if values.notna().any() else 0.0
        )
    for column in spec.categorical_columns:
        mode = data[column].dropna().astype(str).mode()
        initial_defaults[column] = mode.iloc[0] if len(mode) else "__missing__"
    for column in (
        "liquid_grid_min",
        "liquid_grid_max",
        "illiquid_grid_min",
        "illiquid_grid_max",
        "money_scale",
    ):
        if column in data:
            initial_defaults[column] = float(
                pd.to_numeric(data[column], errors="coerce").median()
            )
    initial_defaults.update(accounting_defaults)

    if money_units not in {"model", "data"}:
        raise ValueError("money_units must be 'model' or 'data'.")

    fit_data = prepare_policy_inputs(
        data.copy().reset_index(drop=True),
        feature_spec=spec,
        default_values=initial_defaults,
        accounting_defaults=accounting_defaults,
        money_units=money_units,
    )
    if bool(working_age_only):
        # The solver treats age >= retirement_age as retired. Filtering on the
        # derived indicator preserves complete baseline/treated MPC pairs and
        # guarantees that tuning and validation use the target population.
        working_mask = (
            pd.to_numeric(fit_data["is_retired"], errors="coerce").fillna(1).eq(0)
        )
        fit_data = fit_data.loc[working_mask].reset_index(drop=True)
        if fit_data.empty:
            raise ValueError("Working-age filtering removed every training row.")
    fit_data["model_id"] = fit_data["model_id"].astype(str)
    # Stable ordering makes the model-level holdout split invariant to the
    # completion order of parallel sampling workers.
    fit_data = fit_data.sort_values("model_id", kind="mergesort").reset_index(drop=True)
    feature_missing = [c for c in spec.required_columns if c not in fit_data]
    if feature_missing:
        raise KeyError(f"Feature columns missing from data: {feature_missing}")

    finite_target = np.ones(len(fit_data), dtype=bool)
    for target in required_targets - {"model_id"}:
        finite_target &= np.isfinite(
            pd.to_numeric(fit_data[target], errors="coerce").to_numpy(dtype=float)
        )
    fit_data = fit_data.loc[finite_target].reset_index(drop=True)

    groups = fit_data["model_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError(
            "At least three solved parameterizations are required for separate "
            "training, tuning, and test sets."
        )
    if not 0.0 < float(validation_share) < 1.0:
        raise ValueError("validation_share must be strictly between 0 and 1.")
    if not 0.0 < float(tuning_share) < 1.0:
        raise ValueError("tuning_share must be strictly between 0 and 1.")

    train, tune, test = _load_or_create_model_split(
        fit_data,
        categorical_parameters=categorical_parameters,
        validation_share=validation_share,
        tuning_share=tuning_share,
        random_state=random_state,
        split_path=model_split_path,
    )
    development = pd.concat([train, tune], axis=0, ignore_index=True)

    _status(
        "  model split: "
        f"train={train['model_id'].nunique():,}, "
        f"tune={tune['model_id'].nunique():,}, "
        f"test={test['model_id'].nunique():,}",
        verbose=verbose,
    )

    requested_parameter_power_terms = _dedupe_tuples(parameter_power_terms)
    requested_spline_parameter_power_terms = _dedupe_tuples(
        spline_parameter_power_terms
    )
    requested_spline_parameter_pair_terms = _dedupe_tuples(spline_parameter_pair_terms)
    effective_parameter_power_terms = requested_parameter_power_terms
    effective_spline_parameter_power_terms = requested_spline_parameter_power_terms
    effective_spline_parameter_pair_terms = requested_spline_parameter_pair_terms
    parameter_selection_result: ParameterTermSelectionResult | None = None

    if bool(auto_select_parameter_terms):
        parameter_selection_result = _select_higher_order_parameter_terms(
            train,
            tune,
            spec=spec,
            polynomial_degree=polynomial_degree,
            spline_columns=spline_columns,
            spline_n_knots=spline_n_knots,
            spline_column_n_knots=spline_column_n_knots,
            spline_knot_strategies=spline_knot_strategies,
            spline_knot_powers=spline_knot_powers,
            spline_tensor_pairs=spline_tensor_pairs,
            spline_tensor_triples=spline_tensor_triples,
            spline_categorical_tensor_pairs=spline_categorical_tensor_pairs,
            spline_tensor_parameter_terms=spline_tensor_parameter_terms,
            mandatory_parameter_power_terms=requested_parameter_power_terms,
            mandatory_spline_parameter_power_terms=(
                requested_spline_parameter_power_terms
            ),
            mandatory_spline_parameter_pair_terms=(
                requested_spline_parameter_pair_terms
            ),
            categorical_state_power_terms=categorical_state_power_terms,
            categorical_slope_exclude_patterns=(categorical_slope_exclude_patterns),
            spline_categorical_exclude_patterns=(spline_categorical_exclude_patterns),
            ridge_alphas=ridge_alphas,
            candidate_mpc_weights=candidate_mpc_weights,
            mpc_loss_weight=mpc_loss_weight,
            mpc_objective_max_pairs=mpc_objective_max_pairs,
            mpc_objective_trim_quantile=mpc_objective_trim_quantile,
            mpc_stratify_age_bins=mpc_stratify_age_bins,
            mpc_stratify_liquid_bins=mpc_stratify_liquid_bins,
            mpc_stratify_illiquid_bins=mpc_stratify_illiquid_bins,
            mpc_stratify_target_bins=mpc_stratify_target_bins,
            selection_spline_columns=parameter_selection_spline_columns,
            selection_powers=parameter_selection_powers,
            parameter_include_patterns=(parameter_selection_parameter_patterns),
            parameter_exclude_patterns=(parameter_selection_exclude_patterns),
            include_parameter_powers=(parameter_selection_include_parameter_powers),
            max_train_rows=parameter_selection_max_train_rows,
            max_tune_rows=parameter_selection_max_tune_rows,
            min_mpc_r2_gain=parameter_selection_min_mpc_r2_gain,
            min_level_r2_gain=parameter_selection_min_level_r2_gain,
            max_mpc_r2_loss_for_level_term=(
                parameter_selection_max_mpc_r2_loss_for_level_term
            ),
            min_model_win_share=parameter_selection_min_model_win_share,
            fdr=parameter_selection_fdr,
            bootstrap_reps=parameter_selection_bootstrap_reps,
            bootstrap_confidence=(parameter_selection_bootstrap_confidence),
            max_selected_terms=parameter_selection_max_terms,
            random_state=random_state,
            verbose=verbose,
            show_progress=show_progress,
        )
        effective_parameter_power_terms = (
            parameter_selection_result.parameter_power_terms
        )
        effective_spline_parameter_power_terms = (
            parameter_selection_result.spline_parameter_power_terms
        )
        effective_spline_parameter_pair_terms = (
            parameter_selection_result.spline_parameter_pair_terms
        )

    def make_feature_map() -> Any:
        if model_type == "polynomial":
            return StructuredPolynomialMap(
                spec,
                degree=polynomial_degree,
                state_state_interactions=True,
                state_parameter_interactions=True,
                parameter_parameter_interactions=False,
                categorical_slopes=True,
                categorical_slope_exclude_patterns=tuple(
                    categorical_slope_exclude_patterns
                ),
                spline_columns=tuple(spline_columns),
                spline_n_knots=int(spline_n_knots),
                spline_column_n_knots=tuple(spline_column_n_knots),
                spline_knot_strategies=tuple(spline_knot_strategies),
                spline_knot_powers=tuple(spline_knot_powers),
                spline_degree=3,
                spline_parameter_interactions=True,
                spline_categorical_interactions=True,
                spline_categorical_exclude_patterns=tuple(
                    spline_categorical_exclude_patterns
                ),
                spline_tensor_pairs=tuple(spline_tensor_pairs),
                spline_tensor_triples=tuple(spline_tensor_triples),
                spline_categorical_tensor_pairs=tuple(spline_categorical_tensor_pairs),
                spline_tensor_parameter_terms=tuple(spline_tensor_parameter_terms),
                parameter_power_terms=tuple(effective_parameter_power_terms),
                spline_parameter_power_terms=tuple(
                    effective_spline_parameter_power_terms
                ),
                spline_parameter_pair_terms=tuple(
                    effective_spline_parameter_pair_terms
                ),
                categorical_state_power_terms=tuple(categorical_state_power_terms),
                random_state=random_state,
            )
        if model_type in {"rff", "rbf"}:
            return SmoothRFFMap(
                spec,
                n_components=rff_components,
                random_state=random_state,
            )
        raise ValueError("model_type must be 'polynomial' or 'rff'.")

    _status(
        f"[fit 2/6] Constructing standardized {model_type} features for "
        "target-specific ridge tuning...",
        verbose=verbose,
    )
    feature_start = time.perf_counter()
    tuning_map = make_feature_map()
    Z_train = tuning_map.fit_transform(train)
    Z_tune = tuning_map.transform(tune)
    _status(
        f"  tuning design matrices: train={Z_train.shape}, tune={Z_tune.shape}; "
        f"elapsed: {time.perf_counter() - feature_start:,.1f}s",
        verbose=verbose,
    )

    train_mpc_design = _build_mpc_pair_design(
        train,
        Z_train,
        max_pairs=mpc_objective_max_pairs,
        random_state=random_state + 11,
        trim_quantile=mpc_objective_trim_quantile,
        age_bins=mpc_stratify_age_bins,
        liquid_bins=mpc_stratify_liquid_bins,
        illiquid_bins=mpc_stratify_illiquid_bins,
        target_bins=mpc_stratify_target_bins,
    )
    if any(weight > 0.0 for weight in candidate_mpc_weights):
        if train_mpc_design is None:
            raise ValueError(
                "Direct MPC targeting was requested, but no valid training pairs "
                "were constructed. Increase --mpc-pair-share and rows per model."
            )
        _status(
            f"  direct MPC objective: {train_mpc_design.n_pairs:,} pairs "
            f"({train_mpc_design.n_available:,} available before the cap)",
            verbose=verbose,
        )

    target_transform_names = {
        target: _target_transform_kind(
            target,
            consumption_transform=consumption_transform,
            deposit_transform=deposit_transform,
            liquid_outflow_transform=liquid_outflow_transform,
        )
        for target in ALL_FIT_TARGETS
    }
    tuning_transforms: dict[str, TargetTransform] = {}
    y_train_transformed: dict[str, np.ndarray] = {}
    y_tune_levels: dict[str, np.ndarray] = {}
    for target in ALL_FIT_TARGETS:
        transform = TargetTransform(target_transform_names[target]).fit(
            pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
        )
        tuning_transforms[target] = transform
        y_train_transformed[target] = transform.transform(
            pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
        )
        y_tune_levels[target] = pd.to_numeric(tune[target], errors="coerce").to_numpy(
            dtype=float
        )

    # Compute the expensive design cross-products once. Every alpha, target, and
    # direct-MPC weight below reuses these sufficient statistics.
    tuning_statistics = _compute_ridge_statistics(Z_train, y_train_transformed)
    train_mpc_statistics = _compute_mpc_statistics(train_mpc_design)

    _status(
        f"[fit 3/6] Tuning ridge penalties and the direct MPC objective...",
        verbose=verbose,
    )
    alpha_rows: list[dict[str, Any]] = []
    best_alphas: dict[str, float] = {}
    best_direct_mpc_weights: dict[str, float] = {
        target: 0.0 for target in ALL_FIT_TARGETS
    }
    best_scores = {target: np.inf for target in ALL_FIT_TARGETS}

    tasks: list[tuple[str, float, float]] = []
    for target in ALL_FIT_TARGETS:
        weights = candidate_mpc_weights if target == "consumption" else [0.0]
        for alpha in ridge_alphas:
            for direct_weight in weights:
                tasks.append((target, float(alpha), float(direct_weight)))

    iterator = _progress_iter(
        tasks,
        total=len(tasks),
        description="Ridge/MPC tuning",
        unit="fit",
        enabled=show_progress,
    )

    for position, (target, alpha, direct_weight) in enumerate(iterator, start=1):
        model = _fit_regressor_from_statistics(
            tuning_statistics,
            target=target,
            alpha=alpha,
            mpc_weight=direct_weight,
            pair_statistics=train_mpc_statistics,
        )

        pred_t = np.asarray(model.predict(Z_tune), dtype=float)
        pred = tuning_transforms[target].inverse(pred_t)
        truth = y_tune_levels[target]
        level_nrmse = _relative_rmse(truth, pred)
        combined_score = level_nrmse
        mpc_rmse = np.nan
        mpc_nrmse = np.nan
        mpc_bias = np.nan
        mpc_r2 = np.nan
        n_mpc_pairs = 0

        if target == "consumption":
            pairs = _paired_mpc_table(tune, pred)
            if not pairs.empty:
                exact_mpc = pairs["exact_mpc"].to_numpy(dtype=float)
                predicted_mpc = pairs["predicted_mpc"].to_numpy(dtype=float)
                ok = np.isfinite(exact_mpc) & np.isfinite(predicted_mpc)
                if ok.any():
                    errors = predicted_mpc[ok] - exact_mpc[ok]
                    mpc_rmse = float(np.sqrt(np.mean(errors**2)))
                    mpc_sd = float(np.std(exact_mpc[ok]))
                    mpc_nrmse = mpc_rmse / max(mpc_sd, EPS)
                    mpc_bias = float(np.mean(errors))
                    mpc_r2 = (
                        float(r2_score(exact_mpc[ok], predicted_mpc[ok]))
                        if int(ok.sum()) > 1
                        else np.nan
                    )
                    n_mpc_pairs = int(ok.sum())
                    combined_score = (
                        level_nrmse
                        + float(mpc_loss_weight) * mpc_nrmse
                        + 0.10 * abs(mpc_bias) / max(mpc_sd, EPS)
                    )

        solver_info = getattr(model, "solver_diagnostics_", {}) or {}
        alpha_rows.append(
            {
                "target": target,
                "alpha": alpha,
                "mpc_difference_weight": direct_weight,
                "tuning_nrmse_sd": level_nrmse,
                "tuning_mpc_rmse": mpc_rmse,
                "tuning_mpc_nrmse": mpc_nrmse,
                "tuning_mpc_bias": mpc_bias,
                "tuning_mpc_r2": mpc_r2,
                "tuning_score": combined_score,
                "n_mpc_tuning_pairs": n_mpc_pairs,
                "joint_solver_iterations": solver_info.get("iterations"),
                "joint_solver_stop_code": solver_info.get("istop"),
            }
        )
        if combined_score < best_scores[target]:
            best_scores[target] = combined_score
            best_alphas[target] = alpha
            best_direct_mpc_weights[target] = direct_weight

        if verbose and (_tqdm is None or not show_progress):
            _status(
                f"  {position}/{len(tasks)}: target={target}; alpha={alpha:g}; "
                f"direct MPC weight={direct_weight:g}; score={combined_score:.5f}",
                verbose=verbose,
            )
        elif show_progress and _tqdm is not None:
            iterator.set_postfix(  # type: ignore[attr-defined]
                target=target,
                alpha=f"{alpha:g}",
                mpc_w=f"{direct_weight:g}",
                score=f"{combined_score:.4f}",
            )

    for target in ALL_FIT_TARGETS:
        suffix = (
            f", direct MPC weight={best_direct_mpc_weights[target]:g}"
            if target == "consumption"
            else ""
        )
        _status(
            f"  selected {target} alpha={best_alphas[target]:g}{suffix} "
            f"(tuning score={best_scores[target]:.5f})",
            verbose=verbose,
        )

    # Fit the selected target models on the coefficient-training split and use
    # the tuning split to choose how much the accounting reconciliation should
    # trust direct liquid outflow versus direct deposit.
    tuning_regressions: dict[str, Any] = {}
    for target in ALL_FIT_TARGETS:
        tuning_regressions[target] = _fit_regressor_from_statistics(
            tuning_statistics,
            target=target,
            alpha=best_alphas[target],
            mpc_weight=best_direct_mpc_weights[target],
            pair_statistics=train_mpc_statistics,
        )
    tuning_internal = _primitive_predictions_from_models(
        tuning_map,
        tuning_regressions,
        tuning_transforms,
        tune,
    )

    if liquid_outflow_weight is None:
        weight_grid = np.linspace(0.0, 1.0, 9)
    else:
        if not 0.0 <= float(liquid_outflow_weight) <= 1.0:
            raise ValueError("liquid_outflow_weight must be in [0,1].")
        weight_grid = np.array([float(liquid_outflow_weight)])

    reconciliation_rows: list[dict[str, float]] = []
    selected_outflow_weight = float(weight_grid[0])
    best_reconciliation_score = np.inf
    tuning_bounds = {
        "next_liquid_assets": (
            float(fit_data["liquid_grid_min"].min()),
            float(fit_data["liquid_grid_max"].max()),
        ),
        "next_illiquid_assets": (
            float(fit_data["illiquid_grid_min"].min()),
            float(fit_data["illiquid_grid_max"].max()),
        ),
    }
    for weight in weight_grid:
        reconciled = reconstruct_policy_outputs(
            tune,
            tuning_internal,
            accounting_defaults=accounting_defaults,
            money_units=money_units,
            output_bounds=tuning_bounds,
            liquid_outflow_weight=float(weight),
            project=True,
        )
        deposit_relative = _relative_rmse(
            tune["deposit"],
            reconciled["deposit"],
            benchmark=np.zeros(len(tune)),
        )
        liquid_change_relative = _relative_rmse(
            tune["delta_liquid_assets"],
            reconciled["delta_liquid_assets"],
            benchmark=np.zeros(len(tune)),
        )
        score = (
            float(deposit_reconciliation_loss_weight) * deposit_relative
            + float(liquid_change_loss_weight) * liquid_change_relative
        )
        reconciliation_rows.append(
            {
                "liquid_outflow_weight": float(weight),
                "deposit_relative_rmse": deposit_relative,
                "delta_liquid_relative_rmse": liquid_change_relative,
                "score": score,
            }
        )
        if score < best_reconciliation_score:
            best_reconciliation_score = score
            selected_outflow_weight = float(weight)

    _status(
        f"  selected liquid-outflow reconciliation weight="
        f"{selected_outflow_weight:.3f} "
        f"(tuning score={best_reconciliation_score:.5f})",
        verbose=verbose,
    )

    # Release tuning-stage matrices before constructing the development design.
    # On large cluster runs these arrays dominate peak memory.
    del Z_train, Z_tune, tuning_statistics, train_mpc_design
    del train_mpc_statistics, tuning_regressions, tuning_internal
    gc.collect()

    _status(
        "[fit 4/6] Refitting on training+tuning models and evaluating held-out "
        "parameterizations...",
        verbose=verbose,
    )
    evaluation_start = time.perf_counter()
    evaluation_map = make_feature_map()
    Z_development = evaluation_map.fit_transform(development)
    development_mpc_design = _build_mpc_pair_design(
        development,
        Z_development,
        max_pairs=mpc_objective_max_pairs,
        random_state=random_state + 12,
        trim_quantile=mpc_objective_trim_quantile,
        age_bins=mpc_stratify_age_bins,
        liquid_bins=mpc_stratify_liquid_bins,
        illiquid_bins=mpc_stratify_illiquid_bins,
        target_bins=mpc_stratify_target_bins,
    )
    evaluation_regressions: dict[str, Any] = {}
    evaluation_transforms: dict[str, TargetTransform] = {}
    evaluation_targets: dict[str, np.ndarray] = {}
    for target in ALL_FIT_TARGETS:
        transform = TargetTransform(target_transform_names[target]).fit(
            pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        )
        evaluation_transforms[target] = transform
        evaluation_targets[target] = transform.transform(
            pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        )
    evaluation_statistics = _compute_ridge_statistics(Z_development, evaluation_targets)
    development_mpc_statistics = _compute_mpc_statistics(development_mpc_design)
    for target in ALL_FIT_TARGETS:
        evaluation_regressions[target] = _fit_regressor_from_statistics(
            evaluation_statistics,
            target=target,
            alpha=best_alphas[target],
            mpc_weight=best_direct_mpc_weights[target],
            pair_statistics=development_mpc_statistics,
        )

    internal_test = _primitive_predictions_from_models(
        evaluation_map,
        evaluation_regressions,
        evaluation_transforms,
        test,
    )
    test_pred = reconstruct_policy_outputs(
        test,
        internal_test,
        accounting_defaults=accounting_defaults,
        money_units=money_units,
        output_bounds=tuning_bounds,
        liquid_outflow_weight=selected_outflow_weight,
        project=True,
    )
    validation_metrics = _metrics_table(test, test_pred, "held_out_test_models")
    validation_by_model = _metrics_by_model(
        test,
        test_pred,
        "held_out_test_models",
    )
    _status(
        f"  held-out evaluation complete in "
        f"{time.perf_counter() - evaluation_start:,.1f}s",
        verbose=verbose,
    )
    if verbose:
        for metric_row in validation_metrics.itertuples(index=False):
            suffix = ""
            if pd.notna(metric_row.skill_vs_benchmark):
                suffix = f", skill vs benchmark={metric_row.skill_vs_benchmark:.4f}"
            _status(
                f"    {metric_row.target}: R2={metric_row.r2:.4f}, "
                f"RMSE={metric_row.rmse:.6g}{suffix}",
                verbose=verbose,
            )

    evaluation_output_bounds = {
        target: (
            float(pd.to_numeric(development[target], errors="coerce").min()),
            float(pd.to_numeric(development[target], errors="coerce").max()),
        )
        for target in (
            "consumption",
            "deposit",
            "liquid_outflow",
            "next_liquid_assets",
            "next_illiquid_assets",
        )
    }
    evaluation_metadata = {
        "bundle_format_version": 10,
        "bundle_role": "evaluation",
        "fit_scope": "development_only",
        "true_holdout_bundle": True,
        "fit_models": sorted(development["model_id"].astype(str).unique().tolist()),
        "training_models": sorted(train["model_id"].astype(str).unique().tolist()),
        "tuning_models": sorted(tune["model_id"].astype(str).unique().tolist()),
        "test_models": sorted(test["model_id"].astype(str).unique().tolist()),
        "validation_models": sorted(test["model_id"].astype(str).unique().tolist()),
        "test_models_are_disjoint_from_fit": True,
        "model_split_path": (
            None if model_split_path is None else str(Path(model_split_path))
        ),
        "model_type": model_type,
        "polynomial_degree": int(polynomial_degree),
        "spline_columns": list(spline_columns),
        "spline_n_knots": int(spline_n_knots),
        "spline_column_n_knots": {
            str(column): int(count) for column, count in spline_column_n_knots
        },
        "spline_knot_strategies": {
            str(column): str(strategy) for column, strategy in spline_knot_strategies
        },
        "spline_knot_powers": {
            str(column): float(power) for column, power in spline_knot_powers
        },
        "fitted_spline_knots": dict(getattr(evaluation_map, "spline_knots_", {})),
        "spline_tensor_pairs": [list(pair) for pair in spline_tensor_pairs],
        "spline_tensor_triples": [list(triple) for triple in spline_tensor_triples],
        "spline_categorical_tensor_pairs": [
            list(term) for term in spline_categorical_tensor_pairs
        ],
        "spline_tensor_parameter_terms": [
            list(term) for term in spline_tensor_parameter_terms
        ],
        "parameter_power_terms": [
            list(term) for term in effective_parameter_power_terms
        ],
        "spline_parameter_power_terms": [
            list(term) for term in effective_spline_parameter_power_terms
        ],
        "spline_parameter_pair_terms": [
            list(term) for term in effective_spline_parameter_pair_terms
        ],
        "categorical_state_power_terms": [
            list(term) for term in categorical_state_power_terms
        ],
        "auto_select_parameter_terms": bool(auto_select_parameter_terms),
        "parameter_term_search": (
            []
            if parameter_selection_result is None
            else list(parameter_selection_result.search_rows)
        ),
        "ridge_alphas": dict(best_alphas),
        "mpc_difference_weight_selected": float(best_direct_mpc_weights["consumption"]),
        "liquid_outflow_weight": float(selected_outflow_weight),
        "feature_count": int(evaluation_map.transform(development.iloc[:1]).shape[1]),
        "feature_names": list(getattr(evaluation_map, "feature_names_", [])),
        "state_columns": list(spec.state_columns),
        "engineered_state_columns": list(spec.engineered_state_columns),
        "redundant_state_columns": list(spec.redundant_state_columns),
        "parameter_columns": list(parameters),
        "effective_parameter_columns": list(spec.parameter_columns),
        "categorical_state_columns": list(spec.categorical_state_columns),
        "categorical_parameter_columns": list(spec.categorical_parameter_columns),
        "money_units": money_units,
        "working_age_only": bool(working_age_only),
        "random_state": int(random_state),
    }
    evaluation_bundle = PolicySurrogateBundle(
        feature_map=evaluation_map,
        regressions=evaluation_regressions,
        feature_spec=spec,
        internal_targets=tuple(ALL_FIT_TARGETS),
        target_transforms=evaluation_transforms,
        target_parameterization=target_parameterization,
        output_bounds=evaluation_output_bounds,
        feature_ranges=_feature_range_summary(development, spec),
        default_values=_default_row(development, spec, accounting_defaults),
        accounting_defaults=accounting_defaults,
        validation_metrics=validation_metrics,
        validation_by_model=validation_by_model,
        training_metadata=evaluation_metadata,
        liquid_outflow_weight=float(selected_outflow_weight),
    )
    if evaluation_bundle_path is not None:
        evaluation_bundle.save(evaluation_bundle_path)
        _status(
            f"  saved true-holdout evaluation bundle to {evaluation_bundle_path}",
            verbose=verbose,
        )

    # The validation tables no longer require the development design matrix.
    del evaluation_bundle
    del Z_development, evaluation_statistics, development_mpc_design
    del development_mpc_statistics, evaluation_regressions, evaluation_transforms
    del evaluation_map, internal_test, test_pred
    gc.collect()

    _status(
        "[fit 5/6] Constructing the full design matrix and fitting final smooth "
        "targets...",
        verbose=verbose,
    )
    final_start = time.perf_counter()
    final_map = make_feature_map()
    Z_all = final_map.fit_transform(fit_data)
    final_mpc_design = _build_mpc_pair_design(
        fit_data,
        Z_all,
        max_pairs=mpc_objective_max_pairs,
        random_state=random_state + 13,
        trim_quantile=mpc_objective_trim_quantile,
        age_bins=mpc_stratify_age_bins,
        liquid_bins=mpc_stratify_liquid_bins,
        illiquid_bins=mpc_stratify_illiquid_bins,
        target_bins=mpc_stratify_target_bins,
    )
    final_regressions: dict[str, Any] = {}
    final_transforms: dict[str, TargetTransform] = {}
    final_targets: dict[str, np.ndarray] = {}
    for target in ALL_FIT_TARGETS:
        transform = TargetTransform(target_transform_names[target]).fit(
            pd.to_numeric(fit_data[target], errors="coerce").to_numpy(dtype=float)
        )
        final_transforms[target] = transform
        final_targets[target] = transform.transform(
            pd.to_numeric(fit_data[target], errors="coerce").to_numpy(dtype=float)
        )
    final_statistics = _compute_ridge_statistics(Z_all, final_targets)
    final_mpc_statistics = _compute_mpc_statistics(final_mpc_design)
    for target in ALL_FIT_TARGETS:
        final_regressions[target] = _fit_regressor_from_statistics(
            final_statistics,
            target=target,
            alpha=best_alphas[target],
            mpc_weight=best_direct_mpc_weights[target],
            pair_statistics=final_mpc_statistics,
        )
    _status(
        f"  final design matrix: {Z_all.shape}; elapsed: "
        f"{time.perf_counter() - final_start:,.1f}s",
        verbose=verbose,
    )

    output_bounds = {
        target: (
            float(pd.to_numeric(fit_data[target], errors="coerce").min()),
            float(pd.to_numeric(fit_data[target], errors="coerce").max()),
        )
        for target in (
            "consumption",
            "deposit",
            "liquid_outflow",
            "next_liquid_assets",
            "next_illiquid_assets",
        )
    }
    money_scale_values = (
        pd.to_numeric(fit_data["money_scale"], errors="coerce")
        if "money_scale" in fit_data
        else pd.Series(dtype=float)
    )
    money_scale_values = money_scale_values[
        np.isfinite(money_scale_values) & (money_scale_values > 0)
    ]
    defaults = _default_row(fit_data, spec, accounting_defaults)
    metadata = {
        "bundle_format_version": 10,
        "bundle_role": "production",
        "fit_scope": "all_models",
        "true_holdout_bundle": False,
        "fit_models": sorted(fit_data["model_id"].astype(str).unique().tolist()),
        "test_models_are_disjoint_from_fit": False,
        "evaluation_bundle_filename": (
            None
            if evaluation_bundle_path is None
            else Path(evaluation_bundle_path).name
        ),
        "model_split_path": (
            None if model_split_path is None else str(Path(model_split_path))
        ),
        "saved_grid_format": "compact consumption+deposit with gross and after-tax income",
        "population_sampled_training": True,
        "ridge_solver": "closed-form sufficient-statistics CPU solver",
        "model_type": model_type,
        "polynomial_degree": int(polynomial_degree),
        "spline_columns": list(spline_columns),
        "spline_n_knots": int(spline_n_knots),
        "spline_column_n_knots": {
            str(column): int(count) for column, count in spline_column_n_knots
        },
        "spline_knot_strategies": {
            str(column): str(strategy) for column, strategy in spline_knot_strategies
        },
        "spline_knot_powers": {
            str(column): float(power) for column, power in spline_knot_powers
        },
        "fitted_spline_knots": dict(getattr(final_map, "spline_knots_", {})),
        "spline_tensor_pairs": [list(pair) for pair in spline_tensor_pairs],
        "spline_tensor_triples": [list(triple) for triple in spline_tensor_triples],
        "spline_categorical_tensor_pairs": [
            list(term) for term in spline_categorical_tensor_pairs
        ],
        "spline_tensor_parameter_terms": [
            list(term) for term in spline_tensor_parameter_terms
        ],
        "parameter_power_terms": [
            [parameter, int(power)]
            for parameter, power in effective_parameter_power_terms
        ],
        "spline_parameter_power_terms": [
            [spline, parameter, int(power)]
            for spline, parameter, power in effective_spline_parameter_power_terms
        ],
        "spline_parameter_pair_terms": [
            [spline, parameter_1, parameter_2]
            for spline, parameter_1, parameter_2 in effective_spline_parameter_pair_terms
        ],
        "requested_parameter_power_terms": [
            [parameter, int(power)]
            for parameter, power in requested_parameter_power_terms
        ],
        "requested_spline_parameter_power_terms": [
            [spline, parameter, int(power)]
            for spline, parameter, power in requested_spline_parameter_power_terms
        ],
        "requested_spline_parameter_pair_terms": [
            [spline, parameter_1, parameter_2]
            for spline, parameter_1, parameter_2 in requested_spline_parameter_pair_terms
        ],
        "auto_select_parameter_terms": bool(auto_select_parameter_terms),
        "parameter_selection_spline_columns": list(parameter_selection_spline_columns),
        "parameter_selection_powers": [
            int(power) for power in parameter_selection_powers
        ],
        "parameter_selection_parameter_patterns": list(
            parameter_selection_parameter_patterns
        ),
        "parameter_selection_exclude_patterns": list(
            parameter_selection_exclude_patterns
        ),
        "parameter_selection_include_parameter_powers": bool(
            parameter_selection_include_parameter_powers
        ),
        "parameter_selection_max_train_rows": parameter_selection_max_train_rows,
        "parameter_selection_max_tune_rows": parameter_selection_max_tune_rows,
        "parameter_selection_min_mpc_r2_gain": float(
            parameter_selection_min_mpc_r2_gain
        ),
        "parameter_selection_min_level_r2_gain": float(
            parameter_selection_min_level_r2_gain
        ),
        "parameter_selection_max_mpc_r2_loss_for_level_term": float(
            parameter_selection_max_mpc_r2_loss_for_level_term
        ),
        "parameter_selection_min_model_win_share": float(
            parameter_selection_min_model_win_share
        ),
        "parameter_selection_fdr": float(parameter_selection_fdr),
        "parameter_selection_bootstrap_reps": int(parameter_selection_bootstrap_reps),
        "parameter_selection_bootstrap_confidence": float(
            parameter_selection_bootstrap_confidence
        ),
        "parameter_selection_max_terms": int(parameter_selection_max_terms),
        "parameter_selection_baseline_metrics": (
            None
            if parameter_selection_result is None
            else parameter_selection_result.baseline_metrics
        ),
        "parameter_selection_n_candidates": (
            0
            if parameter_selection_result is None
            else int(parameter_selection_result.n_candidates)
        ),
        "parameter_selection_n_selected": (
            0
            if parameter_selection_result is None
            else int(parameter_selection_result.n_selected)
        ),
        "parameter_term_search": (
            []
            if parameter_selection_result is None
            else list(parameter_selection_result.search_rows)
        ),
        "categorical_state_power_terms": [
            [category, state, int(power)]
            for category, state, power in categorical_state_power_terms
        ],
        "working_age_only": bool(working_age_only),
        "categorical_slope_exclude_patterns": list(categorical_slope_exclude_patterns),
        "spline_categorical_exclude_patterns": list(
            spline_categorical_exclude_patterns
        ),
        "rff_components": int(rff_components),
        "consumption_transform": consumption_transform,
        "deposit_transform": deposit_transform,
        "liquid_outflow_transform": liquid_outflow_transform,
        "mpc_loss_weight": float(mpc_loss_weight),
        "mpc_difference_weight_requested": (
            None if mpc_difference_weight is None else float(mpc_difference_weight)
        ),
        "mpc_difference_weight_grid": [float(x) for x in candidate_mpc_weights],
        "mpc_difference_weight_selected": float(best_direct_mpc_weights["consumption"]),
        "mpc_difference_rows_used": bool(best_direct_mpc_weights["consumption"] > 0.0),
        "mpc_objective_max_pairs": (
            None if mpc_objective_max_pairs is None else int(mpc_objective_max_pairs)
        ),
        "mpc_objective_max_iter": int(mpc_objective_max_iter),
        "mpc_objective_tol": float(mpc_objective_tol),
        "mpc_objective_trim_quantile": float(mpc_objective_trim_quantile),
        "mpc_stratify_age_bins": [float(x) for x in mpc_stratify_age_bins],
        "mpc_stratify_liquid_bins": int(mpc_stratify_liquid_bins),
        "mpc_stratify_illiquid_bins": int(mpc_stratify_illiquid_bins),
        "mpc_stratify_target_bins": int(mpc_stratify_target_bins),
        "mpc_objective_pairs_final": (
            0 if final_mpc_design is None else int(final_mpc_design.n_pairs)
        ),
        "liquid_outflow_weight": float(selected_outflow_weight),
        "liquid_outflow_reconciliation_search": reconciliation_rows,
        "ridge_alphas": best_alphas,
        "ridge_alpha": None,
        "ridge_search": alpha_rows,
        "n_rows": int(len(fit_data)),
        "n_models": int(fit_data["model_id"].nunique()),
        "training_models": sorted(train["model_id"].astype(str).unique().tolist()),
        "tuning_models": sorted(tune["model_id"].astype(str).unique().tolist()),
        "test_models": sorted(test["model_id"].astype(str).unique().tolist()),
        "validation_models": sorted(test["model_id"].astype(str).unique().tolist()),
        "feature_count": int(Z_all.shape[1]),
        "feature_names": list(getattr(final_map, "feature_names_", [])),
        "state_columns": list(spec.state_columns),
        "engineered_state_columns": list(spec.engineered_state_columns),
        "redundant_state_columns": list(spec.redundant_state_columns),
        "model_state_columns": list(spec.model_state_columns),
        "parameter_columns": list(parameters),
        "effective_parameter_columns": list(spec.parameter_columns),
        "categorical_state_columns": list(spec.categorical_state_columns),
        "categorical_parameter_columns": list(spec.categorical_parameter_columns),
        "categorical_columns": list(spec.categorical_columns),
        "test_share": float(validation_share),
        "tuning_share_of_development": float(tuning_share),
        "training_model_count": int(train["model_id"].nunique()),
        "tuning_model_count": int(tune["model_id"].nunique()),
        "test_model_count": int(test["model_id"].nunique()),
        "target_parameterization": target_parameterization,
        "primitive_policy_targets": list(PRIMITIVE_POLICY_TARGETS),
        "auxiliary_policy_targets": list(AUXILIARY_POLICY_TARGETS),
        "asset_choices_reconstructed_from_accounting": True,
        "final_design_standardized": True,
        "parameter_parameter_interactions": False,
        "money_units": money_units,
        "money_scale_median": (
            float(money_scale_values.median()) if len(money_scale_values) else None
        ),
        "money_scale_min": (
            float(money_scale_values.min()) if len(money_scale_values) else None
        ),
        "money_scale_max": (
            float(money_scale_values.max()) if len(money_scale_values) else None
        ),
        "random_state": int(random_state),
    }

    _status(
        f"[fit 6/6] Surrogate fit complete. Total fit time: "
        f"{time.perf_counter() - fit_start:,.1f}s",
        verbose=verbose,
    )
    return PolicySurrogateBundle(
        feature_map=final_map,
        regressions=final_regressions,
        feature_spec=spec,
        internal_targets=tuple(ALL_FIT_TARGETS),
        target_transforms=final_transforms,
        target_parameterization=target_parameterization,
        output_bounds=output_bounds,
        feature_ranges=_feature_range_summary(fit_data, spec),
        default_values=defaults,
        accounting_defaults=accounting_defaults,
        validation_metrics=validation_metrics,
        validation_by_model=validation_by_model,
        training_metadata=metadata,
        liquid_outflow_weight=float(selected_outflow_weight),
    )


# -----------------------------------------------------------------------------
# Convenience end-to-end function and compact CLI inspection
# -----------------------------------------------------------------------------


def train_from_saved_grids(
    model_root: str | Path,
    output_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    schema: GridSchema = GridSchema(),
    rows_per_model: int = 1_000,
    max_total_rows: int = 3_000_000,
    max_models: int | None = None,
    sampling_mode: str = "population",
    population_spec: PopulationSamplingSpec = PopulationSamplingSpec(),
    n_jobs: int = 1,
    mpc_pair_share: float = 0.60,
    mpc_check_sizes: Sequence[float] = (0.02, 0.05, 0.10, 0.188679, 0.20, 0.50),
    parameter_include: Sequence[str] | None = None,
    parameter_exclude: Sequence[str] = (),
    categorical_parameter_include: Sequence[str] | None = None,
    categorical_parameter_exclude: Sequence[str] = (),
    max_solver_distance: float | None = None,
    max_relative_distance: float | None = None,
    max_hjb_residual: float | None = None,
    model_type: str = "polynomial",
    polynomial_degree: int = 2,
    spline_columns: Sequence[str] = DEFAULT_SPLINE_COLUMNS,
    spline_n_knots: int = 8,
    spline_column_n_knots: Sequence[tuple[str, int]] = DEFAULT_SPLINE_COLUMN_N_KNOTS,
    spline_knot_strategies: Sequence[tuple[str, str]] = DEFAULT_SPLINE_KNOT_STRATEGIES,
    spline_knot_powers: Sequence[tuple[str, float]] = DEFAULT_SPLINE_KNOT_POWERS,
    spline_tensor_pairs: Sequence[tuple[str, str]] = DEFAULT_SPLINE_TENSOR_PAIRS,
    spline_tensor_triples: Sequence[
        tuple[str, str, str]
    ] = DEFAULT_SPLINE_TENSOR_TRIPLES,
    spline_categorical_tensor_pairs: Sequence[
        tuple[str, str, str]
    ] = DEFAULT_SPLINE_CATEGORICAL_TENSOR_PAIRS,
    spline_tensor_parameter_terms: Sequence[
        tuple[str, str, str]
    ] = DEFAULT_SPLINE_TENSOR_PARAMETER_TERMS,
    parameter_power_terms: Sequence[tuple[str, int]] = DEFAULT_PARAMETER_POWER_TERMS,
    spline_parameter_power_terms: Sequence[
        tuple[str, str, int]
    ] = DEFAULT_SPLINE_PARAMETER_POWER_TERMS,
    spline_parameter_pair_terms: Sequence[
        tuple[str, str, str]
    ] = DEFAULT_SPLINE_PARAMETER_PAIR_TERMS,
    auto_select_parameter_terms: bool = True,
    parameter_selection_spline_columns: Sequence[
        str
    ] = DEFAULT_PARAMETER_SELECTION_SPLINE_COLUMNS,
    parameter_selection_powers: Sequence[int] = DEFAULT_PARAMETER_SELECTION_POWERS,
    parameter_selection_parameter_patterns: Sequence[str] = (),
    parameter_selection_exclude_patterns: Sequence[str] = (),
    parameter_selection_include_parameter_powers: bool = True,
    parameter_selection_max_train_rows: int | None = 300_000,
    parameter_selection_max_tune_rows: int | None = 150_000,
    parameter_selection_min_mpc_r2_gain: float = 0.005,
    parameter_selection_min_level_r2_gain: float = 0.001,
    parameter_selection_max_mpc_r2_loss_for_level_term: float = 0.001,
    parameter_selection_min_model_win_share: float = 0.55,
    parameter_selection_fdr: float = 0.10,
    parameter_selection_bootstrap_reps: int = 200,
    parameter_selection_bootstrap_confidence: float = 0.95,
    parameter_selection_max_terms: int = 0,
    categorical_state_power_terms: Sequence[
        tuple[str, str, int]
    ] = DEFAULT_CATEGORICAL_STATE_POWER_TERMS,
    categorical_slope_exclude_patterns: Sequence[
        str
    ] = DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS,
    spline_categorical_exclude_patterns: Sequence[
        str
    ] = DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS,
    rff_components: int = 512,
    ridge_alphas: Sequence[float] = (
        1.0e-4,
        1.0e-3,
        1.0e-2,
        0.1,
        1.0,
        10.0,
        100.0,
    ),
    target_parameterization: str = "accounting",
    consumption_transform: str = "identity",
    deposit_transform: str = "asinh",
    liquid_outflow_transform: str = "identity",
    mpc_loss_weight: float = 2.0,
    mpc_difference_weight: float | None = None,
    mpc_difference_weight_grid: Sequence[float] = (
        0.0,
        0.25,
        0.50,
        1.0,
        2.0,
        4.0,
        8.0,
    ),
    mpc_objective_max_pairs: int | None = 500_000,
    mpc_objective_max_iter: int = 200,
    mpc_objective_tol: float = 1.0e-6,
    mpc_objective_trim_quantile: float = 0.0,
    mpc_stratify_age_bins: Sequence[float] = (35.0, 45.0, 55.0, 60.0),
    mpc_stratify_liquid_bins: int = 4,
    mpc_stratify_illiquid_bins: int = 10,
    mpc_stratify_target_bins: int = 5,
    liquid_change_loss_weight: float = 1.0,
    deposit_reconciliation_loss_weight: float = 0.50,
    liquid_outflow_weight: float | None = None,
    state_columns: Sequence[str] = DEFAULT_STATE_COLUMNS,
    engineered_state_columns: Sequence[str] = DEFAULT_ENGINEERED_STATE_COLUMNS,
    redundant_state_columns: Sequence[str] = DEFAULT_REDUNDANT_STATE_COLUMNS,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    working_age_only: bool = True,
    validation_share: float = 0.20,
    tuning_share: float = 0.20,
    model_split_path: str | Path | None = None,
    random_state: int = 123,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict[str, Path]:
    total_start = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_path = (
        Path(model_split_path)
        if model_split_path is not None
        else output / "model_split.json"
    )
    evaluation_bundle_path = output / "policy_surrogate_holdout.joblib"
    _status("[stage 1/4] Building the sampled policy dataset...", verbose=verbose)

    dataset = build_policy_dataset(
        model_root,
        manifest_path=manifest_path,
        schema=schema,
        rows_per_model=rows_per_model,
        max_total_rows=max_total_rows,
        max_models=max_models,
        sampling_mode=sampling_mode,
        population_spec=population_spec,
        n_jobs=n_jobs,
        mpc_pair_share=mpc_pair_share,
        mpc_check_sizes=mpc_check_sizes,
        parameter_include=parameter_include,
        parameter_exclude=parameter_exclude,
        categorical_parameter_include=categorical_parameter_include,
        categorical_parameter_exclude=categorical_parameter_exclude,
        max_solver_distance=max_solver_distance,
        max_relative_distance=max_relative_distance,
        max_hjb_residual=max_hjb_residual,
        random_state=random_state,
        verbose=verbose,
        show_progress=show_progress,
    )
    _status("[stage 2/4] Saving the sampled dataset and catalog...", verbose=verbose)
    dataset_path = _write_dataframe(dataset.data, output / "policy_grid_sample.pkl.gz")
    catalog_path = output / "model_catalog.csv"
    dataset.catalog.to_csv(catalog_path, index=False)
    rejection_summary_path = output / "model_rejection_summary.csv"
    rejected_catalog = dataset.catalog[~dataset.catalog["usable"]].copy()
    if rejected_catalog.empty:
        pd.DataFrame(columns=["rejection_code", "n_models"]).to_csv(
            rejection_summary_path, index=False
        )
    else:
        (
            rejected_catalog.assign(
                rejection_code=rejected_catalog["rejection_code"]
                .replace("", "unspecified")
                .fillna("unspecified")
            )
            .groupby("rejection_code", as_index=False)
            .agg(
                n_models=("model_id", "size"),
                min_solver_distance=("solver_max_distance", "min"),
                median_solver_distance=("solver_max_distance", "median"),
                max_solver_distance=("solver_max_distance", "max"),
            )
            .to_csv(rejection_summary_path, index=False)
        )
    failures_path = output / "grid_load_failures.csv"
    dataset.failures.to_csv(failures_path, index=False)

    _status("[stage 3/4] Fitting and validating the surrogate...", verbose=verbose)
    bundle = fit_policy_surrogate(
        dataset.data,
        parameter_columns=dataset.parameter_columns,
        categorical_parameter_columns=dataset.categorical_parameter_columns,
        state_columns=state_columns,
        engineered_state_columns=engineered_state_columns,
        redundant_state_columns=redundant_state_columns,
        categorical_columns=categorical_columns,
        model_type=model_type,
        polynomial_degree=polynomial_degree,
        spline_columns=spline_columns,
        spline_n_knots=spline_n_knots,
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
        auto_select_parameter_terms=auto_select_parameter_terms,
        parameter_selection_spline_columns=parameter_selection_spline_columns,
        parameter_selection_powers=parameter_selection_powers,
        parameter_selection_parameter_patterns=(parameter_selection_parameter_patterns),
        parameter_selection_exclude_patterns=(parameter_selection_exclude_patterns),
        parameter_selection_include_parameter_powers=(
            parameter_selection_include_parameter_powers
        ),
        parameter_selection_max_train_rows=parameter_selection_max_train_rows,
        parameter_selection_max_tune_rows=parameter_selection_max_tune_rows,
        parameter_selection_min_mpc_r2_gain=(parameter_selection_min_mpc_r2_gain),
        parameter_selection_min_level_r2_gain=(parameter_selection_min_level_r2_gain),
        parameter_selection_max_mpc_r2_loss_for_level_term=(
            parameter_selection_max_mpc_r2_loss_for_level_term
        ),
        parameter_selection_min_model_win_share=(
            parameter_selection_min_model_win_share
        ),
        parameter_selection_fdr=parameter_selection_fdr,
        parameter_selection_bootstrap_reps=(parameter_selection_bootstrap_reps),
        parameter_selection_bootstrap_confidence=(
            parameter_selection_bootstrap_confidence
        ),
        parameter_selection_max_terms=parameter_selection_max_terms,
        categorical_state_power_terms=categorical_state_power_terms,
        categorical_slope_exclude_patterns=categorical_slope_exclude_patterns,
        spline_categorical_exclude_patterns=spline_categorical_exclude_patterns,
        rff_components=rff_components,
        ridge_alphas=ridge_alphas,
        target_parameterization=target_parameterization,
        consumption_transform=consumption_transform,
        deposit_transform=deposit_transform,
        liquid_outflow_transform=liquid_outflow_transform,
        mpc_loss_weight=mpc_loss_weight,
        mpc_difference_weight=mpc_difference_weight,
        mpc_difference_weight_grid=mpc_difference_weight_grid,
        mpc_objective_max_pairs=mpc_objective_max_pairs,
        mpc_objective_max_iter=mpc_objective_max_iter,
        mpc_objective_tol=mpc_objective_tol,
        mpc_objective_trim_quantile=mpc_objective_trim_quantile,
        mpc_stratify_age_bins=mpc_stratify_age_bins,
        mpc_stratify_liquid_bins=mpc_stratify_liquid_bins,
        mpc_stratify_illiquid_bins=mpc_stratify_illiquid_bins,
        mpc_stratify_target_bins=mpc_stratify_target_bins,
        liquid_change_loss_weight=liquid_change_loss_weight,
        deposit_reconciliation_loss_weight=deposit_reconciliation_loss_weight,
        liquid_outflow_weight=liquid_outflow_weight,
        money_units=schema.money_units,
        working_age_only=working_age_only,
        validation_share=validation_share,
        tuning_share=tuning_share,
        model_split_path=split_path,
        evaluation_bundle_path=evaluation_bundle_path,
        random_state=random_state,
        verbose=verbose,
        show_progress=show_progress,
    )
    _status("[stage 4/4] Saving fitted bundle and diagnostics...", verbose=verbose)
    evaluation_bundle = PolicySurrogateBundle.load(evaluation_bundle_path)
    usable_catalog = dataset.catalog[dataset.catalog["usable"]].copy()
    storage_counts = (
        {
            str(name): int(count)
            for name, count in usable_catalog["policy_storage_format"]
            .value_counts()
            .items()
        }
        if "policy_storage_format" in usable_catalog
        else {}
    )

    for fitted_bundle in (bundle, evaluation_bundle):
        fitted_bundle.training_metadata["policy_storage_counts"] = storage_counts
        fitted_bundle.training_metadata["require_compact_policies"] = bool(
            schema.require_compact_policies
        )
        fitted_bundle.training_metadata["money_units"] = schema.money_units
        fitted_bundle.training_metadata["schema_money_units"] = schema.money_units
        fitted_bundle.training_metadata["policy_layout"] = schema.policy_layout
        fitted_bundle.training_metadata["sampling_mode"] = str(sampling_mode)
        fitted_bundle.training_metadata["population_sampling_spec"] = asdict(
            population_spec
        )
        fitted_bundle.training_metadata["working_age_only"] = bool(working_age_only)
        fitted_bundle.training_metadata["sampling_workers"] = int(n_jobs)
        fitted_bundle.training_metadata["model_split_path"] = str(split_path)

    evaluation_bundle.save(evaluation_bundle_path)
    bundle_path = bundle.save(output / "policy_surrogate.joblib")
    metrics_path = output / "validation_metrics.csv"
    bundle.validation_metrics.to_csv(metrics_path, index=False)
    by_model_path = output / "validation_metrics_by_model.csv"
    bundle.validation_by_model.to_csv(by_model_path, index=False)
    parameter_search_path = output / "parameter_term_search.csv"
    parameter_search_rows = bundle.training_metadata.get("parameter_term_search", [])
    parameter_search_columns = [
        "candidate",
        "kind",
        "spline",
        "parameter_1",
        "parameter_2",
        "power",
        "forward_round",
        "selected_in_round",
        "selected",
        "screen_status",
        "n_features",
        "consumption_r2_gain",
        "mpc_r2_gain",
        "score_gain",
        "mpc_model_win_share",
        "mpc_p_value",
        "mpc_q_value",
    ]
    parameter_search_table = pd.DataFrame(parameter_search_rows)
    if parameter_search_table.empty:
        parameter_search_table = pd.DataFrame(columns=parameter_search_columns)
    else:
        ordered = [
            column
            for column in parameter_search_columns
            if column in parameter_search_table.columns
        ]
        remaining = [
            column for column in parameter_search_table.columns if column not in ordered
        ]
        parameter_search_table = parameter_search_table[ordered + remaining]
    parameter_search_table.to_csv(parameter_search_path, index=False)
    summary_path = output / "surrogate_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema": asdict(schema),
                "parameter_columns": dataset.parameter_columns,
                "categorical_parameter_columns": dataset.categorical_parameter_columns,
                "state_columns": list(state_columns),
                "engineered_state_columns": list(engineered_state_columns),
                "redundant_state_columns": list(redundant_state_columns),
                "categorical_columns": list(categorical_columns),
                "rows_per_model": dataset.rows_per_model,
                "sampling_mode": str(sampling_mode),
                "population_sampling_spec": asdict(population_spec),
                "sampling_workers": int(n_jobs),
                "mpc_pair_share": float(mpc_pair_share),
                "mpc_check_sizes": [float(x) for x in mpc_check_sizes],
                "spline_columns": list(spline_columns),
                "spline_n_knots": int(spline_n_knots),
                "spline_column_n_knots": {
                    str(column): int(count) for column, count in spline_column_n_knots
                },
                "spline_knot_strategies": {
                    str(column): str(strategy)
                    for column, strategy in spline_knot_strategies
                },
                "spline_knot_powers": {
                    str(column): float(power) for column, power in spline_knot_powers
                },
                "spline_tensor_pairs": [list(pair) for pair in spline_tensor_pairs],
                "spline_tensor_triples": [
                    list(triple) for triple in spline_tensor_triples
                ],
                "spline_categorical_tensor_pairs": [
                    list(term) for term in spline_categorical_tensor_pairs
                ],
                "spline_tensor_parameter_terms": [
                    list(term) for term in spline_tensor_parameter_terms
                ],
                "requested_parameter_power_terms": [
                    [parameter, int(power)]
                    for parameter, power in parameter_power_terms
                ],
                "requested_spline_parameter_power_terms": [
                    [spline, parameter, int(power)]
                    for spline, parameter, power in spline_parameter_power_terms
                ],
                "requested_spline_parameter_pair_terms": [
                    [spline, parameter_1, parameter_2]
                    for spline, parameter_1, parameter_2 in spline_parameter_pair_terms
                ],
                "selected_parameter_power_terms": bundle.training_metadata.get(
                    "parameter_power_terms", []
                ),
                "selected_spline_parameter_power_terms": bundle.training_metadata.get(
                    "spline_parameter_power_terms", []
                ),
                "selected_spline_parameter_pair_terms": bundle.training_metadata.get(
                    "spline_parameter_pair_terms", []
                ),
                "auto_select_parameter_terms": bool(auto_select_parameter_terms),
                "categorical_state_power_terms": [
                    [category, state, int(power)]
                    for category, state, power in categorical_state_power_terms
                ],
                "working_age_only": bool(working_age_only),
                "mpc_stratify_age_bins": [float(x) for x in mpc_stratify_age_bins],
                "mpc_stratify_liquid_bins": int(mpc_stratify_liquid_bins),
                "mpc_stratify_illiquid_bins": int(mpc_stratify_illiquid_bins),
                "mpc_stratify_target_bins": int(mpc_stratify_target_bins),
                "training_metadata": bundle.training_metadata,
                "feature_ranges": bundle.feature_ranges,
            },
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )
    _status(
        f"Training pipeline complete in {time.perf_counter() - total_start:,.1f}s.",
        verbose=verbose,
    )
    return {
        "bundle": bundle_path,
        "evaluation_bundle": evaluation_bundle_path,
        "model_split": split_path,
        "dataset": dataset_path,
        "catalog": catalog_path,
        "rejection_summary": rejection_summary_path,
        "failures": failures_path,
        "validation_metrics": metrics_path,
        "validation_by_model": by_model_path,
        "parameter_term_search": parameter_search_path,
        "summary": summary_path,
    }


def _inspection_cli() -> None:
    parser = argparse.ArgumentParser(description="Inspect saved HA policy grids.")
    parser.add_argument("model_root", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-models", type=int, default=3)
    args = parser.parse_args()
    dirs = discover_model_dirs(args.model_root, manifest_path=args.manifest)
    for d in dirs[: args.max_models]:
        print(json.dumps(inspect_saved_model(d), indent=2, default=str))


if __name__ == "__main__":
    _inspection_cli()
