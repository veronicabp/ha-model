"""Smooth, accounting-restricted surrogates for saved HA policy grids.

The module reads ``metadata.json`` and ``arrays.npz`` from many solved
heterogeneous-agent models, samples each policy grid with equal weight, and
estimates smooth policy maps over household states and model parameters.

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
import json
import math
import re
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import SplineTransformer
from scipy.sparse.linalg import LinearOperator, lsmr

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
    r"convergence_tol|ct_convergence_tol|derivative_floor|"
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
    r"liquid_col|illiquid_col|show_progress"
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
) -> tuple[bool, str, str]:
    """Return usability, a human-readable reason, and a stable reason code.

    ``max_solver_distance`` is an explicit exclusion rule.  The HA solver that
    generated the grids labels a model ``success`` even when it reaches the
    iteration cap above its convergence tolerance, so a strict distance cutoff
    can legitimately exclude many otherwise readable model directories.
    """

    diag = metadata.get("diagnostics", {}) or {}
    status = diag.get("status")
    if status is not None:
        accepted = {str(s).lower() for s in accepted_statuses}
        if str(status).lower() not in accepted:
            return False, f"solver status={status!r}", "solver_status"

    if max_solver_distance is not None:
        distance = diag.get("max_distance")
        if distance is not None:
            try:
                distance_f = float(distance)
            except (TypeError, ValueError):
                distance_f = np.nan
            if np.isfinite(distance_f) and distance_f > float(max_solver_distance):
                return (
                    False,
                    f"max solver distance {distance_f:.8g} exceeds "
                    f"threshold {float(max_solver_distance):.8g}",
                    "solver_distance",
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
        liquid_drift_key = _find_array_key(
            z, "liquid_drift", schema, required=False
        )
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
        and storage.get("policy_storage_format")
        != "compact_consumption_deposit"
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
        baseline_quantiles = (np.arange(n_pairs, dtype=float) + rng.random(n_pairs)) / n_pairs
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
        "acct__ct_time_step": float(cfg.get("ct_time_step", 1.0)),
        "acct__rb_pos_ct": float(rb_pos),
        "acct__rb_neg_ct": float(rb_neg),
        "acct__ra_ct": float(ra),
        "acct__chi0": float(chi0),
        "acct__chi1": float(chi1),
        "acct__xi": float(xi),
        "acct__illiquid_cost_floor": float(cfg.get("ct_illiquid_cost_floor", 1.0e-6))
        * float(money_factor),
        "acct__policy_drift_clip": float(cfg.get("ct_policy_drift_clip", 0.0)),
        "acct__retirement_age": float(cfg.get("retirement_age", np.inf)),
        "acct__terminal_age": float(cfg.get("max_age", np.inf)),
        "acct__tax_kind": str(tax.get("kind", "none")),
        "acct__tax_flat_rate": float(tax.get("flat_rate", 0.0)),
        "acct__tax_deduction": float(tax.get("deduction", 0.0)) * float(money_factor),
        "acct__tax_payroll_rate": float(tax.get("payroll_rate", 0.0)),
        "acct__tax_progressive_rate": float(tax.get("progressive_rate", 0.0)),
        "acct__tax_progressive_exponent": float(tax.get("progressive_exponent", 1.0)),
        "acct__tax_cap": float(tax.get("tax_cap", 1.0)),
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
    liquid_index: np.ndarray,
    illiquid_index: np.ndarray,
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
    ib = np.asarray(liquid_index, dtype=np.int64)
    ia = np.asarray(illiquid_index, dtype=np.int64)

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

    # Match policy_update exactly at numerical state boundaries.
    sb = np.where(ib == 0, np.maximum(sb, 0.0), sb)
    sb = np.where(ib == len(liquid_grid) - 1, np.minimum(sb, 0.0), sb)
    sa = np.where(ia == 0, np.maximum(sa, 0.0), sa)
    sa = np.where(ia == len(illiquid_grid) - 1, np.minimum(sa, 0.0), sa)

    dt = float(accounting["acct__ct_time_step"])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"Invalid ct_time_step={dt!r} in metadata.")

    b_lo = float(liquid_grid[0]) * float(money_factor)
    b_hi = float(liquid_grid[-1]) * float(money_factor)
    a_lo = float(illiquid_grid[0]) * float(money_factor)
    a_hi = float(illiquid_grid[-1]) * float(money_factor)
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


def sample_one_saved_policy_grid(
    model_dir: str | Path,
    *,
    n_rows: int,
    parameter_values: Mapping[str, Any],
    schema: GridSchema = GridSchema(),
    random_state: int = 0,
    include_optional_policies: bool = True,
    mpc_pair_share: float = 0.40,
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


def build_policy_dataset(
    model_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    schema: GridSchema = GridSchema(),
    rows_per_model: int = 2_000,
    max_total_rows: int = 250_000,
    max_models: int | None = None,
    mpc_pair_share: float = 0.40,
    mpc_check_sizes: Sequence[float] = (0.02, 0.05, 0.10, 0.188679, 0.20, 0.50),
    parameter_include: Sequence[str] | None = None,
    parameter_exclude: Sequence[str] = (),
    categorical_parameter_include: Sequence[str] | None = None,
    categorical_parameter_exclude: Sequence[str] = (),
    max_solver_distance: float | None = None,
    random_state: int = 123,
    output_path: str | Path | None = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> PolicyDatasetResult:
    """Construct a balanced sample from many saved policy grids.

    Every parameterization contributes the same number of rows.  This prevents
    models with denser numerical grids from receiving more statistical weight.
    """

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
        _status(f"  policy storage across all discovered models: {storage_text}", verbose=verbose)

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
        f"[data 3/3] Sampling {effective_rows:,} policy points from each "
        f"of {n_models:,} models (cap: {max_total_rows:,} rows)...",
        verbose=verbose,
    )
    rng = np.random.default_rng(random_state)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    usable_rows = list(usable.iterrows())
    iterator = _progress_iter(
        usable_rows,
        total=n_models,
        description="Sampling grids",
        unit="model",
        enabled=show_progress,
    )
    print_every = max(1, n_models // 10)
    for position, (i, row) in enumerate(iterator, start=1):
        if (
            show_progress
            and _tqdm is None
            and verbose
            and (position == 1 or position == n_models or position % print_every == 0)
        ):
            _status(
                f"  sampled {position:,}/{n_models:,} models",
                verbose=verbose,
            )
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
        try:
            frame = sample_one_saved_policy_grid(
                row["model_dir"],
                n_rows=effective_rows,
                parameter_values=params,
                schema=schema,
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
                mpc_pair_share=mpc_pair_share,
                mpc_check_sizes=mpc_check_sizes,
            )
            frames.append(frame)
        except Exception as exc:
            failures.append(
                {
                    "model_id": str(row["model_id"]),
                    "model_dir": str(row["model_dir"]),
                    "reason": str(exc),
                }
            )

    if not frames:
        failure_table = pd.DataFrame(failures)
        details = failure_table[["model_dir", "reason"]].to_string(index=False)
        raise RuntimeError(
            "Every saved grid failed to load. The loader errors were:\n" + details
        )
    data = pd.concat(frames, ignore_index=True, sort=False)
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
DEFAULT_ENGINEERED_STATE_COLUMNS = (
    "liquid_slack",
    "liquid_assets_to_income",
    "liquid_slack_to_income",
    "illiquid_assets_to_income",
    "after_tax_income_to_income",
    "cash_on_hand_to_income",
    # Lifecycle-horizon variables are critical for MPCs.  The true finite-horizon
    # policy becomes much steeper close to the terminal age, especially after
    # retirement.  Age alone plus a retired dummy cannot represent that shape.
    "years_to_terminal",
    "years_since_retirement",
    "years_to_retirement_positive",
)

# Current gross income is retained as an input/display variable, but is redundant
# in the regression once after-tax income, the persistent labor-income state,
# and the tax parameters are present.  Excluding it materially reduces
# collinearity without changing how the visualization is controlled.
DEFAULT_REDUNDANT_STATE_COLUMNS = ("current_income",)

DEFAULT_SPLINE_COLUMNS = (
    "liquid_slack_to_income",
    "illiquid_assets_to_income",
    "years_to_terminal",
)

# A tensor-product spline lets the liquid-wealth slope vary smoothly with the
# remaining lifecycle horizon.  This replaces the earlier global retiree slope
# shift that generated an artificial second MPC peak.
DEFAULT_SPLINE_TENSOR_PAIRS = (
    ("liquid_slack_to_income", "years_to_terminal"),
)

# Keep is_retired as a level indicator, but do not allow it to create an
# unrestricted discrete jump in every state slope or every liquid spline.
DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS = ("is_retired",)
DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS = ("is_retired",)

# Only parameters that plausibly change the curvature of consumption in liquid
# wealth receive spline interactions.  Interacting every spline with every
# metadata field created a large, weakly regularized basis and produced unstable
# MPC derivatives in earlier versions.
DEFAULT_SPLINE_PARAMETER_PATTERNS = (
    "discount_factor",
    "risk_aversion",
    "borrowing_limit",
    "borrowing_interest_rate",
    "liquid_interest_rate",
    "illiquid_interest_rate",
    "innovation_std",
    "persistence",
    "job_loss_probability",
    "job_finding_probability",
    "unemployment_replacement_rate",
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


class StructuredPolynomialMap:
    """Fully standardized polynomial, spline, and lifecycle tensor basis.

    The crucial addition is a tensor-product spline between liquid slack and
    years to the terminal age.  It lets the MPC vary smoothly over the remaining
    horizon instead of imposing a nearly constant retiree MPC plus a discrete
    retirement jump.  ``is_retired`` remains a level dummy but is excluded from
    category-specific slopes and spline interactions by default.
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
        spline_n_knots: int = 6,
        spline_degree: int = 3,
        spline_parameter_interactions: bool = True,
        spline_categorical_interactions: bool = True,
        spline_categorical_exclude_patterns: Sequence[str] = (
            DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS
        ),
        spline_parameter_patterns: Sequence[str] = DEFAULT_SPLINE_PARAMETER_PATTERNS,
        spline_tensor_pairs: Sequence[tuple[str, str]] = DEFAULT_SPLINE_TENSOR_PAIRS,
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
        self.final_standardize = bool(final_standardize)
        self.scaler_sample_rows = int(max(1, scaler_sample_rows))
        self.random_state = int(random_state)

        self.encoder = StandardizedFeatureEncoder(spec)
        self.spline_transformers_: dict[str, SplineTransformer] = {}
        self.spline_feature_names_: list[str] = []
        self.spline_slices_: dict[str, slice] = {}
        self.active_spline_tensor_pairs_: list[tuple[str, str]] = []
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
        for col in self.spline_columns:
            values = self._raw_filled_column(X, col)
            if np.unique(np.round(values.ravel(), 12)).size < 4:
                continue
            transformer = SplineTransformer(
                n_knots=self.spline_n_knots,
                degree=self.spline_degree,
                knots="quantile",
                extrapolation="linear",
                include_bias=False,
            )
            transformer.fit(values)
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

        names = state_names + parameter_names + cat_names
        numeric_names = state_names + parameter_names
        if self.degree >= 2:
            names += [f"{x}^2" for x in numeric_names]
        if self.degree >= 3:
            names += [f"{x}^3" for x in numeric_names]
        if self.degree >= 4:
            names += [f"{x}^4" for x in numeric_names]
        if self.state_state_interactions:
            names += [
                f"{state_names[i]}*{state_names[j]}"
                for i in range(len(state_names))
                for j in range(i + 1, len(state_names))
            ]
        if self.state_parameter_interactions:
            names += [f"{state}*{parameter}" for state in state_names for parameter in parameter_names]
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
        for left, right in self.active_spline_tensor_pairs_:
            left_names = self.spline_feature_names_[self.spline_slices_[left]]
            right_names = self.spline_feature_names_[self.spline_slices_[right]]
            names += [
                f"tensor({left_name},{right_name})"
                for left_name in left_names
                for right_name in right_names
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
        if self.state_state_interactions and state.shape[1] > 1:
            blocks.extend(
                state[:, i : i + 1] * state[:, j : j + 1]
                for i in range(state.shape[1])
                for j in range(i + 1, state.shape[1])
            )
        if self.state_parameter_interactions and params.shape[1] > 0:
            blocks.append(
                (state[:, :, None] * params[:, None, :]).reshape(len(X), -1)
            )
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
                    (spline[:, :, None] * selected_cats[:, None, :]).reshape(
                        len(X), -1
                    )
                )
            for left, right in self.active_spline_tensor_pairs_:
                left_block = spline[:, self.spline_slices_[left]]
                right_block = spline[:, self.spline_slices_[right]]
                blocks.append(
                    (left_block[:, :, None] * right_block[:, None, :]).reshape(
                        len(X), -1
                    )
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
        self.feature_names_ = [
            f"rbf_{i}" for i in range(self.n_components)
        ]
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
            return np.concatenate([rff, Z], axis=1).astype(
                np.float32, copy=False
            )
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
    after_tax = gross - taxes
    out["labor_income"] = labor
    out["pension_income"] = pension
    out["unemployment_benefits"] = unemployment
    out["taxes"] = taxes
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
    out["liquid_slack_to_income"] = liquid_slack / income_scale
    out["illiquid_assets_to_income"] = illiquid / income_scale
    out["after_tax_income_to_income"] = after_tax / income_scale
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
        + 0.5
        * np.asarray(chi1, dtype=float)
        * d**2
        / np.maximum(a, floor)
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
    c = pd.to_numeric(
        primitive_predictions["consumption"], errors="coerce"
    ).to_numpy(dtype=float)
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
    out["liquid_asset_projection_binding"] = (
        np.abs(next_b - next_b_unclipped) > 1.0e-10
    )
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
        if version != 6:
            raise TypeError(
                "This bundle predates the lifecycle-aware MPC feature format. "
                "Retrain it with the current code."
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

    required = {"model_id", "mpc_pair_id", "mpc_pair_role", "mpc_check_units", "consumption"}
    if not required.issubset(data.columns):
        return pd.DataFrame()

    frame = data.reset_index(drop=True)
    pair_id = pd.to_numeric(frame["mpc_pair_id"], errors="coerce").to_numpy(dtype=float)
    role = pd.to_numeric(frame["mpc_pair_role"], errors="coerce").to_numpy(dtype=float)
    check = pd.to_numeric(frame["mpc_check_units"], errors="coerce").to_numpy(dtype=float)
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
        "years_to_terminal",
        "years_since_retirement",
        "liquid_slack_to_income",
    ):
        if column in frame:
            temp_data[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)

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
        baseline["check"].to_numpy(dtype=float)
        + treated["check"].to_numpy(dtype=float)
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
        ) / pair_check,
        "predicted_mpc": (
            treated["predicted_consumption"].to_numpy(dtype=float)
            - baseline["predicted_consumption"].to_numpy(dtype=float)
        ) / pair_check,
    }
    for column in (
        "age",
        "is_retired",
        "years_to_terminal",
        "years_since_retirement",
        "liquid_slack_to_income",
    ):
        if column in baseline:
            out_data[column] = baseline[column].to_numpy(dtype=float)

    return pd.DataFrame(out_data).loc[good].reset_index(drop=True)


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
    age_bins: Sequence[float] = (45.0, 65.0, 70.0, 75.0),
    liquid_bins: int = 4,
) -> pd.DataFrame:
    """Subsample pairs by model × lifecycle region × liquid-wealth region."""

    if pairs.empty:
        return pairs.copy()

    work = pairs.reset_index(drop=True).copy()
    age_edges = [-np.inf, *sorted(float(x) for x in age_bins), np.inf]
    age_values = pd.to_numeric(
        work["age"] if "age" in work else pd.Series(np.nan, index=work.index),
        errors="coerce",
    )
    work["__age_bin"] = pd.cut(
        age_values,
        bins=age_edges,
        labels=False,
        include_lowest=True,
    ).fillna(-1).astype(int)
    retired = pd.to_numeric(
        work["is_retired"]
        if "is_retired" in work
        else pd.Series(np.nan, index=work.index),
        errors="coerce",
    ).fillna(-1).astype(int)
    work["__retired"] = retired

    liquid = pd.to_numeric(
        work["liquid_slack_to_income"]
        if "liquid_slack_to_income" in work
        else pd.Series(np.nan, index=work.index),
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
            work["__liquid_bin"] = pd.cut(
                liquid,
                bins=quantiles,
                labels=False,
                include_lowest=True,
                duplicates="drop",
            ).fillna(-1).astype(int)
        else:
            work["__liquid_bin"] = 0
    else:
        work["__liquid_bin"] = 0

    work["__mpc_stratum"] = list(
        zip(work["__retired"], work["__age_bin"], work["__liquid_bin"])
    )

    if max_pairs is None or max_pairs <= 0 or len(work) <= int(max_pairs):
        return work.drop(columns=[c for c in work if c.startswith("__")]).reset_index(drop=True)

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
    age_bins: Sequence[float] = (45.0, 65.0, 70.0, 75.0),
    liquid_bins: int = 4,
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
    )

    baseline = pairs['baseline_position'].to_numpy(dtype=np.int64)
    treated = pairs['treated_position'].to_numpy(dtype=np.int64)
    checks = pairs['check_units'].to_numpy(dtype=float)
    targets = pairs['exact_mpc'].to_numpy(dtype=float)

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
    model_ids = pairs.loc[valid, 'model_id'].astype(str).to_numpy(dtype=object)

    q = float(trim_quantile)
    if q < 0.0 or q >= 0.5:
        raise ValueError('mpc_objective_trim_quantile must be in [0, 0.5).')
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
        n_available=n_available,
    )


@dataclass
class JointConsumptionMPCRegressor:
    """Ridge regression fitted jointly to consumption levels and exact MPCs.

    The estimator minimizes

        sum_i (c_i - intercept - Z_i beta)^2
        + w * N/P * sum_p (mpc_p - D_p beta)^2
        + alpha * ||beta||^2,

    where ``D_p = (Z_treated - Z_baseline) / check``.  The intercept cancels
    from the MPC equations and is never ridge-penalized.  A matrix-free LSMR
    solve avoids copying the full level design matrix when derivative rows are
    added.
    """

    alpha: float
    mpc_weight: float
    max_iter: int = 200
    tol: float = 1.0e-6
    coef_: np.ndarray | None = None
    intercept_: float | None = None
    solver_diagnostics_: dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        Z: np.ndarray,
        y: np.ndarray,
        pair_design: MPCPairDesign,
    ) -> 'JointConsumptionMPCRegressor':
        Z_arr = np.asarray(Z, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float64)
        if Z_arr.ndim != 2 or len(y_arr) != len(Z_arr):
            raise ValueError('Z and y have incompatible dimensions.')
        if pair_design is None or pair_design.n_pairs == 0:
            raise ValueError('Joint MPC fitting requires at least one valid pair.')
        if self.alpha < 0.0 or self.mpc_weight <= 0.0:
            raise ValueError('alpha must be nonnegative and mpc_weight positive.')
        if self.max_iter <= 0 or self.tol <= 0.0:
            raise ValueError('max_iter and tol must be positive.')

        D = np.asarray(pair_design.difference, dtype=np.float32)
        m = np.asarray(pair_design.target, dtype=np.float64)
        if D.ndim != 2 or D.shape[1] != Z_arr.shape[1] or len(D) != len(m):
            raise ValueError('MPC pair design has incompatible dimensions.')

        n_level, n_features = Z_arr.shape
        n_pairs = len(m)
        z_mean = np.mean(Z_arr, axis=0, dtype=np.float64)
        y_mean = float(np.mean(y_arr))
        y_centered = y_arr - y_mean

        # Scale the pair block so mpc_weight compares average MPC loss with
        # average level loss and does not depend on how many pair rows happen
        # to be sampled.
        pair_scale = math.sqrt(float(self.mpc_weight) * n_level / n_pairs)

        def matvec(beta: np.ndarray) -> np.ndarray:
            beta = np.asarray(beta, dtype=np.float64)
            centered_level = Z_arr @ beta - float(z_mean @ beta)
            pair_values = pair_scale * (D @ beta)
            return np.concatenate([centered_level, pair_values])

        def rmatvec(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            level_values = values[:n_level]
            pair_values = values[n_level:]
            level_part = Z_arr.T @ level_values - z_mean * float(level_values.sum())
            pair_part = pair_scale * (D.T @ pair_values)
            return np.asarray(level_part + pair_part, dtype=np.float64)

        operator = LinearOperator(
            shape=(n_level + n_pairs, n_features),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )
        rhs = np.concatenate([y_centered, pair_scale * m])
        solution = lsmr(
            operator,
            rhs,
            damp=math.sqrt(float(self.alpha)),
            atol=float(self.tol),
            btol=float(self.tol),
            maxiter=int(self.max_iter),
        )

        coef = np.asarray(solution[0], dtype=np.float64)
        self.coef_ = coef
        self.intercept_ = float(y_mean - z_mean @ coef)
        self.solver_diagnostics_ = {
            'istop': int(solution[1]),
            'iterations': int(solution[2]),
            'residual_norm': float(solution[3]),
            'normal_residual_norm': float(solution[4]),
            'operator_norm': float(solution[5]),
            'condition_number': float(solution[6]),
            'coefficient_norm': float(solution[7]),
            'n_level_rows': int(n_level),
            'n_mpc_pairs': int(n_pairs),
            'mpc_pairs_available': int(pair_design.n_available),
            'mpc_weight': float(self.mpc_weight),
            'alpha': float(self.alpha),
        }
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError('The joint consumption/MPC regressor is not fitted.')
        return float(self.intercept_) + np.asarray(Z, dtype=np.float64) @ self.coef_

    def predict_mpc(
        self,
        Z_baseline: np.ndarray,
        Z_treated: np.ndarray,
        check_sizes: np.ndarray,
    ) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError('The joint consumption/MPC regressor is not fitted.')
        checks = np.asarray(check_sizes, dtype=np.float64)
        if np.any(~np.isfinite(checks)) or np.any(checks <= 0.0):
            raise ValueError('check_sizes must be finite and positive.')
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
        model = Ridge(alpha=float(alpha), fit_intercept=True, solver='lsqr')
        model.fit(Z, y)
        return model

    model = JointConsumptionMPCRegressor(
        alpha=float(alpha),
        mpc_weight=float(mpc_weight),
        max_iter=int(max_iter),
        tol=float(tol),
    )
    model.fit(Z, y, pair_design)
    return model


def _mpc_metric_row_from_pairs(
    data: pd.DataFrame,
    pred: pd.DataFrame,
    *,
    split: str,
    model_id: str | None = None,
) -> dict[str, Any] | None:
    pairs = _paired_mpc_table(data, pred["consumption"].to_numpy(dtype=float))
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
    for model_id, indices in data.groupby("model_id").groups.items():
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
        ids = block["model_id"].astype(str).to_numpy()
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
    spline_n_knots: int = 6,
    spline_tensor_pairs: Sequence[tuple[str, str]] = DEFAULT_SPLINE_TENSOR_PAIRS,
    categorical_slope_exclude_patterns: Sequence[str] = DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS,
    spline_categorical_exclude_patterns: Sequence[str] = DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS,
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
    mpc_loss_weight: float = 1.0,
    mpc_difference_weight: float | None = None,
    mpc_difference_weight_grid: Sequence[float] = (
        0.0,
        0.05,
        0.10,
        0.25,
        0.50,
        1.0,
        2.0,
    ),
    mpc_objective_max_pairs: int | None = 250_000,
    mpc_objective_max_iter: int = 200,
    mpc_objective_tol: float = 1.0e-6,
    mpc_objective_trim_quantile: float = 0.0,
    mpc_stratify_age_bins: Sequence[float] = (45.0, 65.0, 70.0, 75.0),
    mpc_stratify_liquid_bins: int = 4,
    liquid_change_loss_weight: float = 1.0,
    deposit_reconciliation_loss_weight: float = 0.50,
    liquid_outflow_weight: float | None = None,
    money_units: str = "model",
    validation_share: float = 0.20,
    tuning_share: float = 0.20,
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
    if consumption_transform != "identity" and any(x > 0.0 for x in candidate_mpc_weights):
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

    if categorical_parameters:
        model_table = (
            fit_data[["model_id", *categorical_parameters]]
            .drop_duplicates(subset=["model_id"])
            .reset_index(drop=True)
        )
        development_ids, test_ids = _stratified_model_holdout(
            model_table,
            share=validation_share,
            category_columns=categorical_parameters,
            random_state=random_state,
            label="test",
        )
        development = fit_data[
            fit_data["model_id"].astype(str).isin(development_ids)
        ].copy()
        test = fit_data[fit_data["model_id"].astype(str).isin(test_ids)].copy()
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
        train = development[development["model_id"].astype(str).isin(train_ids)].copy()
        tune = development[development["model_id"].astype(str).isin(tune_ids)].copy()
    else:
        test_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_share,
            random_state=random_state,
        )
        development_idx, test_idx = next(test_splitter.split(fit_data, groups=groups))
        development = fit_data.iloc[development_idx].copy()
        test = fit_data.iloc[test_idx].copy()
        development_groups = development["model_id"].astype(str).to_numpy()
        if np.unique(development_groups).size < 2:
            raise ValueError(
                "The requested test share leaves fewer than two development "
                "parameterizations; reduce validation_share."
            )
        tune_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=tuning_share,
            random_state=random_state + 1,
        )
        train_rel, tune_rel = next(
            tune_splitter.split(development, groups=development_groups)
        )
        train = development.iloc[train_rel].copy()
        tune = development.iloc[tune_rel].copy()

    train = train.reset_index(drop=True)
    tune = tune.reset_index(drop=True)
    test = test.reset_index(drop=True)
    development = pd.concat([train, tune], axis=0, ignore_index=True)

    _status(
        "  model split: "
        f"train={train['model_id'].nunique():,}, "
        f"tune={tune['model_id'].nunique():,}, "
        f"test={test['model_id'].nunique():,}",
        verbose=verbose,
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
                spline_degree=3,
                spline_parameter_interactions=True,
                spline_categorical_interactions=True,
                spline_categorical_exclude_patterns=tuple(
                    spline_categorical_exclude_patterns
                ),
                spline_tensor_pairs=tuple(spline_tensor_pairs),
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
        y_tune_levels[target] = pd.to_numeric(
            tune[target], errors="coerce"
        ).to_numpy(dtype=float)

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
        weights = candidate_mpc_weights if target == 'consumption' else [0.0]
        for alpha in ridge_alphas:
            for direct_weight in weights:
                tasks.append((target, float(alpha), float(direct_weight)))

    iterator = _progress_iter(
        tasks,
        total=len(tasks),
        description='Ridge/MPC tuning',
        unit='fit',
        enabled=show_progress,
    )

    for position, (target, alpha, direct_weight) in enumerate(iterator, start=1):
        if target == 'consumption':
            model = _fit_consumption_regressor(
                Z_train,
                y_train_transformed[target],
                alpha=alpha,
                mpc_weight=direct_weight,
                pair_design=train_mpc_design,
                max_iter=mpc_objective_max_iter,
                tol=mpc_objective_tol,
            )
        else:
            model = Ridge(alpha=alpha, fit_intercept=True, solver='lsqr')
            model.fit(Z_train, y_train_transformed[target])

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

        if target == 'consumption':
            pairs = _paired_mpc_table(tune, pred)
            if not pairs.empty:
                exact_mpc = pairs['exact_mpc'].to_numpy(dtype=float)
                predicted_mpc = pairs['predicted_mpc'].to_numpy(dtype=float)
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

        solver_info = getattr(model, 'solver_diagnostics_', {}) or {}
        alpha_rows.append(
            {
                'target': target,
                'alpha': alpha,
                'mpc_difference_weight': direct_weight,
                'tuning_nrmse_sd': level_nrmse,
                'tuning_mpc_rmse': mpc_rmse,
                'tuning_mpc_nrmse': mpc_nrmse,
                'tuning_mpc_bias': mpc_bias,
                'tuning_mpc_r2': mpc_r2,
                'tuning_score': combined_score,
                'n_mpc_tuning_pairs': n_mpc_pairs,
                'joint_solver_iterations': solver_info.get('iterations'),
                'joint_solver_stop_code': solver_info.get('istop'),
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
                alpha=f'{alpha:g}',
                mpc_w=f'{direct_weight:g}',
                score=f'{combined_score:.4f}',
            )

    for target in ALL_FIT_TARGETS:
        suffix = (
            f", direct MPC weight={best_direct_mpc_weights[target]:g}"
            if target == 'consumption'
            else ''
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
        if target == "consumption":
            model = _fit_consumption_regressor(
                Z_train,
                y_train_transformed[target],
                alpha=best_alphas[target],
                mpc_weight=best_direct_mpc_weights[target],
                pair_design=train_mpc_design,
                max_iter=mpc_objective_max_iter,
                tol=mpc_objective_tol,
            )
        else:
            model = Ridge(
                alpha=best_alphas[target],
                fit_intercept=True,
                solver="lsqr",
            )
            model.fit(Z_train, y_train_transformed[target])
        tuning_regressions[target] = model
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
    )
    evaluation_regressions: dict[str, Any] = {}
    evaluation_transforms: dict[str, TargetTransform] = {}
    for target in ALL_FIT_TARGETS:
        transform = TargetTransform(target_transform_names[target]).fit(
            pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        )
        y = transform.transform(
            pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        )
        if target == "consumption":
            model = _fit_consumption_regressor(
                Z_development,
                y,
                alpha=best_alphas[target],
                mpc_weight=best_direct_mpc_weights[target],
                pair_design=development_mpc_design,
                max_iter=mpc_objective_max_iter,
                tol=mpc_objective_tol,
            )
        else:
            model = Ridge(
                alpha=best_alphas[target],
                fit_intercept=True,
                solver="lsqr",
            )
            model.fit(Z_development, y)
        evaluation_regressions[target] = model
        evaluation_transforms[target] = transform

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
    )
    final_regressions: dict[str, Any] = {}
    final_transforms: dict[str, TargetTransform] = {}
    for target in ALL_FIT_TARGETS:
        transform = TargetTransform(target_transform_names[target]).fit(
            pd.to_numeric(fit_data[target], errors="coerce").to_numpy(dtype=float)
        )
        y = transform.transform(
            pd.to_numeric(fit_data[target], errors="coerce").to_numpy(dtype=float)
        )
        if target == "consumption":
            model = _fit_consumption_regressor(
                Z_all,
                y,
                alpha=best_alphas[target],
                mpc_weight=best_direct_mpc_weights[target],
                pair_design=final_mpc_design,
                max_iter=mpc_objective_max_iter,
                tol=mpc_objective_tol,
            )
        else:
            model = Ridge(
                alpha=best_alphas[target],
                fit_intercept=True,
                solver="lsqr",
            )
            model.fit(Z_all, y)
        final_regressions[target] = model
        final_transforms[target] = transform
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
        "bundle_format_version": 6,
        "model_type": model_type,
        "polynomial_degree": int(polynomial_degree),
        "spline_columns": list(spline_columns),
        "spline_n_knots": int(spline_n_knots),
        "spline_tensor_pairs": [list(pair) for pair in spline_tensor_pairs],
        "categorical_slope_exclude_patterns": list(
            categorical_slope_exclude_patterns
        ),
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
        "mpc_difference_weight_selected": float(
            best_direct_mpc_weights["consumption"]
        ),
        "mpc_difference_rows_used": bool(
            best_direct_mpc_weights["consumption"] > 0.0
        ),
        "mpc_objective_max_pairs": (
            None if mpc_objective_max_pairs is None else int(mpc_objective_max_pairs)
        ),
        "mpc_objective_max_iter": int(mpc_objective_max_iter),
        "mpc_objective_tol": float(mpc_objective_tol),
        "mpc_objective_trim_quantile": float(mpc_objective_trim_quantile),
        "mpc_stratify_age_bins": [float(x) for x in mpc_stratify_age_bins],
        "mpc_stratify_liquid_bins": int(mpc_stratify_liquid_bins),
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
    rows_per_model: int = 2_000,
    max_total_rows: int = 250_000,
    max_models: int | None = None,
    mpc_pair_share: float = 0.40,
    mpc_check_sizes: Sequence[float] = (0.02, 0.05, 0.10, 0.188679, 0.20, 0.50),
    parameter_include: Sequence[str] | None = None,
    parameter_exclude: Sequence[str] = (),
    categorical_parameter_include: Sequence[str] | None = None,
    categorical_parameter_exclude: Sequence[str] = (),
    max_solver_distance: float | None = None,
    model_type: str = "polynomial",
    polynomial_degree: int = 2,
    spline_columns: Sequence[str] = DEFAULT_SPLINE_COLUMNS,
    spline_n_knots: int = 6,
    spline_tensor_pairs: Sequence[tuple[str, str]] = DEFAULT_SPLINE_TENSOR_PAIRS,
    categorical_slope_exclude_patterns: Sequence[str] = DEFAULT_CATEGORICAL_SLOPE_EXCLUDE_PATTERNS,
    spline_categorical_exclude_patterns: Sequence[str] = DEFAULT_SPLINE_CATEGORICAL_EXCLUDE_PATTERNS,
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
    mpc_loss_weight: float = 1.0,
    mpc_difference_weight: float | None = None,
    mpc_difference_weight_grid: Sequence[float] = (
        0.0,
        0.05,
        0.10,
        0.25,
        0.50,
        1.0,
        2.0,
    ),
    mpc_objective_max_pairs: int | None = 250_000,
    mpc_objective_max_iter: int = 200,
    mpc_objective_tol: float = 1.0e-6,
    mpc_objective_trim_quantile: float = 0.0,
    mpc_stratify_age_bins: Sequence[float] = (45.0, 65.0, 70.0, 75.0),
    mpc_stratify_liquid_bins: int = 4,
    liquid_change_loss_weight: float = 1.0,
    deposit_reconciliation_loss_weight: float = 0.50,
    liquid_outflow_weight: float | None = None,
    state_columns: Sequence[str] = DEFAULT_STATE_COLUMNS,
    engineered_state_columns: Sequence[str] = DEFAULT_ENGINEERED_STATE_COLUMNS,
    redundant_state_columns: Sequence[str] = DEFAULT_REDUNDANT_STATE_COLUMNS,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_COLUMNS,
    validation_share: float = 0.20,
    tuning_share: float = 0.20,
    random_state: int = 123,
    verbose: bool = True,
    show_progress: bool = True,
) -> dict[str, Path]:
    total_start = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _status("[stage 1/4] Building the sampled policy dataset...", verbose=verbose)

    dataset = build_policy_dataset(
        model_root,
        manifest_path=manifest_path,
        schema=schema,
        rows_per_model=rows_per_model,
        max_total_rows=max_total_rows,
        max_models=max_models,
        mpc_pair_share=mpc_pair_share,
        mpc_check_sizes=mpc_check_sizes,
        parameter_include=parameter_include,
        parameter_exclude=parameter_exclude,
        categorical_parameter_include=categorical_parameter_include,
        categorical_parameter_exclude=categorical_parameter_exclude,
        max_solver_distance=max_solver_distance,
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
        spline_tensor_pairs=spline_tensor_pairs,
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
        liquid_change_loss_weight=liquid_change_loss_weight,
        deposit_reconciliation_loss_weight=deposit_reconciliation_loss_weight,
        liquid_outflow_weight=liquid_outflow_weight,
        money_units=schema.money_units,
        validation_share=validation_share,
        tuning_share=tuning_share,
        random_state=random_state,
        verbose=verbose,
        show_progress=show_progress,
    )
    _status("[stage 4/4] Saving fitted bundle and diagnostics...", verbose=verbose)
    if "policy_storage_format" in dataset.catalog:
        usable_catalog = dataset.catalog[dataset.catalog["usable"]].copy()
        bundle.training_metadata["policy_storage_counts"] = {
            str(name): int(count)
            for name, count in usable_catalog["policy_storage_format"]
            .value_counts()
            .items()
        }
    bundle.training_metadata["require_compact_policies"] = bool(
        schema.require_compact_policies
    )
    bundle.training_metadata["money_units"] = schema.money_units
    bundle.training_metadata["schema_money_units"] = schema.money_units
    bundle.training_metadata["policy_layout"] = schema.policy_layout
    bundle_path = bundle.save(output / "policy_surrogate.joblib")
    metrics_path = output / "validation_metrics.csv"
    bundle.validation_metrics.to_csv(metrics_path, index=False)
    by_model_path = output / "validation_metrics_by_model.csv"
    bundle.validation_by_model.to_csv(by_model_path, index=False)
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
                "mpc_pair_share": float(mpc_pair_share),
                "mpc_check_sizes": [float(x) for x in mpc_check_sizes],
                "spline_tensor_pairs": [list(pair) for pair in spline_tensor_pairs],
                "mpc_stratify_age_bins": [float(x) for x in mpc_stratify_age_bins],
                "mpc_stratify_liquid_bins": int(mpc_stratify_liquid_bins),
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
        "dataset": dataset_path,
        "catalog": catalog_path,
        "rejection_summary": rejection_summary_path,
        "failures": failures_path,
        "validation_metrics": metrics_path,
        "validation_by_model": by_model_path,
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
