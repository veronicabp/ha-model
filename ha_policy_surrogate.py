"""Smooth, accounting-restricted surrogates for saved HA policy grids.

The module reads ``metadata.json`` and ``arrays.npz`` from many solved
heterogeneous-agent models, samples each policy grid with equal weight, and
estimates smooth policy maps over household states and model parameters.

The primitive regression targets are

* consumption; and
* the net deposit from the liquid account into the illiquid account.

Next liquid assets, next illiquid assets, both asset changes, and both drifts are
then reconstructed from the HA model's budget equations.  They are never fitted
as independent choices.  This guarantees internal accounting consistency and
makes held-out liquid-asset performance directly interpretable.

The default feature map is a structured polynomial with ridge regularization. It
contains continuous main effects and powers, state-state interactions,
state-parameter interactions, one-hot categorical variables, and
category-specific slopes.  A smooth random-Fourier RBF alternative is also
available.

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
) -> tuple[bool, str]:
    diag = metadata.get("diagnostics", {}) or {}
    status = diag.get("status")
    if status is not None:
        accepted = {str(s).lower() for s in accepted_statuses}
        if str(status).lower() not in accepted:
            return False, f"solver status={status!r}"
    if max_solver_distance is not None:
        distance = diag.get("max_distance")
        if distance is not None and np.isfinite(float(distance)):
            if float(distance) > float(max_solver_distance):
                return False, f"max solver distance {distance} exceeds threshold"
    return True, ""


def build_model_catalog(
    model_dirs: Sequence[str | Path],
    *,
    schema: GridSchema = GridSchema(),
    max_solver_distance: float | None = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
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
            usable, reason = _model_is_usable(
                meta, max_solver_distance=max_solver_distance
            )
            numeric_params = extract_numeric_parameters(
                meta,
                include_derived=schema.include_derived_metadata_parameters,
            )
            categorical_params = extract_categorical_parameters(
                meta,
                include_derived=schema.include_derived_metadata_parameters,
            )
            row: dict[str, Any] = {
                "model_id": _model_id(meta, d),
                "model_dir": str(d),
                "usable": bool(usable),
                "reason": reason,
                "money_scale": float(meta.get("money_scale", 1.0) or 1.0),
            }
            row.update(numeric_params)
            row.update(categorical_params)
            rows.append(row)
        except Exception as exc:  # catalog all failures rather than aborting
            rows.append(
                {
                    "model_id": d.name,
                    "model_dir": str(d),
                    "usable": False,
                    "reason": f"metadata error: {exc}",
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
        "acct__retirement_age": float(cfg.get("retirement_age", np.inf)),
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
        if not np.isfinite(float(value)) and key != "acct__retirement_age":
            raise ValueError(f"Invalid accounting metadata {key}={value!r}.")
    return values


def sample_one_saved_policy_grid(
    model_dir: str | Path,
    *,
    n_rows: int,
    parameter_values: Mapping[str, Any],
    schema: GridSchema = GridSchema(),
    random_state: int = 0,
    include_optional_policies: bool = True,
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
    if next_liquid_key is None and liquid_drift_key is None:
        raise KeyError(
            "Could not find either a next-liquid-assets policy or liquid_drift. "
            f"Available arrays: {sorted(arrays)}"
        )
    if next_illiquid_key is None and illiquid_drift_key is None:
        raise KeyError(
            "Could not find either a next-illiquid-assets policy or illiquid_drift. "
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

    coords = _stratified_coordinates(canonical_shape, n_rows, rng)
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

    a_floor = max(float(accounting["acct__illiquid_cost_floor"]), EPS)
    adjustment_cost = float(accounting["acct__chi0"]) * np.abs(deposit) + 0.5 * float(
        accounting["acct__chi1"]
    ) * deposit**2 / np.maximum(current_a, a_floor)
    rb = np.where(
        current_b >= 0.0,
        float(accounting["acct__rb_pos_ct"]),
        float(accounting["acct__rb_neg_ct"]),
    )
    liquid_drift_identity = (
        (1.0 - float(accounting["acct__xi"])) * after_tax
        + rb * current_b
        - deposit
        - adjustment_cost
        - consumption
    )
    illiquid_drift_identity = (
        float(accounting["acct__ra_ct"]) * current_a
        + float(accounting["acct__xi"]) * after_tax
        + deposit
    )

    dt = float(accounting["acct__ct_time_step"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"Invalid ct_time_step={dt!r} in metadata.")

    if next_liquid_key is not None:
        next_b_model = _index_policy_array(
            arrays[next_liquid_key], schema.policy_layout, coords
        )
        next_b = np.asarray(next_b_model, dtype=float) * factor
    else:
        next_b = np.clip(
            current_b + dt * liquid_drift_identity,
            liquid_grid[0] * factor,
            liquid_grid[-1] * factor,
        )

    if next_illiquid_key is not None:
        next_a_model = _index_policy_array(
            arrays[next_illiquid_key], schema.policy_layout, coords
        )
        next_a = np.asarray(next_a_model, dtype=float) * factor
    else:
        next_a = np.clip(
            current_a + dt * illiquid_drift_identity,
            illiquid_grid[0] * factor,
            illiquid_grid[-1] * factor,
        )

    data: dict[str, Any] = {
        "model_id": np.repeat(model_id, n_rows),
        "model_dir": np.repeat(str(d), n_rows),
        "education_state": g.astype(int),
        "education": np.asarray([group_values[i] for i in g], dtype=object),
        "age": age_values,
        "years_to_retirement": retirement_age - age_values,
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
        "liquid_drift": liquid_drift_identity,
        "illiquid_drift": illiquid_drift_identity,
        "next_liquid_assets": next_b,
        "next_illiquid_assets": next_a,
        "delta_liquid_assets": next_b - current_b,
        "delta_illiquid_assets": next_a - current_a,
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
    if max_models is not None:
        dirs = dirs[: int(max_models)]
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
    effective_rows = min(int(rows_per_model), max(1, int(max_total_rows) // n_models))
    if effective_rows < 100:
        warnings.warn(
            f"Only {effective_rows} rows/model fit under max_total_rows. "
            "Increase max_total_rows for reliable state coverage.",
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


# Continuous state specification used by default.  After-tax income is included
# because it is the income flow that enters both asset-accounting equations.
# It is nevertheless recomputed from gross income and the selected tax policy
# before every prediction, so the interactive page cannot create a stale or
# internally inconsistent after-tax-income state.
DEFAULT_STATE_COLUMNS = (
    "age",
    "current_income",
    "after_tax_income",
    "labor_income_state",
    "liquid_assets",
    "illiquid_assets",
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
    "liquid_assets",
    "illiquid_assets",
    "total_assets",
    "cash_on_hand",
)


@dataclass(frozen=True)
class FeatureSpec:
    state_columns: tuple[str, ...] = DEFAULT_STATE_COLUMNS
    parameter_columns: tuple[str, ...] = ()
    categorical_state_columns: tuple[str, ...] = DEFAULT_CATEGORICAL_STATE_COLUMNS
    categorical_parameter_columns: tuple[str, ...] = ()
    signed_log_columns: tuple[str, ...] = DEFAULT_MONETARY_COLUMNS

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.categorical_state_columns + self.categorical_parameter_columns
            )
        )

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                self.state_columns + self.parameter_columns + self.categorical_columns
            )
        )


class StandardizedFeatureEncoder:
    """Median imputation, asinh scaling for money variables, and one-hot coding."""

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
        for col in self.spec.state_columns + self.spec.parameter_columns:
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
        state = self._numeric_block(X, self.spec.state_columns)
        params = self._numeric_block(X, self.spec.parameter_columns)

        cat_blocks: list[np.ndarray] = []
        cat_names: list[str] = []
        for col in self.spec.categorical_columns:
            values = X[col].where(X[col].notna(), "__missing__").astype(str).to_numpy()
            known = self.categories_[col]
            # Use standard treatment coding: the first observed category is the
            # baseline, and every remaining level receives a dummy.  This keeps
            # the intercept and category-specific slopes full rank while still
            # representing every category in the polynomial.
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


class StructuredPolynomialMap:
    """A controlled polynomial basis designed for HA policy functions."""

    def __init__(
        self,
        spec: FeatureSpec,
        *,
        degree: int = 2,
        state_state_interactions: bool = True,
        state_parameter_interactions: bool = True,
        parameter_parameter_interactions: bool = False,
        categorical_slopes: bool = True,
    ):
        if degree not in {1, 2, 3, 4}:
            raise ValueError("degree must be 1, 2, 3, or 4")
        self.spec = spec
        self.degree = int(degree)
        self.state_state_interactions = bool(state_state_interactions)
        self.state_parameter_interactions = bool(state_parameter_interactions)
        self.parameter_parameter_interactions = bool(parameter_parameter_interactions)
        self.categorical_slopes = bool(categorical_slopes)
        self.encoder = StandardizedFeatureEncoder(spec)
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame) -> "StructuredPolynomialMap":
        self.encoder.fit(X)
        _, _, _, cat_names = self.encoder.transform_parts(X.iloc[:1])
        s_names = list(self.spec.state_columns)
        p_names = list(self.spec.parameter_columns)
        names = s_names + p_names + cat_names
        if self.degree >= 2:
            names += [f"{x}^2" for x in s_names + p_names]
        if self.degree >= 3:
            names += [f"{x}^3" for x in s_names + p_names]
        if self.degree >= 4:
            names += [f"{x}^4" for x in s_names + p_names]
        if self.state_state_interactions:
            names += [
                f"{s_names[i]}*{s_names[j]}"
                for i in range(len(s_names))
                for j in range(i + 1, len(s_names))
            ]
        if self.state_parameter_interactions:
            names += [f"{s}*{p}" for s in s_names for p in p_names]
        if self.parameter_parameter_interactions:
            names += [
                f"{p_names[i]}*{p_names[j]}"
                for i in range(len(p_names))
                for j in range(i + 1, len(p_names))
            ]
        if self.categorical_slopes:
            names += [f"{cat}*{x}" for cat in cat_names for x in s_names + p_names]
        self.feature_names_ = names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
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
            blocks.append((state[:, :, None] * params[:, None, :]).reshape(len(X), -1))
        if self.parameter_parameter_interactions and params.shape[1] > 1:
            blocks.extend(
                params[:, i : i + 1] * params[:, j : j + 1]
                for i in range(params.shape[1])
                for j in range(i + 1, params.shape[1])
            )
        if self.categorical_slopes and cats.shape[1] > 0 and numeric.shape[1] > 0:
            blocks.append((cats[:, :, None] * numeric[:, None, :]).reshape(len(X), -1))
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

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
    out.setdefault("acct__retirement_age", np.inf)
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

    # Restore any still-missing feature columns after the accounting update.
    for column in feature_spec.required_columns:
        if column not in out:
            out[column] = default_values.get(column, "__missing__")
    return out


def reconstruct_policy_outputs(
    X: pd.DataFrame,
    primitive_predictions: pd.DataFrame,
    *,
    accounting_defaults: Mapping[str, Any],
    money_units: str,
    output_bounds: Mapping[str, tuple[float, float]] | None = None,
    project: bool = True,
) -> pd.DataFrame:
    """Enforce the HA model's accounting identities exactly."""

    if "liquid_assets" not in X or "illiquid_assets" not in X:
        raise KeyError(
            "Accounting reconstruction requires liquid_assets and illiquid_assets."
        )
    if "after_tax_income" not in X:
        raise KeyError("Accounting reconstruction requires after_tax_income.")
    if not set(PRIMITIVE_POLICY_TARGETS).issubset(primitive_predictions.columns):
        raise KeyError("Primitive predictions must contain consumption and deposit.")

    b = pd.to_numeric(X["liquid_assets"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(X["illiquid_assets"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(X["after_tax_income"], errors="coerce").to_numpy(dtype=float)
    c = pd.to_numeric(primitive_predictions["consumption"], errors="coerce").to_numpy(
        dtype=float
    )
    d = pd.to_numeric(primitive_predictions["deposit"], errors="coerce").to_numpy(
        dtype=float
    )
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

    adjustment_cost = chi0 * np.abs(d) + 0.5 * chi1 * d**2 / np.maximum(a, a_floor)
    liquid_return = np.where(b >= 0.0, rb_pos * b, rb_neg * b)
    liquid_drift = (1.0 - xi) * y + liquid_return - d - adjustment_cost - c
    illiquid_drift = ra * a + xi * y + d
    next_b_unclipped = b + dt * liquid_drift
    next_a_unclipped = a + dt * illiquid_drift

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

    if project:
        next_b = np.minimum(np.maximum(next_b_unclipped, liq_lo), liq_hi)
        next_a = np.minimum(np.maximum(next_a_unclipped, ill_lo), ill_hi)
    else:
        next_b = next_b_unclipped
        next_a = next_a_unclipped

    out = pd.DataFrame(index=X.index)
    out["consumption"] = c
    out["deposit"] = d
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
    regressions: dict[str, Ridge]
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
        if version != 2:
            raise TypeError(
                "This bundle predates the final categorical treatment-coding "
                "format. Retrain it with the current code."
            )
        return value


def _feature_range_summary(
    data: pd.DataFrame,
    spec: FeatureSpec,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for col in spec.state_columns + spec.parameter_columns:
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
    for col in spec.state_columns + spec.parameter_columns:
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


def _target_transform_kind(target: str) -> str:
    return "log1p" if target == "consumption" else "asinh"


def _primitive_predictions_from_models(
    feature_map: Any,
    regressions: Mapping[str, Ridge],
    transforms: Mapping[str, TargetTransform],
    X: pd.DataFrame,
) -> pd.DataFrame:
    Z = feature_map.transform(X)
    out: dict[str, np.ndarray] = {}
    for target in PRIMITIVE_POLICY_TARGETS:
        pred_t = np.asarray(regressions[target].predict(Z), dtype=float)
        out[target] = transforms[target].inverse(pred_t)
    return pd.DataFrame(out, index=X.index)


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


def fit_policy_surrogate(
    data: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    categorical_parameter_columns: Sequence[str] = (),
    state_columns: Sequence[str] = DEFAULT_STATE_COLUMNS,
    categorical_columns: Sequence[str] = DEFAULT_CATEGORICAL_STATE_COLUMNS,
    model_type: str = "polynomial",
    polynomial_degree: int = 2,
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
    money_units: str = "model",
    validation_share: float = 0.20,
    tuning_share: float = 0.20,
    random_state: int = 123,
    verbose: bool = True,
    show_progress: bool = True,
) -> PolicySurrogateBundle:
    """Fit and validate a smooth, accounting-restricted policy surrogate."""

    target_parameterization = _normalize_target_parameterization(
        target_parameterization
    )
    required_targets = {
        "consumption",
        "deposit",
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
        parameter_columns=parameters + missing_indicators,
        categorical_state_columns=tuple(categorical_columns),
        categorical_parameter_columns=categorical_parameters,
        signed_log_columns=tuple(
            c for c in DEFAULT_MONETARY_COLUMNS if c in set(state_columns)
        ),
    )

    accounting_defaults = _accounting_defaults_from_data(data)
    initial_defaults: dict[str, Any] = {}
    for column in spec.state_columns + spec.parameter_columns:
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

    # Accounting columns vary row by row when the underlying parameters vary;
    # prepare_policy_inputs therefore uses row values first and defaults only as
    # fallbacks.  Monetary deductions/floors are converted consistently with the
    # units selected by GridSchema.
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
                parameter_parameter_interactions=True,
                categorical_slopes=True,
            )
        if model_type in {"rff", "rbf"}:
            return SmoothRFFMap(
                spec,
                n_components=rff_components,
                random_state=random_state,
            )
        raise ValueError("model_type must be 'polynomial' or 'rff'.")

    _status(
        f"[fit 2/6] Constructing {model_type} features for target-specific "
        "ridge tuning...",
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

    _status(
        f"[fit 3/6] Tuning separate ridge penalties for "
        f"{len(PRIMITIVE_POLICY_TARGETS)} primitive policies...",
        verbose=verbose,
    )
    alpha_rows: list[dict[str, Any]] = []
    best_alphas: dict[str, float] = {}
    tuning_transforms: dict[str, TargetTransform] = {}
    tasks = [
        (target, float(alpha))
        for target in PRIMITIVE_POLICY_TARGETS
        for alpha in ridge_alphas
    ]
    iterator = _progress_iter(
        tasks,
        total=len(tasks),
        description="Ridge tuning",
        unit="fit",
        enabled=show_progress,
    )
    target_best_score = {target: np.inf for target in PRIMITIVE_POLICY_TARGETS}
    target_y_train: dict[str, np.ndarray] = {}
    target_y_tune: dict[str, np.ndarray] = {}
    for target in PRIMITIVE_POLICY_TARGETS:
        transform = TargetTransform(_target_transform_kind(target)).fit(
            pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
        )
        tuning_transforms[target] = transform
        target_y_train[target] = transform.transform(
            pd.to_numeric(train[target], errors="coerce").to_numpy(dtype=float)
        )
        target_y_tune[target] = pd.to_numeric(tune[target], errors="coerce").to_numpy(
            dtype=float
        )

    for position, (target, alpha) in enumerate(iterator, start=1):
        model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
        model.fit(Z_train, target_y_train[target])
        pred_t = np.asarray(model.predict(Z_tune), dtype=float)
        pred = tuning_transforms[target].inverse(pred_t)
        truth = target_y_tune[target]
        ok = np.isfinite(truth) & np.isfinite(pred)
        rmse = float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2)))
        sd = float(np.std(truth[ok]))
        score = rmse / max(sd, EPS)
        alpha_rows.append(
            {
                "target": target,
                "alpha": alpha,
                "tuning_rmse": rmse,
                "tuning_nrmse_sd": score,
            }
        )
        if score < target_best_score[target]:
            target_best_score[target] = score
            best_alphas[target] = alpha
        if verbose and (_tqdm is None or not show_progress):
            _status(
                f"  {position}/{len(tasks)}: target={target}; alpha={alpha:g}; "
                f"NRMSE={score:.5f}",
                verbose=verbose,
            )
        elif show_progress and _tqdm is not None:
            iterator.set_postfix(  # type: ignore[attr-defined]
                target=target,
                alpha=f"{alpha:g}",
                score=f"{score:.4f}",
            )
    for target in PRIMITIVE_POLICY_TARGETS:
        _status(
            f"  selected {target} alpha={best_alphas[target]:g} "
            f"(tuning NRMSE={target_best_score[target]:.5f})",
            verbose=verbose,
        )

    _status(
        "[fit 4/6] Refitting on training+tuning models and evaluating all "
        "primitive and accounting-derived policies on held-out models...",
        verbose=verbose,
    )
    evaluation_start = time.perf_counter()
    development = pd.concat([train, tune], axis=0).sort_index()
    evaluation_map = make_feature_map()
    Z_development = evaluation_map.fit_transform(development)
    evaluation_regressions: dict[str, Ridge] = {}
    evaluation_transforms: dict[str, TargetTransform] = {}
    for target in PRIMITIVE_POLICY_TARGETS:
        transform = TargetTransform(_target_transform_kind(target)).fit(
            pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        )
        y = transform.transform(
            pd.to_numeric(development[target], errors="coerce").to_numpy(dtype=float)
        )
        model = Ridge(
            alpha=best_alphas[target],
            fit_intercept=True,
            solver="lsqr",
        )
        model.fit(Z_development, y)
        evaluation_regressions[target] = model
        evaluation_transforms[target] = transform
    primitive_test = _primitive_predictions_from_models(
        evaluation_map,
        evaluation_regressions,
        evaluation_transforms,
        test,
    )
    test_pred = reconstruct_policy_outputs(
        test,
        primitive_test,
        accounting_defaults=accounting_defaults,
        money_units=money_units,
        output_bounds={
            "next_liquid_assets": (
                float(fit_data["liquid_grid_min"].min()),
                float(fit_data["liquid_grid_max"].max()),
            ),
            "next_illiquid_assets": (
                float(fit_data["illiquid_grid_min"].min()),
                float(fit_data["illiquid_grid_max"].max()),
            ),
        },
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
        "[fit 5/6] Constructing the full design matrix and fitting the final "
        "consumption and deposit surrogates...",
        verbose=verbose,
    )
    final_start = time.perf_counter()
    final_map = make_feature_map()
    Z_all = final_map.fit_transform(fit_data)
    final_regressions: dict[str, Ridge] = {}
    final_transforms: dict[str, TargetTransform] = {}
    for target in PRIMITIVE_POLICY_TARGETS:
        transform = TargetTransform(_target_transform_kind(target)).fit(
            pd.to_numeric(fit_data[target], errors="coerce").to_numpy(dtype=float)
        )
        y = transform.transform(
            pd.to_numeric(fit_data[target], errors="coerce").to_numpy(dtype=float)
        )
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
        "bundle_format_version": 2,
        "model_type": model_type,
        "polynomial_degree": int(polynomial_degree),
        "rff_components": int(rff_components),
        "ridge_alphas": best_alphas,
        # Backwards-readable summary; there is deliberately no common alpha.
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
        "asset_choices_reconstructed_from_accounting": True,
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
        internal_targets=tuple(PRIMITIVE_POLICY_TARGETS),
        target_transforms=final_transforms,
        target_parameterization=target_parameterization,
        output_bounds=output_bounds,
        feature_ranges=_feature_range_summary(fit_data, spec),
        default_values=defaults,
        accounting_defaults=accounting_defaults,
        validation_metrics=validation_metrics,
        validation_by_model=validation_by_model,
        training_metadata=metadata,
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
    parameter_include: Sequence[str] | None = None,
    parameter_exclude: Sequence[str] = (),
    categorical_parameter_include: Sequence[str] | None = None,
    categorical_parameter_exclude: Sequence[str] = (),
    max_solver_distance: float | None = None,
    model_type: str = "polynomial",
    polynomial_degree: int = 2,
    rff_components: int = 512,
    target_parameterization: str = "accounting",
    state_columns: Sequence[str] = DEFAULT_STATE_COLUMNS,
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
    failures_path = output / "grid_load_failures.csv"
    dataset.failures.to_csv(failures_path, index=False)

    _status("[stage 3/4] Fitting and validating the surrogate...", verbose=verbose)
    bundle = fit_policy_surrogate(
        dataset.data,
        parameter_columns=dataset.parameter_columns,
        categorical_parameter_columns=dataset.categorical_parameter_columns,
        state_columns=state_columns,
        categorical_columns=categorical_columns,
        model_type=model_type,
        polynomial_degree=polynomial_degree,
        rff_components=rff_components,
        target_parameterization=target_parameterization,
        money_units=schema.money_units,
        validation_share=validation_share,
        tuning_share=tuning_share,
        random_state=random_state,
        verbose=verbose,
        show_progress=show_progress,
    )
    _status("[stage 4/4] Saving fitted bundle and diagnostics...", verbose=verbose)
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
                "categorical_columns": list(categorical_columns),
                "rows_per_model": dataset.rows_per_model,
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
