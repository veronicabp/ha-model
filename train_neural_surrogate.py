#!/usr/bin/env python3
"""Train a GPU-accelerated neural surrogate for HA policy functions.

This script is designed to sit beside ``ha_policy_surrogate.py`` and reuse its
saved-grid reader, population sampler, accounting preparation, and frozen
model-level split.  It trains a smooth parameter-conditioned mixture-of-experts
network for the two primitive policies:

    * consumption
    * the net deposit from liquid into illiquid wealth

The training objective also targets accounting-implied liquid and illiquid
asset changes, total liquid outflow, finite-difference MPCs, finite-difference
deposit responses, and finite-difference liquid-saving responses.  This keeps
consumption and deposit accurate individually instead of allowing all remaining
error to appear in liquid saving.

Outputs
-------
``neural_policy_surrogate_holdout.pt``
    Development-only model selected on tuning models and evaluated on the
    frozen test models.
``neural_policy_surrogate.pt``
    Production model refit on every usable model with the selected
    hyperparameters.
``neural_grid_search.csv``
    One row per hyperparameter trial.
``neural_validation_metrics.csv``
    Aggregate test metrics for the holdout model.
``neural_validation_by_model.csv``
    Test metrics by solved model.
``neural_training_summary.json``
    Split, data, hyperparameter, and runtime metadata.

The saved bundle is self-contained except that this file and the compatible
``ha_policy_surrogate.py`` must be importable when loading it for prediction.
"""

from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ha_policy_surrogate import (
    DEFAULT_CATEGORICAL_COLUMNS,
    DEFAULT_ENGINEERED_STATE_COLUMNS,
    DEFAULT_REDUNDANT_STATE_COLUMNS,
    DEFAULT_STATE_COLUMNS,
    EPS,
    FeatureSpec,
    GridSchema,
    PopulationSamplingSpec,
    _accounting_defaults_from_data,
    _default_row,
    _feature_range_summary,
    _load_or_create_model_split,
    _paired_mpc_table,
    build_policy_dataset,
    prepare_policy_inputs,
    read_policy_dataset,
    reconstruct_policy_outputs,
)

BUNDLE_FORMAT_VERSION = 1
REQUIRED_TARGET_COLUMNS = (
    "consumption",
    "deposit",
    "delta_liquid_assets",
    "delta_illiquid_assets",
)
ACCOUNTING_COLUMNS = (
    "acct__ct_time_step",
    "acct__rb_pos_ct",
    "acct__rb_neg_ct",
    "acct__ra_ct",
    "acct__chi0",
    "acct__chi1",
    "acct__xi",
    "acct__illiquid_cost_floor",
    "liquid_grid_min",
    "liquid_grid_max",
    "illiquid_grid_min",
    "illiquid_grid_max",
)
LOW_LIQUID_EDGES = np.asarray(
    [
        -np.inf,
        0.0025,
        0.005,
        0.010,
        0.020,
        0.040,
        0.080,
        0.160,
        0.320,
        0.640,
        1.500,
        np.inf,
    ],
    dtype=float,
)


# =============================================================================
# Small utilities
# =============================================================================


def _status(message: str) -> None:
    print(message, flush=True)


def _elapsed(start: float) -> str:
    seconds = max(0.0, time.perf_counter() - start)
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return path
        except Exception as exc:
            fallback = path.with_suffix(".pkl.gz")
            _status(
                f"Parquet write failed for {path.name} ({type(exc).__name__}); "
                f"writing {fallback.name}."
            )
            frame.to_pickle(fallback, compression="gzip")
            return fallback
    if path.name.endswith(".pkl.gz"):
        frame.to_pickle(path, compression="gzip")
        return path
    if path.suffix in {".pkl", ".pickle"}:
        frame.to_pickle(path)
        return path
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
        return path
    fallback = path.with_suffix(".pkl.gz")
    frame.to_pickle(fallback, compression="gzip")
    return fallback


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_float_grid(value: str) -> list[float]:
    values = [float(part) for part in _parse_csv(value)]
    if not values:
        raise argparse.ArgumentTypeError("Expected a nonempty comma-separated grid.")
    return values


def _parse_int_grid(value: str) -> list[int]:
    values = [int(part) for part in _parse_csv(value)]
    if not values:
        raise argparse.ArgumentTypeError("Expected a nonempty comma-separated grid.")
    return values


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part) for part in _parse_csv(value))
    if not values:
        raise argparse.ArgumentTypeError("Expected a nonempty comma-separated list.")
    return values


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    good = np.isfinite(true) & np.isfinite(pred)
    if good.sum() < 2:
        return float("nan")
    true = true[good]
    pred = pred[good]
    denominator = float(np.sum((true - np.mean(true)) ** 2))
    if denominator <= EPS:
        return float("nan")
    return 1.0 - float(np.sum((true - pred) ** 2)) / denominator


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    good = np.isfinite(true) & np.isfinite(pred)
    if not good.any():
        return {
            "n": 0,
            "r2": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "bias": float("nan"),
        }
    error = pred[good] - true[good]
    return {
        "n": int(good.sum()),
        "r2": _safe_r2(true[good], pred[good]),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def _robust_scale(values: np.ndarray, floor: float = 1.0e-4) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float(floor)
    q25, q75 = np.quantile(x, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    std = float(np.std(x))
    median_abs = float(np.median(np.abs(x)))
    candidates = [value for value in (iqr_scale, std, median_abs) if np.isfinite(value)]
    return max(float(floor), max(candidates, default=float(floor)))


def _device_from_args(device_arg: str, require_gpu: bool) -> torch.device:
    requested = str(device_arg).lower().strip()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if require_gpu and device.type != "cuda":
        raise RuntimeError(
            "--require-gpu was supplied, but PyTorch cannot see a CUDA device. "
            "Check the SLURM GPU request, CUDA module, and PyTorch installation."
        )
    return device


def _print_device(device: torch.device) -> None:
    _status(f"[hardware] PyTorch {torch.__version__}")
    _status(f"[hardware] selected device: {device}")
    if device.type == "cuda":
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        _status(
            "[hardware] GPU: "
            f"{properties.name}; memory={properties.total_memory / 1024**3:.1f} GiB; "
            f"CUDA runtime={torch.version.cuda}"
        )
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


# =============================================================================
# Dataset construction and model-level sampling
# =============================================================================


def _infer_parameter_columns(
    data: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    continuous: list[str] = []
    categorical: list[str] = []
    for column in data.columns:
        if not column.startswith("param__"):
            continue
        raw = data[column]
        numeric = pd.to_numeric(raw, errors="coerce")
        coverage = float(numeric.notna().mean())
        if coverage >= 0.80:
            if numeric.nunique(dropna=True) > 1:
                continuous.append(column)
        else:
            values = raw.dropna().astype(str)
            if values.nunique() > 1:
                categorical.append(column)
    return sorted(continuous), sorted(categorical)


def _working_age_filter(data: pd.DataFrame) -> pd.DataFrame:
    if "is_retired" in data:
        retired = pd.to_numeric(data["is_retired"], errors="coerce").fillna(0)
        return data.loc[retired.eq(0)].reset_index(drop=True)
    if {"age", "acct__retirement_age"}.issubset(data.columns):
        age = pd.to_numeric(data["age"], errors="coerce")
        retirement_age = pd.to_numeric(
            data["acct__retirement_age"], errors="coerce"
        )
        return data.loc[age.lt(retirement_age)].reset_index(drop=True)
    return data.reset_index(drop=True)


def _load_or_build_sample(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str], list[str], Path, pd.DataFrame | None]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    default_sample = output_dir / "neural_policy_grid_sample.pkl.gz"
    sample_path = (
        Path(args.sample_path).expanduser().resolve()
        if args.sample_path is not None
        else default_sample
    )

    if sample_path.exists() and (args.reuse_sample or args.sample_path is not None):
        _status(f"[data 1/4] Loading existing sampled policy data: {sample_path}")
        data = read_policy_dataset(sample_path)
        continuous_parameters, categorical_parameters = _infer_parameter_columns(data)
        _status(
            f"[data] loaded {len(data):,} rows from "
            f"{data['model_id'].astype(str).nunique():,} models; "
            f"continuous parameters={len(continuous_parameters)}, "
            f"categorical parameters={len(categorical_parameters)}"
        )
        return (
            data,
            continuous_parameters,
            categorical_parameters,
            sample_path,
            None,
        )

    _status("[data 1/4] Building a new population-weighted policy sample...")
    population_spec = PopulationSamplingSpec(
        poor_hand_to_mouth_share=float(args.poor_htm_share),
        wealthy_hand_to_mouth_share=float(args.wealthy_htm_share),
        broad_support_share=float(args.broad_support_share),
        nonemployment_share=float(args.nonemployment_share),
        negative_liquid_share=float(args.negative_liquid_share),
        borrowing_constraint_anchor_share=float(args.constraint_anchor_share),
        mpc_cross_zero_share=float(args.mpc_cross_zero_share),
        working_age_only=not bool(args.include_retired),
    )
    schema = GridSchema(
        policy_layout=str(args.policy_layout),
        money_units=str(args.money_units),
        require_compact_policies=bool(args.require_compact_policies),
    )
    result = build_policy_dataset(
        args.model_root,
        manifest_path=args.manifest,
        schema=schema,
        rows_per_model=int(args.rows_per_model),
        max_total_rows=int(args.max_total_rows),
        max_models=args.max_models,
        sampling_mode=str(args.sampling_mode),
        population_spec=population_spec,
        n_jobs=int(args.n_jobs),
        mpc_pair_share=float(args.mpc_pair_share),
        mpc_check_sizes=tuple(args.mpc_check_sizes),
        max_solver_distance=args.max_solver_distance,
        max_relative_distance=args.max_relative_distance,
        max_hjb_residual=args.max_hjb_residual,
        random_state=int(args.random_state),
        verbose=True,
        show_progress=not bool(args.no_progress_bar),
    )
    _status("[data 2/4] Saving the sampled data, catalog, and load failures...")
    written_path = _write_dataframe(result.data, default_sample)
    result.catalog.to_csv(output_dir / "neural_model_catalog.csv", index=False)
    result.failures.to_csv(output_dir / "neural_grid_load_failures.csv", index=False)
    return (
        result.data,
        list(result.parameter_columns),
        list(result.categorical_parameter_columns),
        written_path,
        result.catalog,
    )


def _model_balanced_subsample(
    frame: pd.DataFrame,
    max_rows: int | None,
    *,
    random_state: int,
) -> pd.DataFrame:
    """Sample approximately equal rows per model while preserving complete pairs."""

    if max_rows is None or max_rows <= 0 or len(frame) <= int(max_rows):
        return frame.reset_index(drop=True)
    if "model_id" not in frame:
        return frame.sample(n=int(max_rows), random_state=random_state).reset_index(
            drop=True
        )

    groups = list(frame.groupby(frame["model_id"].astype(str), sort=False))
    n_models = len(groups)
    if n_models == 0:
        return frame.iloc[:0].copy()
    base_quota = max(1, int(max_rows) // n_models)
    remainder = max(0, int(max_rows) - base_quota * n_models)
    rng = np.random.default_rng(int(random_state))
    bonus_models = set(rng.permutation(n_models)[:remainder].tolist())
    selected_parts: list[pd.DataFrame] = []

    for position, (_, block_raw) in enumerate(groups):
        block = block_raw.reset_index(drop=False).rename(columns={"index": "__source"})
        quota = base_quota + (1 if position in bonus_models else 0)
        quota = min(quota, len(block))
        if quota >= len(block):
            selected_parts.append(block)
            continue

        pair_id = pd.to_numeric(block.get("mpc_pair_id", -1), errors="coerce")
        role = pd.to_numeric(block.get("mpc_pair_role", -1), errors="coerce")
        pair_rows = block[pair_id.ge(0) & role.isin([0, 1])].copy()
        ordinary = block[~(pair_id.ge(0) & role.isin([0, 1]))].copy()

        pair_ids = pair_rows["mpc_pair_id"].dropna().unique()
        observed_pair_share = len(pair_rows) / max(len(block), 1)
        desired_pair_rows = min(
            2 * len(pair_ids),
            2 * int(math.floor((quota * observed_pair_share) / 2.0)),
        )
        n_pairs = min(len(pair_ids), desired_pair_rows // 2)
        chosen_pair_ids = (
            rng.choice(pair_ids, size=n_pairs, replace=False)
            if n_pairs > 0
            else np.asarray([], dtype=float)
        )
        chosen = pair_rows[pair_rows["mpc_pair_id"].isin(chosen_pair_ids)].copy()

        remaining = max(0, quota - len(chosen))
        if remaining > 0 and len(ordinary):
            ordinary_take = ordinary.sample(
                n=min(remaining, len(ordinary)),
                random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
            )
            chosen = pd.concat([chosen, ordinary_take], ignore_index=True)

        remaining = max(0, quota - len(chosen))
        if remaining >= 2:
            unused = np.setdiff1d(pair_ids, chosen_pair_ids, assume_unique=False)
            extra_pairs = min(len(unused), remaining // 2)
            if extra_pairs > 0:
                selected_extra = rng.choice(unused, size=extra_pairs, replace=False)
                chosen = pd.concat(
                    [chosen, pair_rows[pair_rows["mpc_pair_id"].isin(selected_extra)]],
                    ignore_index=True,
                )

        if len(chosen) > quota:
            # Drop ordinary rows first; never split a complete MPC pair.
            is_pair = pd.to_numeric(chosen["mpc_pair_id"], errors="coerce").ge(0)
            pair_part = chosen[is_pair]
            ordinary_part = chosen[~is_pair]
            room = max(0, quota - len(pair_part))
            if room < len(ordinary_part):
                ordinary_part = ordinary_part.sample(
                    n=room,
                    random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
                )
            chosen = pd.concat([pair_part, ordinary_part], ignore_index=True)

        selected_parts.append(chosen)

    sampled = pd.concat(selected_parts, ignore_index=True)
    if "__source" in sampled:
        sampled = sampled.sort_values("__source", kind="mergesort").drop(
            columns="__source"
        )
    if len(sampled) > int(max_rows):
        # The only possible overrun is at most one complete pair per model.
        _status(
            f"[data] model-balanced pair-preserving sample contains {len(sampled):,} "
            f"rows, slightly above the requested {int(max_rows):,}."
        )
    return sampled.reset_index(drop=True)


# =============================================================================
# Input preprocessing
# =============================================================================


@dataclass
class NeuralInputPreprocessor:
    feature_spec: FeatureSpec
    state_continuous_columns: tuple[str, ...] = ()
    parameter_continuous_columns: tuple[str, ...] = ()
    state_categorical_columns: tuple[str, ...] = ()
    parameter_categorical_columns: tuple[str, ...] = ()
    medians: dict[str, float] = field(default_factory=dict)
    asinh_scales: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    categories: dict[str, list[str]] = field(default_factory=dict)
    fitted: bool = False

    @staticmethod
    def _needs_asinh(column: str, spec: FeatureSpec) -> bool:
        if column in spec.signed_log_columns:
            return True
        return column in {
            "liquid_assets_to_income",
            "liquid_slack_to_income",
            "illiquid_assets_to_income",
            "after_tax_income_to_income",
            "liquid_resources_to_income",
        }

    def fit(self, prepared: pd.DataFrame) -> "NeuralInputPreprocessor":
        self.state_continuous_columns = tuple(self.feature_spec.model_state_columns)
        self.parameter_continuous_columns = tuple(
            self.feature_spec.parameter_columns
        )
        self.state_categorical_columns = tuple(
            self.feature_spec.categorical_state_columns
        )
        self.parameter_categorical_columns = tuple(
            self.feature_spec.categorical_parameter_columns
        )

        continuous = (
            self.state_continuous_columns + self.parameter_continuous_columns
        )
        for column in continuous:
            values = pd.to_numeric(prepared[column], errors="coerce").to_numpy(
                dtype=float
            )
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            values = np.where(np.isfinite(values), values, median)
            self.medians[column] = median
            if self._needs_asinh(column, self.feature_spec):
                nonzero = np.abs(values[np.abs(values) > EPS])
                scale = float(np.median(nonzero)) if len(nonzero) else 1.0
                scale = max(scale, 1.0e-6)
                self.asinh_scales[column] = scale
                values = np.arcsinh(values / scale)
            mean = float(np.mean(values))
            std = float(np.std(values))
            self.means[column] = mean
            self.stds[column] = max(std, 1.0e-6)

        for column in (
            self.state_categorical_columns + self.parameter_categorical_columns
        ):
            values = prepared[column].where(
                prepared[column].notna(), "__missing__"
            ).astype(str)
            observed = sorted(values.unique().tolist())
            # Index zero is always reserved for unknown categories at inference.
            self.categories[column] = ["__unknown__", *observed]

        self.fitted = True
        return self

    def _numeric_block(
        self,
        prepared: pd.DataFrame,
        columns: Sequence[str],
    ) -> np.ndarray:
        out = np.empty((len(prepared), len(columns)), dtype=np.float32)
        for j, column in enumerate(columns):
            values = pd.to_numeric(prepared[column], errors="coerce").to_numpy(
                dtype=float
            )
            values = np.where(np.isfinite(values), values, self.medians[column])
            if column in self.asinh_scales:
                values = np.arcsinh(values / self.asinh_scales[column])
            values = (values - self.means[column]) / self.stds[column]
            out[:, j] = values.astype(np.float32, copy=False)
        return out

    def _categorical_block(
        self,
        prepared: pd.DataFrame,
        columns: Sequence[str],
    ) -> np.ndarray:
        out = np.zeros((len(prepared), len(columns)), dtype=np.int64)
        for j, column in enumerate(columns):
            lookup = {
                value: index for index, value in enumerate(self.categories[column])
            }
            values = prepared[column].where(
                prepared[column].notna(), "__missing__"
            ).astype(str)
            out[:, j] = np.asarray(
                [lookup.get(value, 0) for value in values], dtype=np.int64
            )
        return out

    def transform_prepared(
        self,
        prepared: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("The neural input preprocessor is not fitted.")
        required = (
            self.state_continuous_columns
            + self.parameter_continuous_columns
            + self.state_categorical_columns
            + self.parameter_categorical_columns
        )
        missing = [column for column in required if column not in prepared]
        if missing:
            raise KeyError(f"Prepared neural inputs are missing columns: {missing}")
        return (
            self._numeric_block(prepared, self.state_continuous_columns),
            self._numeric_block(prepared, self.parameter_continuous_columns),
            self._categorical_block(prepared, self.state_categorical_columns),
            self._categorical_block(
                prepared, self.parameter_categorical_columns
            ),
        )

    @property
    def state_cardinalities(self) -> tuple[int, ...]:
        return tuple(len(self.categories[column]) for column in self.state_categorical_columns)

    @property
    def parameter_cardinalities(self) -> tuple[int, ...]:
        return tuple(
            len(self.categories[column])
            for column in self.parameter_categorical_columns
        )

    def to_state(self) -> dict[str, Any]:
        return {
            "feature_spec": {
                "state_columns": list(self.feature_spec.state_columns),
                "engineered_state_columns": list(
                    self.feature_spec.engineered_state_columns
                ),
                "redundant_state_columns": list(
                    self.feature_spec.redundant_state_columns
                ),
                "parameter_columns": list(self.feature_spec.parameter_columns),
                "categorical_state_columns": list(
                    self.feature_spec.categorical_state_columns
                ),
                "categorical_parameter_columns": list(
                    self.feature_spec.categorical_parameter_columns
                ),
                "signed_log_columns": list(self.feature_spec.signed_log_columns),
            },
            "state_continuous_columns": list(self.state_continuous_columns),
            "parameter_continuous_columns": list(
                self.parameter_continuous_columns
            ),
            "state_categorical_columns": list(self.state_categorical_columns),
            "parameter_categorical_columns": list(
                self.parameter_categorical_columns
            ),
            "medians": self.medians,
            "asinh_scales": self.asinh_scales,
            "means": self.means,
            "stds": self.stds,
            "categories": self.categories,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "NeuralInputPreprocessor":
        spec_state = dict(state["feature_spec"])
        spec = FeatureSpec(
            state_columns=tuple(spec_state["state_columns"]),
            engineered_state_columns=tuple(
                spec_state["engineered_state_columns"]
            ),
            redundant_state_columns=tuple(spec_state["redundant_state_columns"]),
            parameter_columns=tuple(spec_state["parameter_columns"]),
            categorical_state_columns=tuple(
                spec_state["categorical_state_columns"]
            ),
            categorical_parameter_columns=tuple(
                spec_state["categorical_parameter_columns"]
            ),
            signed_log_columns=tuple(spec_state["signed_log_columns"]),
        )
        value = cls(feature_spec=spec)
        value.state_continuous_columns = tuple(state["state_continuous_columns"])
        value.parameter_continuous_columns = tuple(
            state["parameter_continuous_columns"]
        )
        value.state_categorical_columns = tuple(
            state["state_categorical_columns"]
        )
        value.parameter_categorical_columns = tuple(
            state["parameter_categorical_columns"]
        )
        value.medians = {str(k): float(v) for k, v in state["medians"].items()}
        value.asinh_scales = {
            str(k): float(v) for k, v in state["asinh_scales"].items()
        }
        value.means = {str(k): float(v) for k, v in state["means"].items()}
        value.stds = {str(k): float(v) for k, v in state["stds"].items()}
        value.categories = {
            str(k): [str(item) for item in values]
            for k, values in state["categories"].items()
        }
        value.fitted = True
        return value


# =============================================================================
# Prepared arrays and MPC-pair arrays
# =============================================================================


@dataclass
class PreparedData:
    state_continuous: np.ndarray
    parameter_continuous: np.ndarray
    state_categorical: np.ndarray
    parameter_categorical: np.ndarray
    consumption_ratio: np.ndarray
    deposit_ratio: np.ndarray
    liquid_outflow_ratio: np.ndarray
    delta_liquid_ratio: np.ndarray
    delta_illiquid_ratio: np.ndarray
    income_scale: np.ndarray
    after_tax_income: np.ndarray
    liquid_assets: np.ndarray
    illiquid_assets: np.ndarray
    liquid_slack_ratio: np.ndarray
    dt: np.ndarray
    rb_pos: np.ndarray
    rb_neg: np.ndarray
    ra: np.ndarray
    chi0: np.ndarray
    chi1: np.ndarray
    xi: np.ndarray
    a_floor: np.ndarray
    liquid_min: np.ndarray
    liquid_max: np.ndarray
    illiquid_min: np.ndarray
    illiquid_max: np.ndarray
    deposit_regime: np.ndarray
    model_ids: np.ndarray | None = None

    def __len__(self) -> int:
        return int(len(self.consumption_ratio))


@dataclass
class PairData:
    baseline: np.ndarray
    treated: np.ndarray
    checks: np.ndarray
    exact_mpc: np.ndarray
    exact_mpd: np.ndarray
    exact_liquid_response: np.ndarray
    weights: np.ndarray
    model_ids: np.ndarray

    def __len__(self) -> int:
        return int(len(self.exact_mpc))


@dataclass(frozen=True)
class LossScales:
    consumption: float
    deposit: float
    liquid_outflow: float
    delta_liquid: float
    delta_illiquid: float
    mpc: float
    mpd: float
    liquid_response: float
    deposit_zero_tolerance: float



def _fit_preprocessor(
    train: pd.DataFrame,
    *,
    feature_spec: FeatureSpec,
    accounting_defaults: Mapping[str, Any],
    default_values: Mapping[str, Any],
    money_units: str,
    max_rows: int,
    random_state: int,
) -> NeuralInputPreprocessor:
    sample = _model_balanced_subsample(
        train,
        max_rows,
        random_state=random_state,
    )
    _status(
        f"[preprocess] fitting input transforms on {len(sample):,} rows "
        f"from {sample['model_id'].astype(str).nunique():,} models"
    )
    prepared = prepare_policy_inputs(
        sample,
        feature_spec=feature_spec,
        default_values=default_values,
        accounting_defaults=accounting_defaults,
        money_units=money_units,
    )
    return NeuralInputPreprocessor(feature_spec).fit(prepared)


def _finite_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float,
) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), float(default), dtype=np.float32)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, float(default))
    return values.astype(np.float32, copy=False)


def _prepare_data_arrays(
    frame: pd.DataFrame,
    *,
    preprocessor: NeuralInputPreprocessor,
    accounting_defaults: Mapping[str, Any],
    default_values: Mapping[str, Any],
    money_units: str,
    chunk_size: int,
    include_model_ids: bool,
    deposit_zero_tolerance: float | None = None,
) -> PreparedData:
    frame = frame.reset_index(drop=True)
    missing_targets = [column for column in REQUIRED_TARGET_COLUMNS if column not in frame]
    if missing_targets:
        raise KeyError(f"Training data are missing targets: {missing_targets}")

    n = len(frame)
    ns = len(preprocessor.state_continuous_columns)
    np_ = len(preprocessor.parameter_continuous_columns)
    ncs = len(preprocessor.state_categorical_columns)
    ncp = len(preprocessor.parameter_categorical_columns)

    state_cont = np.empty((n, ns), dtype=np.float32)
    parameter_cont = np.empty((n, np_), dtype=np.float32)
    state_cat = np.empty((n, ncs), dtype=np.int64)
    parameter_cat = np.empty((n, ncp), dtype=np.int64)

    one_dimensional_names = [
        "consumption_ratio",
        "deposit_ratio",
        "liquid_outflow_ratio",
        "delta_liquid_ratio",
        "delta_illiquid_ratio",
        "income_scale",
        "after_tax_income",
        "liquid_assets",
        "illiquid_assets",
        "liquid_slack_ratio",
        "dt",
        "rb_pos",
        "rb_neg",
        "ra",
        "chi0",
        "chi1",
        "xi",
        "a_floor",
        "liquid_min",
        "liquid_max",
        "illiquid_min",
        "illiquid_max",
    ]
    arrays = {name: np.empty(n, dtype=np.float32) for name in one_dimensional_names}

    start_time = time.perf_counter()
    for start in range(0, n, int(chunk_size)):
        stop = min(start + int(chunk_size), n)
        raw = frame.iloc[start:stop]
        prepared = prepare_policy_inputs(
            raw,
            feature_spec=preprocessor.feature_spec,
            default_values=default_values,
            accounting_defaults=accounting_defaults,
            money_units=money_units,
        )
        s_cont, p_cont, s_cat, p_cat = preprocessor.transform_prepared(prepared)
        state_cont[start:stop] = s_cont
        parameter_cont[start:stop] = p_cont
        state_cat[start:stop] = s_cat
        parameter_cat[start:stop] = p_cat

        permanent = _finite_column(
            prepared,
            "labor_income_state",
            default=float(default_values.get("labor_income_state", 1.0)),
        )
        typical = max(
            1.0e-6,
            abs(float(default_values.get("labor_income_state", 1.0))) * 1.0e-3,
        )
        income_scale = np.maximum(np.abs(permanent), typical).astype(np.float32)
        consumption = _finite_column(prepared, "consumption", default=0.0)
        deposit = _finite_column(prepared, "deposit", default=0.0)
        delta_b = _finite_column(prepared, "delta_liquid_assets", default=0.0)
        delta_a = _finite_column(prepared, "delta_illiquid_assets", default=0.0)
        if "liquid_outflow" in prepared:
            outflow = _finite_column(prepared, "liquid_outflow", default=0.0)
        else:
            a = _finite_column(prepared, "illiquid_assets", default=0.0)
            chi0 = _finite_column(
                prepared,
                "acct__chi0",
                default=float(accounting_defaults.get("acct__chi0", 0.0)),
            )
            chi1 = _finite_column(
                prepared,
                "acct__chi1",
                default=float(accounting_defaults.get("acct__chi1", 0.0)),
            )
            a_floor = _finite_column(
                prepared,
                "acct__illiquid_cost_floor",
                default=float(
                    accounting_defaults.get("acct__illiquid_cost_floor", 1.0e-6)
                ),
            )
            cost = chi0 * np.abs(deposit) + 0.5 * chi1 * deposit**2 / np.maximum(
                a, a_floor
            )
            outflow = consumption + deposit + cost

        arrays["consumption_ratio"][start:stop] = consumption / income_scale
        arrays["deposit_ratio"][start:stop] = deposit / income_scale
        arrays["liquid_outflow_ratio"][start:stop] = outflow / income_scale
        arrays["delta_liquid_ratio"][start:stop] = delta_b / income_scale
        arrays["delta_illiquid_ratio"][start:stop] = delta_a / income_scale
        arrays["income_scale"][start:stop] = income_scale
        arrays["after_tax_income"][start:stop] = _finite_column(
            prepared, "after_tax_income", default=0.0
        )
        arrays["liquid_assets"][start:stop] = _finite_column(
            prepared, "liquid_assets", default=0.0
        )
        arrays["illiquid_assets"][start:stop] = _finite_column(
            prepared, "illiquid_assets", default=0.0
        )
        arrays["liquid_slack_ratio"][start:stop] = _finite_column(
            prepared, "liquid_slack_to_income", default=0.0
        )
        for destination, source, fallback in (
            ("dt", "acct__ct_time_step", 1.0),
            ("rb_pos", "acct__rb_pos_ct", 0.0),
            ("rb_neg", "acct__rb_neg_ct", 0.0),
            ("ra", "acct__ra_ct", 0.0),
            ("chi0", "acct__chi0", 0.0),
            ("chi1", "acct__chi1", 0.0),
            ("xi", "acct__xi", 0.0),
            ("a_floor", "acct__illiquid_cost_floor", 1.0e-6),
            ("liquid_min", "liquid_grid_min", 0.0),
            ("liquid_max", "liquid_grid_max", 1.0),
            ("illiquid_min", "illiquid_grid_min", 0.0),
            ("illiquid_max", "illiquid_grid_max", 1.0),
        ):
            arrays[destination][start:stop] = _finite_column(
                prepared,
                source,
                default=float(accounting_defaults.get(source, fallback)),
            )

        if start == 0 or stop == n or (start // int(chunk_size)) % 10 == 0:
            _status(
                f"[preprocess] transformed {stop:,}/{n:,} rows "
                f"(elapsed {_elapsed(start_time)})"
            )

    if deposit_zero_tolerance is None:
        deposit_scale = _robust_scale(arrays["deposit_ratio"])
        deposit_zero_tolerance = max(1.0e-5, 0.01 * deposit_scale)
    deposit_regime = np.full(n, 1, dtype=np.int64)
    deposit_regime[arrays["deposit_ratio"] < -float(deposit_zero_tolerance)] = 0
    deposit_regime[arrays["deposit_ratio"] > float(deposit_zero_tolerance)] = 2

    model_ids = (
        frame["model_id"].astype(str).to_numpy(dtype=object)
        if include_model_ids
        else None
    )
    return PreparedData(
        state_continuous=state_cont,
        parameter_continuous=parameter_cont,
        state_categorical=state_cat,
        parameter_categorical=parameter_cat,
        consumption_ratio=arrays["consumption_ratio"],
        deposit_ratio=arrays["deposit_ratio"],
        liquid_outflow_ratio=arrays["liquid_outflow_ratio"],
        delta_liquid_ratio=arrays["delta_liquid_ratio"],
        delta_illiquid_ratio=arrays["delta_illiquid_ratio"],
        income_scale=arrays["income_scale"],
        after_tax_income=arrays["after_tax_income"],
        liquid_assets=arrays["liquid_assets"],
        illiquid_assets=arrays["illiquid_assets"],
        liquid_slack_ratio=arrays["liquid_slack_ratio"],
        dt=arrays["dt"],
        rb_pos=arrays["rb_pos"],
        rb_neg=arrays["rb_neg"],
        ra=arrays["ra"],
        chi0=arrays["chi0"],
        chi1=arrays["chi1"],
        xi=arrays["xi"],
        a_floor=arrays["a_floor"],
        liquid_min=arrays["liquid_min"],
        liquid_max=arrays["liquid_max"],
        illiquid_min=arrays["illiquid_min"],
        illiquid_max=arrays["illiquid_max"],
        deposit_regime=deposit_regime,
        model_ids=model_ids,
    )


def _pair_sampling_weights(
    pairs: pd.DataFrame,
    *,
    low_liquidity_scale: float,
    low_liquidity_boost: float,
    mpc_tail_boost: float,
) -> np.ndarray:
    if "liquid_slack_to_income" in pairs:
        slack_series = pairs["liquid_slack_to_income"]
    else:
        slack_series = pd.Series(np.nan, index=pairs.index)
    slack = pd.to_numeric(slack_series, errors="coerce").to_numpy(dtype=float)
    slack = np.where(np.isfinite(slack), np.maximum(slack, 0.0), np.inf)
    low_weight = 1.0 + float(low_liquidity_boost) * np.exp(
        -slack / max(float(low_liquidity_scale), 1.0e-6)
    )

    mpc = pd.to_numeric(pairs["exact_mpc"], errors="coerce").to_numpy(dtype=float)
    finite = mpc[np.isfinite(mpc)]
    if len(finite):
        center = float(np.median(finite))
        mad = float(np.median(np.abs(finite - center)))
        scale = max(1.4826 * mad, float(np.std(finite)), 1.0e-4)
        tail = np.clip(np.abs(mpc - center) / scale, 0.0, 4.0)
    else:
        tail = np.zeros(len(pairs), dtype=float)
    weights = low_weight * (1.0 + float(mpc_tail_boost) * tail)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
    return weights / max(float(np.mean(weights)), EPS)


def _subsample_pairs_by_model(
    pairs: pd.DataFrame,
    max_pairs: int | None,
    *,
    random_state: int,
    weights: np.ndarray,
) -> pd.DataFrame:
    if max_pairs is None or max_pairs <= 0 or len(pairs) <= int(max_pairs):
        result = pairs.copy()
        result["__sampling_weight"] = weights
        return result.reset_index(drop=True)

    work = pairs.reset_index(drop=True).copy()
    work["__sampling_weight"] = weights
    groups = list(work.groupby("model_id", sort=False))
    n_models = len(groups)
    quota = max(1, int(max_pairs) // max(n_models, 1))
    remainder = max(0, int(max_pairs) - quota * n_models)
    rng = np.random.default_rng(int(random_state))
    bonus = set(rng.permutation(n_models)[:remainder].tolist())
    chosen: list[pd.DataFrame] = []
    for position, (_, block) in enumerate(groups):
        take = min(len(block), quota + (1 if position in bonus else 0))
        if take >= len(block):
            chosen.append(block)
            continue
        probabilities = block["__sampling_weight"].to_numpy(dtype=float)
        probabilities = probabilities / probabilities.sum()
        indices = rng.choice(len(block), size=take, replace=False, p=probabilities)
        chosen.append(block.iloc[np.sort(indices)])
    result = pd.concat(chosen, ignore_index=True)
    if len(result) > int(max_pairs):
        result = result.sample(n=int(max_pairs), random_state=random_state)
    return result.reset_index(drop=True)


def _build_pair_data(
    raw: pd.DataFrame,
    *,
    max_pairs: int | None,
    random_state: int,
    low_liquidity_scale: float,
    low_liquidity_boost: float,
    mpc_tail_boost: float,
) -> PairData:
    raw = raw.reset_index(drop=True)
    pairs = _paired_mpc_table(raw)
    if pairs.empty:
        raise ValueError("The selected split contains no complete explicit MPC pairs.")

    baseline = pairs["baseline_position"].to_numpy(dtype=np.int64)
    treated = pairs["treated_position"].to_numpy(dtype=np.int64)
    checks = pairs["check_units"].to_numpy(dtype=float)
    deposit = pd.to_numeric(raw["deposit"], errors="coerce").to_numpy(dtype=float)
    delta_b = pd.to_numeric(
        raw["delta_liquid_assets"], errors="coerce"
    ).to_numpy(dtype=float)
    pairs["exact_mpd"] = (deposit[treated] - deposit[baseline]) / checks
    pairs["exact_liquid_response"] = (
        delta_b[treated] - delta_b[baseline]
    ) / checks
    if "liquid_slack_to_income" not in pairs:
        liquid = pd.to_numeric(raw["liquid_assets"], errors="coerce").to_numpy(dtype=float)
        if "param__config__borrowing_limit" in raw:
            lower = pd.to_numeric(
                raw["param__config__borrowing_limit"], errors="coerce"
            ).to_numpy(dtype=float)
        elif "liquid_grid_min" in raw:
            lower = pd.to_numeric(raw["liquid_grid_min"], errors="coerce").to_numpy(dtype=float)
        else:
            lower = np.zeros(len(raw), dtype=float)
        permanent = pd.to_numeric(
            raw["labor_income_state"], errors="coerce"
        ).to_numpy(dtype=float)
        finite_income = np.abs(permanent[np.isfinite(permanent)])
        typical = float(np.median(finite_income)) if len(finite_income) else 1.0
        income_scale = np.maximum(np.abs(permanent), max(1.0e-6, 1.0e-3 * typical))
        pairs["liquid_slack_to_income"] = (
            liquid[baseline] - lower[baseline]
        ) / income_scale[baseline]

    weights = _pair_sampling_weights(
        pairs,
        low_liquidity_scale=low_liquidity_scale,
        low_liquidity_boost=low_liquidity_boost,
        mpc_tail_boost=mpc_tail_boost,
    )
    pairs = _subsample_pairs_by_model(
        pairs,
        max_pairs,
        random_state=random_state,
        weights=weights,
    )
    final_weights = pairs.pop("__sampling_weight").to_numpy(dtype=np.float32)
    final_weights /= max(float(np.mean(final_weights)), EPS)
    return PairData(
        baseline=pairs["baseline_position"].to_numpy(dtype=np.int64),
        treated=pairs["treated_position"].to_numpy(dtype=np.int64),
        checks=pairs["check_units"].to_numpy(dtype=np.float32),
        exact_mpc=pairs["exact_mpc"].to_numpy(dtype=np.float32),
        exact_mpd=pairs["exact_mpd"].to_numpy(dtype=np.float32),
        exact_liquid_response=pairs["exact_liquid_response"].to_numpy(
            dtype=np.float32
        ),
        weights=final_weights,
        model_ids=pairs["model_id"].astype(str).to_numpy(dtype=object),
    )


def _loss_scales(data: PreparedData, pairs: PairData) -> LossScales:
    deposit_scale = _robust_scale(data.deposit_ratio)
    return LossScales(
        consumption=_robust_scale(data.consumption_ratio),
        deposit=deposit_scale,
        liquid_outflow=_robust_scale(data.liquid_outflow_ratio),
        delta_liquid=_robust_scale(data.delta_liquid_ratio),
        delta_illiquid=_robust_scale(data.delta_illiquid_ratio),
        mpc=_robust_scale(pairs.exact_mpc),
        mpd=_robust_scale(pairs.exact_mpd),
        liquid_response=_robust_scale(pairs.exact_liquid_response),
        deposit_zero_tolerance=max(1.0e-5, 0.01 * deposit_scale),
    )


# =============================================================================
# Neural architecture
# =============================================================================


class CategoricalEmbeddings(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        max_embedding_dim: int,
    ) -> None:
        super().__init__()
        dimensions = [
            min(int(max_embedding_dim), max(2, int(round(math.sqrt(cardinality)))))
            for cardinality in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(int(cardinality), int(dimension))
                for cardinality, dimension in zip(cardinalities, dimensions)
            ]
        )
        self.output_dim = int(sum(dimensions))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if len(self.embeddings) == 0:
            return values.new_zeros((values.shape[0], 0), dtype=torch.float32)
        blocks = [embedding(values[:, j]) for j, embedding in enumerate(self.embeddings)]
        return torch.cat(blocks, dim=1)


class FiLMResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(context_dim, 2 * hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.linear_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        value = self.norm(hidden)
        scale, shift = self.film(context).chunk(2, dim=-1)
        value = value * (1.0 + 0.15 * torch.tanh(scale)) + shift
        value = self.linear_1(value)
        value = F.silu(value)
        value = self.dropout(value)
        value = self.linear_2(value)
        return hidden + self.dropout(value) / math.sqrt(2.0)


@dataclass(frozen=True)
class NeuralModelConfig:
    state_continuous_dim: int
    parameter_continuous_dim: int
    state_cardinalities: tuple[int, ...]
    parameter_cardinalities: tuple[int, ...]
    hidden_dim: int
    depth: int
    n_experts: int
    dropout: float
    embedding_dim: int
    consumption_output_scale: float
    deposit_output_scale: float
    consumption_initial_ratio: float
    deposit_initial_ratio: float


class ConditionalMoEPolicyNetwork(nn.Module):
    """Smooth state network with parameter FiLM modulation and soft experts."""

    def __init__(self, config: NeuralModelConfig) -> None:
        super().__init__()
        self.config = config
        self.state_embeddings = CategoricalEmbeddings(
            config.state_cardinalities, config.embedding_dim
        )
        self.parameter_embeddings = CategoricalEmbeddings(
            config.parameter_cardinalities, config.embedding_dim
        )
        state_input_dim = (
            int(config.state_continuous_dim) + self.state_embeddings.output_dim
        )
        parameter_input_dim = (
            int(config.parameter_continuous_dim)
            + self.parameter_embeddings.output_dim
        )
        context_input_dim = max(parameter_input_dim, 1)

        self.state_encoder = nn.Sequential(
            nn.Linear(max(state_input_dim, 1), config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.parameter_encoder = nn.Sequential(
            nn.Linear(context_input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(
                    config.hidden_dim,
                    config.hidden_dim,
                    config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.gate = nn.Linear(config.hidden_dim, config.n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 2),
                )
                for _ in range(config.n_experts)
            ]
        )
        self.regime_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_dim // 2, 3),
        )
        self.register_buffer(
            "consumption_output_scale",
            torch.tensor(float(config.consumption_output_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "deposit_output_scale",
            torch.tensor(float(config.deposit_output_scale), dtype=torch.float32),
        )
        self._initialize_output_biases()

    def _initialize_output_biases(self) -> None:
        c_scale = max(float(self.config.consumption_output_scale), 1.0e-6)
        c_ratio = max(float(self.config.consumption_initial_ratio) / c_scale, 1.0e-6)
        c_bias = math.log(math.expm1(c_ratio)) if c_ratio < 20.0 else c_ratio
        d_scale = max(float(self.config.deposit_output_scale), 1.0e-6)
        d_bias = float(self.config.deposit_initial_ratio) / d_scale
        for expert in self.experts:
            final = expert[-1]
            if isinstance(final, nn.Linear):
                nn.init.normal_(final.weight, mean=0.0, std=1.0e-3)
                with torch.no_grad():
                    final.bias[0] = c_bias
                    final.bias[1] = d_bias

    def forward(
        self,
        state_continuous: torch.Tensor,
        parameter_continuous: torch.Tensor,
        state_categorical: torch.Tensor,
        parameter_categorical: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state_embedding = self.state_embeddings(state_categorical)
        parameter_embedding = self.parameter_embeddings(parameter_categorical)
        state_input = torch.cat([state_continuous, state_embedding], dim=1)
        parameter_input = torch.cat(
            [parameter_continuous, parameter_embedding], dim=1
        )
        if state_input.shape[1] == 0:
            state_input = state_continuous.new_zeros((len(state_continuous), 1))
        if parameter_input.shape[1] == 0:
            parameter_input = state_continuous.new_zeros((len(state_continuous), 1))

        context = self.parameter_encoder(parameter_input)
        hidden = self.state_encoder(state_input)
        for block in self.blocks:
            hidden = block(hidden, context)

        gate = torch.softmax(self.gate(context), dim=-1)
        expert_outputs = torch.stack(
            [expert(hidden) for expert in self.experts], dim=1
        )
        raw = torch.sum(gate.unsqueeze(-1) * expert_outputs, dim=1)
        consumption_ratio = F.softplus(raw[:, 0]) * self.consumption_output_scale
        deposit_ratio = raw[:, 1] * self.deposit_output_scale
        return {
            "consumption_ratio": consumption_ratio,
            "deposit_ratio": deposit_ratio,
            "deposit_regime_logits": self.regime_head(hidden),
            "gate": gate,
        }


# =============================================================================
# Accounting and losses
# =============================================================================


@dataclass(frozen=True)
class LossWeights:
    consumption: float = 1.0
    deposit: float = 1.0
    liquid_outflow: float = 0.5
    delta_liquid: float = 1.0
    delta_illiquid: float = 0.5
    mpc: float = 8.0
    mpd: float = 1.0
    liquid_response: float = 0.5
    deposit_regime: float = 0.10
    expert_balance: float = 0.01


@dataclass(frozen=True)
class TrialConfig:
    hidden_dim: int
    depth: int
    n_experts: int
    dropout: float
    learning_rate: float
    weight_decay: float
    mpc_loss_weight: float
    deposit_loss_weight: float
    liquid_change_loss_weight: float
    mpd_loss_weight: float


@dataclass
class BatchLosses:
    total: torch.Tensor
    components: dict[str, torch.Tensor]



def _tensor(array: np.ndarray, indices: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array[indices], device=device)


def _forward_indices(
    model: ConditionalMoEPolicyNetwork,
    data: PreparedData,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return model(
        _tensor(data.state_continuous, indices, device),
        _tensor(data.parameter_continuous, indices, device),
        _tensor(data.state_categorical, indices, device).long(),
        _tensor(data.parameter_categorical, indices, device).long(),
    )


def _accounting_from_ratios(
    consumption_ratio: torch.Tensor,
    deposit_ratio: torch.Tensor,
    data: PreparedData,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    income = _tensor(data.income_scale, indices, device)
    consumption = consumption_ratio * income
    deposit = deposit_ratio * income
    after_tax = _tensor(data.after_tax_income, indices, device)
    b = _tensor(data.liquid_assets, indices, device)
    a = _tensor(data.illiquid_assets, indices, device)
    dt = _tensor(data.dt, indices, device)
    rb_pos = _tensor(data.rb_pos, indices, device)
    rb_neg = _tensor(data.rb_neg, indices, device)
    ra = _tensor(data.ra, indices, device)
    chi0 = _tensor(data.chi0, indices, device)
    chi1 = torch.clamp(_tensor(data.chi1, indices, device), min=1.0e-8)
    xi = _tensor(data.xi, indices, device)
    a_floor = torch.clamp(_tensor(data.a_floor, indices, device), min=1.0e-8)
    liquid_min = _tensor(data.liquid_min, indices, device)
    liquid_max = _tensor(data.liquid_max, indices, device)
    illiquid_min = _tensor(data.illiquid_min, indices, device)
    illiquid_max = _tensor(data.illiquid_max, indices, device)

    adjustment_cost = chi0 * torch.abs(deposit) + 0.5 * chi1 * deposit.square() / torch.maximum(
        a, a_floor
    )
    outflow = consumption + deposit + adjustment_cost
    liquid_return = torch.where(b >= 0.0, rb_pos * b, rb_neg * b)
    liquid_drift = (1.0 - xi) * after_tax + liquid_return - outflow
    illiquid_drift = ra * a + xi * after_tax + deposit

    b_tolerance = 1.0e-10 * torch.maximum(
        torch.ones_like(b), torch.maximum(torch.abs(liquid_min), torch.abs(liquid_max))
    )
    a_tolerance = 1.0e-10 * torch.maximum(
        torch.ones_like(a),
        torch.maximum(torch.abs(illiquid_min), torch.abs(illiquid_max)),
    )
    liquid_drift = torch.where(
        b <= liquid_min + b_tolerance,
        torch.clamp(liquid_drift, min=0.0),
        liquid_drift,
    )
    liquid_drift = torch.where(
        b >= liquid_max - b_tolerance,
        torch.clamp(liquid_drift, max=0.0),
        liquid_drift,
    )
    illiquid_drift = torch.where(
        a <= illiquid_min + a_tolerance,
        torch.clamp(illiquid_drift, min=0.0),
        illiquid_drift,
    )
    illiquid_drift = torch.where(
        a >= illiquid_max - a_tolerance,
        torch.clamp(illiquid_drift, max=0.0),
        illiquid_drift,
    )

    next_b = torch.minimum(
        torch.maximum(b + dt * liquid_drift, liquid_min), liquid_max
    )
    next_a = torch.minimum(
        torch.maximum(a + dt * illiquid_drift, illiquid_min), illiquid_max
    )
    return {
        "consumption": consumption,
        "deposit": deposit,
        "liquid_outflow_ratio": outflow / income,
        "delta_liquid_ratio": (next_b - b) / income,
        "delta_illiquid_ratio": (next_a - a) / income,
    }


def _weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    scale: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    residual = (prediction - target) / max(float(scale), 1.0e-8)
    squared = residual.square()
    if weights is None:
        return torch.mean(squared)
    normalized = weights / torch.clamp(torch.mean(weights), min=1.0e-8)
    return torch.mean(normalized * squared)


def _level_losses(
    model: ConditionalMoEPolicyNetwork,
    data: PreparedData,
    indices: np.ndarray,
    *,
    device: torch.device,
    scales: LossScales,
    weights: LossWeights,
    low_liquidity_scale: float,
    low_liquidity_boost: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prediction = _forward_indices(model, data, indices, device)
    accounting = _accounting_from_ratios(
        prediction["consumption_ratio"],
        prediction["deposit_ratio"],
        data,
        indices,
        device,
    )
    slack = torch.clamp(_tensor(data.liquid_slack_ratio, indices, device), min=0.0)
    row_weights = 1.0 + float(low_liquidity_boost) * torch.exp(
        -slack / max(float(low_liquidity_scale), 1.0e-6)
    )
    losses = {
        "consumption": _weighted_mse(
            prediction["consumption_ratio"],
            _tensor(data.consumption_ratio, indices, device),
            scales.consumption,
            row_weights,
        ),
        "deposit": _weighted_mse(
            prediction["deposit_ratio"],
            _tensor(data.deposit_ratio, indices, device),
            scales.deposit,
            row_weights,
        ),
        "liquid_outflow": _weighted_mse(
            accounting["liquid_outflow_ratio"],
            _tensor(data.liquid_outflow_ratio, indices, device),
            scales.liquid_outflow,
            row_weights,
        ),
        "delta_liquid": _weighted_mse(
            accounting["delta_liquid_ratio"],
            _tensor(data.delta_liquid_ratio, indices, device),
            scales.delta_liquid,
            row_weights,
        ),
        "delta_illiquid": _weighted_mse(
            accounting["delta_illiquid_ratio"],
            _tensor(data.delta_illiquid_ratio, indices, device),
            scales.delta_illiquid,
            row_weights,
        ),
        "deposit_regime": F.cross_entropy(
            prediction["deposit_regime_logits"],
            _tensor(data.deposit_regime, indices, device).long(),
        ),
    }
    gate_mean = torch.mean(prediction["gate"], dim=0)
    uniform = torch.full_like(gate_mean, 1.0 / gate_mean.numel())
    losses["expert_balance"] = torch.mean((gate_mean - uniform).square())
    return losses, accounting


def _pair_losses(
    model: ConditionalMoEPolicyNetwork,
    data: PreparedData,
    pairs: PairData,
    pair_indices: np.ndarray,
    *,
    device: torch.device,
    scales: LossScales,
) -> dict[str, torch.Tensor]:
    baseline = pairs.baseline[pair_indices]
    treated = pairs.treated[pair_indices]
    # Evaluate both members in one network call. This avoids two accelerator
    # launches for every pair batch and materially reduces MPC-training time.
    combined_indices = np.concatenate([baseline, treated])
    combined_prediction = _forward_indices(
        model, data, combined_indices, device
    )
    n_pairs = len(baseline)
    baseline_prediction = {
        key: value[:n_pairs] for key, value in combined_prediction.items()
    }
    treated_prediction = {
        key: value[n_pairs:] for key, value in combined_prediction.items()
    }
    baseline_accounting = _accounting_from_ratios(
        baseline_prediction["consumption_ratio"],
        baseline_prediction["deposit_ratio"],
        data,
        baseline,
        device,
    )
    treated_accounting = _accounting_from_ratios(
        treated_prediction["consumption_ratio"],
        treated_prediction["deposit_ratio"],
        data,
        treated,
        device,
    )
    check = _tensor(pairs.checks, pair_indices, device)
    pair_weight = _tensor(pairs.weights, pair_indices, device)
    predicted_mpc = (
        treated_accounting["consumption"]
        - baseline_accounting["consumption"]
    ) / check
    predicted_mpd = (
        treated_accounting["deposit"] - baseline_accounting["deposit"]
    ) / check

    baseline_delta_b = (
        baseline_accounting["delta_liquid_ratio"]
        * _tensor(data.income_scale, baseline, device)
    )
    treated_delta_b = (
        treated_accounting["delta_liquid_ratio"]
        * _tensor(data.income_scale, treated, device)
    )
    predicted_liquid_response = (treated_delta_b - baseline_delta_b) / check
    return {
        "mpc": _weighted_mse(
            predicted_mpc,
            _tensor(pairs.exact_mpc, pair_indices, device),
            scales.mpc,
            pair_weight,
        ),
        "mpd": _weighted_mse(
            predicted_mpd,
            _tensor(pairs.exact_mpd, pair_indices, device),
            scales.mpd,
            pair_weight,
        ),
        "liquid_response": _weighted_mse(
            predicted_liquid_response,
            _tensor(pairs.exact_liquid_response, pair_indices, device),
            scales.liquid_response,
            pair_weight,
        ),
    }


def _combine_losses(
    level: Mapping[str, torch.Tensor],
    pair: Mapping[str, torch.Tensor],
    weights: LossWeights,
) -> BatchLosses:
    components = {**dict(level), **dict(pair)}
    total = (
        weights.consumption * level["consumption"]
        + weights.deposit * level["deposit"]
        + weights.liquid_outflow * level["liquid_outflow"]
        + weights.delta_liquid * level["delta_liquid"]
        + weights.delta_illiquid * level["delta_illiquid"]
        + weights.deposit_regime * level["deposit_regime"]
        + weights.expert_balance * level["expert_balance"]
        + weights.mpc * pair["mpc"]
        + weights.mpd * pair["mpd"]
        + weights.liquid_response * pair["liquid_response"]
    )
    return BatchLosses(total=total, components=components)


# =============================================================================
# Training and evaluation
# =============================================================================


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    predictions: dict[str, np.ndarray] | None = None


def _autocast_context(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype != "none"
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _predict_ratios(
    model: ConditionalMoEPolicyNetwork,
    data: PreparedData,
    *,
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    n = len(data)
    consumption = np.empty(n, dtype=np.float32)
    deposit = np.empty(n, dtype=np.float32)
    regime = np.empty(n, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            stop = min(start + int(batch_size), n)
            indices = np.arange(start, stop, dtype=np.int64)
            with _autocast_context(device, amp_dtype):
                prediction = _forward_indices(model, data, indices, device)
            consumption[start:stop] = (
                prediction["consumption_ratio"].float().cpu().numpy()
            )
            deposit[start:stop] = prediction["deposit_ratio"].float().cpu().numpy()
            regime[start:stop] = (
                prediction["deposit_regime_logits"]
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
    return consumption, deposit, regime


def _accounting_numpy(
    consumption_ratio: np.ndarray,
    deposit_ratio: np.ndarray,
    data: PreparedData,
) -> dict[str, np.ndarray]:
    income = data.income_scale.astype(float)
    consumption = np.asarray(consumption_ratio, dtype=float) * income
    deposit = np.asarray(deposit_ratio, dtype=float) * income
    b = data.liquid_assets.astype(float)
    a = data.illiquid_assets.astype(float)
    y = data.after_tax_income.astype(float)
    adjustment = data.chi0 * np.abs(deposit) + 0.5 * data.chi1 * deposit**2 / np.maximum(
        a, data.a_floor
    )
    outflow = consumption + deposit + adjustment
    liquid_return = np.where(b >= 0.0, data.rb_pos * b, data.rb_neg * b)
    sb = (1.0 - data.xi) * y + liquid_return - outflow
    sa = data.ra * a + data.xi * y + deposit
    b_tol = 1.0e-10 * np.maximum(
        1.0, np.maximum(np.abs(data.liquid_min), np.abs(data.liquid_max))
    )
    a_tol = 1.0e-10 * np.maximum(
        1.0, np.maximum(np.abs(data.illiquid_min), np.abs(data.illiquid_max))
    )
    sb = np.where(b <= data.liquid_min + b_tol, np.maximum(sb, 0.0), sb)
    sb = np.where(b >= data.liquid_max - b_tol, np.minimum(sb, 0.0), sb)
    sa = np.where(a <= data.illiquid_min + a_tol, np.maximum(sa, 0.0), sa)
    sa = np.where(a >= data.illiquid_max - a_tol, np.minimum(sa, 0.0), sa)
    next_b = np.clip(b + data.dt * sb, data.liquid_min, data.liquid_max)
    next_a = np.clip(a + data.dt * sa, data.illiquid_min, data.illiquid_max)
    return {
        "consumption": consumption,
        "deposit": deposit,
        "liquid_outflow": outflow,
        "delta_liquid_assets": next_b - b,
        "delta_illiquid_assets": next_a - a,
    }


def _evaluate_model(
    model: ConditionalMoEPolicyNetwork,
    data: PreparedData,
    pairs: PairData,
    *,
    device: torch.device,
    batch_size: int,
    amp_dtype: str,
    scales: LossScales,
    return_predictions: bool = False,
) -> EvaluationResult:
    consumption_ratio, deposit_ratio, regime_pred = _predict_ratios(
        model,
        data,
        device=device,
        batch_size=batch_size,
        amp_dtype=amp_dtype,
    )
    prediction = _accounting_numpy(consumption_ratio, deposit_ratio, data)
    true = {
        "consumption": data.consumption_ratio * data.income_scale,
        "deposit": data.deposit_ratio * data.income_scale,
        "liquid_outflow": data.liquid_outflow_ratio * data.income_scale,
        "delta_liquid_assets": data.delta_liquid_ratio * data.income_scale,
        "delta_illiquid_assets": data.delta_illiquid_ratio * data.income_scale,
    }
    metrics: dict[str, float] = {}
    for target in true:
        values = _metrics(true[target], prediction[target])
        for name, value in values.items():
            metrics[f"{target}_{name}"] = value

    baseline = pairs.baseline
    treated = pairs.treated
    check = pairs.checks.astype(float)
    predicted_mpc = (
        prediction["consumption"][treated] - prediction["consumption"][baseline]
    ) / check
    predicted_mpd = (
        prediction["deposit"][treated] - prediction["deposit"][baseline]
    ) / check
    predicted_liquid_response = (
        prediction["delta_liquid_assets"][treated]
        - prediction["delta_liquid_assets"][baseline]
    ) / check
    for target, exact, predicted in (
        ("mpc", pairs.exact_mpc, predicted_mpc),
        ("mpd", pairs.exact_mpd, predicted_mpd),
        (
            "liquid_response",
            pairs.exact_liquid_response,
            predicted_liquid_response,
        ),
    ):
        values = _metrics(exact, predicted)
        for name, value in values.items():
            metrics[f"{target}_{name}"] = value

    metrics["deposit_regime_accuracy"] = float(
        np.mean(regime_pred == data.deposit_regime)
    )
    # A fixed scale-normalized selection criterion allows fair comparison when
    # the trial-specific training loss weights differ.
    metrics["selection_score"] = (
        0.50
        * metrics["consumption_rmse"]
        / max(scales.consumption * float(np.median(data.income_scale)), 1.0e-8)
        + 1.00
        * metrics["deposit_rmse"]
        / max(scales.deposit * float(np.median(data.income_scale)), 1.0e-8)
        + 1.00
        * metrics["delta_liquid_assets_rmse"]
        / max(scales.delta_liquid * float(np.median(data.income_scale)), 1.0e-8)
        + 0.25
        * metrics["delta_illiquid_assets_rmse"]
        / max(scales.delta_illiquid * float(np.median(data.income_scale)), 1.0e-8)
        + 3.00 * metrics["mpc_rmse"] / max(scales.mpc, 1.0e-8)
        + 0.50 * metrics["mpd_rmse"] / max(scales.mpd, 1.0e-8)
        + 0.25
        * metrics["liquid_response_rmse"]
        / max(scales.liquid_response, 1.0e-8)
    )
    output_predictions = None
    if return_predictions:
        output_predictions = {
            **prediction,
            "mpc": predicted_mpc,
            "mpd": predicted_mpd,
            "liquid_response": predicted_liquid_response,
            "deposit_regime": regime_pred,
        }
    return EvaluationResult(metrics=metrics, predictions=output_predictions)


def _cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float = 0.05,
):
    warmup_steps = max(1, int(round(total_steps * warmup_fraction)))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1.0e-3, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _loss_weights_from_trial(trial: TrialConfig, args: argparse.Namespace) -> LossWeights:
    return LossWeights(
        consumption=float(args.consumption_loss_weight),
        deposit=float(trial.deposit_loss_weight),
        liquid_outflow=float(args.liquid_outflow_loss_weight),
        delta_liquid=float(trial.liquid_change_loss_weight),
        delta_illiquid=float(args.illiquid_change_loss_weight),
        mpc=float(trial.mpc_loss_weight),
        mpd=float(trial.mpd_loss_weight),
        liquid_response=float(args.liquid_response_loss_weight),
        deposit_regime=float(args.deposit_regime_loss_weight),
        expert_balance=float(args.expert_balance_loss_weight),
    )


def _make_model(
    trial: TrialConfig,
    *,
    preprocessor: NeuralInputPreprocessor,
    data: PreparedData,
    scales: LossScales,
    embedding_dim: int,
) -> ConditionalMoEPolicyNetwork:
    config = NeuralModelConfig(
        state_continuous_dim=data.state_continuous.shape[1],
        parameter_continuous_dim=data.parameter_continuous.shape[1],
        state_cardinalities=preprocessor.state_cardinalities,
        parameter_cardinalities=preprocessor.parameter_cardinalities,
        hidden_dim=int(trial.hidden_dim),
        depth=int(trial.depth),
        n_experts=int(trial.n_experts),
        dropout=float(trial.dropout),
        embedding_dim=int(embedding_dim),
        consumption_output_scale=max(scales.consumption, 1.0e-4),
        deposit_output_scale=max(scales.deposit, 1.0e-4),
        consumption_initial_ratio=float(np.median(data.consumption_ratio)),
        deposit_initial_ratio=float(np.median(data.deposit_ratio)),
    )
    return ConditionalMoEPolicyNetwork(config)


def _train_model(
    model: ConditionalMoEPolicyNetwork,
    train_data: PreparedData,
    train_pairs: PairData,
    *,
    validation_data: PreparedData | None,
    validation_pairs: PairData | None,
    trial: TrialConfig,
    loss_weights: LossWeights,
    scales: LossScales,
    args: argparse.Namespace,
    device: torch.device,
    max_epochs: int,
    patience: int | None,
    trial_label: str,
) -> tuple[ConditionalMoEPolicyNetwork, int, dict[str, float], list[dict[str, Any]]]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(trial.learning_rate),
        weight_decay=float(trial.weight_decay),
        betas=(0.9, 0.99),
    )
    n_level_steps = max(1, math.ceil(len(train_data) / int(args.batch_size)))
    total_optimizer_steps = max(1, n_level_steps * int(max_epochs) // int(args.grad_accum_steps))
    scheduler = _cosine_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_fraction=float(args.warmup_fraction),
    )
    use_scaler = device.type == "cuda" and args.amp_dtype == "fp16"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    rng = np.random.default_rng(int(args.random_state) + abs(hash(trial_label)) % 100000)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_metrics: dict[str, float] = {"selection_score": float("inf")}
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    global_optimizer_step = 0
    start_time = time.perf_counter()

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        level_order = rng.permutation(len(train_data))
        pair_order = rng.permutation(len(train_pairs))
        pair_cursor = 0
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        n_observations = 0
        epoch_start = time.perf_counter()

        for step, start in enumerate(
            range(0, len(train_data), int(args.batch_size)), start=1
        ):
            level_indices = level_order[start : start + int(args.batch_size)]
            if pair_cursor + int(args.pair_batch_size) > len(pair_order):
                pair_order = rng.permutation(len(train_pairs))
                pair_cursor = 0
            pair_indices = pair_order[
                pair_cursor : pair_cursor + int(args.pair_batch_size)
            ]
            pair_cursor += len(pair_indices)

            with _autocast_context(device, args.amp_dtype):
                level_losses, _ = _level_losses(
                    model,
                    train_data,
                    level_indices,
                    device=device,
                    scales=scales,
                    weights=loss_weights,
                    low_liquidity_scale=float(args.low_liquidity_scale),
                    low_liquidity_boost=float(args.level_low_liquidity_boost),
                )
                pair_losses = _pair_losses(
                    model,
                    train_data,
                    train_pairs,
                    pair_indices,
                    device=device,
                    scales=scales,
                )
                batch_losses = _combine_losses(
                    level_losses, pair_losses, loss_weights
                )
                loss = batch_losses.total / int(args.grad_accum_steps)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (
                step % int(args.grad_accum_steps) == 0 or step == n_level_steps
            )
            if should_step:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(args.gradient_clip)
                )
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_optimizer_step += 1

            batch_n = len(level_indices)
            n_observations += batch_n
            sums["total"] = sums.get("total", 0.0) + float(
                batch_losses.total.detach().float().cpu()
            ) * batch_n
            for name, value in batch_losses.components.items():
                sums[name] = sums.get(name, 0.0) + float(
                    value.detach().float().cpu()
                ) * batch_n

            if step == 1 or step % int(args.log_every_steps) == 0 or step == n_level_steps:
                current_lr = optimizer.param_groups[0]["lr"]
                _status(
                    f"[{trial_label}] epoch {epoch:02d}/{max_epochs:02d} "
                    f"step {step:04d}/{n_level_steps:04d} "
                    f"loss={float(batch_losses.total.detach().float().cpu()):.4f} "
                    f"lr={current_lr:.3g}"
                )

        train_summary = {
            f"train_{name}": value / max(n_observations, 1)
            for name, value in sums.items()
        }
        row: dict[str, Any] = {
            "epoch": epoch,
            **train_summary,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }

        if validation_data is not None and validation_pairs is not None:
            evaluation = _evaluate_model(
                model,
                validation_data,
                validation_pairs,
                device=device,
                batch_size=int(args.evaluation_batch_size),
                amp_dtype=str(args.amp_dtype),
                scales=scales,
            )
            row.update({f"validation_{key}": value for key, value in evaluation.metrics.items()})
            score = float(evaluation.metrics["selection_score"])
            _status(
                f"[{trial_label}] epoch {epoch:02d} complete in {_elapsed(epoch_start)}; "
                f"validation score={score:.4f}; "
                f"R²(c)={evaluation.metrics['consumption_r2']:.4f}; "
                f"R²(d)={evaluation.metrics['deposit_r2']:.4f}; "
                f"R²(Δb)={evaluation.metrics['delta_liquid_assets_r2']:.4f}; "
                f"R²(MPC)={evaluation.metrics['mpc_r2']:.4f}"
            )
            if score < float(best_metrics.get("selection_score", np.inf)) - float(
                args.early_stopping_min_delta
            ):
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_metrics = dict(evaluation.metrics)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= int(patience):
                _status(
                    f"[{trial_label}] early stopping after epoch {epoch}; "
                    f"best epoch={best_epoch}."
                )
                history.append(row)
                break
        else:
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = train_summary
            _status(
                f"[{trial_label}] epoch {epoch:02d} complete in {_elapsed(epoch_start)}; "
                f"mean training loss={train_summary['train_total']:.4f}"
            )
        history.append(row)

    model.load_state_dict(best_state)
    _status(
        f"[{trial_label}] training finished in {_elapsed(start_time)}; "
        f"selected epoch={best_epoch}."
    )
    return model, best_epoch, best_metrics, history


# =============================================================================
# Hyperparameter grid
# =============================================================================


def _trial_grid(args: argparse.Namespace) -> list[TrialConfig]:
    product = itertools.product(
        args.hidden_dim_grid,
        args.depth_grid,
        args.n_experts_grid,
        args.dropout_grid,
        args.learning_rate_grid,
        args.weight_decay_grid,
        args.mpc_loss_weight_grid,
        args.deposit_loss_weight_grid,
        args.liquid_change_loss_weight_grid,
        args.mpd_loss_weight_grid,
    )
    trials = [
        TrialConfig(
            hidden_dim=int(hidden),
            depth=int(depth),
            n_experts=int(experts),
            dropout=float(dropout),
            learning_rate=float(lr),
            weight_decay=float(weight_decay),
            mpc_loss_weight=float(mpc_weight),
            deposit_loss_weight=float(deposit_weight),
            liquid_change_loss_weight=float(liquid_weight),
            mpd_loss_weight=float(mpd_weight),
        )
        for (
            hidden,
            depth,
            experts,
            dropout,
            lr,
            weight_decay,
            mpc_weight,
            deposit_weight,
            liquid_weight,
            mpd_weight,
        ) in product
    ]
    rng = np.random.default_rng(int(args.random_state))
    rng.shuffle(trials)
    if int(args.max_grid_trials) > 0 and len(trials) > int(args.max_grid_trials):
        trials = trials[: int(args.max_grid_trials)]
    return trials


# =============================================================================
# Saved bundle and prediction API
# =============================================================================


@dataclass
class NeuralPolicySurrogateBundle:
    model_config: NeuralModelConfig
    model_state: dict[str, torch.Tensor]
    preprocessor: NeuralInputPreprocessor
    loss_scales: LossScales
    default_values: dict[str, Any]
    accounting_defaults: dict[str, Any]
    feature_ranges: dict[str, dict[str, Any]]
    output_bounds: dict[str, tuple[float, float]]
    training_metadata: dict[str, Any]

    def _model(self, device: torch.device) -> ConditionalMoEPolicyNetwork:
        model = ConditionalMoEPolicyNetwork(self.model_config)
        model.load_state_dict(self.model_state)
        return model.to(device).eval()

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        device: str | torch.device = "auto",
        batch_size: int = 65_536,
        project: bool = True,
    ) -> pd.DataFrame:
        if isinstance(device, str):
            target_device = _device_from_args(device, require_gpu=False)
        else:
            target_device = device
        prepared = prepare_policy_inputs(
            frame,
            feature_spec=self.preprocessor.feature_spec,
            default_values=self.default_values,
            accounting_defaults=self.accounting_defaults,
            money_units=str(self.training_metadata.get("money_units", "model")),
        )
        s_cont, p_cont, s_cat, p_cat = self.preprocessor.transform_prepared(prepared)
        model = self._model(target_device)
        n = len(prepared)
        c_ratio = np.empty(n, dtype=np.float32)
        d_ratio = np.empty(n, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n, int(batch_size)):
                stop = min(start + int(batch_size), n)
                prediction = model(
                    torch.as_tensor(s_cont[start:stop], device=target_device),
                    torch.as_tensor(p_cont[start:stop], device=target_device),
                    torch.as_tensor(s_cat[start:stop], device=target_device).long(),
                    torch.as_tensor(p_cat[start:stop], device=target_device).long(),
                )
                c_ratio[start:stop] = prediction["consumption_ratio"].cpu().numpy()
                d_ratio[start:stop] = prediction["deposit_ratio"].cpu().numpy()
        permanent = pd.to_numeric(
            prepared["labor_income_state"], errors="coerce"
        ).to_numpy(dtype=float)
        floor = max(
            1.0e-6,
            1.0e-3 * abs(float(self.default_values.get("labor_income_state", 1.0))),
        )
        income_scale = np.maximum(np.abs(permanent), floor)
        primitive = pd.DataFrame(
            {
                "consumption": c_ratio * income_scale,
                "deposit": d_ratio * income_scale,
            },
            index=prepared.index,
        )
        return reconstruct_policy_outputs(
            prepared,
            primitive,
            accounting_defaults=self.accounting_defaults,
            money_units=str(self.training_metadata.get("money_units", "model")),
            output_bounds=self.output_bounds,
            liquid_outflow_weight=0.0,
            project=project,
        )

    def make_default_row(self) -> pd.DataFrame:
        return pd.DataFrame([self.default_values])

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bundle_format_version": BUNDLE_FORMAT_VERSION,
            "model_config": asdict(self.model_config),
            "model_state": {key: value.detach().cpu() for key, value in self.model_state.items()},
            "preprocessor": self.preprocessor.to_state(),
            "loss_scales": asdict(self.loss_scales),
            "default_values": _jsonable(self.default_values),
            "accounting_defaults": _jsonable(self.accounting_defaults),
            "feature_ranges": _jsonable(self.feature_ranges),
            "output_bounds": _jsonable(self.output_bounds),
            "training_metadata": _jsonable(self.training_metadata),
        }
        torch.save(payload, destination)
        return destination

    @staticmethod
    def load(path: str | Path) -> "NeuralPolicySurrogateBundle":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch before the weights_only keyword.
            payload = torch.load(path, map_location="cpu")
        version = int(payload.get("bundle_format_version", 0))
        if version != BUNDLE_FORMAT_VERSION:
            raise TypeError(
                f"Unsupported neural bundle format {version}; expected "
                f"{BUNDLE_FORMAT_VERSION}."
            )
        config_state = dict(payload["model_config"])
        config_state["state_cardinalities"] = tuple(config_state["state_cardinalities"])
        config_state["parameter_cardinalities"] = tuple(
            config_state["parameter_cardinalities"]
        )
        output_bounds = {
            str(key): (float(value[0]), float(value[1]))
            for key, value in payload.get("output_bounds", {}).items()
        }
        return NeuralPolicySurrogateBundle(
            model_config=NeuralModelConfig(**config_state),
            model_state=payload["model_state"],
            preprocessor=NeuralInputPreprocessor.from_state(
                payload["preprocessor"]
            ),
            loss_scales=LossScales(**payload["loss_scales"]),
            default_values=dict(payload["default_values"]),
            accounting_defaults=dict(payload["accounting_defaults"]),
            feature_ranges=dict(payload["feature_ranges"]),
            output_bounds=output_bounds,
            training_metadata=dict(payload["training_metadata"]),
        )


# =============================================================================
# Validation tables
# =============================================================================


def _validation_tables(
    raw_test: pd.DataFrame,
    prepared_test: PreparedData,
    test_pairs: PairData,
    evaluation: EvaluationResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if evaluation.predictions is None:
        raise ValueError("Validation predictions were not retained.")
    predictions = evaluation.predictions
    true_level = {
        "consumption": prepared_test.consumption_ratio
        * prepared_test.income_scale,
        "deposit": prepared_test.deposit_ratio * prepared_test.income_scale,
        "delta_liquid_assets": prepared_test.delta_liquid_ratio
        * prepared_test.income_scale,
        "delta_illiquid_assets": prepared_test.delta_illiquid_ratio
        * prepared_test.income_scale,
    }
    aggregate_rows: list[dict[str, Any]] = []
    for target, true in true_level.items():
        values = _metrics(true, predictions[target])
        aggregate_rows.append({"target": target, **values})
    for target, exact in (
        ("mpc", test_pairs.exact_mpc),
        ("mpd", test_pairs.exact_mpd),
        ("liquid_response", test_pairs.exact_liquid_response),
    ):
        values = _metrics(exact, predictions[target])
        aggregate_rows.append({"target": target, **values})
    aggregate = pd.DataFrame(aggregate_rows)

    model_ids = raw_test["model_id"].astype(str).to_numpy(dtype=object)
    by_model_rows: list[dict[str, Any]] = []
    for model_id in sorted(set(model_ids.tolist())):
        mask = model_ids == model_id
        for target, true in true_level.items():
            values = _metrics(true[mask], predictions[target][mask])
            by_model_rows.append(
                {"model_id": model_id, "target": target, **values}
            )
    pair_model_ids = test_pairs.model_ids.astype(str)
    for model_id in sorted(set(pair_model_ids.tolist())):
        mask = pair_model_ids == model_id
        for target, exact in (
            ("mpc", test_pairs.exact_mpc),
            ("mpd", test_pairs.exact_mpd),
            ("liquid_response", test_pairs.exact_liquid_response),
        ):
            values = _metrics(exact[mask], predictions[target][mask])
            by_model_rows.append(
                {"model_id": model_id, "target": target, **values}
            )
    return aggregate, pd.DataFrame(by_model_rows)


# =============================================================================
# Main pipeline
# =============================================================================


def _output_bounds(data: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for column in (
        "next_liquid_assets",
        "next_illiquid_assets",
        "consumption",
        "deposit",
    ):
        if column not in data:
            continue
        values = pd.to_numeric(data[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if len(finite):
            bounds[column] = (float(finite.min()), float(finite.max()))
    return bounds


def _fit_bundle_components(
    raw_fit: pd.DataFrame,
    *,
    feature_spec: FeatureSpec,
    money_units: str,
    preprocessor_fit_rows: int,
    random_state: int,
) -> tuple[
    NeuralInputPreprocessor,
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    accounting_defaults = _accounting_defaults_from_data(raw_fit)
    base_sample = _model_balanced_subsample(
        raw_fit,
        min(int(preprocessor_fit_rows), len(raw_fit)),
        random_state=random_state,
    )
    temporary_defaults: dict[str, Any] = {}
    # prepare_policy_inputs needs primitive defaults before the complete default
    # row can be calculated. Medians of direct columns are sufficient here.
    for column in feature_spec.state_columns + feature_spec.parameter_columns:
        if column in base_sample:
            values = pd.to_numeric(base_sample[column], errors="coerce")
            temporary_defaults[column] = (
                float(values.median()) if values.notna().any() else 0.0
            )
    for column in feature_spec.categorical_columns:
        if column in base_sample:
            mode = base_sample[column].dropna().astype(str).mode()
            temporary_defaults[column] = mode.iloc[0] if len(mode) else "__missing__"
    for column in (
        "liquid_grid_min",
        "liquid_grid_max",
        "illiquid_grid_min",
        "illiquid_grid_max",
        "money_scale",
    ):
        if column in base_sample:
            values = pd.to_numeric(base_sample[column], errors="coerce")
            temporary_defaults[column] = (
                float(values.median()) if values.notna().any() else 0.0
            )
    temporary_defaults.update(accounting_defaults)

    prepared_sample = prepare_policy_inputs(
        base_sample,
        feature_spec=feature_spec,
        default_values=temporary_defaults,
        accounting_defaults=accounting_defaults,
        money_units=money_units,
    )
    default_values = _default_row(
        prepared_sample, feature_spec, accounting_defaults
    )
    preprocessor = NeuralInputPreprocessor(feature_spec).fit(prepared_sample)
    feature_ranges = _feature_range_summary(prepared_sample, feature_spec)
    return preprocessor, accounting_defaults, default_values, feature_ranges


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a GPU neural HA policy surrogate with joint consumption, "
            "deposit, liquid-saving, and finite-difference MPC objectives."
        )
    )

    # Data and saved-grid options.
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sample-path", type=Path, default=None)
    parser.add_argument(
        "--reuse-sample",
        action="store_true",
        help="Reuse OUTPUT_DIR/neural_policy_grid_sample.pkl.gz when it exists.",
    )
    parser.add_argument("--policy-layout", default="GHKEBA")
    parser.add_argument("--money-units", choices=("model", "data"), default="model")
    parser.add_argument("--sampling-mode", choices=("population", "grid"), default="population")
    parser.add_argument("--rows-per-model", type=int, default=1_200)
    parser.add_argument("--max-total-rows", type=int, default=6_500_000)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    parser.add_argument("--mpc-pair-share", type=float, default=0.85)
    parser.add_argument(
        "--mpc-check-sizes",
        type=_parse_float_list,
        default=(0.05, 0.10, 0.188679, 0.188679, 0.188679, 0.20, 0.25),
    )
    parser.add_argument("--mpc-cross-zero-share", type=float, default=0.30)
    parser.add_argument("--include-retired", action="store_true")
    parser.add_argument("--poor-htm-share", type=float, default=0.35)
    parser.add_argument("--wealthy-htm-share", type=float, default=0.35)
    parser.add_argument("--broad-support-share", type=float, default=0.10)
    parser.add_argument("--nonemployment-share", type=float, default=0.20)
    parser.add_argument("--negative-liquid-share", type=float, default=0.10)
    parser.add_argument("--constraint-anchor-share", type=float, default=0.30)
    parser.add_argument("--require-compact-policies", action="store_true")
    parser.add_argument("--max-solver-distance", type=float, default=None)
    parser.add_argument("--max-relative-distance", type=float, default=None)
    parser.add_argument("--max-hjb-residual", type=float, default=None)
    parser.add_argument("--no-progress-bar", action="store_true")

    # Frozen model split.
    parser.add_argument("--model-split-file", type=Path, default=None)
    parser.add_argument("--test-share", type=float, default=0.20)
    parser.add_argument("--tuning-share", type=float, default=0.20)

    # Tuning subsamples and pair caps.
    parser.add_argument("--tuning-max-train-rows", type=int, default=1_000_000)
    parser.add_argument("--tuning-max-tune-rows", type=int, default=350_000)
    parser.add_argument("--preprocessor-fit-rows", type=int, default=500_000)
    parser.add_argument("--preprocess-chunk-size", type=int, default=100_000)
    parser.add_argument("--max-training-pairs", type=int, default=2_000_000)
    parser.add_argument("--max-tuning-pairs", type=int, default=250_000)
    parser.add_argument("--max-test-pairs", type=int, default=0)

    # Hardware and optimizer.
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--pair-batch-size", type=int, default=4_096)
    parser.add_argument("--evaluation-batch-size", type=int, default=65_536)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--embedding-dim", type=int, default=16)

    # Hyperparameter grid.
    parser.add_argument("--hidden-dim-grid", type=_parse_int_grid, default=[256, 384])
    parser.add_argument("--depth-grid", type=_parse_int_grid, default=[4, 6])
    parser.add_argument("--n-experts-grid", type=_parse_int_grid, default=[2, 4])
    parser.add_argument("--dropout-grid", type=_parse_float_grid, default=[0.0, 0.05])
    parser.add_argument("--learning-rate-grid", type=_parse_float_grid, default=[3.0e-4, 7.0e-4])
    parser.add_argument("--weight-decay-grid", type=_parse_float_grid, default=[1.0e-6, 1.0e-5])
    parser.add_argument("--mpc-loss-weight-grid", type=_parse_float_grid, default=[5.0, 10.0])
    parser.add_argument("--deposit-loss-weight-grid", type=_parse_float_grid, default=[1.0, 2.0])
    parser.add_argument("--liquid-change-loss-weight-grid", type=_parse_float_grid, default=[1.0, 2.0])
    parser.add_argument("--mpd-loss-weight-grid", type=_parse_float_grid, default=[0.5, 1.0])
    parser.add_argument("--max-grid-trials", type=int, default=12)
    parser.add_argument("--grid-max-epochs", type=int, default=18)
    parser.add_argument("--grid-patience", type=int, default=4)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-4)
    parser.add_argument("--final-epoch-multiplier", type=float, default=1.15)
    parser.add_argument(
        "--minimum-final-epochs",
        type=int,
        default=20,
        help=(
            "Minimum number of epochs for the development and production "
            "refits, even if the tuning model stops unusually early."
        ),
    )
    parser.add_argument(
        "--maximum-final-epochs",
        type=int,
        default=0,
        help="Optional upper bound for final refits; zero means no upper bound.",
    )

    # Fixed components of the multitask objective.
    parser.add_argument("--consumption-loss-weight", type=float, default=1.0)
    parser.add_argument("--liquid-outflow-loss-weight", type=float, default=0.5)
    parser.add_argument("--illiquid-change-loss-weight", type=float, default=0.5)
    parser.add_argument("--liquid-response-loss-weight", type=float, default=0.5)
    parser.add_argument("--deposit-regime-loss-weight", type=float, default=0.10)
    parser.add_argument("--expert-balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--low-liquidity-scale", type=float, default=0.02)
    parser.add_argument("--level-low-liquidity-boost", type=float, default=2.0)
    parser.add_argument("--pair-low-liquidity-boost", type=float, default=4.0)
    parser.add_argument("--mpc-tail-boost", type=float, default=0.75)

    # Run control.
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--skip-production-refit", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.mpc_pair_share < 1.0:
        parser.error("--mpc-pair-share must be in [0,1).")
    if args.batch_size <= 0 or args.pair_batch_size <= 0:
        parser.error("Batch sizes must be positive.")
    if args.grad_accum_steps <= 0:
        parser.error("--grad-accum-steps must be positive.")

    total_start = time.perf_counter()
    _seed_everything(int(args.random_state))
    device = _device_from_args(args.device, bool(args.require_gpu))
    _print_device(device)
    if (
        device.type == "cuda"
        and args.amp_dtype == "bf16"
        and not torch.cuda.is_bf16_supported()
    ):
        _status(
            "[hardware] this GPU does not support bfloat16; falling back to fp16 AMP."
        )
        args.amp_dtype = "fp16"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data, parameter_columns, categorical_parameters, sample_path, catalog = _load_or_build_sample(args)
    if not args.include_retired:
        before = len(data)
        data = _working_age_filter(data)
        _status(f"[data] working-age filter retained {len(data):,}/{before:,} rows")
    data["model_id"] = data["model_id"].astype(str)

    split_path = (
        Path(args.model_split_file).expanduser().resolve()
        if args.model_split_file is not None
        else output_dir / "model_split.json"
    )
    _status(f"[data 3/4] Loading or creating frozen model split: {split_path}")
    raw_train, raw_tune, raw_test = _load_or_create_model_split(
        data,
        categorical_parameters=categorical_parameters,
        validation_share=float(args.test_share),
        tuning_share=float(args.tuning_share),
        random_state=int(args.random_state),
        split_path=split_path,
    )
    _status(
        f"[data] rows train/tune/test = {len(raw_train):,}/"
        f"{len(raw_tune):,}/{len(raw_test):,}; models = "
        f"{raw_train['model_id'].nunique():,}/"
        f"{raw_tune['model_id'].nunique():,}/"
        f"{raw_test['model_id'].nunique():,}"
    )

    feature_spec = FeatureSpec(
        state_columns=tuple(DEFAULT_STATE_COLUMNS),
        engineered_state_columns=tuple(DEFAULT_ENGINEERED_STATE_COLUMNS),
        redundant_state_columns=tuple(DEFAULT_REDUNDANT_STATE_COLUMNS),
        parameter_columns=tuple(parameter_columns),
        categorical_state_columns=tuple(DEFAULT_CATEGORICAL_COLUMNS),
        categorical_parameter_columns=tuple(categorical_parameters),
    )

    # ------------------------------------------------------------------
    # Hyperparameter search on model-balanced tuning subsets.
    # ------------------------------------------------------------------
    _status("[data 4/4] Constructing model-balanced tuning subsets...")
    tuning_train_raw = _model_balanced_subsample(
        raw_train,
        int(args.tuning_max_train_rows),
        random_state=int(args.random_state) + 10,
    )
    tuning_tune_raw = _model_balanced_subsample(
        raw_tune,
        int(args.tuning_max_tune_rows),
        random_state=int(args.random_state) + 11,
    )
    (
        tuning_preprocessor,
        tuning_accounting_defaults,
        tuning_default_values,
        _,
    ) = _fit_bundle_components(
        tuning_train_raw,
        feature_spec=feature_spec,
        money_units=str(args.money_units),
        preprocessor_fit_rows=int(args.preprocessor_fit_rows),
        random_state=int(args.random_state) + 12,
    )
    tuning_train = _prepare_data_arrays(
        tuning_train_raw,
        preprocessor=tuning_preprocessor,
        accounting_defaults=tuning_accounting_defaults,
        default_values=tuning_default_values,
        money_units=str(args.money_units),
        chunk_size=int(args.preprocess_chunk_size),
        include_model_ids=False,
    )
    tuning_train_pairs = _build_pair_data(
        tuning_train_raw,
        max_pairs=int(args.max_training_pairs),
        random_state=int(args.random_state) + 13,
        low_liquidity_scale=float(args.low_liquidity_scale),
        low_liquidity_boost=float(args.pair_low_liquidity_boost),
        mpc_tail_boost=float(args.mpc_tail_boost),
    )
    tuning_scales = _loss_scales(tuning_train, tuning_train_pairs)
    tuning_tune = _prepare_data_arrays(
        tuning_tune_raw,
        preprocessor=tuning_preprocessor,
        accounting_defaults=tuning_accounting_defaults,
        default_values=tuning_default_values,
        money_units=str(args.money_units),
        chunk_size=int(args.preprocess_chunk_size),
        include_model_ids=False,
        deposit_zero_tolerance=tuning_scales.deposit_zero_tolerance,
    )
    tuning_tune_pairs = _build_pair_data(
        tuning_tune_raw,
        max_pairs=int(args.max_tuning_pairs),
        random_state=int(args.random_state) + 14,
        low_liquidity_scale=float(args.low_liquidity_scale),
        low_liquidity_boost=float(args.pair_low_liquidity_boost),
        mpc_tail_boost=float(args.mpc_tail_boost),
    )

    trials = _trial_grid(args)
    _status(
        f"[grid] evaluating {len(trials):,} neural configurations on "
        f"{len(tuning_train):,} training rows and {len(tuning_tune):,} tuning rows"
    )
    grid_rows: list[dict[str, Any]] = []
    best_trial: TrialConfig | None = None
    best_epoch = 0
    best_score = float("inf")

    for trial_number, trial in enumerate(trials, start=1):
        label = f"grid {trial_number:02d}/{len(trials):02d}"
        _status(f"[{label}] configuration: {asdict(trial)}")
        _seed_everything(int(args.random_state) + trial_number)
        model = _make_model(
            trial,
            preprocessor=tuning_preprocessor,
            data=tuning_train,
            scales=tuning_scales,
            embedding_dim=int(args.embedding_dim),
        )
        loss_weights = _loss_weights_from_trial(trial, args)
        model, selected_epoch, metrics, history = _train_model(
            model,
            tuning_train,
            tuning_train_pairs,
            validation_data=tuning_tune,
            validation_pairs=tuning_tune_pairs,
            trial=trial,
            loss_weights=loss_weights,
            scales=tuning_scales,
            args=args,
            device=device,
            max_epochs=int(args.grid_max_epochs),
            patience=int(args.grid_patience),
            trial_label=label,
        )
        row = {
            "trial": trial_number,
            **asdict(trial),
            "selected_epoch": selected_epoch,
            **metrics,
        }
        grid_rows.append(row)
        pd.DataFrame(grid_rows).to_csv(
            output_dir / "neural_grid_search.csv", index=False
        )
        if float(metrics["selection_score"]) < best_score:
            best_score = float(metrics["selection_score"])
            best_trial = trial
            best_epoch = int(selected_epoch)
            _status(
                f"[{label}] new best configuration: score={best_score:.4f}, "
                f"epoch={best_epoch}"
            )
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_trial is None:
        raise RuntimeError("The neural hyperparameter search did not produce a model.")
    final_epochs = max(
        int(args.minimum_final_epochs),
        int(math.ceil(best_epoch * float(args.final_epoch_multiplier))),
    )
    if int(args.maximum_final_epochs) > 0:
        final_epochs = min(final_epochs, int(args.maximum_final_epochs))
    _status(
        f"[grid] selected configuration {asdict(best_trial)}; "
        f"tuning score={best_score:.4f}; final epochs={final_epochs}"
    )

    # Tuning arrays and row-level copies are no longer needed.
    del (
        tuning_train,
        tuning_tune,
        tuning_train_pairs,
        tuning_tune_pairs,
        tuning_train_raw,
        tuning_tune_raw,
    )
    gc.collect()

    # ------------------------------------------------------------------
    # True holdout fit on training+tuning models, then frozen-test evaluation.
    # ------------------------------------------------------------------
    _status("[holdout 1/4] Preparing the development-only fit...")
    raw_development = pd.concat([raw_train, raw_tune], ignore_index=True)
    (
        development_preprocessor,
        development_accounting_defaults,
        development_default_values,
        development_feature_ranges,
    ) = _fit_bundle_components(
        raw_development,
        feature_spec=feature_spec,
        money_units=str(args.money_units),
        preprocessor_fit_rows=int(args.preprocessor_fit_rows),
        random_state=int(args.random_state) + 100,
    )
    development_data = _prepare_data_arrays(
        raw_development,
        preprocessor=development_preprocessor,
        accounting_defaults=development_accounting_defaults,
        default_values=development_default_values,
        money_units=str(args.money_units),
        chunk_size=int(args.preprocess_chunk_size),
        include_model_ids=False,
    )
    development_pairs = _build_pair_data(
        raw_development,
        max_pairs=int(args.max_training_pairs),
        random_state=int(args.random_state) + 101,
        low_liquidity_scale=float(args.low_liquidity_scale),
        low_liquidity_boost=float(args.pair_low_liquidity_boost),
        mpc_tail_boost=float(args.mpc_tail_boost),
    )
    development_scales = _loss_scales(development_data, development_pairs)
    _status("[holdout 2/4] Training the selected development-only network...")
    holdout_model = _make_model(
        best_trial,
        preprocessor=development_preprocessor,
        data=development_data,
        scales=development_scales,
        embedding_dim=int(args.embedding_dim),
    )
    holdout_model, _, _, holdout_history = _train_model(
        holdout_model,
        development_data,
        development_pairs,
        validation_data=None,
        validation_pairs=None,
        trial=best_trial,
        loss_weights=_loss_weights_from_trial(best_trial, args),
        scales=development_scales,
        args=args,
        device=device,
        max_epochs=final_epochs,
        patience=None,
        trial_label="holdout fit",
    )

    _status("[holdout 3/4] Preparing and evaluating the frozen test models...")
    test_data = _prepare_data_arrays(
        raw_test,
        preprocessor=development_preprocessor,
        accounting_defaults=development_accounting_defaults,
        default_values=development_default_values,
        money_units=str(args.money_units),
        chunk_size=int(args.preprocess_chunk_size),
        include_model_ids=True,
        deposit_zero_tolerance=development_scales.deposit_zero_tolerance,
    )
    max_test_pairs = None if int(args.max_test_pairs) <= 0 else int(args.max_test_pairs)
    test_pairs = _build_pair_data(
        raw_test,
        max_pairs=max_test_pairs,
        random_state=int(args.random_state) + 102,
        low_liquidity_scale=float(args.low_liquidity_scale),
        low_liquidity_boost=float(args.pair_low_liquidity_boost),
        mpc_tail_boost=float(args.mpc_tail_boost),
    )
    holdout_evaluation = _evaluate_model(
        holdout_model,
        test_data,
        test_pairs,
        device=device,
        batch_size=int(args.evaluation_batch_size),
        amp_dtype=str(args.amp_dtype),
        scales=development_scales,
        return_predictions=True,
    )
    _status(
        "[holdout] test results: "
        f"R²(c)={holdout_evaluation.metrics['consumption_r2']:.4f}; "
        f"R²(d)={holdout_evaluation.metrics['deposit_r2']:.4f}; "
        f"R²(Δb)={holdout_evaluation.metrics['delta_liquid_assets_r2']:.4f}; "
        f"R²(Δa)={holdout_evaluation.metrics['delta_illiquid_assets_r2']:.4f}; "
        f"R²(MPC)={holdout_evaluation.metrics['mpc_r2']:.4f}; "
        f"R²(MPD)={holdout_evaluation.metrics['mpd_r2']:.4f}"
    )
    aggregate_metrics, by_model_metrics = _validation_tables(
        raw_test,
        test_data,
        test_pairs,
        holdout_evaluation,
    )
    aggregate_metrics.to_csv(
        output_dir / "neural_validation_metrics.csv", index=False
    )
    by_model_metrics.to_csv(
        output_dir / "neural_validation_by_model.csv", index=False
    )
    pd.DataFrame(holdout_history).to_csv(
        output_dir / "neural_holdout_training_history.csv", index=False
    )

    actual_training_models = sorted(
        raw_train["model_id"].astype(str).unique().tolist()
    )
    actual_tuning_models = sorted(
        raw_tune["model_id"].astype(str).unique().tolist()
    )
    actual_test_models = sorted(
        raw_test["model_id"].astype(str).unique().tolist()
    )
    holdout_metadata = {
        "fit_scope": "development_only",
        "training_models": actual_training_models,
        "tuning_models": actual_tuning_models,
        "test_models": actual_test_models,
        "fit_models": sorted(
            set(actual_training_models) | set(actual_tuning_models)
        ),
        "selected_hyperparameters": asdict(best_trial),
        "selected_tuning_epoch": best_epoch,
        "final_epochs": final_epochs,
        "loss_weights": asdict(_loss_weights_from_trial(best_trial, args)),
        "money_units": str(args.money_units),
        "policy_layout": str(args.policy_layout),
        "sample_path": str(sample_path),
        "model_split_path": str(split_path),
        "test_metrics": holdout_evaluation.metrics,
        "created_by": Path(__file__).name,
    }
    holdout_bundle = NeuralPolicySurrogateBundle(
        model_config=holdout_model.config,
        model_state={key: value.detach().cpu() for key, value in holdout_model.state_dict().items()},
        preprocessor=development_preprocessor,
        loss_scales=development_scales,
        default_values=development_default_values,
        accounting_defaults=development_accounting_defaults,
        feature_ranges=development_feature_ranges,
        output_bounds=_output_bounds(raw_development),
        training_metadata=holdout_metadata,
    )
    holdout_path = holdout_bundle.save(
        output_dir / "neural_policy_surrogate_holdout.pt"
    )
    _status(f"[holdout 4/4] Saved true-holdout bundle: {holdout_path}")
    holdout_metrics = dict(holdout_evaluation.metrics)

    del (
        holdout_model,
        development_data,
        development_pairs,
        test_data,
        test_pairs,
        holdout_evaluation,
        raw_development,
        raw_train,
        raw_tune,
        raw_test,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Production refit on all models.
    # ------------------------------------------------------------------
    production_path: Path | None = None
    if not args.skip_production_refit:
        _status("[production 1/3] Preparing all-model production fit...")
        (
            production_preprocessor,
            production_accounting_defaults,
            production_default_values,
            production_feature_ranges,
        ) = _fit_bundle_components(
            data,
            feature_spec=feature_spec,
            money_units=str(args.money_units),
            preprocessor_fit_rows=int(args.preprocessor_fit_rows),
            random_state=int(args.random_state) + 200,
        )
        production_data = _prepare_data_arrays(
            data,
            preprocessor=production_preprocessor,
            accounting_defaults=production_accounting_defaults,
            default_values=production_default_values,
            money_units=str(args.money_units),
            chunk_size=int(args.preprocess_chunk_size),
            include_model_ids=False,
        )
        production_pairs = _build_pair_data(
            data,
            max_pairs=int(args.max_training_pairs),
            random_state=int(args.random_state) + 201,
            low_liquidity_scale=float(args.low_liquidity_scale),
            low_liquidity_boost=float(args.pair_low_liquidity_boost),
            mpc_tail_boost=float(args.mpc_tail_boost),
        )
        production_scales = _loss_scales(production_data, production_pairs)
        _status("[production 2/3] Training the all-model production network...")
        production_model = _make_model(
            best_trial,
            preprocessor=production_preprocessor,
            data=production_data,
            scales=production_scales,
            embedding_dim=int(args.embedding_dim),
        )
        production_model, _, _, production_history = _train_model(
            production_model,
            production_data,
            production_pairs,
            validation_data=None,
            validation_pairs=None,
            trial=best_trial,
            loss_weights=_loss_weights_from_trial(best_trial, args),
            scales=production_scales,
            args=args,
            device=device,
            max_epochs=final_epochs,
            patience=None,
            trial_label="production fit",
        )
        pd.DataFrame(production_history).to_csv(
            output_dir / "neural_production_training_history.csv", index=False
        )
        production_metadata = {
            "fit_scope": "all_models",
            "fit_models": sorted(data["model_id"].astype(str).unique().tolist()),
            "test_models": [],
            "selected_hyperparameters": asdict(best_trial),
            "selected_tuning_epoch": best_epoch,
            "final_epochs": final_epochs,
            "loss_weights": asdict(_loss_weights_from_trial(best_trial, args)),
            "money_units": str(args.money_units),
            "policy_layout": str(args.policy_layout),
            "sample_path": str(sample_path),
            "model_split_path": str(split_path),
            "evaluation_bundle_filename": "neural_policy_surrogate_holdout.pt",
            "created_by": Path(__file__).name,
        }
        production_bundle = NeuralPolicySurrogateBundle(
            model_config=production_model.config,
            model_state={key: value.detach().cpu() for key, value in production_model.state_dict().items()},
            preprocessor=production_preprocessor,
            loss_scales=production_scales,
            default_values=production_default_values,
            accounting_defaults=production_accounting_defaults,
            feature_ranges=production_feature_ranges,
            output_bounds=_output_bounds(data),
            training_metadata=production_metadata,
        )
        production_path = production_bundle.save(
            output_dir / "neural_policy_surrogate.pt"
        )
        _status(f"[production 3/3] Saved production bundle: {production_path}")

    summary = {
        "runtime_seconds": time.perf_counter() - total_start,
        "sample_path": str(sample_path),
        "sample_rows": len(data),
        "sample_models": int(data["model_id"].nunique()),
        "parameter_columns": parameter_columns,
        "categorical_parameter_columns": categorical_parameters,
        "split_path": str(split_path),
        "selected_trial": asdict(best_trial),
        "selected_tuning_epoch": best_epoch,
        "final_epochs": final_epochs,
        "best_tuning_score": best_score,
        "holdout_bundle": str(holdout_path),
        "production_bundle": None if production_path is None else str(production_path),
        "holdout_metrics": holdout_metrics,
        "arguments": vars(args),
    }
    _write_json(summary, output_dir / "neural_training_summary.json")
    _status(
        f"[done] Neural surrogate pipeline completed in {_elapsed(total_start)}."
    )


if __name__ == "__main__":
    main()
