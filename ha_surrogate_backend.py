"""Common loading and inference interface for polynomial and neural HA surrogates.

The Streamlit application should import surrogate bundles through this module
rather than loading ``.joblib`` or ``.pt`` files directly.  The adapter exposes
one stable interface across both model classes:

* ``feature_spec``
* ``default_values``
* ``accounting_defaults``
* ``feature_ranges``
* ``output_bounds``
* ``training_metadata``
* ``validation_metrics``
* ``validation_by_model``
* ``prepare_inputs(frame)``
* ``predict(frame)``

The neural bundle produced by ``train_neural_surrogate.py`` stores validation
results in sibling CSV files.  This loader attaches those tables to the website
adapter and caches the instantiated PyTorch network for repeated interactive
predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ha_policy_surrogate import (
    FeatureSpec,
    PolicySurrogateBundle,
    prepare_policy_inputs,
)

SurrogateType = Literal["auto", "polynomial", "neural"]


@runtime_checkable
class SurrogateBundle(Protocol):
    """Structural interface used by the website and visualization helpers."""

    surrogate_type: str
    surrogate_label: str
    feature_spec: FeatureSpec
    default_values: dict[str, Any]
    accounting_defaults: dict[str, Any]
    feature_ranges: dict[str, dict[str, Any]]
    output_bounds: dict[str, tuple[float, float]]
    training_metadata: dict[str, Any]
    validation_metrics: pd.DataFrame
    validation_by_model: pd.DataFrame

    def prepare_inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        ...

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        project: bool = True,
    ) -> pd.DataFrame:
        ...

    def make_default_row(self) -> pd.DataFrame:
        ...


def normalize_surrogate_type(value: str | None) -> SurrogateType:
    text = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "poly": "polynomial",
        "polynomial": "polynomial",
        "ridge": "polynomial",
        "nn": "neural",
        "neural": "neural",
        "neural_network": "neural",
        "network": "neural",
    }
    if text not in aliases:
        raise ValueError(
            "surrogate_type must be one of: auto, polynomial, neural."
        )
    return aliases[text]  # type: ignore[return-value]


def infer_surrogate_type(path: str | Path) -> Literal["polynomial", "neural"]:
    suffix = Path(path).suffix.lower()
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "neural"
    if suffix in {".joblib", ".pkl", ".pickle"}:
        return "polynomial"
    raise ValueError(
        f"Could not infer the surrogate type from {path!s}. Use an explicit "
        "surrogate_type of 'polynomial' or 'neural'."
    )


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read validation table {path}: {exc}") from exc


def _first_existing(parent: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = parent / name
        if candidate.exists():
            return candidate
    return None


def _metrics_from_metadata(metadata: dict[str, Any]) -> pd.DataFrame:
    """Fallback aggregate metrics when neural validation CSVs were not copied."""

    raw = metadata.get("test_metrics")
    if not isinstance(raw, dict):
        return pd.DataFrame()

    targets = (
        "consumption",
        "deposit",
        "liquid_outflow",
        "delta_liquid_assets",
        "delta_illiquid_assets",
        "mpc",
        "mpd",
        "liquid_response",
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        row: dict[str, Any] = {"target": target}
        found = False
        for metric in ("n", "r2", "rmse", "mae", "bias"):
            key = f"{target}_{metric}"
            if key in raw:
                row[metric] = raw[key]
                found = True
        if found:
            rows.append(row)
    return pd.DataFrame(rows)


def _normalize_validation_metrics(
    metrics: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    out = metrics.copy()
    if out.empty:
        return out
    if "target" not in out:
        raise ValueError("Validation metrics must contain a 'target' column.")
    out["target"] = out["target"].astype(str)
    if "split" not in out:
        out["split"] = split
    return out


def _normalize_validation_by_model(
    metrics: pd.DataFrame,
    *,
    split: str,
) -> pd.DataFrame:
    out = metrics.copy()
    if out.empty:
        return out
    required = {"model_id", "target"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(
            "Model-level validation metrics are missing columns: "
            + ", ".join(sorted(missing))
        )
    out["model_id"] = out["model_id"].astype(str)
    out["target"] = out["target"].astype(str)
    if "split" not in out:
        out["split"] = split
    return out


def _load_neural_validation_tables(
    bundle_path: Path,
    metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent = bundle_path.parent
    aggregate_path = _first_existing(
        parent,
        (
            "neural_validation_metrics.csv",
            "validation_metrics.csv",
        ),
    )
    by_model_path = _first_existing(
        parent,
        (
            "neural_validation_by_model.csv",
            "validation_by_model.csv",
        ),
    )

    aggregate = (
        _read_csv(aggregate_path)
        if aggregate_path is not None
        else _metrics_from_metadata(metadata)
    )
    by_model = (
        _read_csv(by_model_path)
        if by_model_path is not None
        else pd.DataFrame()
    )
    return (
        _normalize_validation_metrics(aggregate, split="held_out_test"),
        _normalize_validation_by_model(by_model, split="held_out_test"),
    )


def _install_neural_model_cache(bundle: Any) -> None:
    """Cache one instantiated PyTorch model per inference device.

    The training bundle's ``predict`` method intentionally reconstructs the
    network from its state dictionary. That is appropriate for isolated batch
    predictions but unnecessarily expensive in Streamlit. This non-persistent
    patch preserves the saved bundle format while reusing the network during the
    app process.
    """

    if getattr(bundle, "_website_model_cache_installed", False):
        return
    original_factory = bundle._model
    runtime_models: dict[str, Any] = {}

    def cached_model(device: Any) -> Any:
        key = str(device)
        if key not in runtime_models:
            runtime_models[key] = original_factory(device)
        return runtime_models[key]

    bundle._model = cached_model
    bundle._website_runtime_models = runtime_models
    bundle._website_model_cache_installed = True


@dataclass
class WebsiteSurrogateBundle:
    """Website-facing adapter around either saved surrogate implementation."""

    raw_bundle: Any
    bundle_path: Path
    surrogate_type: Literal["polynomial", "neural"]
    validation_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    validation_by_model: pd.DataFrame = field(default_factory=pd.DataFrame)
    neural_device: str = "auto"
    neural_batch_size: int = 65_536

    @property
    def surrogate_label(self) -> str:
        return (
            "Polynomial surrogate"
            if self.surrogate_type == "polynomial"
            else "Neural-network surrogate"
        )

    @property
    def short_surrogate_label(self) -> str:
        return "Polynomial" if self.surrogate_type == "polynomial" else "Neural network"

    @property
    def feature_spec(self) -> FeatureSpec:
        if self.surrogate_type == "polynomial":
            return self.raw_bundle.feature_spec
        return self.raw_bundle.preprocessor.feature_spec

    @property
    def default_values(self) -> dict[str, Any]:
        return self.raw_bundle.default_values

    @property
    def accounting_defaults(self) -> dict[str, Any]:
        return self.raw_bundle.accounting_defaults

    @property
    def feature_ranges(self) -> dict[str, dict[str, Any]]:
        return self.raw_bundle.feature_ranges

    @property
    def output_bounds(self) -> dict[str, tuple[float, float]]:
        return self.raw_bundle.output_bounds

    @property
    def training_metadata(self) -> dict[str, Any]:
        return self.raw_bundle.training_metadata

    def prepare_inputs(self, frame: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.raw_bundle, "prepare_inputs"):
            return self.raw_bundle.prepare_inputs(frame)
        return prepare_policy_inputs(
            frame,
            feature_spec=self.feature_spec,
            default_values=self.default_values,
            accounting_defaults=self.accounting_defaults,
            money_units=str(self.training_metadata.get("money_units", "model")),
        )

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        project: bool = True,
    ) -> pd.DataFrame:
        if self.surrogate_type == "polynomial":
            return self.raw_bundle.predict(frame, project=project)
        return self.raw_bundle.predict(
            frame,
            device=self.neural_device,
            batch_size=int(self.neural_batch_size),
            project=project,
        )

    def make_default_row(self) -> pd.DataFrame:
        return pd.DataFrame([self.default_values])

    def __getattr__(self, name: str) -> Any:
        # Forward optional implementation-specific metadata without exposing it
        # as part of the required website protocol.
        return getattr(self.raw_bundle, name)


def load_surrogate_bundle(
    path: str | Path,
    *,
    surrogate_type: str = "auto",
    neural_device: str = "auto",
    neural_batch_size: int = 65_536,
) -> WebsiteSurrogateBundle:
    """Load a polynomial ``.joblib`` or neural ``.pt`` website bundle."""

    bundle_path = Path(path).expanduser().resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Surrogate bundle does not exist: {bundle_path}")

    requested = normalize_surrogate_type(surrogate_type)
    resolved: Literal["polynomial", "neural"] = (
        infer_surrogate_type(bundle_path) if requested == "auto" else requested
    )  # type: ignore[assignment]

    if resolved == "polynomial":
        raw = PolicySurrogateBundle.load(bundle_path)
        validation_metrics = _normalize_validation_metrics(
            raw.validation_metrics,
            split="held_out_test",
        )
        validation_by_model = _normalize_validation_by_model(
            raw.validation_by_model,
            split="held_out_test",
        )
    else:
        try:
            from train_neural_surrogate import NeuralPolicySurrogateBundle
        except ImportError as exc:
            raise ImportError(
                "Neural surrogate inference requires PyTorch and the compatible "
                "train_neural_surrogate.py file to be importable beside the app."
            ) from exc
        raw = NeuralPolicySurrogateBundle.load(bundle_path)
        _install_neural_model_cache(raw)
        validation_metrics, validation_by_model = _load_neural_validation_tables(
            bundle_path,
            dict(raw.training_metadata or {}),
        )

    return WebsiteSurrogateBundle(
        raw_bundle=raw,
        bundle_path=bundle_path,
        surrogate_type=resolved,
        validation_metrics=validation_metrics,
        validation_by_model=validation_by_model,
        neural_device=str(neural_device),
        neural_batch_size=int(neural_batch_size),
    )


__all__ = [
    "SurrogateBundle",
    "SurrogateType",
    "WebsiteSurrogateBundle",
    "infer_surrogate_type",
    "load_surrogate_bundle",
    "normalize_surrogate_type",
]
