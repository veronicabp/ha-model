from __future__ import annotations

"""Create the lightweight Streamlit deployment directory.

Set ``SURROGATE_TYPE`` to ``"polynomial"`` or ``"neural"``. The script copies
that backend's production and true-holdout bundles, model-level validation
sidecars, a public set of exact holdout grids, and a reduced held-out MPC-pair
sample. It can be run once for each backend to place both surrogate types in the
same ``app_data`` directory.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ha_surrogate_backend import load_surrogate_bundle, normalize_surrogate_type

# -----------------------------------------------------------------------------
# User paths and deployment choices
# -----------------------------------------------------------------------------

SURROGATE_TYPE = "neural"  # "polynomial" or "neural"

POLYNOMIAL_SURROGATE_DIR = Path("../../../orcd/scratch/ha_surrogate")
NEURAL_SURROGATE_DIR = Path("../../../orcd/scratch/ha_neural_surrogate")
DEPLOY_DIR = Path("../../../orcd/scratch/deploy")

MAX_PUBLIC_MODELS = 100
PINNED_MODEL_IDS: list[str] = []
PUBLIC_MODEL_IDS_FILE = Path("../../../orcd/scratch/public_model_ids.txt")
REFRESH_PUBLIC_MODEL_IDS = False

APP_DATA_DIR = DEPLOY_DIR / "app_data"
PUBLIC_MODELS_DIR = APP_DATA_DIR / "held_out_models"

BACKEND_CONFIG = {
    "polynomial": {
        "directory": POLYNOMIAL_SURROGATE_DIR,
        "production_bundle": "policy_surrogate.joblib",
        "evaluation_bundle": "policy_surrogate_holdout.joblib",
        "catalogs": ("model_catalog.csv",),
        "samples": (
            "policy_grid_sample.parquet",
            "policy_grid_sample.pkl.gz",
            "policy_grid_sample.pkl",
            "policy_grid_sample.csv",
        ),
        "sidecars": (),
    },
    "neural": {
        "directory": NEURAL_SURROGATE_DIR,
        "production_bundle": "neural_policy_surrogate.pt",
        "evaluation_bundle": "neural_policy_surrogate_holdout.pt",
        "catalogs": ("neural_model_catalog.csv", "model_catalog.csv"),
        "samples": (
            "neural_policy_grid_sample.parquet",
            "neural_policy_grid_sample.pkl.gz",
            "neural_policy_grid_sample.pkl",
            "neural_policy_grid_sample.csv",
        ),
        "sidecars": (
            "neural_validation_metrics.csv",
            "neural_validation_by_model.csv",
        ),
    },
}


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name.endswith(".pkl.gz"):
        return pd.read_pickle(path, compression="gzip")
    if path.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    return pd.read_csv(path)


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find any of {list(names)} in {directory}."
    )


def _load_sample(directory: Path, names: tuple[str, ...]) -> tuple[pd.DataFrame, Path]:
    path = _first_existing(directory, names)
    return _read_table(path), path


def _validate_evaluation_bundle(bundle: object) -> set[str]:
    metadata = dict(getattr(bundle, "training_metadata", {}) or {})
    fit_ids = set(map(str, metadata.get("fit_models", [])))
    test_ids = set(map(str, metadata.get("test_models", [])))
    if metadata.get("fit_scope") != "development_only":
        raise RuntimeError(
            "Evaluation bundle was not fitted on development models only."
        )
    overlap = fit_ids & test_ids
    if not test_ids or overlap:
        raise RuntimeError(
            "Evaluation bundle must have nonempty test models and no fit/test "
            f"overlap; overlap={len(overlap)}."
        )
    return test_ids


def _select_public_ids(
    *,
    ranked_ids: list[str],
    eligible_ids: set[str],
) -> list[str]:
    if PUBLIC_MODEL_IDS_FILE.exists() and not REFRESH_PUBLIC_MODEL_IDS:
        public_ids = [
            line.strip()
            for line in PUBLIC_MODEL_IDS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(public_ids) != len(set(public_ids)):
            raise RuntimeError(f"{PUBLIC_MODEL_IDS_FILE} contains duplicate IDs.")
        invalid = [model_id for model_id in public_ids if model_id not in eligible_ids]
        if invalid:
            raise RuntimeError(
                "Frozen public IDs are absent from the current true holdout set:\n"
                + "\n".join(invalid)
            )
        if len(public_ids) != int(MAX_PUBLIC_MODELS):
            raise RuntimeError(
                f"Frozen file contains {len(public_ids)} IDs but "
                f"MAX_PUBLIC_MODELS={MAX_PUBLIC_MODELS}."
            )
        print(f"Reusing {len(public_ids):,} frozen public model IDs.", flush=True)
        return public_ids

    pinned = [
        str(model_id)
        for model_id in PINNED_MODEL_IDS
        if str(model_id) in eligible_ids
    ]
    pinned_set = set(pinned)
    remaining = [
        model_id
        for model_id in ranked_ids
        if model_id in eligible_ids and model_id not in pinned_set
    ]
    public_ids = (pinned + remaining)[: int(MAX_PUBLIC_MODELS)]
    if len(public_ids) < int(MAX_PUBLIC_MODELS):
        raise RuntimeError(
            f"Requested {MAX_PUBLIC_MODELS} models but found only "
            f"{len(public_ids)} eligible true holdouts."
        )
    PUBLIC_MODEL_IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_MODEL_IDS_FILE.write_text(
        "\n".join(public_ids) + "\n",
        encoding="utf-8",
    )
    print(f"Selected and froze {len(public_ids):,} public model IDs.", flush=True)
    return public_ids


def _mpc_axis_range(sample: pd.DataFrame) -> tuple[float, float] | None:
    required = {
        "model_id",
        "mpc_pair_id",
        "mpc_pair_role",
        "mpc_check_units",
        "consumption",
    }
    if not required.issubset(sample.columns):
        return None
    frame = sample.copy()
    frame["mpc_pair_id"] = pd.to_numeric(frame["mpc_pair_id"], errors="coerce")
    frame["mpc_pair_role"] = pd.to_numeric(frame["mpc_pair_role"], errors="coerce")
    frame["mpc_check_units"] = pd.to_numeric(
        frame["mpc_check_units"], errors="coerce"
    )
    frame["consumption"] = pd.to_numeric(frame["consumption"], errors="coerce")
    frame = frame[
        frame["mpc_pair_id"].ge(0)
        & frame["mpc_pair_role"].isin([0, 1])
        & frame["mpc_check_units"].gt(0)
    ]
    keys = ["model_id", "mpc_pair_id"]
    baseline = frame[frame["mpc_pair_role"].eq(0)][
        keys + ["consumption", "mpc_check_units"]
    ].rename(columns={"consumption": "c0", "mpc_check_units": "check0"})
    treated = frame[frame["mpc_pair_role"].eq(1)][
        keys + ["consumption", "mpc_check_units"]
    ].rename(columns={"consumption": "c1", "mpc_check_units": "check1"})
    pairs = baseline.merge(treated, on=keys, how="inner")
    if pairs.empty:
        return None
    check = 0.5 * (pairs["check0"] + pairs["check1"])
    mpc = (pairs["c1"] - pairs["c0"]) / check
    mpc = mpc[np.isfinite(mpc)]
    if mpc.empty:
        return None
    lo = min(float(mpc.quantile(0.005)), -0.05)
    hi = max(float(mpc.quantile(0.995)), 1.05)
    return lo, hi


def _write_validation_pairs(frame: pd.DataFrame, path: Path) -> Path:
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception:
        fallback = path.with_suffix(".pkl.gz")
        frame.to_pickle(fallback, compression="gzip")
        return fallback


def main() -> None:
    surrogate_type = normalize_surrogate_type(SURROGATE_TYPE)
    if surrogate_type == "auto":
        raise ValueError("SURROGATE_TYPE must be 'polynomial' or 'neural'.")
    config = BACKEND_CONFIG[surrogate_type]
    surrogate_dir = Path(config["directory"]).expanduser().resolve()
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Deploying {surrogate_type} surrogate from {surrogate_dir}", flush=True)
    production_path = surrogate_dir / str(config["production_bundle"])
    evaluation_path = surrogate_dir / str(config["evaluation_bundle"])
    catalog_path = _first_existing(
        surrogate_dir,
        tuple(config["catalogs"]),
    )

    production_bundle = load_surrogate_bundle(
        production_path,
        surrogate_type=surrogate_type,
        neural_device="cpu",
    )
    evaluation_bundle = load_surrogate_bundle(
        evaluation_path,
        surrogate_type=surrogate_type,
        neural_device="cpu",
    )
    test_ids = _validate_evaluation_bundle(evaluation_bundle)

    metrics = evaluation_bundle.validation_by_model.copy()
    if metrics.empty:
        raise RuntimeError(
            "No model-level holdout metrics were found. For the neural backend, "
            "neural_validation_by_model.csv must be beside the .pt bundles."
        )
    catalog = pd.read_csv(catalog_path)
    metrics["model_id"] = metrics["model_id"].astype(str)
    catalog["model_id"] = catalog["model_id"].astype(str)

    held_out = metrics[
        metrics["model_id"].isin(test_ids)
        & metrics["split"].astype(str).str.contains(
            "held|test", case=False, na=False
        )
    ].copy()
    if held_out.empty:
        raise RuntimeError("No true-holdout model metrics were found.")

    ranked = (
        held_out[held_out["target"].isin(["consumption", "deposit"])]
        .pivot_table(index="model_id", columns="target", values="r2", aggfunc="first")
        .reset_index()
    )
    for column in ("consumption", "deposit"):
        if column not in ranked:
            ranked[column] = np.nan
    ranked["mean_r2"] = ranked[["consumption", "deposit"]].mean(axis=1)
    ranked = ranked.sort_values(
        ["mean_r2", "model_id"], ascending=[False, True], kind="mergesort"
    )

    available_ids = set(catalog["model_id"])
    eligible_ids = available_ids & test_ids
    public_ids = _select_public_ids(
        ranked_ids=ranked["model_id"].astype(str).tolist(),
        eligible_ids=eligible_ids,
    )

    public_catalog = catalog[catalog["model_id"].isin(public_ids)].copy()
    public_catalog["__order"] = pd.Categorical(
        public_catalog["model_id"], categories=public_ids, ordered=True
    )
    public_catalog = public_catalog.sort_values("__order").drop(columns="__order")

    if PUBLIC_MODELS_DIR.exists():
        shutil.rmtree(PUBLIC_MODELS_DIR)
    PUBLIC_MODELS_DIR.mkdir(parents=True)

    for row in public_catalog.itertuples(index=False):
        model_id = str(row.model_id)
        source_dir = Path(str(row.model_dir)).expanduser()
        if not source_dir.is_absolute():
            source_dir = (surrogate_dir / source_dir).resolve()
        destination_dir = PUBLIC_MODELS_DIR / model_id
        destination_dir.mkdir(parents=True)
        for filename in ("metadata.json", "arrays.npz"):
            source = source_dir / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing required model file: {source}")
            shutil.copy2(source, destination_dir / filename)

    public_catalog["model_dir"] = (
        "held_out_models/" + public_catalog["model_id"].astype(str)
    )
    public_catalog.to_csv(APP_DATA_DIR / "model_catalog.csv", index=False)

    shutil.copy2(production_path, APP_DATA_DIR / production_path.name)
    shutil.copy2(evaluation_path, APP_DATA_DIR / evaluation_path.name)
    for filename in tuple(config["sidecars"]):
        source = surrogate_dir / filename
        if source.exists():
            shutil.copy2(source, APP_DATA_DIR / filename)
        else:
            print(f"Warning: optional sidecar not found: {source}", flush=True)

    sample, sample_path = _load_sample(
        surrogate_dir,
        tuple(config["samples"]),
    )
    sample["model_id"] = sample["model_id"].astype(str)

    policy_outputs = [
        "consumption",
        "deposit",
        "delta_liquid_assets",
        "delta_illiquid_assets",
        "next_liquid_assets",
        "next_illiquid_assets",
    ]
    axis_rows: list[dict[str, float | str]] = []
    for output in policy_outputs:
        if output not in sample:
            continue
        values = pd.to_numeric(sample[output], errors="coerce")
        values = values[np.isfinite(values)]
        if len(values):
            axis_rows.append(
                {
                    "output": output,
                    "lower": float(values.quantile(0.01)),
                    "upper": float(values.quantile(0.99)),
                }
            )
    mpc_range = _mpc_axis_range(sample)
    if mpc_range is not None:
        axis_rows.append(
            {"output": "mpc", "lower": mpc_range[0], "upper": mpc_range[1]}
        )
    axis_table = pd.DataFrame(axis_rows)
    if axis_table.empty:
        raise RuntimeError("Could not construct output-axis ranges.")
    axis_table.to_csv(APP_DATA_DIR / "output_axis_ranges.csv", index=False)

    pair_id = pd.to_numeric(sample["mpc_pair_id"], errors="coerce")
    pair_role = pd.to_numeric(sample["mpc_pair_role"], errors="coerce")
    keep = (
        sample["model_id"].isin(public_ids)
        & pair_id.ge(0)
        & pair_role.isin([0, 1])
    )
    heldout_pairs = sample.loc[keep].copy()
    if heldout_pairs.empty:
        raise RuntimeError("No MPC pair rows exist for the deployed holdout models.")

    for stale_name in ("heldout_mpc_pairs.parquet", "heldout_mpc_pairs.pkl.gz"):
        stale = APP_DATA_DIR / stale_name
        if stale.exists():
            stale.unlink()
    validation_path = _write_validation_pairs(
        heldout_pairs,
        APP_DATA_DIR / "heldout_mpc_pairs.parquet",
    )

    print(f"Deployment data created in: {APP_DATA_DIR}", flush=True)
    print(f"Source sample: {sample_path}", flush=True)
    print(f"Exact true-holdout models copied: {len(public_catalog):,}", flush=True)
    print(f"Held-out MPC-pair rows: {len(heldout_pairs):,}", flush=True)
    print(f"Output-axis ranges saved: {len(axis_table):,}", flush=True)
    print(
        "Reduced validation file size: "
        f"{validation_path.stat().st_size / 1024**2:.1f} MB "
        f"({validation_path.name})",
        flush=True,
    )


if __name__ == "__main__":
    main()
