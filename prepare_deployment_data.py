# %%
from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Change these paths.
# ---------------------------------------------------------------------

SURROGATE_DIR = Path(
    "/Users/vbp/Dropbox (Personal)/research/research-data/iceland-data/ha_smooth"
)
DEPLOY_DIR = Path("/Users/vbp/Dropbox (Personal)/research/github-projects/ha-model")

MAX_PUBLIC_MODELS = 500
PINNED_MODEL_IDS: list[str] = []

APP_DATA_DIR = DEPLOY_DIR / "app_data"
PUBLIC_MODELS_DIR = APP_DATA_DIR / "held_out_models"


# %%
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

bundle_path = SURROGATE_DIR / "policy_surrogate.joblib"
catalog_path = SURROGATE_DIR / "model_catalog.csv"

bundle = joblib.load(bundle_path)
metrics = bundle.validation_by_model.copy()
catalog = pd.read_csv(catalog_path)

metrics["model_id"] = metrics["model_id"].astype(str)
catalog["model_id"] = catalog["model_id"].astype(str)

split = metrics["split"].astype(str).str.lower()
held_out = metrics[
    split.str.contains("held", na=False) | split.str.contains("test", na=False)
].copy()

if held_out.empty:
    raise RuntimeError("No held-out model IDs were found in validation_by_model.")

# Rank models deterministically, while always retaining explicitly pinned IDs.
ranked = (
    held_out[held_out["target"].isin(["consumption", "deposit"])]
    .pivot_table(index="model_id", columns="target", values="r2", aggfunc="first")
    .reset_index()
)
for column in ("consumption", "deposit"):
    if column not in ranked:
        ranked[column] = pd.NA
ranked["mean_r2"] = ranked[["consumption", "deposit"]].mean(axis=1)
ranked = ranked.sort_values(["mean_r2", "model_id"], ascending=[False, True])

available_ids = set(catalog["model_id"])
pinned = [
    str(model_id) for model_id in PINNED_MODEL_IDS if str(model_id) in available_ids
]
remaining = [
    model_id
    for model_id in ranked["model_id"].astype(str)
    if model_id in available_ids and model_id not in set(pinned)
]
public_ids = (pinned + remaining)[: int(MAX_PUBLIC_MODELS)]

if not public_ids:
    raise RuntimeError("No held-out models were found in model_catalog.csv.")

public_catalog = catalog[catalog["model_id"].isin(public_ids)].copy()
public_catalog["__order"] = pd.Categorical(
    public_catalog["model_id"], categories=public_ids, ordered=True
)
public_catalog = public_catalog.sort_values("__order").drop(columns="__order")


# %%
# Replace the exact-model directory so stale models do not remain deployed.
if PUBLIC_MODELS_DIR.exists():
    shutil.rmtree(PUBLIC_MODELS_DIR)
PUBLIC_MODELS_DIR.mkdir(parents=True)

for row in public_catalog.itertuples(index=False):
    model_id = str(row.model_id)
    source_dir = Path(str(row.model_dir))
    destination_dir = PUBLIC_MODELS_DIR / model_id

    if not source_dir.exists():
        raise FileNotFoundError(f"Could not find model directory: {source_dir}")

    destination_dir.mkdir(parents=True)
    for filename in ("metadata.json", "arrays.npz"):
        source = source_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing required model file: {source}")
        shutil.copy2(source, destination_dir / filename)

public_catalog["model_dir"] = "held_out_models/" + public_catalog["model_id"].astype(
    str
)
public_catalog.to_csv(APP_DATA_DIR / "model_catalog.csv", index=False)

shutil.copy2(bundle_path, APP_DATA_DIR / "policy_surrogate.joblib")


# %%
# Create a small household-level validation file containing only MPC-pair rows
# for exact models that are actually deployed. The website does not need the
# full training sample.
sample_candidates = [
    SURROGATE_DIR / "policy_grid_sample.parquet",
    SURROGATE_DIR / "policy_grid_sample.pkl.gz",
    SURROGATE_DIR / "policy_grid_sample.pkl",
    SURROGATE_DIR / "policy_grid_sample.csv",
]
sample_path = next((path for path in sample_candidates if path.exists()), None)

if sample_path is None:
    raise FileNotFoundError("Could not find policy_grid_sample beside the bundle.")

if sample_path.suffix == ".parquet":
    sample = pd.read_parquet(sample_path)
elif sample_path.name.endswith(".pkl.gz"):
    sample = pd.read_pickle(sample_path, compression="gzip")
elif sample_path.suffix in {".pkl", ".pickle"}:
    sample = pd.read_pickle(sample_path)
else:
    sample = pd.read_csv(sample_path)

sample["model_id"] = sample["model_id"].astype(str)
pair_id = pd.to_numeric(sample["mpc_pair_id"], errors="coerce")
pair_role = pd.to_numeric(sample["mpc_pair_role"], errors="coerce")

keep = sample["model_id"].isin(public_ids) & pair_id.ge(0) & pair_role.isin([0, 1])
heldout_pairs = sample.loc[keep].copy()
del sample

if heldout_pairs.empty:
    raise RuntimeError("No held-out MPC-pair rows were found for deployed models.")

heldout_pairs.to_parquet(
    APP_DATA_DIR / "heldout_mpc_pairs.parquet",
    index=False,
)

# Data for axes
# Old website default axis ranges: 0.5th to 99.5th percentiles.
policy_outputs = [
    "consumption",
    "deposit",
    "delta_liquid_assets",
    "delta_illiquid_assets",
    "next_liquid_assets",
    "next_illiquid_assets",
]

axis_rows = []

for output in policy_outputs:
    if output not in sample:
        continue

    values = pd.to_numeric(
        sample[output],
        errors="coerce",
    )
    values = values[np.isfinite(values)]

    if len(values):
        axis_rows.append(
            {
                "output": output,
                "lower": float(values.quantile(0.005)),
                "upper": float(values.quantile(0.995)),
            }
        )

pd.DataFrame(axis_rows).to_csv(
    APP_DATA_DIR / "output_axis_ranges.csv",
    index=False,
)

# Remove obsolete full-sample files from the deployment directory.
for filename in (
    "policy_grid_sample.parquet",
    "policy_grid_sample.pkl.gz",
    "policy_grid_sample.pkl",
    "policy_grid_sample.csv",
):
    path = APP_DATA_DIR / filename
    if path.exists():
        path.unlink()

print(f"Deployment data created in: {APP_DATA_DIR}")
print(f"Exact held-out models copied: {len(public_catalog):,}")
print(f"Held-out MPC-pair rows: {len(heldout_pairs):,}")
print(
    "Reduced validation file size: "
    f"{(APP_DATA_DIR / 'heldout_mpc_pairs.parquet').stat().st_size / 1024**2:.1f} MB"
)
