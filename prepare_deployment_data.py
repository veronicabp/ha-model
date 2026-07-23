# %%
from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import pandas as pd

# ---------------------------------------------------------------------
# Change these three paths.
# ---------------------------------------------------------------------

SURROGATE_DIR = Path(
    "/Users/vbp/Dropbox (Personal)/research/research-data/iceland-data/ha_smooth"
)

ORIGINAL_CATALOG = SURROGATE_DIR / "model_catalog.csv"

DEPLOY_DIR = Path("/Users/vbp/Dropbox (Personal)/research/github-projects/ha-model")
APP_DATA_DIR = DEPLOY_DIR / "app_data"
PUBLIC_MODELS_DIR = APP_DATA_DIR / "held_out_models"


# %%
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

bundle_path = SURROGATE_DIR / "policy_surrogate.joblib"
bundle = joblib.load(bundle_path)

metrics = bundle.validation_by_model.copy()

if metrics.empty:
    raise RuntimeError("The surrogate bundle contains no per-model validation results.")

metrics["model_id"] = metrics["model_id"].astype(str)

split = metrics["split"].astype(str).str.lower()

# Accommodate either "held_out" or "test" as the split label.
held_out_mask = split.str.contains("held", na=False) | split.str.contains(
    "test", na=False
)

held_out_ids = list(set(metrics.loc[held_out_mask, "model_id"]))[:500]

if not held_out_ids:
    raise RuntimeError("No held-out model IDs were found in validation_by_model.")

catalog = pd.read_csv(ORIGINAL_CATALOG)
catalog["model_id"] = catalog["model_id"].astype(str)

public_catalog = catalog[catalog["model_id"].isin(held_out_ids)].copy()

if public_catalog.empty:
    raise RuntimeError("None of the held-out models were found in model_catalog.csv.")

if "model_dir" not in public_catalog:
    raise RuntimeError("model_catalog.csv has no model_dir column.")

for row in public_catalog.itertuples(index=False):
    model_id = str(row.model_id)
    source_dir = Path(str(row.model_dir))

    if not source_dir.exists():
        raise FileNotFoundError(f"Could not find model directory: {source_dir}")

    destination_dir = PUBLIC_MODELS_DIR / model_id

    print(f"Copying {model_id}")
    shutil.copytree(
        source_dir,
        destination_dir,
        dirs_exist_ok=True,
    )

# The online app will resolve these relative to app_data/.
public_catalog["model_dir"] = "held_out_models/" + public_catalog["model_id"].astype(
    str
)

public_catalog.to_csv(
    APP_DATA_DIR / "model_catalog.csv",
    index=False,
)

# Copy the bundle and plotting sample.
shutil.copy2(
    bundle_path,
    APP_DATA_DIR / "policy_surrogate.joblib",
)

sample_candidates = [
    SURROGATE_DIR / "policy_grid_sample.pkl.gz",
    SURROGATE_DIR / "policy_grid_sample.parquet",
    SURROGATE_DIR / "policy_grid_sample.pkl",
    SURROGATE_DIR / "policy_grid_sample.csv",
]

for source in sample_candidates:
    if source.exists():
        shutil.copy2(source, APP_DATA_DIR / source.name)
        break

print()
print(f"Deployment data created in: {APP_DATA_DIR}")
print(f"Held-out models copied: {len(public_catalog):,}")

# %%
