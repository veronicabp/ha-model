# %%
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error

OUTPUT_DIR = Path("app_data")

BUNDLE_PATH = OUTPUT_DIR / "policy_surrogate.joblib"
SAMPLE_PATH = OUTPUT_DIR / "policy_grid_sample.pkl.gz"
METRICS_PATH = OUTPUT_DIR / "validation_metrics_by_model.csv"

CHECK_DOLLARS = 10_000
DOLLARS_PER_MODEL_UNIT = 53_000

N_MODELS = 20
N_STATES_PER_MODEL = 1_000
SEED = 123

# Adjust after looking at the plotted distribution.
SECOND_PEAK = (0.18, 0.30)

# Set to a list of model IDs to inspect particular models.
MODEL_IDS = None

from ha_policy_surrogate import (
    PolicySurrogateBundle,
    read_policy_dataset,
)

from ha_mpc_distribution import (
    load_exact_policy_grid,
    align_population_to_exact_grid,
    compute_exact_grid_check_mpcs,
    compute_check_mpcs,
    predict_exact_grid_consumption,
)

bundle = PolicySurrogateBundle.load(BUNDLE_PATH)
sample = read_policy_dataset(SAMPLE_PATH)

sample["model_id"] = sample["model_id"].astype(str)

money_units = bundle.training_metadata.get("money_units", "model")
policy_layout = bundle.training_metadata.get("policy_layout", "GHKEBA")
test_models = set(map(str, bundle.training_metadata["test_models"]))

sample_test = sample[sample["model_id"].isin(test_models)].copy()

print("Held-out models:", sample_test["model_id"].nunique())
print("Held-out rows:", len(sample_test))
print(
    "Direct MPC weight:",
    bundle.training_metadata.get("mpc_difference_weight_selected"),
)
# %%
# %%
from tqdm import tqdm

# Calculate exact and surrogate MPCs on identical states.
model_ids = sample_test["model_id"].unique()

comparisons = []

state_columns = [
    "education",
    "age",
    "income_state",
    "employment_state",
    "labor_income_state",
    "current_income",
    "liquid_assets",
    "illiquid_assets",
]

for model_number, model_id in tqdm(enumerate(model_ids), total=len(model_ids)):

    model_data = sample_test[sample_test["model_id"] == model_id].copy()

    duplicate_columns = [
        column for column in state_columns if column in model_data.columns
    ]

    model_data = model_data.drop_duplicates(subset=duplicate_columns)

    model_data = model_data.sample(
        n=min(N_STATES_PER_MODEL, len(model_data)),
        random_state=SEED + model_number,
    ).reset_index(drop=True)

    model_dir = Path(model_data["model_dir"].iloc[0])

    exact_grid = load_exact_policy_grid(
        model_dir,
        policy_layout=policy_layout,
    )

    aligned = align_population_to_exact_grid(
        exact_grid,
        model_data,
        money_units=money_units,
    )

    exact = compute_exact_grid_check_mpcs(
        exact_grid,
        aligned,
        check_amount_dollars=CHECK_DOLLARS,
        dollars_per_model_unit=DOLLARS_PER_MODEL_UNIT,
        money_units=money_units,
    )

    surrogate = compute_check_mpcs(
        bundle,
        aligned,
        check_amount_dollars=CHECK_DOLLARS,
        dollars_per_model_unit=DOLLARS_PER_MODEL_UNIT,
        money_units=money_units,
    )

    # Raw predictions let us test whether truncating consumption at zero
    # generates the extra peak.
    baseline = bundle.prepare_inputs(aligned.copy())

    check_units = (
        CHECK_DOLLARS / DOLLARS_PER_MODEL_UNIT
        if money_units == "model"
        else CHECK_DOLLARS
    )

    treated = baseline.copy()

    treated_liquid = baseline["liquid_assets"].to_numpy(dtype=float) + check_units

    treated_liquid = np.minimum(
        treated_liquid,
        baseline["liquid_grid_max"].to_numpy(dtype=float),
    )

    treated["liquid_assets"] = treated_liquid
    treated = bundle.prepare_inputs(treated)

    raw0 = bundle.predict(baseline, project=False)
    raw1 = bundle.predict(treated, project=False)

    out = aligned.copy()

    out["model_id"] = model_id
    out["model_dir"] = str(model_dir)

    out["true_c0"] = exact["baseline_consumption"].to_numpy()
    out["true_c1"] = exact["post_check_consumption"].to_numpy()
    out["true_mpc"] = exact["mpc"].to_numpy()

    out["surrogate_c0"] = surrogate["baseline_consumption"].to_numpy()
    out["surrogate_c1"] = surrogate["post_check_consumption"].to_numpy()
    out["surrogate_mpc"] = surrogate["mpc"].to_numpy()

    out["surrogate_c0_raw"] = raw0["consumption"].to_numpy()
    out["surrogate_c1_raw"] = raw1["consumption"].to_numpy()

    out["surrogate_mpc_raw"] = (
        out["surrogate_c1_raw"] - out["surrogate_c0_raw"]
    ) / check_units

    out["true_check_clipped"] = exact["check_state_clipped"].to_numpy()

    out["surrogate_check_clipped"] = surrogate["check_state_clipped"].to_numpy()

    # Copy engineered states used by the regression.
    for column in bundle.feature_spec.engineered_state_columns:
        if column in baseline:
            out[column] = baseline[column].to_numpy()

    comparisons.append(out)


comparison = pd.concat(
    comparisons,
    ignore_index=True,
)

comparison["c0_error"] = comparison["surrogate_c0"] - comparison["true_c0"]

comparison["c1_error"] = comparison["surrogate_c1"] - comparison["true_c1"]

comparison["mpc_error"] = comparison["surrogate_mpc"] - comparison["true_mpc"]

comparison["mpc_error_raw"] = comparison["surrogate_mpc_raw"] - comparison["true_mpc"]

comparison["abs_mpc_error"] = comparison["mpc_error"].abs()

print("Rows compared:", len(comparison))
# %%
# %%
# Overall accuracy and the consumption-error decomposition.

check_units = (
    CHECK_DOLLARS / DOLLARS_PER_MODEL_UNIT if money_units == "model" else CHECK_DOLLARS
)

summary = pd.DataFrame(
    {
        "outcome": [
            "baseline consumption",
            "post-check consumption",
            "MPC",
            "raw MPC",
        ],
        "r2": [
            r2_score(
                comparison["true_c0"],
                comparison["surrogate_c0"],
            ),
            r2_score(
                comparison["true_c1"],
                comparison["surrogate_c1"],
            ),
            r2_score(
                comparison["true_mpc"],
                comparison["surrogate_mpc"],
            ),
            r2_score(
                comparison["true_mpc"],
                comparison["surrogate_mpc_raw"],
            ),
        ],
        "rmse": [
            mean_squared_error(
                comparison["true_c0"],
                comparison["surrogate_c0"],
            )
            ** 0.5,
            mean_squared_error(
                comparison["true_c1"],
                comparison["surrogate_c1"],
            )
            ** 0.5,
            mean_squared_error(
                comparison["true_mpc"],
                comparison["surrogate_mpc"],
            )
            ** 0.5,
            mean_squared_error(
                comparison["true_mpc"],
                comparison["surrogate_mpc_raw"],
            )
            ** 0.5,
        ],
        "bias": [
            comparison["c0_error"].mean(),
            comparison["c1_error"].mean(),
            comparison["mpc_error"].mean(),
            comparison["mpc_error_raw"].mean(),
        ],
    }
)

print(summary)

print(
    "\nCorrelation of baseline and treated consumption errors:",
    comparison[["c0_error", "c1_error"]].corr().iloc[0, 1],
)

# This should be almost exactly zero.
identity_error = (
    comparison["mpc_error"]
    - (comparison["c1_error"] - comparison["c0_error"]) / check_units
)

print(
    "Maximum MPC decomposition error:",
    identity_error.abs().max(),
)

print(
    "Share affected by nonnegative-consumption projection:",
    np.mean(
        np.abs(comparison["surrogate_mpc"] - comparison["surrogate_mpc_raw"]) > 1e-10
    ),
)

print(
    "Share with clipped treated liquid assets:",
    comparison["surrogate_check_clipped"].mean(),
)
# %%
# %%
# Per-model MPC accuracy.

model_results = []

for model_id, group in comparison.groupby("model_id"):

    model_results.append(
        {
            "model_id": model_id,
            "n": len(group),
            "true_mean": group["true_mpc"].mean(),
            "surrogate_mean": group["surrogate_mpc"].mean(),
            "true_sd": group["true_mpc"].std(),
            "surrogate_sd": group["surrogate_mpc"].std(),
            "bias": group["mpc_error"].mean(),
            "rmse": np.sqrt(np.mean(group["mpc_error"] ** 2)),
            "r2": r2_score(
                group["true_mpc"],
                group["surrogate_mpc"],
            ),
        }
    )

model_results = pd.DataFrame(model_results).sort_values("r2")

print(model_results)
# %%
# %%
# Plot both distributions using identical bins and identify KDE peaks.

pooled = np.concatenate(
    [
        comparison["true_mpc"].to_numpy(),
        comparison["surrogate_mpc"].to_numpy(),
    ]
)

lo, hi = np.quantile(pooled, [0.005, 0.995])
bins = np.linspace(lo, hi, 60)
density_grid = np.linspace(lo, hi, 500)

true_density = gaussian_kde(comparison["true_mpc"])(density_grid)

surrogate_density = gaussian_kde(comparison["surrogate_mpc"])(density_grid)

true_peak_indices, _ = find_peaks(
    true_density,
    prominence=0.05 * true_density.max(),
)

surrogate_peak_indices, _ = find_peaks(
    surrogate_density,
    prominence=0.05 * surrogate_density.max(),
)

print(
    "True KDE peaks:",
    density_grid[true_peak_indices],
)

print(
    "Surrogate KDE peaks:",
    density_grid[surrogate_peak_indices],
)

plt.figure(figsize=(9, 5))

plt.hist(
    comparison["true_mpc"],
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="True",
)

plt.hist(
    comparison["surrogate_mpc"],
    bins=bins,
    density=True,
    histtype="step",
    linewidth=2,
    label="Surrogate",
)

plt.plot(
    density_grid,
    true_density,
    label="True KDE",
)

plt.plot(
    density_grid,
    surrogate_density,
    label="Surrogate KDE",
)

plt.xlabel("MPC")
plt.ylabel("Density")
plt.legend()
plt.show()
# %%
# %%
# True MPC against surrogate MPC.

plot_lo = np.quantile(
    np.concatenate(
        [
            comparison["true_mpc"],
            comparison["surrogate_mpc"],
        ]
    ),
    0.005,
)

plot_hi = np.quantile(
    np.concatenate(
        [
            comparison["true_mpc"],
            comparison["surrogate_mpc"],
        ]
    ),
    0.995,
)

plt.figure(figsize=(6, 6))

plt.scatter(
    comparison["true_mpc"],
    comparison["surrogate_mpc"],
    s=6,
    alpha=0.25,
)

plt.plot(
    [plot_lo, plot_hi],
    [plot_lo, plot_hi],
)

plt.xlim(plot_lo, plot_hi)
plt.ylim(plot_lo, plot_hi)
plt.xlabel("True MPC")
plt.ylabel("Surrogate MPC")
plt.show()
# %%
# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

from ha_policy_surrogate import PolicySurrogateBundle
from ha_mpc_distribution import (
    PopulationAssumptions,
    compute_check_mpcs,
    compute_exact_grid_check_mpcs,
    generate_synthetic_population,
    infer_money_unit_info,
    load_exact_policy_grid,
    mpc_summary,
    set_parameter_values,
)

BUNDLE_PATH = Path("app_data/policy_surrogate.joblib")

MODEL_ID = "fd565ff2f4a3b9"

CHECK_AMOUNT = 10_000
BINS = 60
# %%
# %%
bundle = PolicySurrogateBundle.load(BUNDLE_PATH)

catalog = pd.read_csv(BUNDLE_PATH.parent / "model_catalog.csv")

catalog["model_id"] = catalog["model_id"].astype(str)

selected = catalog[catalog["model_id"] == str(MODEL_ID)]

if len(selected) != 1:
    raise ValueError(f"Found {len(selected)} catalog rows for model_id={MODEL_ID}")

selected = selected.iloc[0]

model_dir = Path(str(selected["model_dir"])).expanduser()

if not model_dir.is_absolute():
    model_dir = BUNDLE_PATH.parent / model_dir

model_dir = model_dir.resolve()

policy_layout = bundle.training_metadata.get(
    "policy_layout",
    "GHKEBA",
)

unit_info = infer_money_unit_info(
    bundle,
    bundle_path=BUNDLE_PATH,
)

money_units = unit_info.money_units
dollars_per_model_unit = unit_info.dollars_per_model_unit

print("Model directory:", model_dir)
print("Money units:", money_units)
print("Dollars per model unit:", dollars_per_model_unit)
# %%
# %%
# Start from the bundle defaults and replace all parameters with those
# belonging to the selected solved model.

model_base = dict(bundle.default_values)

for column in bundle.feature_spec.parameter_columns:
    if column.endswith("__missing"):
        model_base[column] = 0
    elif column in selected.index and pd.notna(selected[column]):
        model_base[column] = float(selected[column])

for column in bundle.feature_spec.categorical_parameter_columns:
    if column in selected.index and pd.notna(selected[column]):
        model_base[column] = str(selected[column])
# %%
# %%
# Edit these assumptions to choose the population distribution.

assumptions = PopulationAssumptions(
    n_households=20_000,
    seed=20260717,
    median_income_dollars=60_000,
    permanent_income_log_sd=0.50,
    transitory_income_log_sd=0.20,
    high_education_share=0.55,
    nonemployment_share=0.08,
    poor_hand_to_mouth_share=0.25,
    wealthy_hand_to_mouth_share=0.15,
    poor_liquid_ratio_median=0.015,
    poor_illiquid_ratio_median=0.10,
    wealthy_htm_liquid_ratio_median=0.010,
    wealthy_htm_illiquid_ratio_median=2.50,
    regular_liquid_ratio_median=0.25,
    regular_illiquid_ratio_median=1.25,
    liquid_ratio_log_sd=0.75,
    illiquid_ratio_log_sd=0.85,
    negative_liquid_share=0.08,
)

population = generate_synthetic_population(
    bundle,
    base_values=model_base,
    assumptions=assumptions,
    dollars_per_model_unit=dollars_per_model_unit,
    money_units=money_units,
)

population["population_type"].value_counts(normalize=True)
# %%
# %%
# Exact-model MPCs.

exact_grid = load_exact_policy_grid(
    model_dir,
    policy_layout=policy_layout,
)

exact_response = compute_exact_grid_check_mpcs(
    exact_grid,
    population,
    check_amount_dollars=CHECK_AMOUNT,
    dollars_per_model_unit=dollars_per_model_unit,
    money_units=money_units,
)

# exact_response contains the population aligned to this model's solved
# age, income, employment, and education states.
# %%
# %%
# Surrogate MPCs on exactly the same aligned household states and with
# exactly the selected model's parameters.

surrogate_population = set_parameter_values(
    exact_response.copy(),
    bundle,
    model_base,
)

surrogate_response = compute_check_mpcs(
    bundle,
    surrogate_population,
    check_amount_dollars=CHECK_AMOUNT,
    dollars_per_model_unit=dollars_per_model_unit,
    money_units=money_units,
)
# %%
# %%
exact_mpc = pd.to_numeric(
    exact_response["mpc"],
    errors="coerce",
).to_numpy(dtype=float)

surrogate_mpc = pd.to_numeric(
    surrogate_response["mpc"],
    errors="coerce",
).to_numpy(dtype=float)

finite = np.isfinite(exact_mpc) & np.isfinite(surrogate_mpc)

exact_mpc = exact_mpc[finite]
surrogate_mpc = surrogate_mpc[finite]

error = surrogate_mpc - exact_mpc

summary = pd.DataFrame(
    {
        "statistic": [
            "households",
            "true mean MPC",
            "surrogate mean MPC",
            "true median MPC",
            "surrogate median MPC",
            "MPC bias",
            "MPC RMSE",
            "MPC correlation",
            "MPC R2",
        ],
        "value": [
            len(exact_mpc),
            exact_mpc.mean(),
            surrogate_mpc.mean(),
            np.median(exact_mpc),
            np.median(surrogate_mpc),
            error.mean(),
            np.sqrt(np.mean(error**2)),
            np.corrcoef(exact_mpc, surrogate_mpc)[0, 1],
            r2_score(exact_mpc, surrogate_mpc),
        ],
    }
)

summary
# %%
# %%
# Plot in the same form as the website: exact distribution as bars and
# surrogate distribution as a line through the same histogram bins.

pooled = np.concatenate([exact_mpc, surrogate_mpc])

lo, hi = np.quantile(
    pooled,
    [0.005, 0.995],
)

lo = min(float(lo), -0.05)
hi = max(float(hi), 1.05)

edges = np.linspace(
    lo,
    hi,
    BINS + 1,
)

centers = 0.5 * (edges[:-1] + edges[1:])

widths = np.diff(edges)

exact_density, _ = np.histogram(
    exact_mpc,
    bins=edges,
    density=True,
)

surrogate_density, _ = np.histogram(
    surrogate_mpc,
    bins=edges,
    density=True,
)

plt.figure(figsize=(10, 6))

plt.bar(
    centers,
    exact_density,
    width=widths,
    alpha=0.55,
    label="Solved model",
)

plt.plot(
    centers,
    surrogate_density,
    linewidth=3,
    label="Polynomial surrogate",
)

plt.axvline(
    exact_mpc.mean(),
    linestyle="--",
    label=f"Model mean: {exact_mpc.mean():.3f}",
)

plt.axvline(
    surrogate_mpc.mean(),
    linestyle=":",
    label=f"Surrogate mean: {surrogate_mpc.mean():.3f}",
)

plt.xlim(lo, hi)
plt.xlabel("Marginal propensity to consume")
plt.ylabel("Density")
plt.title(f"Model {MODEL_ID}: MPC distribution from a " f"${CHECK_AMOUNT:,.0f} check")
plt.legend()
plt.tight_layout()
plt.show()
# %%
