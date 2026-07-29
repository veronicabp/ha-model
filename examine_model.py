# %%
from pathlib import Path

import numpy as np
import pandas as pd

from ha_policy_surrogate import PolicySurrogateBundle
from ha_mpc_distribution import (
    PopulationAssumptions,
    compute_check_mpcs,
    compute_exact_grid_check_mpcs,
    generate_synthetic_population,
    infer_money_unit_info,
    load_exact_policy_grid,
    set_parameter_values,
)

BUNDLE_PATH = Path("app_data/policy_surrogate.joblib")
CATALOG_PATH = Path("app_data/model_catalog.csv")

# MODEL_ID = "955b358266a191"
MODEL_ID = "aad44501c6f4eb"
CHECK_DOLLARS = 10_000
N_HOUSEHOLDS = 20_000
SEED = 20260717


# %%
bundle = PolicySurrogateBundle.load(BUNDLE_PATH)

catalog = pd.read_csv(CATALOG_PATH)
catalog["model_id"] = catalog["model_id"].astype(str)

# for model_id in catalog["model_id"].sample(frac=1, replace=False, random_state=2):
#     row = catalog.loc[catalog["model_id"] == model_id].iloc[0]
#     row_dict = row.to_dict()
#     if row_dict["param__policy__income__persistence"] == 0.9:
#         break

row = catalog.loc[catalog["model_id"] == MODEL_ID].iloc[0]
model_dir = Path(str(row["model_dir"]))

if not model_dir.is_absolute():
    model_dir = BUNDLE_PATH.parent / model_dir

model_dir = model_dir.resolve()


# Set the surrogate parameters to this particular solved model.
model_parameters = dict(bundle.default_values)

for column in bundle.feature_spec.parameter_columns:
    if column.endswith("__missing"):
        model_parameters[column] = 0
    elif column in row.index and pd.notna(row[column]):
        model_parameters[column] = float(row[column])

for column in bundle.feature_spec.categorical_parameter_columns:
    if column in row.index and pd.notna(row[column]):
        model_parameters[column] = str(row[column])


unit_info = infer_money_unit_info(
    bundle,
    bundle_path=BUNDLE_PATH,
)

money_units = unit_info.money_units
dollars_per_model_unit = unit_info.dollars_per_model_unit

print(model_dir)
print(money_units, dollars_per_model_unit)

# %%
assumptions = PopulationAssumptions(
    n_households=N_HOUSEHOLDS,
    seed=SEED,
    # Website defaults:
    median_income_dollars=60_000,
    poor_hand_to_mouth_share=0.3,
    wealthy_hand_to_mouth_share=0.2,
    # negative_liquid_share=0.0,
)

population = generate_synthetic_population(
    bundle,
    base_values=model_parameters,
    assumptions=assumptions,
    dollars_per_model_unit=dollars_per_model_unit,
    money_units=money_units,
)

population = set_parameter_values(
    population,
    bundle,
    model_parameters,
)


exact_grid = load_exact_policy_grid(
    model_dir,
    policy_layout=bundle.training_metadata.get(
        "policy_layout",
        "GHKEBA",
    ),
)

exact_response = compute_exact_grid_check_mpcs(
    exact_grid,
    population,
    check_amount_dollars=CHECK_DOLLARS,
    dollars_per_model_unit=dollars_per_model_unit,
    money_units=money_units,
)


# Evaluate the surrogate on the states after they have been aligned to the
# selected model's exact age, income, employment, and education grids.
surrogate_population = set_parameter_values(
    exact_response.copy(),
    bundle,
    model_parameters,
)

surrogate_response = compute_check_mpcs(
    bundle,
    surrogate_population,
    check_amount_dollars=CHECK_DOLLARS,
    dollars_per_model_unit=dollars_per_model_unit,
    money_units=money_units,
)

# %%
household_mpcs = exact_response.copy()

household_mpcs["household_id"] = np.arange(len(household_mpcs))

household_mpcs["model_id"] = MODEL_ID

household_mpcs["true_mpc"] = pd.to_numeric(
    exact_response["mpc"],
    errors="coerce",
)

household_mpcs["surrogate_mpc"] = pd.to_numeric(
    surrogate_response["mpc"],
    errors="coerce",
)

household_mpcs["mpc_error"] = (
    household_mpcs["surrogate_mpc"] - household_mpcs["true_mpc"]
)

household_mpcs["true_baseline_consumption"] = exact_response["baseline_consumption"]

household_mpcs["true_post_check_consumption"] = exact_response["post_check_consumption"]

household_mpcs["surrogate_baseline_consumption"] = surrogate_response[
    "baseline_consumption"
]

household_mpcs["surrogate_post_check_consumption"] = surrogate_response[
    "post_check_consumption"
]


columns = [
    "household_id",
    "model_id",
    "population_type",
    "age",
    "is_retired",
    "employment_state",
    "current_income",
    "after_tax_income",
    "labor_income_state",
    "liquid_assets",
    "illiquid_assets",
    "true_mpc",
    "surrogate_mpc",
    "mpc_error",
    "true_baseline_consumption",
    "true_post_check_consumption",
    "surrogate_baseline_consumption",
    "surrogate_post_check_consumption",
]

columns = [column for column in columns if column in household_mpcs.columns]

household_mpcs = household_mpcs[columns]

household_mpcs

# %%
household_mpcs.groupby(
    "population_type",
    observed=True,
).agg(
    households=("true_mpc", "size"),
    true_mean_mpc=("true_mpc", "mean"),
    surrogate_mean_mpc=("surrogate_mpc", "mean"),
    mean_error=("mpc_error", "mean"),
    rmse=(
        "mpc_error",
        lambda x: np.sqrt(np.mean(np.asarray(x) ** 2)),
    ),
)

# %%
household_mpcs.to_csv("test.csv")

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# plt.hist(household_mpcs.true_mpc, alpha=0.5, density=True, bins=np.linspace(0, 0.6, 20))
# plt.hist(
#     household_mpcs.surrogate_mpc, alpha=0.5, density=True, bins=np.linspace(0, 0.6, 20)
# )
# %%

household_mpcs["liquid_to_income"] = household_mpcs[
    "liquid_assets"
]  # / household_mpcs["after_tax_income"]

# sns.regplot(
#     household_mpcs[household_mpcs.liquid_to_income < 2],
#     x="liquid_to_income",
#     y="true_mpc",
#     x_bins=20,
#     fit_reg=False,
# )
# sns.regplot(
#     household_mpcs[household_mpcs.liquid_to_income < 2],
#     x="liquid_to_income",
#     y="surrogate_mpc",
#     x_bins=20,
#     color="red",
#     fit_reg=False,
# )

# %%
sns.regplot(
    household_mpcs[
        (household_mpcs.employment_state == 1)
        & (household_mpcs.is_retired == 0)
        # & (household_mpcs.liquid_to_income < 1)
    ],
    x="liquid_to_income",
    y="true_mpc",
    x_bins=100,
    fit_reg=False,
)
sns.regplot(
    household_mpcs[
        (household_mpcs.employment_state == 1)
        & (household_mpcs.is_retired == 0)
        # & (household_mpcs.liquid_to_income < 1)
    ],
    x="liquid_to_income",
    y="surrogate_mpc",
    x_bins=100,
    fit_reg=False,
)
# %%
