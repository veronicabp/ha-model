# HA policy surrogate: consistent release

This directory contains a complete, mutually compatible training, evaluation,
deployment, and Streamlit stack for the lifecycle HA policy surrogate.

## Files

- `ha_model_grid.py`: HA solver and solved-grid construction. This is the latest
  uploaded solver and is unchanged by the surrogate refactor.
- `ha_policy_surrogate.py`: saved-grid reader, population sampler, feature map,
  tuning, true-holdout evaluation fit, and final production fit.
- `train_surrogate.py`: command-line interface for surrogate training.
- `train_surrogate.sh`: SLURM example using the recommended defaults.
- `evaluate_surrogate.py`: MPC and curvature diagnostics. It only evaluates the
  development-only holdout bundle.
- `ha_policy_visualization.py`: policy-slice and exact-grid helpers.
- `ha_mpc_distribution.py`: synthetic-population and exact/surrogate MPC tools.
- `policy_explorer.py`: Streamlit application.
- `construct_export_data.py`: creates a lightweight `app_data` directory for
  deployment and copies both surrogate bundles plus selected exact holdout grids.

## Training outputs

Training creates two distinct bundles:

- `policy_surrogate.joblib`: production bundle refitted on all usable models.
- `policy_surrogate_holdout.joblib`: evaluation bundle fitted only on training
  and tuning models. Its declared test models are disjoint from its fit sample.

It also creates:

- `model_split.json`: frozen model-level train/tune/test split. Existing tuning
  and test IDs remain fixed across later runs; newly added models enter training.
- `parameter_term_search.csv`: sequential forward-search results.
- `policy_grid_sample.pkl.gz` or Parquet equivalent.
- `model_catalog.csv`, validation tables, failure tables, and summary JSON.

Bundle format version is 10. Use the same `ha_policy_surrogate.py` and compatible
NumPy/SciPy/scikit-learn versions in training and deployment environments.

## Mandatory basis

The default feature map includes the terms that were consistently important in
prior holdout diagnostics:

- splines in raw liquid assets, liquid slack / permanent income, illiquid assets
  / permanent income, pre-tax income / permanent income, post-tax income /
  permanent income, and working-age age;
- an exact positive-part hinge at zero liquid assets;
- liquid-state tensor interactions with income, illiquid wealth, and age;
- a liquid-slack × pre-tax-income × illiquid-wealth spline tensor;
- unemployed-specific liquid-slack × income and liquid-slack × illiquid-wealth
  spline tensors;
- liquid-state spline interactions with risk-aversion squared;
- liquid-state spline interactions with discount factor × risk aversion and
  convex adjustment cost × risk aversion;
- employed/unemployed-specific zero-asset slopes and quadratic curvature.

Redundant working-age horizon variables and cash-on-hand identities are not
included as independent regression states.

## Automatic higher-order parameter selection

After fitting the mandatory basis, training performs greedy forward selection on
tuning models. In each round it screens every remaining eligible parameter power
and parameter pair, each modulated by the liquid-state spline bases. The best
term is accepted only when it clears the requested MPC R² gain, model-win-share,
bootstrap, multiple-testing, and combined-loss criteria. The expanded baseline
is then rebuilt before the next round. With `--parameter-selection-max-terms 0`,
selection continues until no remaining term passes.

The screening row caps retain every model and sample rows approximately evenly
within model while preserving complete MPC pairs.

## Population and MPC sampling

The default application is working-age only: `age < retirement_age`, matching
how the solver defines retirement. The sampler oversamples hand-to-mouth states,
working-age unemployment, low liquid slack, MPC tails, and transfers that move a
household from negative to positive liquid wealth.

## Recommended cluster run

Edit the paths in `train_surrogate.sh`, then submit:

```bash
sbatch train_surrogate.sh
```

The shell script uses 1,000 rows per model, up to 3 million rows, 60% paired MPC
rows, 30% cross-zero pairs, target-tail stratification, and tuning over direct
MPC objective weights.

## True-holdout diagnostics

The evaluator accepts either bundle path. When given the production bundle, it
automatically loads its sibling holdout bundle and refuses to proceed unless the
fit/test overlap is zero.

```bash
python evaluate_surrogate.py \
  --bundle ../../../orcd/scratch/ha_surrogate/policy_surrogate.joblib \
  --output-dir ../../../orcd/scratch/ha_evaluation
```

## Deployment

Edit paths and the desired number of exact public models in
`construct_export_data.py`, then run:

```bash
python construct_export_data.py
```

The resulting `app_data` contains both bundles. Ordinary website pages use the
production bundle. Validation and exact-grid comparison pages use the true-
holdout bundle. The Grid versus polynomial page supports MPC as an outcome,
calculated as the exact and surrogate consumption response to the same liquid
transfer.
