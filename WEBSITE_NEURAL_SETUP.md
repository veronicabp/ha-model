# Website setup: polynomial and neural surrogates

Place these code files in the same directory:

- `policy_explorer.py`
- `ha_surrogate_backend.py`
- `ha_policy_visualization.py`
- `ha_mpc_distribution.py`
- `ha_policy_surrogate.py`
- `train_neural_surrogate.py`

The neural loader imports the model architecture and bundle class from the same
`train_neural_surrogate.py` that created the `.pt` file. This prevents the
website architecture from drifting away from the training architecture.

## Neural deployment files

The neural `app_data` directory should contain:

- `neural_policy_surrogate.pt`
- `neural_policy_surrogate_holdout.pt`
- `neural_validation_metrics.csv`
- `neural_validation_by_model.csv`
- `model_catalog.csv` or `neural_model_catalog.csv`
- `heldout_mpc_pairs.parquet` or `heldout_mpc_pairs.pkl.gz`
- `held_out_models/<model_id>/metadata.json`
- `held_out_models/<model_id>/arrays.npz`

Set `SURROGATE_TYPE = "neural"` in `construct_export_data.py` and run it to
construct these deployment files.

## Run

The app includes a sidebar selector for **Polynomial** versus **Neural network**.
You can also choose the initial backend from the command line.

```bash
streamlit run policy_explorer.py -- \
  --surrogate-type neural \
  --neural-bundle app_data/neural_policy_surrogate.pt \
  --neural-evaluation-bundle app_data/neural_policy_surrogate_holdout.pt \
  --neural-device auto
```

For the polynomial backend:

```bash
streamlit run policy_explorer.py -- \
  --surrogate-type polynomial \
  --polynomial-bundle app_data/policy_surrogate.joblib \
  --polynomial-evaluation-bundle app_data/policy_surrogate_holdout.joblib
```

`--bundle` and `--evaluation-bundle` remain supported as backward-compatible
generic path overrides.
