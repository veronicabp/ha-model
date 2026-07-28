#!/bin/bash
#SBATCH --job-name=ha-surrogate
#SBATCH --cpus-per-task=32
#SBATCH --partition=sched_mit_econ
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ha-surrogate-%j.out

set -euo pipefail
mkdir -p logs

MODEL_ROOT="../../../orcd/scratch/ha_models"
OUTPUT_DIR="../../../orcd/scratch/ha_surrogate"
CPUS="${SLURM_CPUS_PER_TASK:-1}"

# Sampling workers are explicitly limited to one BLAS thread in
# ha_policy_surrogate.py. The parent fit uses --fit-threads below.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python -u train_surrogate.py \
  --model-root "$MODEL_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --policy-layout GHKEBA \
  --money-units model \
  --sampling-mode population \
  --rows-per-model 1000 \
  --max-total-rows 3000000 \
  --n-jobs "$CPUS" \
  --fit-threads "$CPUS" \
  --model-type polynomial \
  --polynomial-degree 2 \
  --spline-knots 8 \
  --nonemployment-share 0.20 \
  --poor-htm-share 0.35 \
  --wealthy-htm-share 0.35 \
  --broad-support-share 0.10 \
  --mpc-pair-share 0.60 \
  --mpc-cross-zero-share 0.30 \
  --mpc-check-sizes "0.10,0.188679,0.20,0.25" \
  --mpc-loss-weight 2.0 \
  --mpc-difference-weight-grid "0,0.25,0.5,1,2,4,8" \
  --mpc-objective-max-pairs 500000 \
  --mpc-stratify-age-bins "35,45,55,60" \
  --mpc-stratify-liquid-bins 5 \
  --mpc-stratify-illiquid-bins 5 \
  --mpc-stratify-target-bins 5 \
  --ridge-alphas "0.0001,0.001,0.01,0.1,1" \
  --parameter-selection-max-train-rows 300000 \
  --parameter-selection-max-tune-rows 150000 \
  --parameter-selection-max-terms 0 \
  --consumption-transform identity \
  --deposit-transform asinh \
  --liquid-outflow-transform identity \
  --test-share 0.20 \
  --tuning-share 0.20
