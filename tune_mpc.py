import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local imports
METRIC_FILE  = Path("outputs/mpc_metric.json")
LOG_DIR      = Path("outputs/_data_logs")
TUNE_LOG_DIR = Path("outputs/tune_datalog")
TIMEOUT = 300

# Compute the metric for a single simulation run with given MPC parameters
def run_simulation(Q_pos: float, R: float, N: int) -> float:
    cmd = [sys.executable, "run_simulation.py",
           f"--Q_pos={Q_pos}", f"--R={R}", f"--N={N}", "--no_viz"]
    try:
        subprocess.run(cmd, timeout=TIMEOUT, check=False)
    except (subprocess.TimeoutExpired, Exception):
        return 1000.0

    if not METRIC_FILE.exists():
        return 1000.0

    with open(METRIC_FILE) as f:
        return json.load(f)["metric"]

# Save a copy of the latest log CSV with a name encoding the trial parameters and metric
def save_trial_log(trial_number: int, Q_pos: float, R: float, N: int, metric: float) -> None:
    csvs = sorted(glob.glob(str(LOG_DIR / "*.csv")))
    if not csvs:
        return
    TUNE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(csvs[-1])
    dst = TUNE_LOG_DIR / f"trial_{trial_number:03d}_Q{Q_pos:.4f}_R{R:.6f}_N{N}_dist{metric:.2f}.csv"
    shutil.copy(src, dst)

# Objective function for Optuna to minimize
def objective(trial: optuna.Trial) -> float:
    Q_pos = trial.suggest_float("Q_pos", 0.01, 10.0, log=True)
    R     = trial.suggest_float("R",     0.0001, 1.0, log=True)
    N     = trial.suggest_int  ("N",     2, 5)

    metric = run_simulation(Q_pos, R, N)
    save_trial_log(trial.number, Q_pos, R, N, metric)
    print(f"Trial {trial.number:3d} | Q_pos={Q_pos:.4f}  R={R:.5f}  N={N}  metric={metric:.3f} m")
    return metric

# Main entry point: run Optuna optimization
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name="mpc_tuning",
        storage="sqlite:///outputs/mpc_tuning.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=args.trials)

    best = study.best_params
    print(f"\nBest: Q_pos={best['Q_pos']:.5f}  R={best['R']:.6f}  N={best['N']}"
          f"Error: {study.best_value:.3f} m from perch")
    print(f"\nRe-run best:")
    print(f"  python run_simulation.py --Q_pos={best['Q_pos']:.5f} --R={best['R']:.6f} --N={best['N']}")
