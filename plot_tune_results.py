import glob
import re
from pathlib import Path

from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TUNE_LOG_DIR = Path("outputs/tune_datalog")
PERCH = np.array([0., 0.])  # [north, down]


def parse_metric(filename: str) -> float:
    m = re.search(r"_dist([\d.]+)\.csv$", filename)
    return float(m.group(1)) if m else float("inf")


def parse_params(filename: str) -> dict:
    q = re.search(r"_Q([\d.]+)_",  filename)
    r = re.search(r"_R([\d.]+)_",  filename)
    n = re.search(r"_N(\d+)_",     filename)
    return {
        "Q_pos": float(q.group(1)) if q else None,
        "R":     float(r.group(1)) if r else None,
        "N":     int(n.group(1))   if n else None,
    }


def main():
    csvs = sorted(glob.glob(str(TUNE_LOG_DIR / "*.csv")))
    if not csvs:
        print(f"No CSVs found in {TUNE_LOG_DIR}")
        return

    metrics = [parse_metric(p) for p in csvs]
    best_idx    = int(np.argmin(metrics))
    best_name   = Path(csvs[best_idx]).stem
    best_params = parse_params(csvs[best_idx])

    print(f"\nBest trial:  {best_name}")
    print(f"  Q_pos = {best_params['Q_pos']}")
    print(f"  R     = {best_params['R']}")
    print(f"  N     = {best_params['N']}")
    print(f"  dist  = {metrics[best_idx]:.3f} m from perch")
    print(f"\nRe-run best:")
    print(f"  python run_simulation.py --Q_pos={best_params['Q_pos']} --R={best_params['R']} --N={best_params['N']}\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Bayesian Tuning — {len(csvs)} trials\nBest: {best_name}")

    ax_traj, ax_vel, ax_theta, ax_ctrl = (
        axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    )
    ax_traj: Axes

    for i, (csv_path, metric) in enumerate(zip(csvs, metrics)):
        df = pd.read_csv(csv_path)
        t       = df["t"].to_numpy()
        north   = df["north"].to_numpy()
        down    = df["down"].to_numpy()
        u       = df["u"].to_numpy()
        w       = df["w"].to_numpy()
        theta   = df["theta_deg"].to_numpy()
        tailalt = np.degrees(df["del_tail_alt_rad"].to_numpy())
        splay   = np.degrees(df["del_tail_splay_rad"].to_numpy())

        is_best = (i == best_idx)
        kw = dict(linewidth=2.0, alpha=1.0, zorder=3) if is_best else dict(linewidth=0.8, alpha=0.15, color="gray", zorder=1)

        ax_traj.plot(north, down, **kw)
        ax_vel.plot(t, u, **kw)
        ax_vel.plot(t, w, **(kw | ({"linestyle": "--"} if is_best else {})))
        ax_theta.plot(t, theta, **kw)
        ax_ctrl.plot(t, tailalt, **kw)
        ax_ctrl.plot(t, splay,   **(kw | ({"linestyle": "--"} if is_best else {})))

    # Perch marker + target zone
    ax_traj.plot(PERCH[0], -PERCH[1], "r*", markersize=14, zorder=5, label="Perch")
    ax_traj.add_patch(mpatches.Circle((PERCH[0], PERCH[1]), 0.5,
                                      color="red", fill=False, linestyle="--",
                                      linewidth=1.5, zorder=5, label="0.5 m target zone"))

    # Best-trial legend entries
    best_df = pd.read_csv(csvs[best_idx])
    ax_traj.plot([], [], color="C0", linewidth=2, label=f"Best (dist={metrics[best_idx]:.2f} m)")
    ax_traj.plot([], [], color="gray", alpha=0.4, linewidth=1, label="Other trials")

    ax_vel.plot([], [], color="C0",              linewidth=2, label="u (best)")
    ax_vel.plot([], [], color="C0", linestyle="--", linewidth=2, label="w (best)")
    ax_ctrl.plot([], [], color="C0",              linewidth=2, label="tailalt (best)")
    ax_ctrl.plot([], [], color="C0", linestyle="--", linewidth=2, label="splay (best)")

    for ax, title, ylabel in [
        (ax_traj,  "Trajectory",      "Altitude (m)"),
        (ax_vel,   "Body velocities", "m/s"),
        (ax_theta, "Pitch angle",     "theta (deg)"),
        (ax_ctrl,  "Controls",        "deg"),
    ]:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(fontsize=7)

    ax_traj.set_xlim(-17, 2)
    ax_traj.set_ylim(-10, 5)
    ax_traj.set_xlabel("North (m)")
    ax_traj.invert_yaxis()
    ax_vel.set_xlabel("Time (s)")
    ax_theta.set_xlabel("Time (s)")
    ax_ctrl.set_xlabel("Time (s)")

    plt.tight_layout()
    out = Path("outputs/tune_summary.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
