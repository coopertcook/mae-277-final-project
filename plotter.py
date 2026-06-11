import sys
import glob
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


LOG_DIR = "outputs/_data_logs"


def load_latest_log():
    files = sorted(glob.glob(f"{LOG_DIR}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {LOG_DIR}")
    return files[-1]


def plot_log(mpc_predicted_states, csv_path: str, perch_location: np.ndarray = None, save_path: str = None):
    '''
    Plot longitudinal simulation results from a GIFT log CSV.

    csv_path:       path to the CSV log file
    perch_location: [north, down] target (optional)
    '''
    df = pd.read_csv(csv_path)

    mpc_times = np.arange(0, mpc_predicted_states["timestep"]*(mpc_predicted_states["N"] + 0.5), mpc_predicted_states["timestep"])
    t     = df["t"].to_numpy()
    north = df["north"].to_numpy()
    down  = df["down"].to_numpy()
    u     = df["u"].to_numpy()
    w     = df["w"].to_numpy()
    theta = df["theta_deg"].to_numpy()
    tailalt = df["del_tail_alt_rad"].to_numpy()
    splay   = df["del_tail_splay_rad"].to_numpy()

    perch_north = perch_location[0] if perch_location is not None else None
    perch_alt   = perch_location[1] if perch_location is not None else None  # down → altitude

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Longitudinal Simulation\n{csv_path.split('/')[-1]}")

    # Trajectory: north (x-axis) vs altitude (y-axis)
    ax: Axes = axes[0, 0]
    ax.plot(north, down, "b-", linewidth=2)
    ax.plot(north[0], down[0], "go", markersize=10, label="Start")
    if perch_location is not None:
        ax.plot(perch_north, perch_alt, "r*", markersize=14, label="Perch target")
        ax.add_patch(mpatches.Circle((perch_north, perch_alt), 0.5,
                                     color="red", fill=False, linestyle="--",
                                     linewidth=1.5, label="0.5 m target zone"))
    for predicted_states in mpc_predicted_states["states"]:
        ax.plot(predicted_states[3, :], predicted_states[4, :], ":")
        ax.scatter(predicted_states[3, 0], predicted_states[4, 0])
    ax.set_xlabel("North (m)")
    ax.set_ylabel("Down (m)")
    ax.invert_yaxis()

    ax.set_title("Trajectory  (longitudinal plane)")
    ax.legend()
    ax.grid(True)

    # Body velocities
    ax = axes[0, 1]
    ax.plot(t, u, label="u  [body forward, m/s]")
    ax.plot(t, w, label="w  [body down, m/s]")
    if perch_location is not None:
        ax.axhline(0, color="r", linestyle="--", linewidth=1, label="target u=0 (perch)")
    for start_time, predicted_states in zip(mpc_predicted_states["time"], mpc_predicted_states["states"]):
        ax.plot(start_time + mpc_times, predicted_states[0, :], ":")
        ax.plot(start_time + mpc_times, predicted_states[1, :], ":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("m/s")
    ax.set_title("Body velocities")
    ax.legend()
    ax.grid(True)

    # Pitch angle
    ax = axes[1, 0]
    ax.plot(t, theta)
    for start_time, predicted_states in zip(mpc_predicted_states["time"], mpc_predicted_states["states"]):
        ax.plot(start_time + mpc_times, np.degrees(predicted_states[5, :]), ":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("theta (deg)")
    ax.set_title("Pitch angle")
    ax.grid(True)

    # Controls
    ax = axes[1, 1]
    ax.plot(t, np.degrees(tailalt), label="tailalt (deg)")
    ax.plot(t, np.degrees(splay),   label="splay (deg)")
    for start_time, predicted_controls in zip(mpc_predicted_states["time"], mpc_predicted_states["controls"]):
        ax.plot(mpc_times[:-1] + start_time, np.degrees(predicted_controls[0, :]), ":")
        ax.plot(mpc_times[:-1] + start_time, np.degrees(predicted_controls[1, :]), ":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("deg")
    ax.set_title("Applied controls")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    # Usage: python plotter.py [csv_path] [perch_north] [perch_down]
    # Example: python plotter.py outputs/_data_logs/sim_....csv 15 0
    csv_path = sys.argv[1] if len(sys.argv) > 1 else load_latest_log()
    perch = None
    if len(sys.argv) == 4:
        perch = np.array([float(sys.argv[2]), float(sys.argv[3])])
    print(f"Plotting: {csv_path}")
    plot_log(csv_path, perch_location=perch)
