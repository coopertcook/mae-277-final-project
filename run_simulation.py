import argparse
import time
import casadi as ca
import numpy as np
from copy import deepcopy
from rich import print

from GIFT.core.aircraft import Aircraft
from GIFT.core.simulator import Simulator
from GIFT.processing.animator import Animation
from GIFT.processing.plotter import SimulationPlotter
from GIFT.utils.utilities import TrajectoryData, standardize_simulator_data

from utilities import convert_GIFT_AB_to_QP_AB, SuppressOutput
from form_LPVQP_matrices import form_LPVQP_matrices
from plotter import plot_log, load_latest_log

# CLI args: defaults match current hand-tuned values so running directly is unchanged
_p = argparse.ArgumentParser(add_help=False)
_ = _p.add_argument("--Q_pos", type=float, default=1)
_ = _p.add_argument("--R",     type=float, default=0.5)
_ = _p.add_argument("--N",     type=int,   default=5)
_ = _p.add_argument("--no_viz", action="store_true")
_args, _ = _p.parse_known_args()

animator_type: str = "matplotlib"
visualize: bool = not _args.no_viz
vehicle: str = "raptor_glider_3dof"

aircraft = Aircraft(type=vehicle, config_overrides={})
simulation = Simulator(aircraft, dynamics="3DOF")
LPV_dynamics = Simulator(aircraft, enable_logging=False, dynamics="LPV3").dynamics

t_start = 0
t_end = 10
dt = 0.01 # GIFT timestep
Ts = 0.01 # MPC timestep
perch_location = np.array([0., 0.]) # [north, down] (using GIFT location information)
N = _args.N
aircraft.x[[3, 5]] = np.array([-15., 0.])
aircraft.u[1] = 35 * np.pi / 180

time_start = time.time()


mpc_predicted_states = {
    "timestep": Ts,
    "N": N,
    "time": [],
    "states": [],
    "controls": [],
}
last_time = 0
_min_dist = float("inf")
times = np.arange(t_start, t_end + dt, dt)
starting_time = time.time()
for i, t in enumerate(times[:-1]):
    current_state = np.append(simulation.aircraft.x[[0, 2, 7, 3, 5, 10]], 1)
    current_control = simulation.aircraft.u

    states_to_fetch = [current_state]
    controls_to_fetch = [current_control]

    # Pre-step check (avoid solving QP when already arrived)
    _dist = np.linalg.norm(current_state[[3, 4]] - perch_location, ord=2, axis=0)
    _min_dist = min(_min_dist, _dist)
    if _dist <= 0.5:
        last_time = t
        break
    if _min_dist < 3.0 and _dist > _min_dist + 3.0:
        last_time = t
        break
    if _dist < np.max(perch_location) + 1:  # further than initial distance — clearly diverging
        last_time = t
        break
    if time.time() - 2 > starting_time:
        break

    starting_time = time.time()

    try:
        N_nums = list(range(1, N+1))
        for N_num in N_nums:
            print(f">>> Solving QP for N = {N_num} at time {t}...")
            states_to_fetch = states_to_fetch[:N_num]
            controls_to_fetch = controls_to_fetch[:N_num]

            # Fetch the A and B matrices for each of the state and control vectors for the current problem.
            A_qps: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
            B_qps: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
            for state, control in zip(states_to_fetch, controls_to_fetch):
                # Convert the QP state to the state notation used in GIFT
                GIFT_state = np.zeros((12))
                GIFT_control = np.zeros((2))
                GIFT_state[[0, 2, 7, 3, 5, 10]] = state[:-1]
                GIFT_control[[0, 1]] = control

                # Get initial body-frame, nearest trim A and B matrices
                x_trim, u_trim, A, B = LPV_dynamics.get_body_AB(GIFT_state, GIFT_control, last_time)
                u_trim = np.array([0, u_trim[1]])

                # Convert the matrices for the QP
                A_qp, B_qp = convert_GIFT_AB_to_QP_AB(x_trim, u_trim, A, B, Ts)

                # Rotate body velocities (u,w) to inertial (NED) frame using current pitch angle.
                theta = state[5] # pitch angle is at index 5 in the QP state vector
                A_qp[[3, 3], [0, 1]] = [ Ts * np.cos(theta), Ts * np.sin(theta)]
                A_qp[[4, 4], [0, 1]] = [-Ts * np.sin(theta), Ts * np.cos(theta)]
                # A_qp[4, 5] *= -1
                # A_qp[[0, 1, 2], 0] += state[[0, 1, 2]]

                # B_qp = B_qp[:, [0]]
                B_qp[:5, :] *= -1

                # Add the QP A and B matrices to the lists
                A_qps.append(A_qp)
                B_qps.append(B_qp)
            
            # Running cost on distance from perch: (x_pos - 25)^2 + (z_pos + 10)^2
            Q = np.zeros((7, 7))
            Q[[0, 1, 2], [0, 1, 2]] = 0
            Q[[3, 4], [3, 4]] = _args.Q_pos

            # Create the control cost matrix (2x2 since control is [tailalt, splay])
            R = np.eye(B_qps[0].shape[1]) * _args.R

            # Create the terminal cost matrix
            P = np.zeros((7, 7))
            P[[3, 4], [3, 4]] += _args.Q_pos

            # Create the constraint matrices
            Aug_x = np.array([
                # [-1, 0, 0, 0, 0, 0, 0,   -5],
                # [ 1, 0, 0, 0, 0, 0, 0,   16],
                # [ 0,-1, 0, 0, 0, 0, 0,    5],
                # [ 0, 1, 0, 0, 0, 0, 0,    5],
                [ 0, 0, 0, 0, 0, 0, 0,    5],
                [ 0, 0, 0, 0, 0, 0, 0,    5],
                # [ 0, 0, 0, 0, 0, 1, 0,    45 * np.pi / 180],  # theta <=  45 deg
                # [ 0, 0, 0, 0, 0,-1, 0,    45 * np.pi / 180],  # theta >= -45 deg
                # [ 0, 0, 1, 0, 0, 0, 0,    20 * np.pi / 180],  # q <=  20 deg/s
                # [ 0, 0,-1, 0, 0, 0, 0,    20 * np.pi / 180],  # q >= -20 deg/s
            ])
            Aug_u = np.array([
                [ 0,-1,   -20 * np.pi / 180],  # splay >= 20 deg (near trim ~26 deg)
                [ 0, 1,    35 * np.pi / 180],  # splay <= 35 deg (near trim ~26 deg)
                [-1, 0,    25 * np.pi / 180],
                [ 1, 0,    25 * np.pi / 180],
                # [-1,    25 * np.pi / 180],
                # [ 1,    25 * np.pi / 180],
            ])

            # Get the QP matrices
            H, L, G, W, T, IMPC, Sx, Su = form_LPVQP_matrices(
                np.stack(A_qps),
                np.stack(B_qps),
                np.stack([Q]*N_num),
                np.stack([R]*N_num),
                P, Aug_x, Aug_u
            )

            # Initialize the CasADi sparse matrices
            h = ca.DM(H)
            a = ca.DM(G)

            with SuppressOutput():
                # Create the quadratic program problem using CasADi's `conic` class.
                qp = {}
                qp['h'] = h.sparsity() # Passes only the sparsity patterns
                qp['a'] = a.sparsity() # Passes only the sparsity patterns
                S = ca.conic('S', 'qpoases', qp)

                # Calculate the current linear term vector and QP constraint right-hand side matrix.
                g = 2 * L @ current_state[:, np.newaxis]
                uba = W + T @ current_state[:, np.newaxis]

                # Solve the QP
                sol = S(h=h, g=g, a=a, uba=uba)
                U_opt = np.array(sol['x'])

            # Get the QP predicted states and control inputs
            x_pred = Sx @ current_state[:, np.newaxis] + Su @ U_opt
            x_pred = x_pred.reshape(N_num+1, len(current_state)).T # Each row: time step N, each column: state variable
            u_pred = U_opt.reshape(N_num, -1).T
            # u_pred = np.vstack([u_pred, np.ones_like(u_pred)*current_control[1]])

            # Append all the new states and controls to lists for the next QP loop
            states_to_fetch = [states_to_fetch[0]]
            controls_to_fetch = [controls_to_fetch[0]]
            states_to_fetch.extend([x_pred[:, i] for i in range(1, x_pred.shape[1])])
            controls_to_fetch.extend([u_pred[:, i] for i in range(u_pred.shape[1])])
        
        mpc_predicted_states["time"].append(last_time)
        mpc_predicted_states["states"].append(x_pred)
        mpc_predicted_states["controls"].append(u_pred)
        uk = IMPC @ U_opt
        # uk[0, 0]
        simulation.aircraft.u[0] = u_pred[0, 0]
        simulation.aircraft.u[1] = u_pred[1, 0]

        # Advance the simulation by one step with the new control input
        _ = simulation.execute(t_start=t, t_step=1e-3, t_end=t+dt, write_history=False)
        last_time = t + dt

        # Post-step check: catches overshoot within the same timestep
        _post_state = np.append(aircraft.x[[0, 2, 7, 3, 5, 10]], 1)
        _post_dist = np.linalg.norm(_post_state[[3, 4]] - perch_location, ord=2, axis=0)
        _min_dist = min(_min_dist, _post_dist)
        if _post_dist <= 0.5 or (_min_dist < 3.0 and _post_dist > _min_dist + 3.0):
            break
    except (RuntimeError, KeyboardInterrupt):
        print("!!! Error occurred. Stopping simulation... !!!")
        break

raw_sim_data = simulation.execute(t_start=last_time, t_end=last_time + dt, write_history=True)
sim_data = standardize_simulator_data(raw_sim_data, aircraft=aircraft)

result = TrajectoryData(
        name=f"System",
        data=sim_data,
        config=deepcopy(aircraft.config),
    )
time_end = time.time()
print(f">>> Simulation completed in {time_end - time_start:.2f} seconds")

# Compute and save metric for Bayesian optimisation
import json, glob, os, pandas as pd
_csv_files = sorted(glob.glob("outputs/_data_logs/*.csv"))
if _csv_files:
    _df = pd.read_csv(_csv_files[-1])
    _north = _df["north"].to_numpy()
    _down  = _df["down"].to_numpy()
    _t_max = float(_df["t"].max())
    _dists = np.sqrt((_north - perch_location[0])**2 + (_down - perch_location[1])**2)
    
    if np.min(_dists) <= 2:
        _metric = _t_max # float(_dists.min())
    else:
        _metric = 100
    # _metric += max(0.0, 1.0 - _t_max) * 20.0   # heavy penalty for runs shorter than 2s (crashes)
else:
    _metric = 1000.0
os.makedirs("outputs", exist_ok=True)
with open("outputs/mpc_metric.json", "w") as _f:
    json.dump({"metric": _metric, "Q_pos": _args.Q_pos, "R": _args.R, "N": _args.N,
               "t_max": _t_max if _csv_files else 0.0}, _f)
print(f">>> Metric (min dist to perch): {_metric:.3f} sec")

# if visualize:
#     result.data = result.data.drop_nulls()
#     # use the results dictionary to visualize the data
#     plots = SimulationPlotter(
#         results=[result],
#         vehicle=vehicle,
#     )
#     plots.show_all()

if visualize:
    plot_log(mpc_predicted_states, load_latest_log(), perch_location=perch_location)
