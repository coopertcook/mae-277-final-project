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

from utilities import convert_GIFT_AB_to_QP_AB
from form_LPVQP_matrices import form_LPVQP_matrices


animator_type: str = "matplotlib"
visualize: bool = True
vehicle: str = "raptor_glider_3dof"

aircraft = Aircraft(type=vehicle, config_overrides={})
simulation = Simulator(aircraft, dynamics="3DOF")
LPV_dynamics = Simulator(aircraft, enable_logging=False, dynamics="LPV3").dynamics

t_start = 0
t_end = 5
dt = 0.1
perch_location = np.array([25., -10.]) # [north, down] (using GIFT location information)
N = 5

time_start = time.time()

times = np.arange(t_start, t_end + dt, dt)
for i, t in enumerate(times[:-1]):
    current_state = np.append(aircraft.x[[0, 2, 7, 3, 5, 10]], 1)
    current_control = np.append(aircraft.u, 1)

    states_to_fetch = [current_state]
    controls_to_fetch = [current_control]
    
    for N_num in range(1, N+1):
        print(f">>> Solving QP for N = {N_num} at time {t}...")

        # Fetch the A and B matrices for each of the state and control vectors for the current problem.
        A_qps: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
        B_qps: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
        for state, control in zip(states_to_fetch, controls_to_fetch):
            # Convert the QP state to the state notation used in GIFT
            GIFT_state = np.zeros((12))
            GIFT_control = np.zeros((2))
            GIFT_state[[0, 2, 7, 3, 5, 10]] = state[:-1]
            GIFT_control[[0, 1]] = control[:-1]

            # Get initial body-frame, nearest trim A and B matrices
            dx, du, A, B = LPV_dynamics.get_body_AB(GIFT_state, GIFT_control)
            
            # Convert the matrices for the QP
            A_qp, B_qp = convert_GIFT_AB_to_QP_AB(dx, du, A, B, 0.1)

            # Add the QP A and B matrices to the lists
            A_qps.append(A_qp)
            B_qps.append(B_qp)

        # Running cost on distance from perch: (x_pos - 25)^2 + (z_pos + 10)^2
        Q = np.zeros((7, 7))
        Q[[3, 4], [3, 4]] = 1
        Q[[3, 4], -1] = -perch_location
        Q[-1, [3, 4]] = -perch_location

        # Create the control cost matrix (2x2 since control is [tailalt, splay])
        R = np.eye(2) * 0.001

        # Create the terminal cost matrix
        P = np.zeros((7, 7))
        P[[3, 4], [3, 4]] += 1
        P[[3, 4], -1] = -perch_location
        P[-1, [3, 4]] = -perch_location

        # Create the constraint matrices
        Aug_x = np.array([
            # [-1, 0, 0, 0, 0, 0, 0,    5],
            # [ 1, 0, 0, 0, 0, 0, 0,   16],
            [ 0,-1, 0, 0, 0, 0, 0,    5],
            [ 0, 1, 0, 0, 0, 0, 0,    5],
        ])
        Aug_u = np.array([
            [-1, 0,    12 * np.pi / 180],
            [ 1, 0,    12 * np.pi / 180],
            [ 0,-1,    10 * np.pi / 180],
            [ 0, 1,    45 * np.pi / 180],
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
        
        # Create the quadratic program problem using CasADi's `conic` class.
        qp = {}
        qp['h'] = h.sparsity() # Passes only the sparsity patterns
        qp['a'] = a.sparsity() # Passes only the sparsity patterns
        S = ca.conic('S', 'qpoases', qp)

        # Calculate the current linear term vector and constraint right-hand side matrix.
        g = np.reshape(np.einsum('ij, j -> i', L, current_state), (L.shape[0], 1))
        uba = (W + np.einsum('ij, j -> i', T, current_state)[:, np.newaxis])

        # Solve the QP
        sol = S(h=h, g=g, a=a, uba=uba)
        U_opt = np.array(sol['x'])

        # Get the QP predicted states and control inputs
        x_pred = Sx @ current_state[:, np.newaxis] + Su @ U_opt
        x_pred = x_pred.reshape(N_num+1, len(current_state)).T # Each row: time step N, each column: state variable
        u_pred = U_opt.reshape(N_num, -1).T

        # Append all the new states and controls to lists for the next QP loop
        states_to_fetch = [states_to_fetch[0]]
        controls_to_fetch = [controls_to_fetch[0]]
        states_to_fetch.extend([x_pred[:, i] for i in range(1, N_num+1)])   # Pretty sure this is messed up
        controls_to_fetch.extend([u_pred[:, i] for i in range(N_num)])

    uk = IMPC @ U_opt

    # print(aircraft.control_view.as_dict())

    aircraft.u[0] = uk[0, 0]
    aircraft.u[1] = uk[1, 0]

    _ = simulation.execute(t_start=t, t_end=t+dt, t_step=1e-2, write_history=False)
    # aircraft.u[0] += 0.01

raw_sim_data = simulation.execute(t_start=times[-1], t_end=times[-1] + dt, write_history=True)
sim_data = standardize_simulator_data(raw_sim_data, aircraft=aircraft)

result = TrajectoryData(
        name=f"System",
        data=sim_data,
        config=deepcopy(aircraft.config),
    )
time_end = time.time()
print(f">>> Simulation completed in {time_end - time_start:.2f} seconds")

if visualize:
    result.data = result.data.drop_nulls()
    # use the results dictionary to visualize the data
    plots = SimulationPlotter(
        results=[result],
        vehicle=vehicle,
    )
    plots.show_all()

    # print(">>> Starting Matplotlib animation...")
    # Animation([result]).animate(speed=0.4)
