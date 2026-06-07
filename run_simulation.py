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

from utilities import convert_GIFT_AB
from form_LPVQP_matrices import form_LPVQP_matrices


animator_type: str = "matplotlib"
visualize: bool = False
vehicle: str = "raptor_glider_3dof"

aircraft = Aircraft(type=vehicle, config_overrides={})
simulation = Simulator(aircraft, dynamics="3DOF")
LPV_dynamics = Simulator(aircraft, enable_logging=False, dynamics="LPV3").dynamics

t_start = 0
t_end = 5
dt = 0.1
perch_location = np.array([25., 5.]) # [north, down] (using GIFT location information)
N = 20

time_start = time.time()

times = np.arange(t_start, t_end + dt, dt)
for i, t in enumerate(times[:-1]):
    current_state = np.append(aircraft.x[[0, 2, 7, 3, 5, 10]], 1)
    current_control = np.append(aircraft.u, 1)
    
    # Get body-frame, nearest trim A and B matrices
    dx, du, A, B = LPV_dynamics.get_body_AB(aircraft.x, aircraft.u)

    # Convert the matrices for the MPC
    A_qp, B_qp = convert_GIFT_AB(dx, du, A, B, 0.01)
    
    # Create the state cost matrix
    Q = np.zeros((7, 7))
    Q[[2, 3, 4], [2, 3, 4]] += 1
    
    # Create the state cost matrix
    R = np.zeros((3, 3))
    
    # Create the terminal cost matrix
    P = np.zeros((7, 7))
    P[[3, 4], [3, 4]] += 1
    P[[3, 4], -1] = -perch_location
    P[-1, [3, 4]] = -perch_location

    # Create the constraint matrices
    Aug_x = np.zeros((2, 8))
    Aug_u = np.zeros((2, 4))

    # Get the QP matrices
    H, L, G, W, T, IMPC = form_LPVQP_matrices(np.stack([A_qp]*N), np.stack([B_qp]*N), np.stack([Q]*N), np.stack([R]*N), P, Aug_x, Aug_u)
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
    uk = IMPC @ U_opt

    print(uk)

    _ = simulation.execute(t_start=t, t_end=t+dt, t_step=1e-2, write_history=False)
    aircraft.u[0] += 0.01

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
