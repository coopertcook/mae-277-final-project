import time
from copy import deepcopy
from rich import print

from GIFT.core.aircraft import Aircraft
from GIFT.core.simulator import Simulator
from GIFT.processing.animator import Animation
from GIFT.processing.plotter import SimulationPlotter
from GIFT.utils.utilities import TrajectoryData, standardize_simulator_data


animator_type: str = "matplotlib"
visualize: bool = True
vehicle: str = "raptor_glider_3dof"

time_start = time.time()

aircraft = Aircraft(type=vehicle, config_overrides={})
simulation = Simulator(aircraft, dynamics="3DOF")
LPV_dynamics = Simulator(aircraft, enable_logging=False, dynamics="LPV3").dynamics

raw_sim_data = simulation.execute(t_start=1, t_end=2)
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
