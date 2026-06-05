import time
from copy import deepcopy
from rich import print

from gift.GIFT.core.aircraft import Aircraft
from gift.GIFT.core.simulator import Simulator
from gift.GIFT.processing.animator import Animation
from gift.GIFT.processing.plotter import SimulationPlotter
from gift.GIFT.utils.utilities import TrajectoryData, standardize_simulator_data


animator_type: str = "matplotlib"
visualize: bool = True
vehicle: str = "raptor_glider_3dof"

time_start = time.time()

aircraft = Aircraft(type=vehicle, config_overrides={})
simulation = Simulator(aircraft, dynamics="LPV3")
LPV_dynamics = Simulator(aircraft, enable_logging=False, dynamics="LPV3").dynamics

raw_sim_data = simulation.execute()
sim_data = standardize_simulator_data(raw_sim_data, aircraft=aircraft)

result = TrajectoryData(
        name=f"System",
        data=sim_data,
        config=deepcopy(aircraft.config),
    )
time_end = time.time()
print(f">>> Simulation completed in {time_end - time_start:.2f} seconds")

if visualize:
    # use the results dictionary to visualize the data
    plots = SimulationPlotter(
        results=[result],
        vehicle=vehicle,
    )
    plots.show_all()

    # print(">>> Starting Matplotlib animation...")
    # Animation([result]).animate(speed=0.4)

