import scipy
import numpy as np


def convert_GIFT_AB(dx, du, Ac, Bc, Ts: float):
    A_MPC = np.zeros((7, 7))
    
    # Move A matrix to the proper indices in MPC A matrix
    A_MPC[[[0], [1], [2], [5]], [0, 1, 2, 5]] = Ac
    
    # Calculate trim contribution and place it in last column
    A_dx = Ac @ dx[:, np.newaxis]
    A_MPC[[[0], [1], [2], [5]], [6]] = -A_dx

    # Since we integrate x and z in the state vector, we add those 1s to the matrix
    A_MPC[[3, 4], [0, 1]] += 1
    A_MPC[[3, 4, 5], [3, 4, 5]] += 1
    A_MPC[-1, -1] += 1


    B_MPC = np.zeros((7, 3))
    B_MPC[[[0], [1], [2], [5]], [0, 1]] = Bc
    B_du = Bc @ du[:, np.newaxis]
    B_MPC[[[0], [1], [2], [5]], [2]] = -B_du

    Ad = scipy.linalg.expm(A_MPC * Ts)
    Bd = np.linalg.inv(A_MPC) @ (Ad - np.eye(Ad.shape[0])) @ B_MPC

    return Ad, Bd
    