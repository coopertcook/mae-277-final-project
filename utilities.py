import scipy
import numpy as np


def convert_GIFT_AB_to_QP_AB(
    dx: np.ndarray[tuple[int, int], np.dtype[np.float64]], 
    du: np.ndarray[tuple[int, int], np.dtype[np.float64]], 
    Ac: np.ndarray[tuple[int, int], np.dtype[np.float64]], 
    Bc: np.ndarray[tuple[int, int], np.dtype[np.float64]], 
    Ts: float):
    '''
    x -> [u, w, q, x, z, theta, 1]
    u -> [u_tailalt, u_splay, 1]
    '''
    Ad = scipy.linalg.expm(Ac * Ts)
    Bd = np.linalg.inv(Ac) @ (Ad - np.eye(Ad.shape[0])) @ Bc
    
    A_MPC = np.zeros((7, 7))
    
    # Move A matrix to the proper indices in MPC A matrix
    A_MPC[[[0], [1], [2], [5]], [0, 1, 2, 5]] = Ad
    
    # Calculate trim contribution and place it in last column
    A_dx = Ad @ dx[:, np.newaxis]
    A_MPC[[[0], [1], [2], [5]], [6]] = -A_dx

    # Since we integrate x and z in the state vector, we add those 1s to the matrix
    A_MPC[[3, 4], [0, 1]] += 1
    A_MPC[[3, 4, 5], [3, 4, 5]] += 1
    A_MPC[-1, -1] += 1

    B_MPC = np.zeros((7, 3))
    
    # Set up the B matrix
    B_MPC[[[0], [1], [2], [5]], [0, 1]] = Bd
    B_du = Bd @ du[:, np.newaxis]
    B_MPC[[[0], [1], [2], [5]], [2]] = -B_du

    return A_MPC, B_MPC
    