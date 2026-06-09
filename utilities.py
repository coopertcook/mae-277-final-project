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
    # Use scipy's ZOH discretization to avoid inv(Ac) failing when Ac is singular
    # (e.g. theta_dot = q gives a zero eigenvalue in Ac)
    Ad, Bd, _, _, _ = scipy.signal.cont2discrete(
        (Ac, Bc, np.zeros_like(Ac), np.zeros((Ac.shape[0], Bc.shape[1]))),
        Ts, method='zoh'
    )

    A_MPC = np.zeros((7, 7))

    # Place the 4x4 LPV block [u, w, q, theta] into MPC state indices [0,1,2,5]
    A_MPC[[[0], [1], [2], [5]], [0, 1, 2, 5]] = Ad

    # Last column absorbs both trim offsets so the QP only optimizes real controls
    # -(Ad @ dx + Bd @ du) = -(A_trim*x_trim + B_trim*u_trim)
    A_dx = Ad @ dx[:, np.newaxis]
    B_du = Bd @ du[:, np.newaxis]
    A_MPC[[[0], [1], [2], [5]], [6]] = -A_dx - B_du

    # Position integration: x_pos_{k+1} = x_pos_k + u_k*Ts,  z_pos_{k+1} = z_pos_k + w_k*Ts
    # A_MPC[[3, 4], [0, 1]] += Ts
    # Cooper's intuiation: the position integration also depends on theta, so we add that term to A_MPC as well
    theta = dx[3] 
    A_MPC[[3, 4], [0]] += Ts * np.array([np.cos(theta), np.sin(theta)])
    A_MPC[[3, 4], [1]] += Ts * np.array([-np.sin(theta), np.cos(theta)])
    A_MPC[[3, 4], [3, 4]] += 1
    A_MPC[-1, -1] += 1

    # B_MPC is 7x2 — no homogeneous column, trim offset moved into A_MPC last column above
    B_MPC = np.zeros((7, 2))
    B_MPC[[[0], [1], [2], [5]], [0, 1]] = Bd

    return A_MPC, B_MPC
    