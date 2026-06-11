import scipy
import numpy as np


def convert_GIFT_AB_to_QP_AB(
    x_trim: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    u_trim: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    Ac: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    Bc: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    Ts: float):
    '''
    x -> [u, w, q, x, z, theta, 1]
    u -> [u_tailalt, u_splay]
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
    A_trim = Ad @ x_trim[:, np.newaxis]
    B_trim = Bd @ u_trim[:, np.newaxis]
    A_trim[[0, 1, 2], 0] -= x_trim[[0, 1, 2]]
    A_MPC[[[0], [1], [2], [5]], [-1]] = - A_trim - B_trim
    # print(Ad)
    # print(Bd)
    # print(A_trim)
    # print(B_trim)

    # print(Ad @ np.array([[14,  2, 0, 0]]).T)
    # print(Bd @ (np.array([[12, 12]]).T * np.pi / 180))

    # Position integration: identity terms only.
    # Velocity coupling (rows 3,4 × cols 0,1) is applied by the caller with the correct theta.
    A_MPC[[3, 4], [3, 4]] = 1
    A_MPC[-1, -1] += 1

    # B_MPC is 7x2 — no homogeneous column, trim offset moved into A_MPC last column above
    B_MPC = np.zeros((7, 2))
    B_MPC[[[0], [1], [2], [5]], [0, 1]] = Bd

    return A_MPC, B_MPC
    

import os
import sys

# Define a context manager to redirect stdout and stderr temporarily
class SuppressOutput:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self.stdout
        sys.stderr = self.stderr