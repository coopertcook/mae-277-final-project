import scipy
import numpy as np


def form_LPVQP_matrices(
    A: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],   # System A matrices (N x m x m)
    B: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],   # System B matrices (N x m x n)
    Q: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],   # State cost (N x m x m)
    R: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],   # Control cost (N x n x n)
    P: np.ndarray[tuple[int, int], np.dtype[np.float64]],   # Terminal cost (m x m)
    Aug_x: np.ndarray[tuple[int, int], np.dtype[np.float64]],   # State constraints in augmented matrix form (p x m+1)
    Aug_u: np.ndarray[tuple[int, int], np.dtype[np.float64]],   # Control constraints in augmented matrix form (q x n+1)
    ): # Horizon is implied from length of A/B/Q/R/P
    '''
    Calculate the quadratic program matrices for a linear time-varying problem with constraints.
    '''
    N = A.shape[0]
    m = A.shape[1]
    n = B.shape[2]

    # --------- QP Matrices ----------------------------------------
    Sx = np.vstack([np.eye(m, dtype=np.float64) for _ in range(N + 1)])
    Su = np.zeros(((N + 1) * m, N * n), dtype=np.float64)

    for i in range(0, N):
        # Set the ith A block of the stacked state influence vector
        # The next block is the matrix product of the previous matrix and the next A matrix
        Sx[(i * m + m):(i * m + 2 * m)] = Sx[(i * m):(i * m + m), :] @ A[i, :, :]

        # Set the element in the ith block of the block diagonal of Su
        # This element is just B_i
        Su[(i * m + m):(i * m + 2 * m), (i * n):(i * n + n)] = B[i, :, :]
        
        for j in range(i + 1, N):
            # Set the ith diagonal of the stacked control influence matrix
            Su[(j * m + m):(j * m + 2 * m), (i * n):(i * n + n)] = A[j, :, :] @ Su[((j-1) * m + m):((j-1) * m + 2 * m), (i * n):(i * n + n)]

    Q_bar = scipy.linalg.block_diag(*[Q[i, :, :] for i in range(N)], P)

    R_bar = scipy.linalg.block_diag(*[R[i, :, :] for i in range (N)])
    
    H = Su.T @ (Q_bar @ Su) + R_bar
    F = Sx.T @ (Q_bar @ Su)
    L = F.T

    # --------- Constraints ------------------------------------------
    Ax = Aug_x[:, :-1]
    bx = Aug_x[:, [-1]]
    Gx = np.zeros((Ax.shape[0]*N, B.shape[2]*N))
    Ex = np.zeros((Ax.shape[0]*N, A.shape[2]))

    for i in range(N):
        Gx[i*Ax.shape[0]:(i+1)*Ax.shape[0], :] =  Ax @ Su[(i+1)*m:(i+2)*m, :]
        Ex[i*Ax.shape[0]:(i+1)*Ax.shape[0], :] = -Ax @ Sx[(i+1)*m:(i+2)*m, :]
    wx = np.tile(bx, (N, 1))

    Au = Aug_u[:, :-1]
    bu = Aug_u[:, [-1]]
    Gu = scipy.linalg.block_diag(*[Au]*N)
    wu = np.tile(bu, (N, 1))

    G = np.vstack((Gu, Gx))
    W = np.vstack((wu, wx))
    T = np.vstack((np.zeros((Gu.shape[0], m)), Ex))

    IMPC = np.zeros((n, n*N))
    np.fill_diagonal(IMPC[:n, :n], 1)

    return H, L, G, W, T, IMPC, Sx, Su

if __name__ == "__main__":

    # Initial system A matrix
    A = np.array([
        [1, 1],
        [0, 1],
    ], dtype=np.float64)

    # Initial system B matrix
    B = np.array([
        [0],
        [1],
    ], dtype=np.float64)

    # Initial state cost
    Q = np.eye(2)

    # Initial control cost
    R = np.array([[0.1]])

    # Initial terminal cost
    P = np.eye(2)*1

    # State constraints augmented matrix
    # Aug_x = np.zeros((A.shape[0], A.shape[0] + 1))
    Aug_x = np.hstack((Q, np.ones((2, 1))))

    # Control constraints augmented matrix
    Aug_u = np.zeros((B.shape[0], B.shape[1] + 1))

    # Steps to horizon
    N = 6

    # A and B matrices for each step
    As = np.stack([A for _ in range(N)])
    Bs = np.stack([B for _ in range(N)])
    Qs = np.stack([Q]*N)
    Rs = np.stack([R]*N)

    result = form_LPVQP_matrices(As, Bs, Qs, Rs, P, Aug_x, Aug_u)
    print(result)