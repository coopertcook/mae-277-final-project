import scipy
import control
import casadi as ca
import numpy as np

from form_LPVQP_matrices import form_LPVQP_matrices


def x_dot(
    t: int, 
    x: np.ndarray[tuple[int], np.dtype[np.float64]], 
    u: np.ndarray[tuple[int], np.dtype[np.float64]], 
    m: float, 
    k: float
    ):
    '''
    Returns the state derivative of the non-linear system, given the 
    current state and control vectors and the mass and spring constants.
    '''
    A = np.array([
        [0, 1],
        [-k/m, 0],
    ], dtype=np.float64)
    
    B = np.array([
        [0],
        [1/m]
    ], dtype=np.float64)

    x_dot_x = np.einsum('ij, j -> i', A, x)
    x_dot_u = np.einsum('ij, j -> i', B, u)

    return x_dot_x + x_dot_u


def do_problem(
    N: int,         # Horizon time steps
    Ts: float,      # Time step size
    
    pos_n: float,   # Lower position constraint
    pos_p: float,   # Upper position constraint
    
    con_n: float,   # Lower control constraint
    con_p: float,   # Upper control constraint
    
    m_plant: float, # Plant mass
    k_plant: float, # Plant stiffness

    m_model: float, # Model mass
    k_model: float, # Model stiffness
    ):
    '''
    Solves and simulates the constrained tracking MPC problem of a mass-
    spring system using quadratic programming.
    '''
    # Continuous-time state space representation
    Ac = np.array([
        [0, 1],
        [-k_model/m_model, 0],
    ], dtype=np.float64)
    Bc = np.array([
        [0],
        [1/m_model]
    ], dtype=np.float64)

    # Convert continuous-time to discrete-time.
    Ad = scipy.linalg.expm(Ac * Ts)
    Bd = np.linalg.inv(Ac) @ (Ad - np.eye(2)) @ Bc


    # ----------------- Construct Augmented Model -----------------------
    # The augmented A matrix includes the relationships between the error
    # term `ek`, the control term `uk`, and the position component, `x1`.
    # Since `ek = x1,k - rk`, and `rk` does not change between steps of the
    # QP, the change in `ek` is attributed only to the change in `x1`. So,
    # `ek+1 = ek + dx1,k`. Then `x1,k+1 = x1,k + dx1,k`.
    A = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ], dtype=np.float64)
    A[:2, :2] = Ad

    # The augmented B matrix includes the relationships between the control
    # input change and the state vector. Because in this formulation, only 
    # the control term itself is affected by the change in control, the B
    # matrix only shows a relationship in its fourth component.
    B = np.array([
        [0],
        [0],
        [0],
        [1],
        [0],
    ], dtype=np.float64)
    B[:2, [0]] = Bd

    # Create state and control cost relationships
    Q = np.diag([0, 0, 1, 0, 0])
    R = np.ones((1, 1))

    # Get the solution to the Ricatti equation for a discrete-time LQR and
    # form the terminal cost matrix.
    _, Pdxe, _ = control.dlqr(A[:3, :3], B[:3, [0]], Q[:3, :3], R)
    P = np.zeros_like(A, dtype=np.float64)
    P[:3, :3] = Pdxe


    # ------------------- Setup Constraint Matrices --------------------
    # Set up the constraints in matrix format. We use a large number, lN, as
    # an approximation for infinity. This number is large compared to what 
    # we know are reasonable bounds of the state and control vectors. For
    # conciseness, I've opted to represent them as augmented matrices, where
    # the last column is the right hand side of the `A*x <= b` constraint
    # system of linear inequalities.
    lN = 10
    Aug_x = np.array([
        [-1, 0, 0, 0, 0,   lN],
        [ 1, 0, 0, 0, 0,   lN],
        [ 0,-1, 0, 0, 0,   lN],
        [ 0, 1, 0, 0, 0,   lN],
        [ 0, 0,-1, 0, 0,   lN],
        [ 0, 0, 1, 0, 0,   lN],
        [ 0, 0, 0,-1, 0, -con_n],
        [ 0, 0, 0, 1, 0,  con_p],
        [ 0, 0, 0, 0,-1, -pos_n],
        [ 0, 0, 0, 0, 1,  pos_p],
    ], dtype=np.float64)
    Aug_u = np.array([
        [-1,   lN],
        [ 1,   lN],
    ], dtype=np.float64)


    # -------------------- Formulate QP Problem -------------------------
    # To solve this problem, we will use a quadratic program, so we need to
    # calculate the quadratic program matrices.
    H, L, G, W, T, IMPC = form_LPVQP_matrices(A, B, Q, R, P, Aug_x, Aug_u, N)

    # Generate sequence of position commands. These will be set every N steps.
    np.random.seed(3)
    r = (np.random.random((9)) - 0.5)*0.19
    r = np.insert(r, 4, -0.25)
    num_steps_per_rk = int(10 / Ts)

    # Create a matrix to store the system states
    t = np.zeros((num_steps_per_rk * len(r) + 1), dtype=np.float64)
    X = np.zeros((4, t.shape[0]), dtype=np.float64)
    # In this matrix, I will store the x1, x2, and u histories

    # Set up initial condition (this is partially redundant right now)
    x0_qp = np.array([0, 0, 0, 0, 0], dtype=np.float64) # Initial condition


    # -------------------- Setup the CasADi QP --------------------
    h = ca.DM(H)
    a = ca.DM(G)

    # Create the quadratic program problem using CasADi's `conic` class.
    qp = {}
    qp['h'] = h.sparsity() # Passes only the sparsity patterns
    qp['a'] = a.sparsity() # Passes only the sparsity patterns
    S = ca.conic('S', 'qpoases', qp, {'printLevel': 'none'})

    # Iterate through all the desired time steps, solving the QP each time.
    t_step = 0
    exceptions: list[str] = []
    for t_step in range(num_steps_per_rk * len(r)):
        try:
            # Determine the control input for this time step
            rk = r[t_step // num_steps_per_rk]

            # Update the initial condition with the last computed state
            x0_qp[:2] = X[:2, t_step] - X[:2, t_step - 1]   # delta x1k and x2k
            x0_qp[3] = X[2, t_step]     # uk
            x0_qp[4] = X[0, t_step] # x1k
            x0_qp[2] = x0_qp[4] - rk    # ek

            # Calculate the current linear term vector and constraint right-hand side matrix.
            g = np.reshape(np.einsum('ij, j -> i', L, x0_qp), (L.shape[0], 1))
            uba = (W + np.einsum('ij, j -> i', T, x0_qp)[:, np.newaxis])

            # Solve the QP
            sol = S(h=h, g=g, a=a, uba=uba)
            U_opt = np.array(sol['x'])

            # Simulate QP optimal control vector
            # Update current initial state for the non-linear simulation for this time step.
            x0_sim = X[:, t_step]
            duk = IMPC @ U_opt
            u = X[2, t_step] + duk

            # Get the state derivative callback function
            func = lambda t, y: x_dot(t, y, u[0, [0]], m_plant, k_plant)
            sol = scipy.integrate.solve_ivp(func, [0, Ts], x0_sim[:2], t_eval=np.array([Ts]))

            # Save time and state variables to history
            t[t_step+1] = (t_step+1)*Ts
            X[:2, t_step+1] = sol.y[:2, 0]
            X[2, t_step+1] = u[0, 0]
            X[3, t_step+1] = rk

        # Sometimes the optimizer fails to find a feasible solution, so here
        # I catch those errors, repeat the same step, and report the errors
        # at the end.
        except RuntimeError as e:
            t[t_step+1] = (t_step+1)*Ts
            X[:2, t_step+1] = X[:2, t_step]
            X[2, t_step+1] = X[2, t_step]
            X[3, t_step+1] = X[3, t_step]
            exceptions.append(f"Exception raised at {x0_qp}: {e}")

    # print(exceptions)
    print("Number of exceptions: ", len(exceptions))

    return t, X
