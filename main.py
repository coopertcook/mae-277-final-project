import numpy as np
import matplotlib.pyplot as plt

from do_problem import do_problem


# ------------------- BASE PERFORMANCE --------------------
# Set up standard problem with model matching plant
pos_n, pos_p = -0.2, 0.2
con_n, con_p = pos_n, pos_p
t, x = do_problem(
    N = 15, Ts = 0.1,
    pos_n = pos_n, pos_p = pos_p,
    con_n = con_n, con_p = con_p,
    m_plant = 1, k_plant = 1,
    m_model = 1, k_model = 1,
)

# Generate figure
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)

_ = ax.plot(t[:], x[0, :], label="Position")
_ = ax.plot(t[:], x[2, :], label="Input Force")
_ = ax.plot(t[:], x[3, :], linestyle='--', label="Position Command")
_ = ax.hlines([pos_n, pos_p], 0, t.max(), linestyle='-.', color="black", label="Position/Input Constraint")

_ = ax.set_xlabel("Time (t)")
_ = ax.set_title("Tracking MPC On a Mass-Spring System")
_ = ax.legend()

fig.tight_layout()
fig.savefig("standard.png", dpi=300)



# --------------------- ROBUSTNESS -----------------------
# Set up standard problem with model mis-matching plant
pos_n, pos_p = -0.2, 0.2
con_n, con_p = pos_n, pos_p
t, x = do_problem(
    N = 15, Ts = 0.1,
    pos_n = pos_n, pos_p = pos_p,
    con_n = con_n, con_p = con_p,
    m_plant = 0.8, k_plant = 1.2,   # Modified plant
    m_model = 1, k_model = 1,
)

# Generate figure
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)

_ = ax.plot(t[:], x[0, :], label="Position")
_ = ax.plot(t[:], x[2, :], label="Input Force")
_ = ax.plot(t[:], x[3, :], linestyle='--', label="Position Command")
_ = ax.hlines([pos_n, pos_p], 0, t.max(), linestyle='-.', color="black", label="Position/Input Constraint")

_ = ax.set_xlabel("Time (t)")
_ = ax.set_title("Plant-Model Mis-Match")
_ = ax.legend()

fig.tight_layout()
fig.savefig("robustness_original.png", dpi=300)



# Set up standard problem with model mis-matching plant and improved
# control authority.
pos_n, pos_p = -0.2, 0.2
con_n, con_p = -0.25, 0.25          # Modified control authority
t, x = do_problem(
    N = 15, Ts = 0.1,
    pos_n = pos_n, pos_p = pos_p,
    con_n = con_n, con_p = con_p,
    m_plant = 0.8, k_plant = 1.2,   # Modified plant
    m_model = 1, k_model = 1,
)

# Generate figure
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)

_ = ax.plot(t[:], x[0, :], label="Position")
_ = ax.plot(t[:], x[2, :], label="Input Force")
_ = ax.plot(t[:], x[3, :], linestyle='--', label="Position Command")
_ = ax.hlines([pos_n, pos_p], 0, t.max(), linestyle='-.', color="black", label="Position Constraint")
_ = ax.hlines([con_n, con_p], 0, t.max(), linestyle='-.', color="red", label="Input Constraint")

_ = ax.set_xlabel("Time (t)")
_ = ax.set_title("Plant-Model Mis-Match, Improved Control Authority")
_ = ax.legend()

fig.tight_layout()
fig.savefig("robustness_authority.png", dpi=300)



# Set up standard problem with model mis-matching plant
pos_n, pos_p = -0.2, 0.2
con_n, con_p = pos_n, pos_p
t, x = do_problem(
    N = 15, Ts = 0.1,
    pos_n = pos_n, pos_p = pos_p,
    con_n = con_n, con_p = con_p,
    m_plant = 1, k_plant = 1,   # Modified plant
    m_model = 0.8, k_model = 1.2,
)

# Generate figure
fig = plt.figure(figsize=(12,7))
ax = fig.add_subplot(111)

_ = ax.plot(t[:], x[0, :], label="Position")
_ = ax.plot(t[:], x[2, :], label="Input Force")
_ = ax.plot(t[:], x[3, :], linestyle='--', label="Position Command")
_ = ax.hlines([pos_n, pos_p], 0, t.max(), linestyle='-.', color="black", label="Position/Input Constraint")

_ = ax.set_xlabel("Time (t)")
_ = ax.set_title("Plant-Model Mis-Match")
_ = ax.legend()

fig.tight_layout()
fig.savefig("robustness_opposite.png", dpi=300)
