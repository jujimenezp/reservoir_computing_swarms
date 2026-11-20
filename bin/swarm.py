import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# Parameters
r_r = 1. # Repulsion radius
r_a = 1. # Alignment radius
r_p = 2. # Predator radius
s = 10. # Speed reference for friction force
dt = 0.02
tf = 50
trail_len = 5

N = 200 # Number of boids

x_h = np.zeros((2,1)) # Homing position

# Forces Weights
K_r = 2.
K_a = 0.01
K_h = 2.
K_f = 20.
K_p = 100.

# Force limitations parameters
alpha = 200
B = 0.1

# Lorenz system parameters
sigma = 10.
rho = 28.
beta = 8./3.

# Adjacency matrices
def adj_matrix(x,R):
    adjacency_mat = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(i+1,N):
            if np.linalg.norm(x[:,i]-x[:,j]) < R:
                adjacency_mat[i,j] = 1
                adjacency_mat[j,i] = 1
    return adjacency_mat
x = np.zeros((2,N))
adj_mat_alignment = adj_matrix(x,r_a)
adj_mat_repulsion = adj_matrix(x,r_r)

# Forces
def update_forces(x,v):
    F = np.zeros((2,N))
    for i in range(N):
        for j in range(i+1, N):
            # Repulsion
            if adj_mat_repulsion[i,j] == 1 and np.linalg.norm(x[:,i] - x[:,j]) > 1e-12:
                F[:,i] += K_r * (x[:,i] - x[:,j]) / (np.sum((x[:,i]-x[:,j])**2))
            # Alignment
            if adj_mat_alignment[i,j] == 1:
                F[:,i] += K_a * (v[:,j] - v[:,i])
            F[:,j] += -F[:,i]
        # Homing
        F[:,i] += K_h * (x_h[:,0] - x[:,i])
        # Friction
        speed = np.linalg.norm(v[:,i])
        if speed > 1e-12:
            F[:,i] -= K_f * (v[:,i]/speed) * (np.linalg.norm(v[:,i]) - s) / s
        # Predator
        distance_predator = np.linalg.norm(x[:,i] - x_p[:2])
        if distance_predator < r_p and distance_predator > 1e-12:
            F[:,i] += K_p * (x[:,i] - x_p[:2]) / (distance_predator**2)
    return alpha * np.tanh(B*F)

# ----------------------
# Predator trajectory
# ----------------------
def lorenz_system_step(x_p,dt):
    x,y,z = x_p
    x_new = x_p[0] + dt * (sigma*(y-x))
    y_new = x_p[1] +  dt * (x*(rho - z) - y)
    z_new = x_p[2] + dt * (x*y - beta*z)
    return [x_new, y_new,z_new]

# ===============
# Initialization
# ===============
rng = np.random.default_rng(seed=123)

x = rng.random((2,N))*10 - 5
v = np.zeros((2,N))
x_p = np.array([8,8,27]).reshape(3,1)

print("Rescaling Lorenz system trajectory")

x_p_traj = [x_p[:,0]]
for t in np.arange(0,tf,dt):
    x_p_traj.append(lorenz_system_step(x_p_traj[-1],dt))

x_p_traj = np.array(x_p_traj)
x_p_traj = 2* (x_p_traj-np.average(x_p_traj, axis=0)) / np.std(x_p_traj,axis=0)
x_p = x_p_traj[0]

adj_mat_alignment = adj_matrix(x,r_a)
adj_mat_repulsion = adj_matrix(x,r_r)
F = update_forces(x,v)

# ===============
# Visualization
# ===============
trails = np.zeros((trail_len,N,2))
trails[0] = x.T
fig, ax = plt.subplots(1,1,layout='tight')
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)
boids_plot = ax.scatter(x[0,:], x[1,:], color = 'black', zorder = 1, s=10)
predator_plot = ax.scatter(x_p[0],x_p[1], color='red', zorder=2, s=10)
t_plot = ax.text(0.88,0.94,f't={0}', transform=ax.transAxes, fontsize = 10, bbox = dict(boxstyle="round", facecolor="white"))
#plt.ion()

# GIF
writer = FFMpegWriter(fps=15)

# ==============
# Main loop
# ==============
i=0
print("Main loop starting")
with writer.saving(fig, "results/swarm.mp4", dpi=150):
    for t in np.arange(0,tf,dt):
        # Update predator position
        x_p = x_p_traj[i]
        i += 1

        # Update forces
        adj_mat_alignment = adj_matrix(x,r_a)
        adj_mat_repulsion = adj_matrix(x,r_r)
        F = update_forces(x,v)

        # Update boids positions and velocities
        v += dt*F
        x += dt*v

        # Update plot
        t_plot.set_text(f't={t:.0f}')
        boids_plot.set_offsets(x.T)
        predator_plot.set_offsets(x_p[:2])
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.02)

        #GIF
        writer.grab_frame()


print("Loop completed :D")
plt.close(fig)