import numpy as np
import matplotlib.pyplot as plt

# Parameters
r_r = 1. # Repulsion radius
r_a = 1. # Alignment radius
r_p = 2. # Predator radius
eps = 0.001 # Value to avoid division by zero
s = 10. # Speed reference for friction force
dt = 0.02

N = 200 # Number of boids
x = np.zeros((2,N)) # Positions
v = np.zeros((2,N)) # Velocities

x_h = np.array([[0],[0]]) # Homing position
x_p = np.array([[0],[0]]) # Predator position

# Forces Weights
K_r = 1.
K_a = 0.1
K_h = 2.
K_f = 20.
K_p = 100.

# ===============
# Initialization
# ===============
rng = np.random.default_rng(seed=123)
x = rng.standard_normal((2,N)) * 10. + 5.
v = rng.random((2,N)) * 2. - 1.

# Adjacency matrices
def adj_matrix(x,R):
    adjacency_mat = np.zeros((N,N), dtype=int)
    for i in range(N):
        for j in range(i+1,N):
            if np.linalg.norm(x[:,i]-x[:,j]) < R:
                adjacency_mat[i,j] = 1
                adjacency_mat[j,i] = 1
    return adjacency_mat
adj_mat_alignment = adj_matrix(x,r_a)
adj_mat_repulsion = adj_matrix(x,r_r)

# Forces
def update_forces(x,v):
    F = np.zeros((2,N))
    for i in range(N):
        for j in range(i+1, N):
            # Repulsion
            if adj_mat_repulsion[i,j] == 1:
                F[:,i] += K_r * (x[:,i] - x[:,j]) / (np.sum((x[:,i]-x[:,j])**2)+eps**2)
            # Alignment
            if adj_mat_alignment[i,j] == 1:
                F[:,i] += K_a * (v[:,j] - v[:,i])
            F[:,j] += -F[:,i]
        # Homing
        F[:,i] += K_h * (x_h[:,0] - x[:,i])
        # Friction
        F[:,i] -= K_f * v[:,i] * (np.linalg.norm(v[:,i]) - s) / s
        # Predator
        if np.linalg.norm(x[:,i] - x_p[:,0]) < r_p:
            F[:,i] += K_p * (x[:,i] - x_p[:,0]) / (np.sum((x[:,i] - x_p[:,0])**2)+eps**2)
    return F

            

# ===============
# Visualization
# ===============
fig, ax = plt.subplots(1,1,layout='tight')
ax.scatter(x[0,:], x[1,:])
plt.show()