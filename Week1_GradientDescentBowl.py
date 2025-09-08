# ----------------------------------------
# Gradient Descent Animation in 3D
# Cost function J(w,b) surface (the "bowl")
# Red ball = current (w,b) as it updates
# Dataset: (1,2), (2,3)
# ----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Training data
X = [1, 2]
Y = [2, 3]
m = len(X)

# Cost function
def compute_cost(w, b):
    cost = 0
    for i in range(m):
        cost += (w * X[i] + b - Y[i]) ** 2
    return cost / (2 * m)

# Compute cost surface for ranges of w and b
w_vals = np.linspace(-1, 3, 50)
b_vals = np.linspace(-1, 3, 50)
W, B = np.meshgrid(w_vals, b_vals)
J_vals = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J_vals[i, j] = compute_cost(W[i, j], B[i, j])

# Initialize parameters
w = 0.0
b = 0.0
alpha = 0.1
iterations = 20

# Store path of (w,b,J)
path_w = [w]
path_b = [b]
path_J = [compute_cost(w, b)]

# Run gradient descent (precompute path)
for step in range(iterations):
    preds = [w * X[i] + b for i in range(m)]
    errors = [preds[i] - Y[i] for i in range(m)]

    dw = (1/m) * sum(errors[i] * X[i] for i in range(m))
    db = (1/m) * sum(errors)

    w = w - alpha * dw
    b = b - alpha * db

    path_w.append(w)
    path_b.append(b)
    path_J.append(compute_cost(w, b))

# --- Set up 3D figure ---
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')

# Surface
ax.plot_surface(W, B, J_vals, cmap='viridis', alpha=0.6)

# Path line (static, red)
ax.plot(path_w, path_b, path_J, color='red', alpha=0.5)

# Red ball (moving point)
point, = ax.plot([], [], [], 'ro', markersize=8)

# Axis labels
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Cost J(w,b)')
ax.set_title('Gradient Descent Animation on Cost Surface')

# --- Animation function ---
def update(frame):
    point.set_data([path_w[frame]], [path_b[frame]])
    point.set_3d_properties([path_J[frame]])
    return point,

# Animate
ani = FuncAnimation(fig, update, frames=len(path_w), interval=500, blit=True, repeat=False)

plt.show()