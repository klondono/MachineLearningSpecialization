# ----------------------------------------
# Gradient Descent Full Visualization (4 charts)
# 1. Line fitting noisy data
# 2. Cost shrinking over iterations
# 3. 3D cost surface (the bowl)
# 4. Contour plot (top view of the bowl)
# ----------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Training data (noisy line around y = x + 1)
np.random.seed(0)
X = np.array([1, 2, 3, 4, 5])
Y = X + 1 + np.random.normal(0, 0.2, size=5)
m = len(X)

# Cost function
def compute_cost(w, b):
    cost = 0
    for i in range(m):
        cost += (w * X[i] + b - Y[i]) ** 2
    return cost / (2 * m)

# Gradient descent setup
w, b = 0.0, 0.0
alpha = 0.1
iterations = 20

# Storage
path_w, path_b, path_J = [w], [b], [compute_cost(w, b)]

# Precompute gradient descent path
for step in range(iterations):
    preds = [w * X[i] + b for i in range(m)]
    errors = [preds[i] - Y[i] for i in range(m)]

    dw = (1/m) * sum(errors[i] * X[i] for i in range(m))
    db = (1/m) * sum(errors)

    w = w - alpha * dw
    b = b - alpha * db
    J = compute_cost(w, b)

    path_w.append(w)
    path_b.append(b)
    path_J.append(J)

# --- Cost surface for 3D and contour plots ---
w_vals = np.linspace(-1, 3, 100)
b_vals = np.linspace(-1, 3, 100)
W, B = np.meshgrid(w_vals, b_vals)
J_vals = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J_vals[i, j] = compute_cost(W[i, j], B[i, j])

# --- Setup figure with 4 subplots ---
fig = plt.figure(figsize=(20, 5))

# 1. Line fitting
ax1 = fig.add_subplot(1, 4, 1)
ax1.scatter(X, Y, color='red', label='Training data')
line, = ax1.plot([], [], color='blue', label='Fitted line')
ax1.set_xlim(0, 6)
ax1.set_ylim(min(Y)-1, max(Y)+1)
title1 = ax1.set_title("Line Fitting")
ax1.legend()
ax1.grid(True)

# 2. Cost shrinking
ax2 = fig.add_subplot(1, 4, 2)
ax2.set_xlim(0, iterations)
ax2.set_ylim(0, max(path_J)+0.5)
cost_line, = ax2.plot([], [], color='green', label='Cost J(w,b)')
title2 = ax2.set_title("Cost Shrinking")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Cost J(w,b)")
ax2.legend()
ax2.grid(True)

# 3. 3D bowl
ax3 = fig.add_subplot(1, 4, 3, projection='3d')
ax3.plot_surface(W, B, J_vals, cmap='viridis', alpha=0.6)
ax3.set_xlabel('w')
ax3.set_ylabel('b')
ax3.set_zlabel('Cost J(w,b)')
ax3.set_title("3D Cost Function")
ax3.plot(path_w, path_b, path_J, color='red', alpha=0.5)  # path line
point3d, = ax3.plot([], [], [], 'ro', markersize=8)       # moving ball

# 4. Contour plot (top view)
ax4 = fig.add_subplot(1, 4, 4)
contour = ax4.contour(W, B, J_vals, levels=30, cmap='viridis')
ax4.set_xlabel('w')
ax4.set_ylabel('b')
ax4.set_title("Contour Plot (Top View)")
ax4.set_xlim(-2, 4)   # <-- zoomed out horizontally
ax4.set_ylim(-2, 4)   # <-- zoomed out vertically
ax4.plot(path_w, path_b, color='red', alpha=0.5)          # path line
point2d, = ax4.plot([], [], 'ro', markersize=8)      # moving dot

# --- Animation update function ---
def update(frame):
    # 1. Update fitted line
    x_line = np.linspace(0, 6, 100)
    y_line = path_w[frame] * x_line + path_b[frame]
    line.set_data(x_line, y_line)

    # 2. Update cost curve
    cost_line.set_data(range(frame+1), path_J[:frame+1])

    # 3. Update 3D ball
    point3d.set_data([path_w[frame]], [path_b[frame]])
    point3d.set_3d_properties([path_J[frame]])

    # 4. Update contour dot
    point2d.set_data([path_w[frame]], [path_b[frame]])

    # Update titles dynamically
    title1.set_text(f"Line Fitting (iter {frame}, w={path_w[frame]:.2f}, b={path_b[frame]:.2f})")
    title2.set_text(f"Cost Shrinking (J={path_J[frame]:.4f})")

    return line, cost_line, point3d, point2d, title1, title2

# Animate
ani = FuncAnimation(fig, update, frames=len(path_w), interval=600, blit=False, repeat=False)

plt.tight_layout()
plt.show()