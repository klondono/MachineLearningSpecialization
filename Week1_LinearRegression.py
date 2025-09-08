# ----------------------------------------
# Gradient Descent Visualization
# - Left: line fitting training data
# - Right: cost function parabola (vs w)
# Dataset: (1,2), (2,3)
# ----------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import time

# Training data
X = [1, 2]
Y = [2, 3]
m = len(X)

# Initialize parameters
w = 0.0
b = 0.0

# Learning rate and iterations
alpha = 0.1
iterations = 20

# --- Define cost function J(w,b) ---
def compute_cost(w, b):
    cost = 0
    for i in range(m):
        cost += (w * X[i] + b - Y[i]) ** 2
    return cost / (2 * m)

# --- Precompute cost values for plotting parabola (fixing b=0 for simplicity) ---
w_vals = np.linspace(-1, 3, 100)
cost_vals = [compute_cost(wi, b=0) for wi in w_vals]

# --- Setup plots ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for step in range(iterations):
    # --- Predictions ---
    preds = [w * X[i] + b for i in range(m)]

    # --- Errors ---
    errors = [preds[i] - Y[i] for i in range(m)]

    # --- Cost ---
    J = compute_cost(w, b)

    # --- Gradients ---
    dw = (1/m) * sum(errors[i] * X[i] for i in range(m))
    db = (1/m) * sum(errors)

    # --- Update parameters ---
    w = w - alpha * dw
    b = b - alpha * db

    # --- Plot 1: Data and Line ---
    ax1.clear()
    ax1.scatter(X, Y, color='red', label='Training data')
    line_x = [0, 3]
    line_y = [w * x + b for x in line_x]
    ax1.plot(line_x, line_y, color='blue', label=f'Line iter {step+1}')
    ax1.set_title(f'Iteration {step+1}\nCost={J:.4f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)

    # --- Plot 2: Cost function parabola ---
    ax2.clear()
    ax2.plot(w_vals, cost_vals, color='green', label='Cost function J(w)')
    ax2.scatter(w, compute_cost(w, 0), color='red', zorder=5, label='Current w')
    ax2.set_title('Cost vs w (b=0)')
    ax2.set_xlabel('w')
    ax2.set_ylabel('J(w)')
    ax2.legend()
    ax2.grid(True)

    # --- Draw and pause ---
    plt.draw()
    plt.pause(0.5)

plt.ioff()
plt.show()