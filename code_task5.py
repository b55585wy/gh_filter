import numpy as np
from matplotlib import pyplot as plt
from numpy.random import randn

def g_h_filter(data, x0, dx, g, h, dt):
    """ G-H Filter implementation """
    x_est = x0
    results = []
    for z in data:
        # prediction step
        x_pred = x_est + dx * dt
        dx = dx

        # update step
        residual = z - x_pred
        dx += h * (residual / dt)
        x_est = x_pred + g * residual
        results.append(x_est)
    return results

def gen_data(x0, dx, count, noise_factor, accel=0.):
    """ Generate data with acceleration """
    zs = []
    for i in range(count):
        zs.append(x0 + accel * (i**2) / 2 + dx * i + randn() * noise_factor)
        dx += accel
    return zs

# Given measurements
z = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + [15]*50  # There are 50 '15's in the array

# Filter parameters
x0 = 0
dx = 0
dt = 1.
h = 0.02

# Different g values
g_values = [0.1, 0.4, 0.8]

# Plotting
plt.figure(figsize=(15, 7))

for g in g_values:
    filtered_data = g_h_filter(z, x0, dx, g, h, dt)
    plt.plot(filtered_data, label=f'Filtered (g={g})')

plt.scatter(range(len(z)), z, color='black', marker='o', label='Measured Data')
plt.title('G-H Filter Performance with Different g Values')
plt.xlabel('Time Step')
plt.ylabel('Measurement Value')
plt.legend()
plt.grid(True)
plt.show()
