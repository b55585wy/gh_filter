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
        dx = dx + h * ((z - x_pred) / dt)
        # update step
        x_est = x_pred + g * (z - x_pred)
        results.append(x_est)
    return results

def gen_data(x0, dx, count, noise_factor, accel=0.):
    """ Generate data with acceleration """
    zs = []
    for i in range(count):
        zs.append(x0 + accel * (i**2) / 2 + dx * i + randn() * noise_factor)
        dx += accel
    return zs

def main():
    # Given measurements
    z = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] + [15]*50  # There are 50 '15's in the array

    # Filter parameters
    x0 = 0
    dx = 0
    dt = 1.
    g = 0.4  # Fixed g value for all scenarios

    # Different h and dx values for comparison
    h_dx_values = [(0, 0.05), (2, 0.05), (2, 0.0)]

    # Plotting
    plt.figure(figsize=(15, 7))

    for dx, h in h_dx_values:
        filtered_data = g_h_filter(z, x0, dx, g, h, dt)
        plt.plot(filtered_data, label=f'Filtered (dx={dx}, h={h})')

    plt.scatter(range(len(z)), z, color='black', marker='o', label='Measured Data')
    plt.title('G-H Filter Performance with Different h and dx Values')
    plt.xlabel('Time Step')
    plt.ylabel('Measurement Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
