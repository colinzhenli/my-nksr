import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data from the table
solver_times = np.array([0.19, 0.41, 62, 1.16, 1.11, 0.25, 0.93])  # in seconds
growed_voxels = np.array([23, 104, 176, 334, 354, 45, 266]) * 1e3  # in thousands

# Model function
def model(N, a, b):
    return a * N + b * N**(3/2)

# Fit the model to the data
popt, pcov = curve_fit(model, growed_voxels, solver_times)

# Extract fitted parameters
a, b = popt

# Print the results
print(f"Fitted parameters: a = {a}, b = {b}")

# Plot the data and the fitted curve
plt.scatter(growed_voxels, solver_times, label='Data', color='red')
plt.plot(growed_voxels, model(growed_voxels, *popt), label='Fitted model', color='blue')
plt.xlabel('Growed Voxels (N)')
plt.ylabel('Solver Time (t)')
plt.legend()
plt.show()
