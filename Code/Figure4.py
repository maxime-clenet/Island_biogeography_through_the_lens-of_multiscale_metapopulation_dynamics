import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logser
from scipy.optimize import curve_fit

# Function to generate species abundance distribution
def reg_colo_rate(M, theta):
    r = logser.rvs(theta, size=M)
    return r / sum(r) * M

# Function to calculate probability of an unoccupied site
def calculate_p0(n, c_in, c_out, e, C_S):
    beta = (n * c_out * C_S) / c_in
    return (1 - c_in / e) ** beta

# Arrhenius species-area relationship function
def arrhenius_law(x, c, z):
    return c * x**z

# Define parameters
n_values = range(1, 301)
c_in = 0.8
c_out_values = np.linspace(0.0001, 0.01, num=10)
e = 1
C_S = 1
M = 2000
theta_values = [0.00001, 0.9]  # Values of theta to compare

# Initialize storage for z values and confidence intervals
z_values_dict = {}
z_errors_dict = {}
critical_value = 2.576  # 99% confidence interval

# Loop over theta values
for theta in theta_values:
    distrib = reg_colo_rate(M, theta)  # Generate species abundance distribution
    z_values = []
    z_errors = []

    for c_out in c_out_values:
        S = []
        for n in n_values:
            S_value = sum(1 - calculate_p0(n, c_in, c_out, e, s) for s in distrib)
            S.append(S_value)
        
        # Fit to Arrhenius law
        popt, pcov = curve_fit(arrhenius_law, n_values, S)
        
        # Extract z value and its 99% confidence interval
        z = popt[1]
        z_std_err = np.sqrt(pcov[1, 1])
        z_values.append(z)
        z_errors.append(critical_value * z_std_err)

    # Store results for this theta
    z_values_dict[theta] = z_values
    z_errors_dict[theta] = z_errors

# Plot z as a function of c_out for both theta values with confidence intervals
plt.figure(figsize=(10, 6))
colors = ['blue', 'red']
labels = ['Uniform', 'Heterogeneous']
for i, theta in enumerate(theta_values):
    plt.plot(c_out_values, z_values_dict[theta], marker='o', color=colors[i], label=rf'{labels[i]}')
    
    # Create shaded confidence interval
    upper_bound = np.array(z_values_dict[theta]) + np.array(z_errors_dict[theta])
    lower_bound = np.array(z_values_dict[theta]) - np.array(z_errors_dict[theta])
    plt.fill_between(c_out_values, lower_bound, upper_bound, color=colors[i], alpha=0.3)

# Labels and title
plt.xlabel(r'Mainland-island colonization rate ($c_{out}$)', fontsize=15)
plt.ylabel(r'$z$', fontsize=15)
# plt.title(r'Heterogeneous case: Comparison of $	heta = 0.001$ and $	heta = 0.9$', fontsize=15)
plt.grid(True)
plt.legend()
plt.show()