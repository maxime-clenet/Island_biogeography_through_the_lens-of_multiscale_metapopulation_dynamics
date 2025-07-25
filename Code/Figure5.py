import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, comb
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def calculate_p0(n, c_in, c_out, e, C_S):
    """
    Computes the probability of an empty island state (p_0) using a combinatorial approach.

    Parameters:
    - n (int): Number of patches on the island
    - c_in (float): Within-island colonization rate
    - c_out (float): Mainland-island colonization rate
    - e (float): Extinction rate
    - C_S (float): Scaling constant for species colonization

    Returns:
    - p0_n (float): Probability of an empty island at equilibrium
    """
    alpha = c_in / (n * e)
    beta = (n * c_out * C_S) / c_in
    
    # Compute summation term using vectorized operations
    k_values = np.arange(n + 1)
    binomial_coefficients = comb(n, k_values)
    gamma_ratios = gamma(k_values + beta) / gamma(beta)
    summation = np.sum(binomial_coefficients * gamma_ratios * (alpha ** k_values))

    return 1 / summation


# Parameters
n_values = np.arange(1, 151)  # Use numpy array for better performance
c_in = 1
c_out_values = np.linspace(0.0001, 0.01, num=10)
e = 0.8
C_S = 1
M = 2000


def arrhenius_law(x, c, z):
    """Arrhenius species-area relationship (power law)."""
    return c * x**z


def gleason_law(x, a, b):
    """Gleason species-area relationship (logarithmic model)."""
    return a + b * np.log(x)


# Create the main figure
fig, ax = plt.subplots(figsize=(10, 6))

# Compute and plot species richness as a function of area for different c_out values
for c_out in c_out_values:
    M_values = M * (1 - np.array([calculate_p0(n, c_in, c_out, e, C_S) for n in n_values]))
    ax.plot(n_values, M_values, label=f'$c_{{out}}={c_out:.7f}$')

ax.set_xlabel('Area (n)', fontsize=25)
ax.set_ylabel(r'Species richness $(S_\infty)$', fontsize=25)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True)

plt.tight_layout()
plt.show()


# Calculate R² values for both Arrhenius and Gleason models
r2_arrhenius_values = []
r2_gleason_values = []

for c_out in c_out_values:
    M_values = M * (1 - np.array([calculate_p0(n, c_in, c_out, e, C_S) for n in n_values]))

    # Fit to Arrhenius model
    popt, _ = curve_fit(arrhenius_law, n_values, M_values)
    fitted_values_arrhenius = arrhenius_law(n_values, *popt)
    r2_arrhenius_values.append(r2_score(M_values, fitted_values_arrhenius))

    # Fit to Gleason model
    popt, _ = curve_fit(gleason_law, n_values, M_values)
    fitted_values_gleason = gleason_law(n_values, *popt)
    r2_gleason_values.append(r2_score(M_values, fitted_values_gleason))

# Plot R² comparison
plt.figure(figsize=(10, 6))
plt.plot(c_out_values, r2_arrhenius_values, marker='o', label='Arrhenius')
plt.plot(c_out_values, r2_gleason_values, marker='x', label='Gleason')
plt.xlabel(r'Mainland-island colonization rate ($c_{out}$)', fontsize=25)
plt.ylabel(r'$R^2$', fontsize=25)
plt.legend()
plt.grid(True)
plt.ylim(0.72, 1.03)
plt.show()


# Compute z-values and confidence intervals
z_values = []
z_errors = []
critical_value = 2.576  # 99% confidence level

for c_out in c_out_values:
    M_values = M * (1 - np.array([calculate_p0(n, c_in, c_out, e, C_S) for n in n_values]))

    # Fit to Arrhenius model
    popt, pcov = curve_fit(arrhenius_law, n_values, M_values)
    
    # Extract z-value and its 99% confidence interval
    z = popt[1]
    z_std_err = np.sqrt(pcov[1, 1])
    z_values.append(z)
    z_errors.append(critical_value * z_std_err)

# Plot z-values with confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(c_out_values, z_values, marker='o', label='Fitted z')

# Create shaded region for confidence intervals
upper_bound = np.array(z_values) + np.array(z_errors)
lower_bound = np.array(z_values) - np.array(z_errors)
plt.fill_between(c_out_values, lower_bound, upper_bound, color='skyblue', alpha=0.3, label='99% Confidence Interval')

plt.xlabel(r'Mainland-island colonization rate ($c_{out}$)', fontsize=25)
plt.ylabel(r'$z$', fontsize=25)
plt.grid(True)
plt.legend()
plt.show()