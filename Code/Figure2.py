import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 50        # Size of the island (total number of patches)
c_in = 0.8    # Within-island colonization rate
e = 1         # Extinction rate
C_S = 1       # Scaling constant for out-group contribution

def compute_lambda(n, c_in, c_out, C_S):
    """
    Computes the vector of transition rates (λ) for species colonization 
    and immigration processes in the system.

    Parameters:
    - n: int, total number of patches (island size)
    - c_in: float, within-island colonization rate
    - c_out: float, mainland-island colonization rate
    - C_S: float, scaling constant for external contribution

    Returns:
    - lambda_values: numpy array of shape (n+1,) with transition rates
    """
    k_values = np.arange(n + 1)  # Possible number of occupied patches
    lambda_values = c_in * k_values / n * (n - k_values) + c_out * C_S * (n - k_values)
    lambda_values[-1] = 0  # No transition from full occupancy
    return lambda_values

def compute_mu(n, e):
    """
    Computes the vector of extinction rates (μ) for the system.

    Parameters:
    - n: int, total number of patches
    - e: float, extinction rate

    Returns:
    - mu_values: numpy array of shape (n+1,) with extinction rates
    """
    return e * np.arange(n + 1)

def compute_probabilities(n, lambda_values, mu_values):
    """
    Computes the steady-state probabilities of species occupancy levels.

    Parameters:
    - n: int, total number of patches
    - lambda_values: numpy array, transition (colonization) rates λ
    - mu_values: numpy array, extinction rates μ

    Returns:
    - probabilities: numpy array, steady-state probabilities for each occupancy level
    """
    gamma_values = lambda_values[:n] / mu_values[1:]  # λ_k / μ_k ratio
    S = np.sum(np.cumprod(np.insert(gamma_values, 0, 1)))  # Compute normalization factor
    
    # Compute probabilities recursively
    probabilities = np.zeros(n + 1)
    probabilities[0] = 1 / (1 + S)  # Base case

    for k in range(1, n + 1):
        probabilities[k] = gamma_values[k - 1] * probabilities[k - 1]
    
    return probabilities

# Values of c_out to test
c_out_values = [0.005, 0.03, 0.1]

# Plotting
plt.figure(figsize=(10, 6))

for c_out in c_out_values:
    # Compute transition rates and steady-state probabilities
    lambda_values = compute_lambda(n, c_in, c_out, C_S)
    mu_values = compute_mu(n, e)
    probabilities = compute_probabilities(n, lambda_values, mu_values)
    
    # Plot the probabilities
    plt.plot(range(n+1), probabilities, markersize=5, linestyle='-', label=f'$c_{{out}} = {c_out}$')

# Enhance the plot
plt.xlabel("Number of occupied patches ($k$)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()



