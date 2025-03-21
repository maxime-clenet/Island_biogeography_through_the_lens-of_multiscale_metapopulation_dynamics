import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import logser
from scipy.special import gamma, comb
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Function to calculate p0
def calculate_p0(n, c_in, c_out, e, C_S):
    beta = (n * c_out * C_S) / c_in
    result = (1 - c_in / e) ** beta
    return result

# Function to generate log-series distribution
def reg_colo_rate(M, theta):
    r = logser.rvs(theta, size=M)
    return r / sum(r) * M

# Power-law (Arrhenius) and logarithmic (Gleason) fitting functions
def arrhenius_law(x, c, z):
    return c * x**z

def gleason_law(x, a, b):
    return a + b * np.log(x)

# Parameters
n_values = range(1, 301)
c_in = 0.8
e = 1
M = 2000
C_S = 1
c_out_values = np.linspace(0.0001, 0.01, num=10)

# --- Homogeneous Case ---
plt.figure(figsize=(10, 6))

for c_out in c_out_values:
    S = [M * (1 - calculate_p0(n, c_in, c_out, e, 1)) for n in n_values]
    plt.plot(n_values, S, label=rf'$c_{{out}} = {c_out}$')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Area (n)', fontsize=15)
plt.ylabel(r'Species richness $(S_\infty)$', fontsize=15)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- R² Comparison (Homogeneous Case) ---
plt.figure(figsize=(10, 6))
r2_arrhenius_values = []
r2_gleason_values = []

for c_out in c_out_values:
    S = [M * (1 - calculate_p0(n, c_in, c_out, e, 1)) for n in n_values]

    # Fit to Arrhenius law
    popt, _ = curve_fit(arrhenius_law, n_values, S)
    r2_arrhenius_values.append(r2_score(S, arrhenius_law(np.array(n_values), *popt)))

    # Fit to Gleason law
    popt, _ = curve_fit(gleason_law, n_values, S)
    r2_gleason_values.append(r2_score(S, gleason_law(np.array(n_values), *popt)))

plt.plot(c_out_values, r2_arrhenius_values, marker='o', label='Arrhenius')
plt.plot(c_out_values, r2_gleason_values, marker='x', label='Gleason')
plt.xlabel(r'Mainland-island colonization rate ($c_{out}$)', fontsize=15)
plt.ylabel(r'$R^2$', fontsize=15)
plt.legend()
plt.grid(True)
plt.ylim(0.72, 1.03)
plt.tight_layout()
plt.show()

# --- Heterogeneous Case ---
theta = 0.9
distrib = reg_colo_rate(M, theta)

fig, ax = plt.subplots(figsize=(10, 6))

for c_out in c_out_values:
    S = [sum(1 - calculate_p0(n, c_in, c_out, e, s) for s in distrib) for n in n_values]
    ax.plot(n_values, S, label=f'$c_{{out}}={c_out:.7f}$')

ax.set_xlabel('Area (n)', fontsize=15)
ax.set_ylabel(r'Species richness $(S_\infty)$', fontsize=15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True)

ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower right")
ax_inset.hist(distrib, bins=30, density=True, alpha=0.75, color='skyblue', edgecolor='black')
ax_inset.set_ylabel('Density', fontsize=10)
ax_inset.set_title(f'Log-Series Distribution', fontsize=10)

plt.tight_layout()
plt.show()

# --- R² Comparison (Heterogeneous Case) ---
plt.figure(figsize=(10, 6))
r2_arrhenius_values = []
r2_gleason_values = []

for c_out in c_out_values:
    S = [sum(1 - calculate_p0(n, c_in, c_out, e, s) for s in distrib) for n in n_values]

    # Fit to Arrhenius law
    popt, _ = curve_fit(arrhenius_law, n_values, S)
    r2_arrhenius_values.append(r2_score(S, arrhenius_law(np.array(n_values), *popt)))

    # Fit to Gleason law
    popt, _ = curve_fit(gleason_law, n_values, S)
    r2_gleason_values.append(r2_score(S, gleason_law(np.array(n_values), *popt)))

plt.plot(c_out_values, r2_arrhenius_values, marker='o', label='Arrhenius')
plt.plot(c_out_values, r2_gleason_values, marker='x', label='Gleason')
plt.xlabel(r'Mainland-island colonization rate ($c_{out}$)', fontsize=15)
plt.ylabel(r'$R^2$', fontsize=15)
plt.legend()
plt.grid(True)
plt.ylim(0.72, 1.03)
plt.show()