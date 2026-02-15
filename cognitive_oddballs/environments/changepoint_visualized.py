import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Reproducibility
np.random.seed(123)

# Time axis
T = 300
t = np.arange(T)

# ----- Latent state with change-points -----
latent = np.zeros(T)

change_points = [0, 60, 120, 180, 240]
state_levels = np.cumsum(np.random.normal(0, 2.0, size=len(change_points)))

for i in range(len(change_points)):
    start = change_points[i]
    end = change_points[i + 1] if i + 1 < len(change_points) else T
    latent[start:end] = state_levels[i]

# ----- Random-walk volatility -----
log_var = np.zeros(T)
for i in range(1, T):
    log_var[i] = log_var[i - 1] + np.random.normal(0, 0.05)

sigma = np.exp(0.5 * log_var)

# ----- Observations -----
y = latent + np.random.normal(0, sigma)

# Inject a few large deviations so oddballs are visible
oddball_idx = np.random.choice(T, size=15, replace=False)
y[oddball_idx] += np.random.choice([-1, 1], size=15) * 5.0 * sigma[oddball_idx]

# ----- Define normal vs oddball (purely by variance) -----
k = 3.0  # variance threshold
is_oddball = np.abs(y - latent) > k * sigma
is_normal = ~is_oddball

# ----- Plot -----
sns.set(style="whitegrid")

plt.figure(figsize=(12, 6))

# Latent state
plt.plot(t, latent, color="black", linewidth=2, label="Latent state")

# Variance band (± k sigma)
plt.fill_between(
    t, latent - k * sigma, latent + k * sigma, color="black", alpha=0.1, label="±3σ band"
)

# Normal observations (inside variance)
sns.scatterplot(x=t[is_normal], y=y[is_normal], color="blue", s=30, label="Normal observations")

# Oddballs (outside variance)
sns.scatterplot(x=t[is_oddball], y=y[is_oddball], color="red", s=60, label="Oddballs")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Normal vs Oddball Observations Relative to Latent State Variance")
plt.legend()
plt.tight_layout()
plt.show()
