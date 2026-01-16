from matplotlib import pyplot as plt
import numpy as np
import warnings

# At the top of your file, enable all numpy warnings
np.seterr(all="raise")  # or 'raise' to stop execution
warnings.filterwarnings("error")  # Convert warnings to exceptions


class HGF:
    """Hierarchical Gaussian Filter, according to Mathys et al. (2011, 2014)"""

    def __init__(self, kappa, omega, theta):
        # Store parameters
        self.kappa = kappa
        self.omega = omega
        self.theta = theta

        # Initialize states - see Simulation on page 11
        self.m2 = 0.0  # $\mu_{2}^{(n)}$, n = 0, ..., k - 1
        self.m3 = 0.0  # $\mu_{3}^{(n)}$, n = 0, ..., k - 1
        self.s2 = 1.0  # $\sigma_{2}^{(n)}$, n = 0, ..., k - 1
        self.s3 = 1.0  # $\sigma_{3}^{(n)}$, n = 0, ..., k - 1

        self.m1_hat_prev = 0

        self.counter = 0

        self.history = {"m1": [], "m2": [], "m3": [], "s2": [], "s3": [], "u": []}

    def _predictions(self) -> tuple:
        """Using the intenral model from (k-1) to make prediction about (k) before observing the next sensory value (u^{(k)})."""
        # Eq. 24:$\hat{m})_{1}^{(k)} = sigmoid(\mu_{2}^{(k-1)})$
        # The output of the sigmoid is clipped to [1e-10, 1-1e-10] to prevent numerical issues in 1 / s1_hat!
        m1_hat = sigmoid(self.m2)
          # Expected posterior parameter mu (believe about the future state of the environment)

        self.m1_hat_prev = m1_hat

        # Eq 26: $\hat{\sigma}_{1}^{(k-1)} = \hat{\mu}_{1}^{(k-1)} * (1 - \hat{\mu}_{1}^{(k-1)})$
        s1_hat = (
            self.m1_hat_prev * (1 - self.m1_hat_prev)
        )  # Expected posterior parameter sigma (uncertainty about the future state of the environment)

        # Division by zero undetected?!
        pi1_hat = 1 / s1_hat
        # prevent underflow, small to big number

        # Eq 27: $\sigma_{2}^{(k-1)} + e^{\kappa * \mu_{3}^{(k-1)} + \omega}$ -> no $\sigma_{3}^{(k-1)}$ present!
        s2_hat = self.s2 + np.exp(self.kappa * self.m3 + self.omega)
        pi2_hat = 1 / s2_hat  # prevent underflow, small to big number

        # Eq 31: $\hat{\pi}_{3}^{(k)} = 1 / (\sigma_{3}^{(k-1)} + \theta)$
        pi3_hat = 1 / (self.s3 + self.theta)

        return pi1_hat, pi2_hat, pi3_hat, m1_hat

    def update(self, u: int):
        """Update beliefs given binary input u (0 or 1) for point in time k."""

        # STEP 1: Make predictions (using previous trial's values)
        pi1_hat, pi2_hat, pi3_hat, m1_hat = self._predictions()

        # STEP 2: Update LEVEL 1
        ## Eq. 21: $\mu_{1}^{(k)}$ = u^{k}
        self.m1 = float(u)

        # STEP 3: Compute prediction error at Level 1
        ## Eq. 25: $\delta_{1}^{(k)} = \mu_{1}^{(k)} - \hat{\mu}_{1}^{(k)}$
        d1 = self.m1 - m1_hat

        # STEP 4: Update Level 2 (m2 and s2)
        ## Eq. 22: $\frac{1}{\frac{1}{\hat{\sigma}_2^{(k)}} + \sigma_{1}^{(k)}}$
        s2_prev = self.s2
        # self.s2 = 1 / (1 / (s2_hat) + s1_hat) # Values get too little underflow!

        pi2 = pi2_hat + (1 / pi1_hat)
        self.s2 = 1 / pi2

        ## Eq. 23: $\mu_{2}^{(k-1)} + \sigma_{2}^{(k)} * \delta_{1}^{(k)}$
        m2_prev = self.m2
        self.m2 = self.m2 + self.s2 * d1  # m2 becomes weird after

        # STEP 5: Compute prediction error at Level 2 and helper equations
        ## Eq. 32: $w_2^{(k)} = \frac{e^{\kappa m_3^{(k-1)} + \omega}}{e^{\kappa m_3^{(k-1)} + \omega} + s_2^{(k-1)}}$
        w2 = np.exp(self.kappa * self.m3 + self.omega) / (
            np.exp(self.kappa * self.m3 + self.omega) + s2_prev
        )

        ## Eq. 33: r_2^{(k)} = \frac{e^{\kappa m_3^{(k-1)} + \omega} - s_2^{(k-1)}}{e^{\kappa m_3^{(k-1)} + \omega} + s_2^{(k-1)}}
        r2 = (np.exp(self.kappa * self.m3 + self.omega) - s2_prev) / (
            np.exp(self.kappa * self.m3 + self.omega) + s2_prev
        )

        ## Eq. 34: $\delta_{2}^{(k)} = \frac{\sigma_{2}^{(k)} + (\mu_{2}^{(k)} - \mu_{2}^{(k-1)})^2}{\sigma_{2}^{(k-1)} + e^{\kappa * \mu_{3}^{(k-1)} + \omega}} - 1$
        d2 = (self.s2 + (self.m2 - m2_prev) ** 2) / (
            s2_prev + np.exp(self.kappa * self.m3 + self.omega)
        ) - 1

        # STEP 6: Update Level 3 (m3 and s3)
        ## Eq. 29: $\pi_{3} = \hat{\pi}_{3}^{(k)} + \frac{\kappa^2}{2} * w_{2}^{(k)} * (w_{2}^{(k)} + r_{2}^{(k)} * \delta_{2}^{(k)})$
        pi3 = pi3_hat + (self.kappa**2 / 2) * w2 * (w2 + r2 * d2)

        self.s3 = 1 / pi3

        ## Eq. 30: $\mu_{3}^{(k)} = \mu_{3}^{(k-1)} + \sigma_{3}^{(k)} * \frac{\kappa}{2} * w_{2}^{(k)} * \delta_{2}^{(k)}$
        self.m3 = self.m3 + self.s3 * (self.kappa / 2) * w2 * d2

        # Store history
        self.history["u"].append(u)
        self.history["m1"].append(self.m1)
        self.history["m2"].append(self.m2)
        self.history["m3"].append(self.m3)
        self.history["s2"].append(self.s2)
        self.history["s3"].append(self.s3)

        self.counter += 1

    def run_simulation(self, inputs):
        for u in inputs:
            self.update(u)
        return self.history

    def plot_results(self, true_prob=None):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        trials = np.arange(len(self.history["u"]))

        # Level 1: Inputs and belief
        ax = axes[0]
        ax.scatter(trials, self.history["u"], c="green", s=10, label="Input u", alpha=0.6)
        m1_prob = [sigmoid(m2) for m2 in self.history["m2"]]
        # m1_prob = self.history["m1"]
        ax.plot(trials, m1_prob, "r-", linewidth=2, label="Posterior E[x1=1] = s(m2)")
        if true_prob is not None:
            ax.plot(trials, true_prob, "k-", linewidth=1, label="True p(x1=1)")
        ax.set_ylabel("Level 1: x1")
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Level 2: Tendency
        ax = axes[1]
        ax.plot(trials, self.history["m2"], "r-", linewidth=2, label="Posterior E[x2] = m2")
        s2_array = np.array(self.history["s2"])
        m2_array = np.array(self.history["m2"])
        ax.fill_between(
            trials,
            m2_array - np.sqrt(s2_array),
            m2_array + np.sqrt(s2_array),
            alpha=0.2,
            color="red",
        )
        ax.set_ylabel("Level 2: x2 (tendency)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Level 3: Log-volatility
        ax = axes[2]
        ax.plot(trials, self.history["m3"], "b-", linewidth=2, label="Posterior E[x3] = m3")
        s3_array = np.array(self.history["s3"])
        s3_array[s3_array < 0] = 0
        m3_array = np.array(self.history["m3"])
        ax.fill_between(
            trials,
            m3_array - np.sqrt(s3_array),
            m3_array + np.sqrt(s3_array),
            alpha=0.2,
            color="blue",
        )
        ax.set_ylabel("Level 3: x3 (log-volatility)")
        ax.set_xlabel("Trial")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_reference_scenario():
    """Generate input sequence matching Figure 5."""
    np.random.seed(42)

    # Stage 1: 100 trials with p=0.5
    p1 = 0.5
    stage1 = np.random.rand(100) < p1
    true_prob1 = np.full(100, p1)

    # Stage 2: 120 trials with alternating high/low probability
    stage2 = []
    true_prob2 = []
    for i in range(6):
        p = 0.9 if i % 2 == 0 else 0.1
        stage2.extend(np.random.rand(20) < p)
        true_prob2.extend([p] * 20)

    # Stage 3: Repeat stage 1
    stage3 = np.random.rand(100) < p1
    true_prob3 = np.full(100, p1)

    inputs = np.concatenate([stage1, stage2, stage3]).astype(int)
    true_prob = np.concatenate([true_prob1, true_prob2, true_prob3])

    return inputs, true_prob


if __name__ == "__main__":
    # Create filter with reference parameters
    hgf = HGF(kappa=1.8, omega=-3, theta=0.5)

    # Generate inputs
    inputs, true_prob = generate_reference_scenario()

    hgf.run_simulation(inputs)

    # Plot results
    fig = hgf.plot_results(true_prob=true_prob)
    plt.savefig("hgf_reference_scenario.png", dpi=150, bbox_inches="tight")
