# model.py
"""e.g.
    # Parameterization:
    # theta[0] = hazard_rate (linear)
    # theta[1] = log_sigma_obs
    # theta[2] = log_sigma_mu

    def set_parameters_cma(model: ChangePointModelVariational, theta: np.ndarray) -> None:
        model.hazard_rate = theta[0]
        model.sigma_obs = np.exp(theta[1])
        model.sigma_mu = np.exp(theta[2])"""

# paramOpt.py
"""models = {
    "CPM": ChangePointModelVariational(mu0=250, sigma0=50, obs_noise=5, w1=0.5, w2=0.5, h=0.1),
    "gHGF": WeberModel(node4=True, node_4_type="volatility_parent", n4_p=3.0),
    "HGF": HGFPaper2Gaussian(
        eta=0.005, s=15.0**2, mu1_init=0.0, sig1_init=10.0, mu2_init=-4.0, sig2_init=1.0
    ),
}"""

"""print("Optimizing Change Point Model on Change Point Environments")
cma_optimization(ChangePointModelVariational(), cp_envs)
print("\nOptimizing HGF Model on Change Point Environments")
cma_optimization(HGFPaper2Gaussian(), cp_envs)
print("\nOptimizing Weber Model on Change Point Environments")
cma_optimization(WeberModel(), cp_envs)

print("\nOptimizing HGF Model on Random Walk Environments")
cma_optimization(ChangePointModelVariational(), rw_envs)
print("\nOptimizing HGF Model on Random Walk Environments")
cma_optimization(HGFPaper2Gaussian(), rw_envs)
print("\nOptimizing Weber Model on Random Walk Environments")   
cma_optimization(WeberModel(), rw_envs)"""
# TODO: How do I make this so that it works for the different model classes?
"""theta_best = es_result[0]  # best parameter vector
hazard_best = theta_best[0]
sigma_obs_best = np.exp(theta_best[1])
sigma_mu_best = np.exp(theta_best[2])

optimal_params[model_cls.__name__] = {
    "hazard_rate": hazard_best,
    "sigma_obs": sigma_obs_best,
    "sigma_mu": sigma_mu_best,
}"""
"""def make_objective(envs):
    def obj(theta):
        return objective_function_cma_theta(
            theta=theta,
            model_cls=ChangePointModelVariational,
            envs=envs,
        )
    return obj"""

# feed that into pyCMA-ES optimizer: param vector x
    # need to figure out which parameters to fit for each model 
    # need to define bounds for each model parameter
    # need objective function that takes in param vector x and returns average surprise
"""
def objective_function(model: Model, environment_generator: Callable, param_bounds: dict, n_envs: int = 1000, n_trials: int = 100) -> float:
    total_surprise = 0.0
    for _ in range(n_envs):
        observations = environment_generator(n_trials)
        model_output = model.run(observations)
        # possibly replace with variational free energy calculation, etc.
        surprise = -np.sum(np.log(model_output['predicted_likelihood'] + 1e-10))  # Avoid log(0)
        total_surprise += surprise
    return total_surprise / n_envs

def optimize_model_parameters(model: Model, environment_generator: Callable, param_bounds: dict, n_envs: int = 1000, n_trials: int = 100, max_iterations: int = 100) -> dict:
    def cma_objective_function(x):
        # Map x to model parameters
        param_dict = {key: x[i] for i, key in enumerate(param_bounds.keys())}
        # Set model parameters
        model.set_parameters(param_dict)
        # Evaluate objective function
        return objective_function(model, environment_generator, param_bounds, n_envs, n_trials)

    # Initial guess: midpoint of bounds
    x0 = [(bounds[0] + bounds[1]) / 2 for bounds in param_bounds.values()]
    # Bounds for CMA-ES
    lower_bounds = [bounds[0] for bounds in param_bounds.values()]
    upper_bounds = [bounds[1] for bounds in param_bounds.values()]

    # CMA-ES optimization
    es = cma.CMAEvolutionStrategy(x0, 0.5, {'bounds': [lower_bounds, upper_bounds], 'maxiter': max_iterations})
    es.optimize(cma_objective_function)

    # Get best parameters
    best_params = es.result.xbest
    optimized_param_dict = {key: best_params[i] for i, key in enumerate(param_bounds.keys())}
    return optimized_param_dict"""
# repeat until optimal parameters found
