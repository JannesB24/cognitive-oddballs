# 🧠 Cognitive Oddballs

Final mini-project for the "Models of learning and decision making in changing environments" seminar with Prof. Weber at the University of Osnabrück in the winter term 2025/26.

# 📋 Short Description of the Project

In this study project we are interested in the behavior and performance of two popular cognitive model types in differently structured volatile environments. Additionally to the comparison of performance we are interested in model identification based on behavioral outputs.  Previously this was investigated in diffusive and switching environments without outliers (Markovic and Kiebel 2016). We extend this investigation to diffusive and switching environments with outliers. Therefore we integrate the so called Oddball task in a diffusive and a switching environment. Oddballs are outliers of the assumed range of variance/noise. We define the two differently structured environments: 1. the diffusive random walk based environment with oddballs; 2. the switching change point-based environment with oddballs. The comparison of the effect of differently structured oddball environments on learning and behavior is inspired by Foucault et al. (2025). Similar to Markovic & Kiebel (2016) we implement two environments, versions of  the Hierarchical Gaussian Filter (see Mathys et al 2011, 2014) and a Changepoint Model (Nassar et al. 2016) and we compare them and do model recovery based on simulated behavioral experiments in the Oddball versions of the noisy environments. 

The novelty provided in our approach is that we extend the workflow provided by Markovic and Kiebel (2016) to the oddball paradigm. We do model comparison and model recovery in the oddball task with differently structured noisy/volatile environments. This is worth investigating because it not yet clear if models show different behavioral outputs in the oddball paradigms in differently structured noisy environments. If their behavior is identifiably different we should be able to infer the computational model given we have behavioral outputs of the model.

# Features
* Extensions of Change-point and Gaussian random walk environment to feature oddballs, incl. visualization
* Model implementations:
    * A Change-point model, as described in Nassar et al. (2016) + adaptions for comparability as described in Markovic & Kiebel (2016)
    * A Hierarchical Gaussian Filter (HGF), as described in Mathys et al. (2011)
    * A Generalized HGF, as described in Weber et al. (2023)
* Model evaluation including:
    * Parameter optimization using CMA-ES, as described in Marković & Kiebel (2016).
    * Parameter recovery
    * Model comparison using RMSE and Variational Free Energy as metrics, inspired by Marković & Kiebel (2016).
    * Model recovery using MLE-BIC and Bayesian Inference (Laplace Approximation), as described in Marković & Kiebel (2016).

# 🛠️ Development Setup

## 🐍 Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running Code

Due to the independence and long runtime of many of the components, environment visualization, parameter optimization, model evaluation, and model recovery can only be run separately.

To visualize the Change-point environment, run:

```bash
python cognitive_oddballs/environments/change_point_oddball.py
```
or 

```bash
python -m cognitive_oddballs.environments.change_point_oddball
```

For the Gaussian random walk environment, simply change the file name to reflect the environment. 
In the following, we will use only one syntax, but of course, you can use either

To perform parameter optimization, please set the amount of simulations (```n_simulations```) and trials (```n_trials```) you want to optimize for. To manage runtime, it is currently set to only 7 simulations à 100 trials. Running it takes around 4 hours, but the amount of simulations is not enough for a proper parameter optimization. To match Marković & Kiebel (2016), adjust to 1000 simulations à 100 trials. However, without paralellisation, expect at least a week of runtime per model type.
After setting ```n_simulations``` and ```n_trials``` in main, please run:

```bash
python -m cognitive_oddballs.pipeline.paramOpt
```

To perform model evaluation, again please set the amount of simulations (```n_simulations```) and trials (```n_trials```). With the given setting of 100 trials and 1000 simulations, expect a runtime of 10-15 minutes. If you want to use the optimized parameters gathered before, please update the parameters inside the ```models``` dictionary of ```experiment_changepoint```, and ```experiment_randomwalk```, respectively.
After adjusting model parameters, please run:

```bash
python -m cognitive_oddballs.pipeline.eval
```

To perform parameter and model recovery, please run:

```bash
python -m cognitive_oddballs.pipeline.recovery_optimized
```
The script runs a 2 x 2 x 2 x 2 design experiment and plots confusion matrices and summary statistics
    * two models: the Hierachical Gaussian Filter (Mathys 2011 version) and the Change-Point Model (Nassar et al. 2016 model
    with Variational Inference version a la Markovic & Kiebel 2016)
    * two environments: an random-walk environment with oddballs and a change-point environment with oddballs
    * two simulation lengths: 100 simulations with 100 trials & 100 simulations with 500 trials
    * two response noise levels: low response noise (r_sigma = 2) and high response noise (r_sigma = 10)

While many optimisations were applied in this script it still runs 5 to 14+ hours depending on the machine. Optimisations applied:
    * Numba JIT for inner HGF update loop
    * Numba JIT for inner CPM update loop
    * Vectorised belief arrays (no per-trial Python dicts)
    * Parallel grid search via joblib
    * Parallel across conditions via joblib
    * Coarse-to-fine grid refinement
    * Early termination of hopeless grid points
    * Analytical Hessian diagonal approximation
    * Pre-allocated arrays everywhere
    * Reduced Python object overhead


# Structure
* __environments__
    * __change_point_oddball.py__: Generate and visualize helicopter change-point environment with occasional oddball bag drops.
    * __changepoint_visualized.py__: Used for creating the plots that were part of the seminar presentation.
    * __random_walk_visualized.py__: Used for creating the plots that were part of the seminar presentation.
    * __random_walk_oddball.py__: Generate and visualize helicopter Gaussian random walk environment with occasional oddball bag drops.
    * __visualizer.py__: Utils to visualize and summarize given helicopter environment in terminal.
* __models__
    * __change_point_model_variational.py__: Change-point model with variational inference to allow comparibility to HGFs, as described in Marković & Kiebel (2016). Implements model interface.
    * __change_point_nassar_2016.py__: Normative Bayesian learning model from Nassar et al. (2016) for helicopter task.
    * __model.py__: Abstract class to define model interface. Enforces run method to ensure compatibility for evaluation, and various methods to allow for CMA-ES compatibility for parameter optimization.
    * __weber_model.py__: 3- to 5-node gHGF implementation for helicopter task using PyHGF. Implements model interface and includes various diagnostic functions.
    * __hgf__
        * __hgf2_gaussian.py__: 2-level HGF implementation, as defined in Mathys et al. (2011) Adapted for helicopter task and comparability to implement model interface.
        * __hgf3.py__: Alternate 3-level HGF implementation for helicopter task following Mathys et al. (2011, 2014) style. (???)
* __pipeline__
    * __eval.py__: Pipeline for evaluating given models on oddball task, using RSME and Variational Free Energy as metrics, following Marković & Kiebel (2016).
    * __paramOpt.py__: Pipeline to perform parameter optimization for given models on oddball task using CMA-ES, following Marković & Kiebel (2016).
    * __recovery_optimized.py__: script we used to produce the results of the model identification
        * runs a 2 x 2 x 2 x 2 design experiment (detailed in "running code" above) and plots confusion matrices and summary statistics
        * While many optimisations were applied (detailed in "running code" above) in this script it still runs 5 to 14+ hours depending on the machine. 
    * __results__: Contains results from model eval (.json and figures) and results from parameter optimization. cma_params_{environment}_cpm contains parameter results for only CPM after 500 simulations à 100 trials, cma_params_{environment} contains parameter results after 7 simulations à 100 trials, both conducted using CMA-ES.
* __supplementary_materials__
    * __testing_weber_model.py__: Various diagnostic functions for gHGF to test functioning and further development. Comment/uncomment as needed.


## ✨ Code Quality with Ruff

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and code formatting.

### Installation

```bash
pip install -r requirements-dev.txt
```

or directly via pip:

```bash
pip install ruff
```

### Usage

**Linting:**
```bash
ruff check .
```

**Formatting:**
```bash
ruff format .
```

**Auto-fix issues:**
```bash
ruff check --fix .
```

### ⚙️ Configuration

Ruff is configured in `pyproject.toml` or `ruff.toml` at the project root.