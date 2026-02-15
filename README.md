# 🧠 Cognitive Oddballs

Final mini-project for the "Models of learning and decision making in changing environments" seminar with Prof. Weber at the University of Osnabrück in the winter term 2025/26.

# 📋 Short Description of the Project

In this study project we are interested in the behavior and performance of two popular cognitive model types in differently structured volatile environments. Additionally to the comparison of performance we are interested in model identification based on behavioral outputs.  Previously this was investigated in diffusive and switching environments without outliers (Markovic and Kiebel 2016). We extend this investigation to diffusive and switching environments with outliers. Therefore we integrate the so called Oddball task in a diffusive and a switching environment. Oddballs are outliers of the assumed range of variance/noise. We define the two differently structured environments: 1. the diffusive random walk based environment with oddballs; 2. the switching change point-based environment with oddballs. The comparison of the effect of differently structured oddball environments on learning and behavior is inspired by Foucault et al. (2025). Similar to Markovic & Kiebel (2016) we implement two environments, versions of  the Hierarchical Gaussian Filter (see Mathys et al 2011, 2014) and a Changepoint Model (Nassar et al. 2016) and we compare them and do model recovery based on simulated behavioral experiments in the Oddball versions of the noisy environments. 

The novelty provided in our approach is that we extend the workflow provided by Markovic and Kiebel (2016) to the oddball paradigm. We do model comparison and model recovery in the oddball task with differently structured noisy/volatile environments. This is worth investigating because it not yet clear if models show different behavioral outputs in the oddball paradigms in differently structured noisy environments. If their behavior is identifiably different we should be able to infer the computational model given we have behavioral outputs of the model.


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

To run the main script:

```bash
python cognitive_oddballs/main.py
```
or 

```bash
python -m cognitive_oddballs.main
```

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


# Structure
* __main.py__
* __utils.py__
## environments
* __change_point_oddball.py__: Generate and visualize helicopter change-point environment with occasional oddball bag drops.
* __changepoint_visualized.py__: ???
* __random_walk_visualized.py__: ???
* __random_walk_oddball.py__: Generate and visualize helicopter Gaussian random walk environment with occasional oddball bag drops.
* __visualizer.py__: Utils to visualize and summarize given helicopter environment in terminal.
## model_recovery
* __main_recov.py__: TBD
* __plots.py__: TBD
### data
### evaluation
### experiments
### inference
### models
## models
* __change_point_model_variational.py__: Change-point model with variational inference to allow comparibility to HGFs, as described in Marković & Kiebel (2016). Implements model interface.
* __change_point_nassar_2016.py__: Normative Bayesian learning model from Nassar et al. (2016) for helicopter task.
* __cpm_markovic_adjusted.py__: ???
* __model.py__: Abstract class to define model interface. Enforces run method to ensure compatibility for evaluation, and various methods to allow for CMA-ES compatibility for parameter optimization.
* __robust_ghgf.py__: Alternate 3-level gHGF implementation for helicopter task using PyHGF. (???)
* __weber_model.py__: 3- to 5-node gHGF implementation for helicopter task using PyHGF. Implements model interface and includes various diagnostic functions.
### hgf
* __hgf2_gaussian.py__: 2-level HGF implementation, as defined in Mathys et al. (2011) Adapted for helicopter task and comparability to implement model interface.
* __hgf3.py__: Alternate 3-level HGF implementation for helicopter task following Mathys et al. (2011, 2014) style. (???)
## pipeline
* __eval.py__: Pipeline for evaluating given models on oddball task, using RSME and Variational Free Energy as metrics, following Marković & Kiebel (2016).
* __paramOpt.py__: Pipeline to perform parameter optimization for given models on oddball task using CMA-ES, following Marković & Kiebel (2016).
* __recovery.py__: TBD
* __recovery2.py__: TBD
### results
## supplementary_materials
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