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
