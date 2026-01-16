# Cognitive Oddballs

Final mini-project for the "Models of learning and decision making in changing environments" seminar with Prof. Weber at the University of Osnabrück in the winter term 2025/26.

# Short Description of the project

In this study project we focus on the performance of different architectures of cognitive models within differently structured volatile environments. As the central task we chose the Oddball task which we define in two versions: 1. random walk based version of the oddball task; 2. change point-based version of the oddball task. The comparison of differently structured environments and their effect on learning is motivated by Foucault et al. 2025. In their paper they posit that humans do not only learn what aspects change in a volatile environment but also how these aspects change. We are interested if their findings can be further validated in an oddball environment. Oddballs are essentially outliers that lie out of the assumed range of variance/noise. Can humans and cognitive models handle these outliers while still keeping up with changes in differently structured volatile environments? Furthermore can they learn how environments change?

We evaluate four models within these environments: Two versions of the Hierarchical Gaussian Filter (see Mathys et al. 2011, 2014; Weber et al. 2023) and two versions of change-point models (see Nassar et al. 2010, 2016, 2022).


# Development Setup

## Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Code Quality with Ruff

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

### Configuration

Ruff is configured in `pyproject.toml` or `ruff.toml` at the project root.
